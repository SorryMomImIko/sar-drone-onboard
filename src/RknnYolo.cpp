#include "RknnYolo.hpp"
#include "rknn_api.h"   // из RKNN runtime SDK (rknpu2/include) - см. README про версии
#include <fstream>
#include <stdexcept>
#include <algorithm>

// ВАЖНО: сигнатуры rknn_input/rknn_output ниже соответствуют классическому
// C-API rknn-toolkit2 (rknn_inputs_set/rknn_outputs_get). В некоторых версиях
// SDK для RK3588 есть более новый zero-copy API (rknn_create_mem/rknn_set_io_mem) -
// он быстрее (без лишней копии в NPU), но сигнатуры другие. Если у вас
// установлен именно такой SDK, сверьтесь с примерами в rknpu2/examples
// из вашей версии тулкита - код ниже придётся адаптировать под неё.

namespace {

struct Letterbox { float scale; int dx, dy; };

Letterbox letterbox(const cv::Mat& src, cv::Mat& dst, int S) {
    float r = std::min(S / (float)src.cols, S / (float)src.rows);
    int nw = std::lround(src.cols * r), nh = std::lround(src.rows * r);
    cv::Mat rs;
    cv::resize(src, rs, {nw, nh});
    dst = cv::Mat(S, S, CV_8UC3, cv::Scalar(114, 114, 114));
    int dx = (S - nw) / 2, dy = (S - nh) / 2;
    rs.copyTo(dst(cv::Rect(dx, dy, nw, nh)));
    return {r, dx, dy};
}

std::vector<char> readFile(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) throw std::runtime_error("не найден файл модели: " + path);
    size_t size = (size_t)f.tellg();
    f.seekg(0);
    std::vector<char> buf(size);
    f.read(buf.data(), (std::streamsize)size);
    return buf;
}

} // namespace

struct RknnYolo::Impl {
    rknn_context ctx = 0;
    int input_size;
    int num_classes;
    float conf_thresh, nms_thresh;
};

RknnYolo::RknnYolo(const std::string& model_path, int input_size,
                    int num_classes, float conf_thresh, float nms_thresh) {
    impl_ = new Impl();
    impl_->input_size  = input_size;
    impl_->num_classes = num_classes;
    impl_->conf_thresh  = conf_thresh;
    impl_->nms_thresh   = nms_thresh;

    auto model_data = readFile(model_path);
    int ret = rknn_init(&impl_->ctx, model_data.data(), (uint32_t)model_data.size(), 0, nullptr);
    if (ret != RKNN_SUCC) {
        delete impl_;
        throw std::runtime_error(
            "rknn_init провалился, код " + std::to_string(ret) +
            " - чаще всего это рассинхрон версии librknnrt.so и NPU-драйвера "
            "на плате (сверьте cat /sys/kernel/debug/rknpu/version с версией SDK)");
    }
}

RknnYolo::~RknnYolo() {
    if (impl_) {
        if (impl_->ctx) rknn_destroy(impl_->ctx);
        delete impl_;
    }
}

std::vector<Detection> RknnYolo::infer(const cv::Mat& frame) {
    const int S = impl_->input_size;

    cv::Mat padded;
    Letterbox lb = letterbox(frame, padded, S);

    // Модель ожидает RGB, если при rknn.config() на этапе конвертации
    // задавались mean/std в предположении RGB-входа (стандартно для Ultralytics).
    cv::Mat rgb;
    cv::cvtColor(padded, rgb, cv::COLOR_BGR2RGB);

    rknn_input inputs[1] = {};
    inputs[0].index = 0;
    inputs[0].type  = RKNN_TENSOR_UINT8;
    inputs[0].fmt   = RKNN_TENSOR_NHWC;
    inputs[0].size  = (uint32_t)(S * S * 3);
    inputs[0].buf   = rgb.data;

    rknn_inputs_set(impl_->ctx, 1, inputs);
    rknn_run(impl_->ctx, nullptr);

    // Модель собрана с РАЗДЕЛЁННЫМИ выходами (см. rknn/README_RKNN.md):
    //   выход 0: [1, 4, 8400] - cx, cy, w, h в пикселях входа 640
    //   выход 1: [1, nc, 8400] - вероятности классов, sigmoid уже внутри графа
    // Так сделано потому, что при одном общем тензоре квантизация int8 берёт
    // масштаб по рамкам (значения до 640) и вероятности 0..1 схлопываются в ноль.
    rknn_output outputs[2] = {};
    outputs[0].want_float = 1;
    outputs[1].want_float = 1;
    rknn_outputs_get(impl_->ctx, 2, outputs, nullptr);

    const int nc = impl_->num_classes;
    // 8400 анкоров стандартно для входа 640x640 (страйды 8/16/32: 80x80+40x40+20x20).
    // Если input_size другой, пересчитайте: sum((S/8)^2 + (S/16)^2 + (S/32)^2).
    const int A = (S / 8) * (S / 8) + (S / 16) * (S / 16) + (S / 32) * (S / 32);

    const float* bx = (const float*)outputs[0].buf;   // [4, A], по строкам
    const float* cl = (const float*)outputs[1].buf;   // [nc, A], по строкам

    std::vector<cv::Rect> boxes;
    std::vector<float> scores;
    std::vector<int> classes;

    for (int i = 0; i < A; ++i) {
        int best = 0;
        float conf = cl[i];
        for (int c = 1; c < nc; ++c) {
            float v = cl[(size_t)c * A + i];
            if (v > conf) { conf = v; best = c; }
        }
        if (conf < impl_->conf_thresh) continue;

        float cx = bx[i];
        float cy = bx[(size_t)1 * A + i];
        float w  = bx[(size_t)2 * A + i];
        float h  = bx[(size_t)3 * A + i];

        // обратное преобразование letterbox -> координаты исходного кадра
        float x0 = (cx - w * 0.5f - lb.dx) / lb.scale;
        float y0 = (cy - h * 0.5f - lb.dy) / lb.scale;

        boxes.emplace_back(cv::Point((int)x0, (int)y0),
                            cv::Size((int)(w / lb.scale), (int)(h / lb.scale)));
        scores.push_back(conf);
        classes.push_back(best);
    }

    std::vector<int> keep;
    cv::dnn::NMSBoxes(boxes, scores, impl_->conf_thresh, impl_->nms_thresh, keep);

    std::vector<Detection> result;
    result.reserve(keep.size());
    for (int idx : keep) {
        Detection d;
        d.class_id   = classes[idx];
        d.confidence = scores[idx];
        d.cx = boxes[idx].x + boxes[idx].width  / 2.0f;
        d.cy = boxes[idx].y + boxes[idx].height / 2.0f;
        d.w  = (float)boxes[idx].width;
        d.h  = (float)boxes[idx].height;
        result.push_back(d);
    }

    rknn_outputs_release(impl_->ctx, 2, outputs);
    return result;
}
