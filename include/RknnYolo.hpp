#pragma once
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>

struct Detection {
    int class_id;
    float confidence;
    float cx, cy;   // центр рамки в пикселях ИСХОДНОГО кадра (не 640x640)
    float w, h;      // размер рамки в пикселях исходного кадра
};

// Обёртка над RKNN runtime для YOLOv8-подобной модели (выход [1, 4+nc, 8400]).
class RknnYolo {
public:
    // model_path   - путь к .rknn файлу
    // input_size   - сторона квадратного входа модели (обычно 640)
    // num_classes  - число классов, под которое обучена модель
    RknnYolo(const std::string& model_path, int input_size,
             int num_classes, float conf_thresh = 0.25f, float nms_thresh = 0.45f);
    ~RknnYolo();

    // frame - BGR-кадр произвольного размера (letterbox делается внутри)
    std::vector<Detection> infer(const cv::Mat& frame);

    RknnYolo(const RknnYolo&) = delete;
    RknnYolo& operator=(const RknnYolo&) = delete;

private:
    struct Impl;
    Impl* impl_;
};
