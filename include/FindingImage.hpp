#pragma once
// FindingImage.hpp - кадр находки: результат YOLO, нарисованный на картинке.
//
// На землю уходит ИМЕННО ТОТ КАДР, на котором сработал детектор, с
// нарисованной рамкой, названием класса и уверенностью. Оператор должен
// видеть то же, что увидела сеть, иначе проверить находку он не может:
// координата без картинки - это просто точка, по которой надо кого-то послать.
//
// Полный кадр 1280x720 при качестве 72 - это 63 КБ, 54 куска, с тремя
// повторами 188 КБ на одну находку. Это много, поэтому кадр уменьшается.
// Замеры (текстура травы с фигурой, куски по 1200 байт):
//
//   ширина  качество   байт  кусков  x3 повтора
//     1280        65  49393      42     145 КБ
//      960        65  17541      15      51 КБ
//      800        65  10735       9      31 КБ   <- принято
//      640        65   6455       6      19 КБ
//      480        65   3667       4      11 КБ
//
// 800 пикселей: человек, занимавший в исходном кадре 60 пикселей по высоте,
// остаётся 37-пиксельной фигурой - различим. Ниже 640 начинает пропадать
// именно то, ради чего кадр и отправляется.
//
// Полный кадр в исходном размере при этом сохраняется на карту платы: канала
// он не занимает, а после посадки по нему разбирают ложные срабатывания.
//
// ПОДПИСИ ТОЛЬКО ЛАТИНИЦЕЙ: putText из OpenCV не умеет кириллицу и нарисует
// вместо неё вопросительные знаки.

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

#include "RknnYolo.hpp"      // struct Detection

namespace sar {

// Тип картинки в заголовке куска (см. ImageTx.hpp)
enum : uint8_t { IMG_FRAME = 1, IMG_CROP = 2 };

struct FindingImageParams {
    int   frame_width = 800;   // ширина уменьшенного кадра для отправки
    int   quality     = 65;    // качество JPEG для отправки
    int   full_quality = 85;   // качество полного кадра на карту платы
    float crop_margin = 2.0f;  // во сколько раз расширить рамку для вырезки
    int   crop_side   = 256;   // сторона вырезки
    int   crop_quality = 72;
};

// Контур находки: четыре точки. Для YOLO это углы рамки, для ArUco -
// углы самого маркера, поэтому рисование одно на оба случая.
using Outline = std::vector<cv::Point2f>;

inline Outline box_outline(const Detection& d) {
    const float x0 = d.cx - d.w / 2, y0 = d.cy - d.h / 2;
    const float x1 = d.cx + d.w / 2, y1 = d.cy + d.h / 2;
    return { {x0, y0}, {x1, y0}, {x1, y1}, {x0, y1} };
}

// Рисует результат на копии кадра: контур и подпись.
// text - только латиница: putText из OpenCV не умеет кириллицу.
inline cv::Mat annotate(const cv::Mat& frame, const Outline& outline,
                        const std::string& text_s) {
    cv::Mat vis = frame.clone();
    const cv::Rect full(0, 0, vis.cols, vis.rows);
    const cv::Scalar green(0, 255, 0);

    std::vector<cv::Point> pts;
    pts.reserve(outline.size());
    for (const auto& p : outline) pts.emplace_back(cvRound(p.x), cvRound(p.y));
    // Рисуем линиями, а не polylines: тот в разных версиях OpenCV трактует
    // один вектор точек по-разному и может упасть в работе, а не при сборке.
    if (pts.size() >= 3)
        for (size_t i = 0; i < pts.size(); ++i)
            cv::line(vis, pts[i], pts[(i + 1) % pts.size()], green, 2);

    // Подпись ставим у верхнего левого угла контура.
    cv::Rect box = cv::boundingRect(pts) & full;
    char text[80];
    std::snprintf(text, sizeof(text), "%s", text_s.c_str());

    // Подпись над рамкой, а если рамка у верхнего края - под ней,
    // иначе текст уезжает за кадр и его не видно.
    int base = 0;
    const cv::Size ts = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.6, 2, &base);
    const int ty = (box.y - 6 > ts.height) ? box.y - 6 : box.y + box.height + ts.height + 6;
    // Тёмная подложка: на светлом фоне зелёный текст иначе не читается.
    cv::rectangle(vis, cv::Rect(box.x, ty - ts.height - 4, ts.width + 6, ts.height + 8) & full,
                  cv::Scalar(0, 0, 0), cv::FILLED);
    cv::putText(vis, text, cv::Point(box.x + 3, ty),
                cv::FONT_HERSHEY_SIMPLEX, 0.6, green, 2);
    return vis;
}

// Кадр находки для отправки: результат YOLO, уменьшенный и сжатый.
// Пустой результат = кодирование не удалось (записать в лог, не падать).
inline std::vector<uint8_t> encode_finding_frame(const cv::Mat& frame,
                                                 const Outline& outline,
                                                 const std::string& label,
                                                 const FindingImageParams& p = {}) {
    if (frame.empty()) return {};

    cv::Mat vis = annotate(frame, outline, label);
    if (vis.cols > p.frame_width) {
        const double k = double(p.frame_width) / vis.cols;
        // INTER_AREA - единственная интерполяция, которая при уменьшении не
        // съедает тонкие детали; рамка толщиной 2 пикселя после неё видна.
        cv::resize(vis, vis, cv::Size(), k, k, cv::INTER_AREA);
    }

    std::vector<uint8_t> jpeg;
    const std::vector<int> params{cv::IMWRITE_JPEG_QUALITY, p.quality};
    if (!cv::imencode(".jpg", vis, jpeg, params)) return {};
    return jpeg;
}

// Вырезка вокруг находки, тоже с рамкой - для совсем узкого канала
// (~6 КБ против 10 КБ) или как дополнение к кадру: деталь крупным планом.
inline std::vector<uint8_t> encode_finding_crop(const cv::Mat& frame,
                                                const Outline& outline,
                                                const std::string& label,
                                                const FindingImageParams& p = {}) {
    if (frame.empty()) return {};

    cv::Mat vis = annotate(frame, outline, label);

    std::vector<cv::Point> pts;
    for (const auto& q : outline) pts.emplace_back(cvRound(q.x), cvRound(q.y));
    const cv::Rect bb = cv::boundingRect(pts);
    const float ccx = bb.x + bb.width / 2.f, ccy = bb.y + bb.height / 2.f;

    float side = float(std::max(bb.width, bb.height)) * p.crop_margin;
    side = std::max(side, float(p.crop_side) / 2.f);
    side = std::min(side, float(std::min(vis.cols, vis.rows)));

    const int half = int(side / 2.f);
    int w = half * 2, h = half * 2;
    if (w <= 0 || h <= 0) return {};
    // Центрируем на цели и сдвигаем внутрь кадра, если вылезли за край.
    // Сдвиг, а не обрезка: иначе у цели у края кадра пропадает полкартинки.
    const int x = std::max(0, std::min(int(ccx) - half, vis.cols - w));
    const int y = std::max(0, std::min(int(ccy) - half, vis.rows - h));

    cv::Mat crop = vis(cv::Rect(x, y, w, h)).clone();
    if (std::max(crop.cols, crop.rows) > p.crop_side) {
        const double k = double(p.crop_side) / std::max(crop.cols, crop.rows);
        cv::resize(crop, crop, cv::Size(), k, k, cv::INTER_AREA);
    }

    std::vector<uint8_t> jpeg;
    const std::vector<int> params{cv::IMWRITE_JPEG_QUALITY, p.crop_quality};
    if (!cv::imencode(".jpg", crop, jpeg, params)) return {};
    return jpeg;
}

// Полный кадр в исходном размере - на карту платы, для разбора после посадки.
// Возвращает путь или пустую строку.
inline std::string save_full_frame(const cv::Mat& frame, const Outline& outline,
                                   const std::string& label,
                                   const std::string& dir, uint32_t event_id,
                                   const FindingImageParams& p = {}) {
    if (frame.empty()) return {};

    char name[160];
    std::snprintf(name, sizeof(name), "%s/find_%06u_%s.jpg",
                  dir.c_str(), event_id, label.c_str());

    const std::vector<int> params{cv::IMWRITE_JPEG_QUALITY, p.full_quality};
    if (!cv::imwrite(name, annotate(frame, outline, label), params)) return {};
    return name;
}

} // namespace sar
