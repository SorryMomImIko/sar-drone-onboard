#pragma once
// ArucoFinder.hpp - поиск маркеров ArUco на процессоре.
//
// Работает ПАРАЛЛЕЛЬНО нейросети: пока NPU считает YOLO, процессор ищет
// маркеры. Это настоящая параллельность, а не чередование - и rknn, и
// детектор ArUco считают вне интерпретатора и друг друга не ждут.
//
// Словарь DICT_4X4_50 и уточнение углов до субпикселя - те же, что в
// model_detection/aruco/aruco_detect.py, чтобы результат на плате совпадал
// с тем, что получалось в Python.
//
// ПОЧЕМУ ПОДТВЕРЖДЕНИЙ НУЖНО МЕНЬШЕ, ЧЕМ ДЛЯ ЧЕЛОВЕКА. В коде маркера есть
// собственная проверка целостности: случайный квадрат на земле не пройдёт
// её и маркером не станет. Поэтому двух кадров подряд достаточно, тогда как
// для YOLO нужно пять.

#include <opencv2/opencv.hpp>

// В OpenCV 4.7 детектор переехал из contrib в основную сборку и сменил API.
// На плате может оказаться и то, и другое, поэтому поддержаны обе версии.
#if CV_VERSION_MAJOR > 4 || (CV_VERSION_MAJOR == 4 && CV_VERSION_MINOR >= 7)
  #include <opencv2/objdetect/aruco_detector.hpp>
  #define SAR_ARUCO_NEW_API 1
#else
  #include <opencv2/aruco.hpp>
  #define SAR_ARUCO_NEW_API 0
#endif

#include <vector>

namespace sar {

struct ArucoHit {
    int   id = -1;
    float cx = 0, cy = 0;          // центр маркера в пикселях исходного кадра
    float side_px = 0;             // средняя сторона - грубая мера дальности
    std::vector<cv::Point2f> corners;
};

class ArucoFinder {
public:
    explicit ArucoFinder(int dict_id = cv::aruco::DICT_4X4_50) {
#if SAR_ARUCO_NEW_API
        params_ = cv::aruco::DetectorParameters();
        params_.cornerRefinementMethod = cv::aruco::CORNER_REFINE_SUBPIX;
        detector_ = cv::aruco::ArucoDetector(
            cv::aruco::getPredefinedDictionary(dict_id), params_);
#else
        dict_ = cv::aruco::getPredefinedDictionary(dict_id);
        params_ = cv::aruco::DetectorParameters::create();
        params_->cornerRefinementMethod = cv::aruco::CORNER_REFINE_SUBPIX;
#endif
    }

    std::vector<ArucoHit> detect(const cv::Mat& frame) {
        std::vector<ArucoHit> out;
        if (frame.empty()) return out;

        // Детектору нужна полутоновая картинка. Делаем её сами: иначе
        // OpenCV сделает то же самое внутри, но на каждом вызове.
        cv::cvtColor(frame, gray_, cv::COLOR_BGR2GRAY);

        std::vector<std::vector<cv::Point2f>> corners;
        std::vector<int> ids;
#if SAR_ARUCO_NEW_API
        detector_.detectMarkers(gray_, corners, ids);
#else
        cv::aruco::detectMarkers(gray_, dict_, corners, ids, params_);
#endif
        out.reserve(ids.size());
        for (size_t i = 0; i < ids.size(); ++i) {
            if (corners[i].size() != 4) continue;
            ArucoHit h;
            h.id = ids[i];
            h.corners = corners[i];
            for (const auto& p : corners[i]) { h.cx += p.x; h.cy += p.y; }
            h.cx /= 4.f; h.cy /= 4.f;
            // Средняя длина стороны: по ней видно, насколько маркер мелкий,
            // а значит насколько шатким будет его центр.
            float per = 0;
            for (int k = 0; k < 4; ++k)
                per += float(cv::norm(corners[i][k] - corners[i][(k + 1) % 4]));
            h.side_px = per / 4.f;
            out.push_back(std::move(h));
        }
        return out;
    }

private:
    cv::Mat gray_;
#if SAR_ARUCO_NEW_API
    cv::aruco::DetectorParameters params_;
    cv::aruco::ArucoDetector detector_;
#else
    cv::Ptr<cv::aruco::Dictionary> dict_;
    cv::Ptr<cv::aruco::DetectorParameters> params_;
#endif
};

} // namespace sar
