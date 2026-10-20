#pragma once
// Finding.hpp - находка и её отправка на землю.
//
// Один и тот же путь для всех трёх видов находок: человек, задымление,
// маркер ArUco. Разница только в том, кто их нашёл - нейросеть на NPU или
// детектор маркеров на процессоре. Дальше всё одинаково:
//
//     поза на момент съёмки кадра -> пиксель в широту и долготу
//     -> кадр с контуром и подписью -> очередь на карте -> земля по TCP
//
// Вызывается из ДВУХ потоков одновременно, поэтому номер события атомарный,
// а всё состояние - внутри аргументов.

#include <atomic>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

#include <opencv2/opencv.hpp>

#include "FcTelemetry.hpp"
#include "FindingImage.hpp"
#include "GeoProject.hpp"
#include "ReliableLink.hpp"

namespace sar {

// Кадр вместе с моментом своей съёмки. Без метки времени геопривязка
// привязывала бы находку к телеметрии "на сейчас", а кадр снят раньше.
struct StampedFrame {
    cv::Mat  img;
    uint64_t t_us = 0;   // монотонные часы, момент попадания света на сенсор
};

struct FindingEvent {
    std::string label;        // person, smoke, aruco - ТОЛЬКО латиница, она идёт на кадр
    Outline     outline;      // контур: рамка YOLO или углы маркера
    float       cx = 0, cy = 0;   // точка, по которой считается геопозиция
    float       confidence = 1.f;
    int         class_id = -1;    // 0 person, 1 car, 2 smoke, -1 для ArUco
    int         aruco_id = -1;    // номер маркера, иначе -1
};

struct EmitterConfig {
    // Угол обзора, а НЕ готовые параметры камеры: фокусное в пикселях
    // зависит от размера кадра, а размер становится известен только когда
    // пришёл первый кадр. Зашитый размер здесь означал бы тихую ошибку
    // геопривязки при любом другом разрешении камеры.
    double           hfov_deg = 62.0;
    Mat3             R_mount;
    int64_t          mono_to_utc_us = 0;
    std::string      finds_dir = "finds";
    bool             send_frame = true;
    bool             send_crop  = false;
    bool             save_full  = true;
};

class FindingEmitter {
public:
    FindingEmitter(FcTelemetry& tel, ReliableLink& link, EmitterConfig cfg)
        : tel_(tel), link_(link), cfg_(std::move(cfg)) {}

    void emit(const StampedFrame& frame, const FindingEvent& ev) {
        const uint32_t event_id = ++counter_;

        // Телеметрия НА МОМЕНТ СЪЁМКИ ЭТОГО КАДРА, а не на "сейчас": между
        // съёмкой и этой строкой прошло несколько инференсов.
        // Параметры камеры - по РЕАЛЬНОМУ размеру этого кадра.
        const auto cam = CameraIntrinsics::from_fov(cfg_.hfov_deg,
                                                    frame.img.cols, frame.img.rows);
        Pose pose{};
        GeoPoint geo{};
        bool geo_valid = false;
        if (tel_.pose_at_frame(frame.t_us, pose)) {
            // Луч выше горизонта (дрон сильно накренён) - геопривязки нет.
            if (auto g = pixel_to_ground(ev.cx, ev.cy, pose, cam, cfg_.R_mount)) {
                geo = *g;
                geo_valid = true;
            }
        }

        // Время события = время КАДРА в UTC. Не время, когда досчитали:
        // на землю нужен момент, когда человека видели.
        const uint64_t ts_us = uint64_t(int64_t(frame.t_us) + cfg_.mono_to_utc_us);

        // МЕТКА ПЕРВОЙ. Координаты весят сотню байт и на узком канале уйдут
        // за доли секунды, кадр - секунды. Оператор узнаёт "где" сразу.
        const std::string payload = to_json(ev, geo, ts_us, event_id, geo_valid);
        if (!link_.enqueue(REC_META, event_id, payload))
            std::cerr << "событие " << event_id << ": МЕТКА НЕ ПОПАЛА В ОЧЕРЕДЬ\n";

        // КАДР НАХОДКИ - то, что увидел детектор, с контуром и подписью.
        char label[80];
        if (ev.aruco_id >= 0)
            std::snprintf(label, sizeof(label), "%s %d", ev.label.c_str(), ev.aruco_id);
        else
            std::snprintf(label, sizeof(label), "%s %.2f", ev.label.c_str(), ev.confidence);

        if (cfg_.send_frame) {
            const auto jpeg = encode_finding_frame(frame.img, ev.outline, label);
            if (!jpeg.empty()) link_.enqueue(REC_FRAME, event_id, jpeg);
            else std::cerr << "событие " << event_id << ": кадр не закодировался\n";
        }
        if (cfg_.send_crop) {
            const auto jpeg = encode_finding_crop(frame.img, ev.outline, label);
            if (!jpeg.empty()) link_.enqueue(REC_CROP, event_id, jpeg);
        }
        if (cfg_.save_full)
            save_full_frame(frame.img, ev.outline, ev.label, cfg_.finds_dir, event_id);

        std::ostringstream line;
        line << (geo_valid ? "Найдено: " : "Найдено БЕЗ КООРДИНАТ: ") << payload
             << (link_.connected() ? " | в очереди: " : " | СВЯЗИ НЕТ, в очереди: ")
             << link_.queued() << "\n";
        std::cout << line.str() << std::flush;   // одной записью: пишут два потока
    }

private:
    std::string to_json(const FindingEvent& ev, const GeoPoint& g,
                        uint64_t ts_us, uint32_t event_id, bool geo_valid) const {
        std::ostringstream ss;
        ss << "{"
           << "\"event_id\":" << event_id << ","
           << "\"kind\":\"" << ev.label << "\","
           << "\"class_id\":" << ev.class_id << ","
           << "\"aruco_id\":" << ev.aruco_id << ","
           << "\"timestamp_us\":" << ts_us << ","
           << "\"confidence\":" << ev.confidence << ","
           << "\"geo_valid\":" << (geo_valid ? "true" : "false") << ","
           << "\"fix_type\":" << int(tel_.fix_type()) << ","
           << "\"sats\":" << int(tel_.sats());
        // Пометка тестовой позы: координата подставная, это должно быть
        // видно на земле, а не только в консоли борта.
        if (tel_.fake_pose()) ss << ",\"fake_pose\":true";
        // Координаты шлём ТОЛЬКО когда им можно верить. Иначе на земле
        // появится метка в случайном месте, и её поедут проверять.
        if (geo_valid)
            ss << std::setprecision(9)
               << ",\"lat\":" << g.lat << ",\"lon\":" << g.lon;
        ss << "}";
        return ss.str();
    }

    FcTelemetry&  tel_;
    ReliableLink& link_;
    EmitterConfig cfg_;
    std::atomic<uint32_t> counter_{0};
};

} // namespace sar
