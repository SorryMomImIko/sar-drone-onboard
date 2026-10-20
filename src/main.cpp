#include <atomic>
#include <cstdlib>
#include <chrono>
#include <deque>
#include <cstdio>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <map>
#include <thread>
#include <sys/stat.h>

#include <gst/gst.h>
#include <opencv2/opencv.hpp>

#include "ArucoFinder.hpp"
#include "FcTelemetry.hpp"
#include "Finding.hpp"
#include "GstCamera.hpp"
#include "LatestSlot.hpp"
#include "ReliableLink.hpp"
#include "RknnYolo.hpp"

// ======================= НАСТРОЙКИ - ПОМЕНЯТЬ ПОД СЕБЯ =======================

// CSI-камера на libcamera. Для USB/V4L2-камеры заменить на:
//   "v4l2src device=/dev/video0 ! video/x-raw,width=1280,height=720,framerate=30/1 "
//   "! queue max-size-buffers=1 leaky=downstream ! appsink name=sink"
// Если камера отдаёт MJPEG, перед appsink нужен "! jpegdec ! videoconvert".
static const std::string CAMERA_PIPELINE =
    "libcamerasrc ! video/x-raw,format=NV12,width=1280,height=720,framerate=30/1 "
    "! queue max-size-buffers=1 leaky=downstream "
    "! appsink name=sink";


// --- Детектор -----------------------------------------------------------------
static const std::string MODEL_PATH = "best_i8_split.rknn";
static const int   MODEL_INPUT_SIZE = 640;
static const int   NUM_CLASSES      = 3;        // person / car / smoke
static const float CONF_THRESHOLD   = 0.35f;
static const float NMS_THRESHOLD    = 0.45f;

// Что считаем находкой: человек и задымление. Машины (1) не цель поиска.
static const int TARGET_CLASS_IDS[] = {0, 2};
static const char* CLASS_LABELS[] = {"person", "car", "smoke"};

// Подряд идущих детекций до подтверждения. Для нейросети пять: одиночные
// ложные срабатывания отсеиваются. Для ArUco двух достаточно - в коде
// маркера есть собственная проверка целостности.
// ПОДТВЕРЖДЕНИЕ НАХОДКИ - ПО ОКНУ ВРЕМЕНИ, А НЕ ПО КАДРАМ ПОДРЯД.
// "Подряд" ломается на ровном месте: цель на секунду закрыл куст, детектор
// моргнул на одном кадре - счётчик обнулился, и человек под деревьями может
// не подтвердиться никогда, хотя виден в половине кадров. Окно это терпит.
// Три срабатывания за две секунды - это ~13 кадров при 6.5 fps: одиночный
// ложняк так не проходит, а настоящая цель набирает их с запасом.
static const int    CONFIRM_HITS     = 3;
static const double CONFIRM_WINDOW_S = 2.0;
// Та же находка не чаще раза в столько секунд. Без этого человек, попавший
// в кадр на минуту, уехал бы на землю полторы сотни раз: кадр 11 КБ, канал
// забился бы повторами одного и того же. Трекера здесь нет, поэтому "та же"
// - это "тот же класс", а не "тот же человек".
static const double FINDING_REPEAT_S = 5.0;
static const int ARUCO_CONFIRM_FRAMES = 2;

// --- ArUco --------------------------------------------------------------------
// Словарь тот же, что в model_detection/aruco - DICT_4X4_50.
// SAR_ARUCO=0 выключает поиск маркеров без пересборки
static const bool ARUCO_ENABLED =
    !(std::getenv("SAR_ARUCO") && std::string(std::getenv("SAR_ARUCO")) == "0");
// Повторно слать тот же маркер не чаще, чем раз в столько секунд: маркер
// лежит на месте и попадает в кадр десятками кадров подряд.
static const int  ARUCO_REPEAT_S = 15;

// --- Камера -------------------------------------------------------------------
// Угол обзора по горизонтали - из даташита объектива. Ошибка 5 % даёт 1.3 м
// на земле с высоты 80 м, и она СИСТЕМАТИЧЕСКАЯ - по кадрам не усреднится.
static const double CAMERA_HFOV_DEG = 62.0;
// Углы установки камеры относительно корпуса, градусы.
// 0, 0 = смотрит строго вниз, верх кадра - вперёд по полёту.
// Ошибка 2 градуса даёт 2.9 м на земле с высоты 80 м, тоже систематическая.
static const double CAMERA_TILT_DEG = 0.0;
static const double CAMERA_PAN_DEG  = 0.0;

// ЗАДЕРЖКА ВИДЕОТРАКТА - принятая константа, не измеренная. Это время от
// попадания света на сенсор до появления кадра в GStreamer: экспозиция,
// вычитывание, драйвер. CSI 30-60 мс, USB 60-150 мс, RTSP 150-400 мс.
// Цена ошибки: 100 мс = 1.5-2.2 м на земле с высоты 80 м.
// Признак неверной константы - ПОСТОЯННЫЙ СНОС меток вдоль направления полёта.
static const uint64_t CAPTURE_LATENCY_US = 120000;

// --- Полётник -----------------------------------------------------------------
//   "msp"     - Betaflight / INAV (Foxeer F405 V2 - это он)
//   "mavlink" - ArduPilot / PX4, собирается только с -DSAR_WITH_MAVLINK=ON
static const std::string FC_PROTOCOL = "msp";
// По USB полётник появляется как CDC-ACM: обычный последовательный порт,
// скорость значения не имеет. Номер в /dev/ttyACM0 зависит от порядка
// подключения, поэтому брать путь из /dev/serial/by-id/ - он привязан к
// серийному номеру устройства:   ls -l /dev/serial/by-id/
// Пустая строка = работать без полётника (находки уйдут без координат).
// Пустое значение = работать без полётника:  SAR_FC_URI= ./sar_yolo_pipeline
static const std::string FC_URI =
    std::getenv("SAR_FC_URI") ? std::getenv("SAR_FC_URI") : "/dev/ttyACM0";

// --- Земля --------------------------------------------------------------------
// Дрон подключается к станции САМ и переподключается после каждого обрыва,
// поэтому её адрес постоянный, а ground_server.py запущен до взлёта.
// Доставка с гарантией: находка лежит в QUEUE_DIR, пока земля не подтвердит.
// Адрес наземной станции. Переопределяется переменной окружения, чтобы не
// пересобирать программу ради смены IP:  SAR_GROUND_IP=192.168.3.72 ./sar_yolo_pipeline
static const std::string GROUND_STATION_IP   =
    std::getenv("SAR_GROUND_IP") ? std::getenv("SAR_GROUND_IP") : "192.168.3.72";
static const uint16_t    GROUND_STATION_PORT = 5700;
static const std::string QUEUE_DIR           = "queue";

// --- ТЕСТОВАЯ ПОЗА (наземные проверки) ----------------------------------------
// Включается ТОЛЬКО переменной окружения, по умолчанию выключена:
//     SAR_FAKE_POSE=1 ./sar_yolo_pipeline ...
//     SAR_FAKE_POSE=60.005388,30.373863,30   - свои широта, долгота, высота
// Нужна, чтобы проверить цепочку "находка -> координата -> земля" на столе:
// без фикса GPS и с барометром, не обнулённым взведением, геопривязка не
// отрабатывает, и проверять нечего.
// Умолчание - башня Политеха, высота 30 м.
static const double FAKE_LAT = 60.005388;
static const double FAKE_LON = 30.373863;
static const float  FAKE_ALT = 30.0f;

// На землю уходит кадр, уменьшенный до 800 пикселей (~11 КБ, с повторами
// 33 КБ). Полный кадр пишется на карту платы - для разбора после посадки.
static const bool        SEND_FINDING_FRAME = true;
static const bool        SEND_FINDING_CROP  = false;
static const bool        SAVE_FULL_FRAME    = true;
static const std::string FINDS_DIR          = "finds";

// ==============================================================================

static std::atomic<bool> g_running{true};

static const char* class_label(int class_id) {
    const int n = int(sizeof(CLASS_LABELS) / sizeof(CLASS_LABELS[0]));
    return (class_id >= 0 && class_id < n) ? CLASS_LABELS[class_id] : "obj";
}

static bool is_target(int class_id) {
    for (const int c : TARGET_CLASS_IDS)
        if (c == class_id) return true;
    return false;
}

// Поток поиска маркеров: идёт на процессоре, пока NPU занят нейросетью.
static void aruco_loop(LatestSlot<sar::StampedFrame>& slot,
                       sar::FindingEmitter& emitter) {
    sar::ArucoFinder finder;
    std::map<int, uint64_t> last_sent_us;     // номер маркера -> когда отправляли
    std::map<int, int>      hits;             // номер маркера -> кадров подряд

    sar::StampedFrame frame;
    while (g_running && slot.take(frame)) {
        const auto found = finder.detect(frame.img);

        std::map<int, bool> seen_now;
        for (const auto& h : found) {
            seen_now[h.id] = true;
            if (++hits[h.id] < ARUCO_CONFIRM_FRAMES) continue;

            // Маркер лежит на земле и попадает в кадр подряд десятки раз.
            // Шлём его не чаще раза в ARUCO_REPEAT_S, иначе забьём канал
            // одной и той же находкой.
            const uint64_t now = sar::now_us();
            auto it = last_sent_us.find(h.id);
            if (it != last_sent_us.end() &&
                now - it->second < uint64_t(ARUCO_REPEAT_S) * 1000000ull) continue;
            last_sent_us[h.id] = now;

            sar::FindingEvent ev;
            ev.label      = "aruco";
            ev.outline    = h.corners;
            ev.cx         = h.cx;
            ev.cy         = h.cy;
            ev.confidence = 1.f;      // код маркера проверяется, догадок нет
            ev.class_id   = -1;
            ev.aruco_id   = h.id;
            emitter.emit(frame, ev);
        }
        // Маркер пропал из кадра - счётчик подтверждений обнуляем.
        for (auto& kv : hits)
            if (!seen_now.count(kv.first)) kv.second = 0;
    }
}

int main(int argc, char** argv) {
    gst_init(&argc, &argv);

    // Источник можно передать аргументом - это позволяет проверить весь
    // пайплайн без камеры, прогнав папку с кадрами:
    //   ./sar_yolo_pipeline "multifilesrc location=kadry/%06d.jpg index=0 \
    //      caps=image/jpeg,framerate=10/1 ! jpegdec ! videoconvert \
    //      ! video/x-raw,format=NV12 ! appsink name=sink"
    const std::string pipeline = (argc > 1) ? argv[1] : CAMERA_PIPELINE;
    std::cout << "источник: " << pipeline << "\n";

    GstCamera cam;
    if (!cam.open(pipeline)) {
        std::cerr << "Не удалось открыть камеру\n";
        return 1;
    }

    RknnYolo yolo(MODEL_PATH, MODEL_INPUT_SIZE, NUM_CLASSES,
                  CONF_THRESHOLD, NMS_THRESHOLD);

    // Телеметрия с полётника: свой поток и два кольцевых буфера
    // (углы и координаты отдельно). См. FcTelemetry.hpp.
    sar::FcTelemetry telemetry;
    bool have_fc = false;
    // Тестовая поза: разбираем переменную окружения до открытия полётника.
    bool fake_pose = false;
    double fake_lat = FAKE_LAT, fake_lon = FAKE_LON;
    float  fake_alt = FAKE_ALT;
    if (const char* fp = std::getenv("SAR_FAKE_POSE")) {
        if (std::string(fp) != "0" && *fp) {
            fake_pose = true;
            double a = 0, b = 0, c = 0;
            if (std::sscanf(fp, "%lf,%lf,%lf", &a, &b, &c) == 3) {
                fake_lat = a; fake_lon = b; fake_alt = float(c);
            }
        }
    }
    if (FC_URI.empty()) {
        std::cout << "полётник: отключён настройкой, координат не будет\n";
    } else if (telemetry.open(FC_PROTOCOL, FC_URI)) {
        std::cout << "полётник: " << FC_URI << ", протокол " << telemetry.protocol()
                  << " - поток пошёл\n";
        have_fc = true;
    } else {
        // Не падаем: детекция работает и без координат, находку всё равно
        // надо показать оператору. Но предупреждаем громко.
        std::cerr << "полётник: НЕ УДАЛОСЬ открыть " << FC_URI << " по протоколу "
                  << FC_PROTOCOL << " - геопривязки не будет (проверь имя порта, "
                     "права: группа dialout, и что протокол реализован)\n";
    }

    if (fake_pose) {
        telemetry.set_fake_position(fake_lat, fake_lon, fake_alt);
        std::cerr << "\n*** ТЕСТОВАЯ ПОЗА: координаты подставные - "
                  << fake_lat << ", " << fake_lon << ", высота " << fake_alt
                  << " м ***\n*** В ПОЛЁТ С ЭТИМ НЕ ВЫПУСКАТЬ: метки находок"
                     " будут указывать не туда ***\n\n";
    }

    // Канал на землю: своя очередь, свой поток, подтверждения от земли.
    // Потоки детекции в него только кладут и никогда не ждут.
    sar::ReliableLink link(GROUND_STATION_IP, GROUND_STATION_PORT, QUEUE_DIR);
    link.start();
    std::cout << "наземная станция: " << GROUND_STATION_IP << ":" << GROUND_STATION_PORT
              << ", очередь в " << QUEUE_DIR << "/\n";
    if (SAVE_FULL_FRAME) ::mkdir(FINDS_DIR.c_str(), 0755);

    // Сдвиг монотонных часов к всемирному времени: метки времени кадров живут
    // на CLOCK_MONOTONIC, а на землю надо отдавать настоящее UTC. Считаем
    // разницу один раз - за полёт она уползёт на единицы миллисекунд.
    const int64_t mono_to_utc_us =
        int64_t(std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count())
        - int64_t(sar::now_us());

    sar::EmitterConfig ecfg;
    ecfg.hfov_deg = CAMERA_HFOV_DEG;   // параметры считаются от реального кадра
    ecfg.R_mount = sar::R_body_from_cam(CAMERA_TILT_DEG, CAMERA_PAN_DEG);
    ecfg.mono_to_utc_us = mono_to_utc_us;
    ecfg.finds_dir  = FINDS_DIR;
    ecfg.send_frame = SEND_FINDING_FRAME;
    ecfg.send_crop  = SEND_FINDING_CROP;
    ecfg.save_full  = SAVE_FULL_FRAME;
    sar::FindingEmitter emitter(telemetry, link, ecfg);

    // Два слота: нейросеть и маркеры берут кадры независимо и друг друга не
    // ждут. Кто не успел - пропускает кадр, а не задерживает второго.
    LatestSlot<sar::StampedFrame> yolo_slot, aruco_slot;

    std::thread capture_thread([&] {
        cv::Mat img;
        uint64_t pts_us = 0;
        while (g_running && cam.read(img, &pts_us)) {
            // PTS - момент, когда кадр попал в GStreamer; из него уже вычтено
            // ожидание в очередях. Вычитаем задержку сенсора и драйвера и
            // получаем момент самой съёмки.
            const uint64_t t = (pts_us != 0) ? pts_us : sar::now_us();
            sar::StampedFrame sf;
            sf.img  = img.clone();
            sf.t_us = (t > CAPTURE_LATENCY_US) ? t - CAPTURE_LATENCY_US : 0;
            if (ARUCO_ENABLED) aruco_slot.put(sf);
            yolo_slot.put(std::move(sf));
        }
    });

    std::thread aruco_thread;
    if (ARUCO_ENABLED)
        aruco_thread = std::thread(aruco_loop, std::ref(aruco_slot), std::ref(emitter));

    // Живая телеметрия на землю: раз в секунду, мимо очереди находок.
    // Нужна, чтобы оператор видел, где дрон, всё время, а не только в
    // моменты находок. Потеря отдельной посылки не важна - значение
    // через секунду всё равно сменится.
    std::thread live_thread([&] {
        // Посылка уходит ВСЕГДА, даже когда полётника нет или он молчит.
        // Иначе оператор видит одно и то же пустое место в трёх разных
        // случаях - дрон не на связи, полётник не подключён, GPS без фикса -
        // а это три разных решения на земле. Поля, которых нет, просто
        // отсутствуют в JSON; земля это разбирает.
        while (g_running) {
            std::ostringstream js;
            js << "{\"kind\":\"telemetry\""
               << ",\"timestamp_us\":"
               << uint64_t(int64_t(sar::now_us()) + mono_to_utc_us)
               << ",\"fc_open\":" << (have_fc ? "true" : "false");

            bool geo_ok = false;
            if (have_fc) {
                const auto s = telemetry.live_state();
                js << ",\"fc_link\":" << (s.fc_link ? "true" : "false")
                   << ",\"msp_messages\":" << s.messages;

                if (s.have_att)
                    js << std::setprecision(4)
                       << ",\"roll_deg\":"  << s.roll  * 57.29578
                       << ",\"pitch_deg\":" << s.pitch * 57.29578
                       << ",\"yaw_deg\":"   << s.yaw   * 57.29578;

                // Высота - СНАРУЖИ блока координат: в MSP она приходит своим
                // ответом и есть даже без GPS. Внутри блока она пропадала бы
                // ровно тогда, когда полезнее всего - на столе и при потере
                // фикса в воздухе.
                if (s.have_alt)
                    js << std::setprecision(4) << ",\"alt_m\":" << s.alt_m;

                if (s.fake) js << ",\"fake_pose\":true";

                if (s.have_pos) {
                    js << ",\"fix_type\":" << int(s.fix_type)
                       << ",\"sats\":" << int(s.sats);
                    // Координату отдаём только когда ей можно верить: точка
                    // с фиксом 0 - это нули посреди Атлантики, и на карте
                    // она хуже, чем отсутствие точки.
                    geo_ok = s.fix_type >= 3 && s.sats >= 6;
                    if (geo_ok)
                        js << std::setprecision(9)
                           << ",\"lat\":" << s.lat << ",\"lon\":" << s.lon;
                }
            }
            js << ",\"geo_valid\":" << (geo_ok ? "true" : "false") << "}";
            link.set_live(js.str());

            std::this_thread::sleep_for(std::chrono::milliseconds(500));
        }
    });

    // Времена срабатываний по каждому классу - скользящее окно.
    std::deque<uint64_t> hit_times[NUM_CLASSES];
    uint64_t last_emit_us[NUM_CLASSES]{};
    int frames = 0;
    double infer_total = 0.0;
    auto t_start = std::chrono::steady_clock::now();

    sar::StampedFrame frame;
    while (g_running && yolo_slot.take(frame)) {
        auto t0 = std::chrono::steady_clock::now();
        auto detections = yolo.infer(frame.img);
        infer_total += std::chrono::duration<double>(
            std::chrono::steady_clock::now() - t0).count();
        ++frames;

        if (frames % 30 == 0) {
            const double wall = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - t_start).count();
            std::cout << "кадров " << frames
                      << " | инференс " << (infer_total / frames * 1000.0) << " мс"
                      << " | сквозной темп " << (frames / wall) << " fps"
                      << " | детекций " << detections.size()
                      << " | земля " << (link.connected() ? "на связи" : "МОЛЧИТ")
                      << ", доставлено " << link.delivered()
                      << ", в очереди " << link.queued();
            if (have_fc) {
                std::cout << " | полётник " << (telemetry.link_ok() ? "на связи" : "МОЛЧИТ")
                          << ", фикс " << int(telemetry.fix_type())
                          << "/" << int(telemetry.sats()) << " спутн.";
            }
            std::cout << std::endl;
        }

        // Лучшая детекция ПО КАЖДОМУ искомому классу отдельно: человек и
        // задымление - разные находки, подтверждаться должны независимо.
        Detection best[NUM_CLASSES]{};
        float     best_conf[NUM_CLASSES]{};
        bool      found[NUM_CLASSES]{};
        for (const auto& d : detections) {
            if (d.class_id < 0 || d.class_id >= NUM_CLASSES) continue;
            if (!is_target(d.class_id)) continue;
            if (d.confidence > best_conf[d.class_id]) {
                best[d.class_id]      = d;
                best_conf[d.class_id] = d.confidence;
                found[d.class_id]     = true;
            }
        }

        for (const int cls : TARGET_CLASS_IDS) {
            // CONFIRM_HITS срабатываний за CONFIRM_WINDOW_S - фильтр
            // одиночных ложняков. Это НЕ трекер: если целей в кадре несколько
            // и они не связаны между кадрами, поверх нужен ByteTrack или SORT.
            const uint64_t now = sar::now_us();
            auto& hits = hit_times[cls];
            if (found[cls]) hits.push_back(now);
            // Выбрасываем всё, что вышло за окно.
            const uint64_t window_us = uint64_t(CONFIRM_WINDOW_S * 1e6);
            while (!hits.empty() && now - hits.front() > window_us) hits.pop_front();

            if (int(hits.size()) < CONFIRM_HITS) continue;
            if (last_emit_us[cls] &&
                now - last_emit_us[cls] < uint64_t(FINDING_REPEAT_S * 1e6)) continue;
            last_emit_us[cls] = now;
            hits.clear();

            const Detection& d = best[cls];
            sar::FindingEvent ev;
            ev.label      = class_label(d.class_id);
            ev.outline    = sar::box_outline(d);
            ev.cx         = d.cx;
            ev.cy         = d.cy;
            ev.confidence = best_conf[cls];
            ev.class_id   = d.class_id;
            emitter.emit(frame, ev);
        }
    }

    g_running = false;
    yolo_slot.stop();
    aruco_slot.stop();
    capture_thread.join();
    if (aruco_thread.joinable()) aruco_thread.join();
    if (live_thread.joinable()) live_thread.join();
    telemetry.stop();
    // Канал останавливаем последним: неотправленное останется в QUEUE_DIR
    // и уйдёт при следующем запуске.
    link.stop();
    return 0;
}
