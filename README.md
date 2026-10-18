# Бортовая система поиска людей, задымлений и маркеров ArUco

Orange Pi 3B на дроне: камера → YOLOv8 на NPU и ArUco на процессоре →
при находке на землю по WiFi/TCP уходит кадр с обводкой и подписью
и координаты дрона.

## Как это работает

```
камера (CSI/USB)
   └─ GstCamera          кадр + момент съёмки (PTS минус задержка видеотракта)
       └─ два LatestSlot  свежий кадр каждому детектору, старые теряются
           ├─ RknnYolo    человек и задымление, NPU, ~56 мс, ~16 fps
           └─ ArucoFinder маркеры DICT_4X4_50, процессор, параллельно NPU
                 └─ FindingEmitter - общий путь для всех трёх видов находок:
                     FcTelemetry.pose_at_frame()  поза НА МОМЕНТ СЪЁМКИ кадра
                     GeoProject.pixel_to_ground() пиксель -> широта/долгота
                     FindingImage.encode_finding_frame()  кадр с контуром, ~11 КБ
                     ReliableLink.enqueue()       очередь на карте -> земля по TCP

полётник Foxeer F405 V2 (Betaflight)
   └─ USB /dev/ttyACM*
       └─ MspThread    опрос MSP: углы 25 Гц, высота 10 Гц, GPS 5 Гц
           └─ два кольцевых буфера с интерполяцией по времени
```

## Файлы

| Файл | Что делает |
|---|---|
| `src/main.cpp` | настройки и сборка всего вместе, два потока |
| `include/GstCamera.hpp` | камера через GStreamer, момент съёмки из PTS |
| `include/LatestSlot.hpp` | развязка потоков, только свежий кадр |
| `include/RknnYolo.hpp`, `src/RknnYolo.cpp` | YOLO на NPU |
| `include/ArucoFinder.hpp` | маркеры ArUco на процессоре, DICT_4X4_50 |
| `include/Finding.hpp` | общий путь находки: геопривязка, кадр, отправка |
| `include/FindingImage.hpp` | контур и подпись на кадре; JPEG |
| `include/FcTelemetry.hpp` | поза на момент кадра, честный отказ |
| `include/ReliableLink.hpp` | доставка с гарантией: очередь, ACK, переотправка |
| `include/geo/MspThread.hpp` | опрос Betaflight по MSP |
| `include/geo/TelemetryBuffers.hpp` | два буфера + интерполяция |
| `include/geo/GeoProject.hpp` | пиксель -> точка на земле |
| `include/geo/TelemetrySource.hpp`, `TelemetrySourceFactory.hpp` | граница протокола |
| `include/geo/MavlinkThread.hpp` | ArduPilot/PX4, по умолчанию не собирается |
| `tools/ground_server.py` | наземная станция: приём и подтверждение |
| `tools/sniff_serial.py` | определить протокол полётника по дампу |

Описания: `MSP_BETAFLIGHT.md` - телеметрия, `LINK.md` - канал на землю и кадры.

## Сборка на плате

```sh
sudo apt install build-essential cmake git pkg-config \
     libopencv-dev libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev \
     gstreamer1.0-plugins-good gstreamer1.0-libcamera

mkdir -p build && cd build && cmake .. && make -j4
mkdir -p ../finds ../queue
./sar_yolo_pipeline
```

Если `cv::aruco` не найдётся, поставить `libopencv-contrib-dev` (в OpenCV до
версии 4.7 детектор маркеров лежал в contrib; код поддерживает обе версии API).

Заголовки MAVLink **не нужны**: полётник на Betaflight. Если понадобится
ArduPilot - `cmake .. -DSAR_WITH_MAVLINK=ON`, предварительно склонировав
`c_library_v2` в `third_party/mavlink`.

## Что настроить перед первым запуском

В начале `src/main.cpp`:

| Константа | Сейчас | Чем проверяется |
|---|---|---|
| `FC_URI` | `/dev/ttyACM0` | `ls -l /dev/serial/by-id/`, брать путь оттуда |
| `GROUND_STATION_IP` | `192.168.1.100` | адрес ноутбука с `ground_server.py` |
| `CAMERA_HFOV_DEG` | 62 | даташит объектива; 5 % = 1.3 м на 80 м |
| `CAMERA_TILT_DEG` / `PAN_DEG` | 0, 0 | как камера стоит; 2° = 2.9 м на 80 м |
| `CAPTURE_LATENCY_US` | 120 мс | полёт по маршруту туда-обратно, этап 6 |
| `ARUCO_ENABLED` | вкл | словарь `DICT_4X4_50`, тот же, что в `model_detection/aruco` |
| `ARUCO_REPEAT_S` | 15 с | как часто повторно слать один и тот же маркер |

Последние три - **систематические**: по кадрам не усредняются.

## Наземная станция

```sh
python tools/ground_server.py --port 5700 --dir poisk
```

Запускать до взлёта: дрон подключается сам и сам переподключается после
обрывов. Пишет `event_NNNNNN_frame.jpg` и `events.jsonl`.

## Чего в системе нет

- **Трекера.** `CONFIRM_FRAMES` подряд - фильтр одиночных ложняков, а не
  сопровождение: два человека в кадре дадут две отдельные находки.
  Для ArUco трекер не нужен: маркеры различаются по номеру, и повторы
  гасятся задержкой `ARUCO_REPEAT_S`.
- **Учёта рельефа.** Высота барометрическая от точки взлёта (в Betaflight
  дальномера нет), перепад под пятном поиска в неё не входит.
- **Метрик точности детектора** на размеченном наборе.
- **Измеренной точности геопривязки в поле.** Все цифры - из моделирования
  и расчёта бюджета ошибок.
