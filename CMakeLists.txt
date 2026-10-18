cmake_minimum_required(VERSION 3.16)
project(sar_yolo_pipeline CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(PkgConfig REQUIRED)

# На образах Rockchip gstreamer-video-1.0.pc требует librga (2D-ускоритель RK).
# Если её dev-пакета нет, pkg-config падает - тогда берём только два модуля,
# а gstvideo подключаем напрямую: заголовки те же, из plugins-base-dev.
pkg_check_modules(GSTVIDEO IMPORTED_TARGET gstreamer-video-1.0)
if(GSTVIDEO_FOUND)
    pkg_check_modules(GST REQUIRED IMPORTED_TARGET
        gstreamer-1.0 gstreamer-app-1.0 gstreamer-video-1.0)
    set(GST_EXTRA_LIBS "")
else()
    message(WARNING "gstreamer-video-1.0.pc недоступен (обычно нет librga-dev); линкуем gstvideo-1.0 напрямую")
    pkg_check_modules(GST REQUIRED IMPORTED_TARGET
        gstreamer-1.0 gstreamer-app-1.0)
    set(GST_EXTRA_LIBS gstvideo-1.0)
endif()
find_package(OpenCV REQUIRED)

# Путь к распакованному RKNN runtime SDK (репозиторий rknn-toolkit2,
# папка rknpu2) - поменять под то, куда его положили на плате.
set(RKNN_ROOT "${CMAKE_SOURCE_DIR}/third_party/rknpu2" CACHE PATH "Путь к RKNN runtime SDK")

add_executable(sar_yolo_pipeline
    src/main.cpp
    src/RknnYolo.cpp
)

# Полётник на этом дроне - Betaflight, протокол MSP: заголовков не требует,
# всё в include/geo/MspThread.hpp.
#
# Поддержка MAVLink (ArduPilot/PX4) выключена по умолчанию, чтобы на плате не
# держать лишнего. Если понадобится:
#   cd third_party
#   git clone --depth 1 https://github.com/mavlink/c_library_v2.git mavlink
#   cmake .. -DSAR_WITH_MAVLINK=ON
option(SAR_WITH_MAVLINK "Поддержка полётников на MAVLink (ArduPilot/PX4)" OFF)
if(SAR_WITH_MAVLINK)
    set(MAVLINK_ROOT "${CMAKE_SOURCE_DIR}/third_party/mavlink" CACHE PATH "Путь к mavlink/c_library_v2")
    if(NOT EXISTS "${MAVLINK_ROOT}/common/mavlink.h")
        message(FATAL_ERROR
            "SAR_WITH_MAVLINK=ON, но заголовков нет в ${MAVLINK_ROOT}\n"
            "  cd third_party && git clone --depth 1 https://github.com/mavlink/c_library_v2.git mavlink")
    endif()
    target_compile_definitions(sar_yolo_pipeline PRIVATE SAR_WITH_MAVLINK)
endif()


find_package(Threads REQUIRED)   # потоки телеметрии и отправки на землю

target_include_directories(sar_yolo_pipeline PRIVATE
    include
    include/geo
    ${RKNN_ROOT}/include
)
if(SAR_WITH_MAVLINK)
    target_include_directories(sar_yolo_pipeline PRIVATE ${MAVLINK_ROOT} ${MAVLINK_ROOT}/common)
endif()

target_link_libraries(sar_yolo_pipeline PRIVATE
    PkgConfig::GST
    ${GST_EXTRA_LIBS}
    ${OpenCV_LIBS}
    ${RKNN_ROOT}/lib/librknnrt.so
    Threads::Threads
)
