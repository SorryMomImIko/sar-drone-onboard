#pragma once
#include <gst/gst.h>
#include <gst/app/gstappsink.h>
#include <gst/video/video.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>

// Обёртка над GStreamer appsink с забором кадра без лишней копии
// (одна конверсия NV12->BGR вместо двух). pipeline-строка должна
// заканчиваться на "appsink name=sink" и отдавать формат NV12.
//
// Примеры источников:
//   CSI-камера на libcamera:  libcamerasrc ! video/x-raw,format=NV12,...
//   USB/V4L2-камера:          v4l2src device=/dev/video0 ! video/x-raw,...
// Если камера отдаёт MJPEG, перед appsink нужен ! jpegdec ! videoconvert
// и логика read() ниже (которая ожидает сырой NV12) не подойдёт как есть.
class GstCamera {
public:
    bool open(const std::string& pipeline_desc) {
        GError* err = nullptr;
        pipeline_ = gst_parse_launch(pipeline_desc.c_str(), &err);
        if (!pipeline_) {
            std::cerr << "gst_parse_launch: " << (err ? err->message : "unknown error") << "\n";
            if (err) g_error_free(err);
            return false;
        }
        sink_ = gst_bin_get_by_name(GST_BIN(pipeline_), "sink");
        if (!sink_) {
            std::cerr << "В пайплайне не найден элемент appsink с name=sink\n";
            return false;
        }
        gst_app_sink_set_max_buffers(GST_APP_SINK(sink_), 1);
        gst_app_sink_set_drop(GST_APP_SINK(sink_), TRUE);
        gst_base_sink_set_sync(GST_BASE_SINK(sink_), FALSE);

        return gst_element_set_state(pipeline_, GST_STATE_PLAYING) != GST_STATE_CHANGE_FAILURE;
    }

    // out — свежий BGR-кадр. Возвращает false при ошибке/остановке потока.
    //
    // t_us (необязательный) — МОМЕНТ СЪЁМКИ кадра в микросекундах по тем же
    // монотонным часам (CLOCK_MONOTONIC), по которым живёт телеметрия.
    // Берётся как base_time + PTS буфера, то есть из него уже вычтено время,
    // которое кадр простоял в очередях GStreamer. Задержку самого сенсора и
    // драйвера PTS не содержит — её вычитает вызывающий (CAPTURE_LATENCY_US).
    // Если PTS недоступен (некоторые источники его не ставят), пишет 0 —
    // вызывающий должен подставить текущее время.
    bool read(cv::Mat& out, uint64_t* t_us = nullptr) {
        GstSample* sample = gst_app_sink_pull_sample(GST_APP_SINK(sink_));
        if (!sample) return false;

        GstBuffer* buf  = gst_sample_get_buffer(sample);
        GstCaps*   caps = gst_sample_get_caps(sample);

        if (t_us) {
            // GstSystemClock по умолчанию монотонный, в наносекундах —
            // та же шкала, что clock_gettime(CLOCK_MONOTONIC).
            const GstClockTime pts  = GST_BUFFER_PTS(buf);
            const GstClockTime base = gst_element_get_base_time(pipeline_);
            *t_us = (GST_CLOCK_TIME_IS_VALID(pts) && GST_CLOCK_TIME_IS_VALID(base))
                    ? (base + pts) / 1000ull : 0ull;
        }

        GstVideoInfo info;
        gst_video_info_from_caps(&info, caps);

        // gst_video_frame_map (не gst_buffer_map!) правильно учитывает
        // stride/padding, который GStreamer почти всегда добавляет.
        GstVideoFrame vf;
        if (!gst_video_frame_map(&vf, &info, buf, GST_MAP_READ)) {
            gst_sample_unref(sample);
            return false;
        }

        int w = GST_VIDEO_FRAME_WIDTH(&vf);
        int h = GST_VIDEO_FRAME_HEIGHT(&vf);
        int stride = GST_VIDEO_FRAME_PLANE_STRIDE(&vf, 0);
        uint8_t* y = (uint8_t*)GST_VIDEO_FRAME_PLANE_DATA(&vf, 0);

        // NV12 в памяти: Y-плоскость (h строк), затем интерливед UV (h/2 строк).
        cv::Mat nv12(h * 3 / 2, w, CV_8UC1, y, stride);
        cv::cvtColor(nv12, out, cv::COLOR_YUV2BGR_NV12); // единственная копия

        gst_video_frame_unmap(&vf);
        gst_sample_unref(sample);
        return true;
    }

    ~GstCamera() {
        if (pipeline_) {
            gst_element_set_state(pipeline_, GST_STATE_NULL);
            gst_object_unref(pipeline_);
        }
    }

private:
    GstElement* pipeline_ = nullptr;
    GstElement* sink_ = nullptr;
};
