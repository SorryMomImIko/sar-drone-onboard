#pragma once
// FcTelemetry.hpp - телеметрия с полётника для пайплайна.
//
// Держит два кольцевых буфера и поток опроса/чтения (какой именно протокол -
// решает фабрика). Наружу отдаёт позу НА МОМЕНТ СЪЁМКИ КАДРА.
//
// Разница между "на момент кадра" и "последнее, что пришло" по
// моделированию: 1.5 м против 3.5 м медианной ошибки на земле с высоты 80 м.

#include "TelemetryBuffers.hpp"        // sar::AttitudeBuffer, PositionBuffer, Pose
#include "TelemetrySourceFactory.hpp"  // выбор протокола

#include <memory>
#include <string>

namespace sar {

class FcTelemetry {
public:
    ~FcTelemetry() { stop(); }

    // protocol: "msp" (Betaflight / INAV) или "mavlink" (ArduPilot / PX4)
    // uri:      "/dev/ttyACM0", лучше полный путь из /dev/serial/by-id/
    bool open(const std::string& protocol, const std::string& uri) {
        src_ = make_telemetry_source(protocol, att_, pos_);
        if (!src_) return false;                 // протокол не реализован
        if (!src_->open(uri)) return false;
        src_->start();
        return true;
    }

    void stop() { if (src_) src_->stop(); }

    // Поза на момент съёмки кадра. false = честный отказ:
    // дыра в телеметрии, нет фикса GPS или высота неизвестна.
    bool pose_at_frame(uint64_t t_us, Pose& out) {
        auto p = sar::pose_at(att_, pos_, t_us);
        if (!p) return false;

        fix_  = p->fix_type;
        sats_ = p->sats;
        if (p->fix_type < 3) return false;           // фикса нет - координате не верим
        if (!(p->alt_for_geo() > 1.0f)) return false; // высота неизвестна или дрон на земле

        out = *p;
        return true;
    }

    // --- что показывать в консоли
    bool        link_ok()      const { return src_ && src_->link_ok(); }
    bool        clock_locked() const { return src_ && src_->clock_locked(); }
    uint64_t    messages()     const { return src_ ? src_->messages() : 0; }
    const char* protocol()     const { return src_ ? src_->name() : "нет"; }
    uint8_t     fix_type()     const { return fix_; }
    uint8_t     sats()         const { return sats_; }

private:
    AttitudeBuffer att_;
    PositionBuffer pos_;
    std::unique_ptr<ITelemetrySource> src_;
    uint8_t fix_ = 0, sats_ = 0;
};

} // namespace sar
