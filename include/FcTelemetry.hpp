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

    // ---------------- ТЕСТОВАЯ ПОЗА ----------------
    // Подменяет координаты и высоту ради наземных проверок: без GPS и без
    // полёта геопривязка иначе не отрабатывает вовсе, и проверить цепочку
    // "находка -> координата -> земля" негде.
    //
    // УГЛЫ НЕ ПОДМЕНЯЮТСЯ. Они настоящие, с полётника: наклони плату - метка
    // на земле поедет. Если подменить и их, проверка перестанет что-либо
    // проверять и превратится в печать заранее известного числа.
    //
    // Режим виден снаружи: в каждой метке и в телеметрии стоит "fake_pose",
    // земля показывает это крупно. Иначе однажды дрон улетит с подставной
    // координатой, и спасатели поедут к башне Политеха вместо леса.
    void set_fake_position(double lat, double lon, float alt_m) {
        fake_ = true; flat_ = lat; flon_ = lon; falt_ = alt_m;
    }
    bool fake_pose() const { return fake_; }

    // Поза на момент съёмки кадра. false = честный отказ:
    // дыра в телеметрии, нет фикса GPS или высота неизвестна.
    bool pose_at_frame(uint64_t t_us, Pose& out) {
        if (fake_) return fake_pose_at(t_us, out);
        auto p = sar::pose_at(att_, pos_, t_us);
        if (!p) return false;

        fix_  = p->fix_type;
        sats_ = p->sats;
        if (p->fix_type < 3) return false;           // фикса нет - координате не верим
        if (!(p->alt_for_geo() > 1.0f)) return false; // высота неизвестна или дрон на земле

        out = *p;
        return true;
    }

    // Состояние для ЖИВОЙ телеметрии на землю.
    //
    // ЗДЕСЬ НЕЛЬЗЯ ИСПОЛЬЗОВАТЬ pose_at: он требует ОБА буфера сразу и
    // отказывает целиком, если пуст любой. Для находки это правильно -
    // без координаты находка бессмысленна. Для живой телеметрии это
    // ошибка: на столе и в помещении GPS не отвечает (а если в прошивке
    // нет USE_GPS, то и вовсе отвечает "не знаю такой команды"), буфер
    // координат остаётся пустым - и оператор не видит НИЧЕГО, хотя углы
    // и связь с полётником есть. Поэтому буферы читаются независимо.
    //
    // latest(), а не interpolate(): интерполяция требует двух записей, а
    // показать оператору первую же пришедшую посылку важнее точности
    // индикатора на 0.04 с.
    struct LiveState {
        bool     fc_link   = false;   // полётник отвечает по MSP
        uint64_t messages  = 0;
        bool     have_att  = false;
        bool     have_pos  = false;
        float    roll = 0.f, pitch = 0.f, yaw = 0.f;
        uint64_t att_t_us  = 0;
        double   lat = 0.0, lon = 0.0;
        float    alt_m     = 0.f;
        bool     have_alt  = false;
        uint8_t  fix_type  = 0;
        uint8_t  sats      = 0;
        bool     fake      = false;   // координаты подставные
    };

    LiveState live_state() {
        LiveState s;
        s.fake = fake_;
        s.fc_link  = link_ok();
        s.messages = messages();

        if (auto a = att_.latest()) {
            s.have_att = true;
            s.roll = a->roll; s.pitch = a->pitch; s.yaw = a->yaw;
            s.att_t_us = a->t_us;
        }
        // Высота - сначала из отдельного ответа полётника: она есть и без
        // GPS. Координатная высота ниже, если придёт, её перепишет.
        if (src_) {
            float a = 0.f;
            if (src_->baro_alt_m(a)) { s.alt_m = a; s.have_alt = true; }
        }
        if (auto p = pos_.latest()) {
            s.have_pos = true;
            s.lat = p->lat; s.lon = p->lon;
            s.fix_type = p->fix_type; s.sats = p->sats;
            // Та же логика, что в Pose::alt_for_geo: дальномер, если есть,
            // иначе барометр. В Betaflight дальномера нет, но код один.
            const float alt = std::isnan(p->alt_agl) ? p->alt_rel : p->alt_agl;
            s.have_alt = !std::isnan(alt);
            if (s.have_alt) s.alt_m = alt;
            fix_ = p->fix_type; sats_ = p->sats;
        }
        if (fake_) {
            s.have_pos = true;
            s.lat = flat_; s.lon = flon_;
            s.alt_m = falt_; s.have_alt = true;
            s.fix_type = 3; s.sats = 12;
        }
        return s;
    }

    // --- что показывать в консоли
    bool        link_ok()      const { return src_ && src_->link_ok(); }
    bool        clock_locked() const { return src_ && src_->clock_locked(); }
    uint64_t    messages()     const { return src_ ? src_->messages() : 0; }
    const char* protocol()     const { return src_ ? src_->name() : "нет"; }
    uint8_t     fix_type()     const { return fix_; }
    uint8_t     sats()         const { return sats_; }

private:
    // Поза с подставными координатами: углы берём настоящие, если они есть.
    bool fake_pose_at(uint64_t t_us, Pose& out) {
        Pose r;
        auto a = att_.interpolate(t_us);
        if (!a) a = att_.latest();        // одной записи хватает для проверки
        if (a) { r.roll = a->roll; r.pitch = a->pitch; r.yaw = a->yaw; }
        r.t_us = t_us;
        r.lat = flat_; r.lon = flon_;
        r.alt_rel = falt_; r.alt_agl = NAN;
        r.fix_type = 3; r.sats = 12;
        fix_ = 3; sats_ = 12;
        out = r;
        return true;
    }

    AttitudeBuffer att_;
    PositionBuffer pos_;
    std::unique_ptr<ITelemetrySource> src_;
    uint8_t fix_ = 0, sats_ = 0;
    bool   fake_ = false;
    double flat_ = 0.0, flon_ = 0.0;
    float  falt_ = 0.f;
};

} // namespace sar
