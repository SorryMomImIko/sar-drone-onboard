// TelemetryBuffers.hpp - кольцевые буферы телеметрии с интерполяцией по времени.
//
// Пишутся из потока MAVLink, читаются из потока обработки кадров.
// Единая шкала времени: CLOCK_MONOTONIC платы, микросекунды.
//
// БУФЕРА ДВА, И ЭТО ПРИНЦИПИАЛЬНО.
// Углы приходят сообщением ATTITUDE, координаты - GLOBAL_POSITION_INT,
// с разной частотой и со своими метками времени. Если складывать их в одну
// запись (брать последнюю известную координату в момент прихода угла),
// координата окажется на полпериода старше угла. При 10 Гц это до 100 мс,
// то есть полтора метра систематического сноса на скорости 15 м/с.
// Ошибка была найдена прогоном на модели; подробности в docs/.
#pragma once

#include <array>
#include <cmath>
#include <cstdint>
#include <ctime>
#include <mutex>
#include <optional>

namespace sar {

inline uint64_t now_us() {
    timespec ts{};
    clock_gettime(CLOCK_MONOTONIC, &ts);   // монотонные часы: не прыгают от NTP
    return static_cast<uint64_t>(ts.tv_sec) * 1000000ull + ts.tv_nsec / 1000;
}

// Нормализация угла в (-pi, pi]. Без неё интерполяция рыскания через север
// даёт скачок на 360 градусов.
inline float wrap_pi(float a) {
    while (a > 3.14159265f)  a -= 6.28318531f;
    while (a <= -3.14159265f) a += 6.28318531f;
    return a;
}

// ---------------------------------------------------------------------------
// Выборки
// ---------------------------------------------------------------------------

struct AttitudeSample {
    uint64_t t_us  = 0;     // момент ИЗМЕРЕНИЯ в шкале платы
    float    roll  = 0.f;   // рад
    float    pitch = 0.f;
    float    yaw   = 0.f;
};

struct PositionSample {
    uint64_t t_us     = 0;
    double   lat      = 0.0;   // градусы
    double   lon      = 0.0;
    float    alt_msl  = 0.f;   // м над уровнем моря
    float    alt_rel  = 0.f;   // м над точкой взлёта (барометр)
    float    alt_agl  = NAN;   // м над рельефом (дальномер), NaN если нет
    uint8_t  fix_type = 0;     // 3 = 3D fix; меньше - координатам не верим
    uint8_t  sats     = 0;
};

// Полное состояние на момент кадра - то, что уходит в геопривязку.
struct Pose {
    uint64_t t_us     = 0;
    double   lat      = 0.0;
    double   lon      = 0.0;
    float    alt_rel  = 0.f;
    float    alt_agl  = NAN;
    float    roll     = 0.f;
    float    pitch    = 0.f;
    float    yaw      = 0.f;
    uint8_t  fix_type = 0;
    uint8_t  sats     = 0;

    // Высота для геопривязки: дальномер, если он есть, иначе барометр.
    float alt_for_geo() const { return std::isnan(alt_agl) ? alt_rel : alt_agl; }
};

// ---------------------------------------------------------------------------
// Интерполяция выборок. Отдельная функция на каждый тип: у углов своя
// арифметика (разность нормализуется в +-pi), у координат обычная линейная.
// ---------------------------------------------------------------------------

inline AttitudeSample lerp_sample(const AttitudeSample& a,
                                  const AttitudeSample& b, double k) {
    const float kf = static_cast<float>(k);
    AttitudeSample r = a;
    r.roll  = a.roll  + wrap_pi(b.roll  - a.roll ) * kf;
    r.pitch = a.pitch + wrap_pi(b.pitch - a.pitch) * kf;
    r.yaw   = wrap_pi(a.yaw + wrap_pi(b.yaw - a.yaw) * kf);
    return r;
}

inline PositionSample lerp_sample(const PositionSample& a,
                                  const PositionSample& b, double k) {
    const float kf = static_cast<float>(k);
    PositionSample r = a;
    r.lat     = a.lat + (b.lat - a.lat) * k;
    r.lon     = a.lon + (b.lon - a.lon) * k;
    r.alt_msl = a.alt_msl + (b.alt_msl - a.alt_msl) * kf;
    r.alt_rel = a.alt_rel + (b.alt_rel - a.alt_rel) * kf;
    if (!std::isnan(a.alt_agl) && !std::isnan(b.alt_agl))
        r.alt_agl = a.alt_agl + (b.alt_agl - a.alt_agl) * kf;
    r.fix_type = (a.fix_type < b.fix_type) ? a.fix_type : b.fix_type;  // пессимистично
    return r;
}

// ---------------------------------------------------------------------------
// Кольцевой буфер
// ---------------------------------------------------------------------------

template <typename S, size_t N = 256>
class RingBuffer {
public:
    // Максимальная дыра, при которой ещё отвечаем крайней записью.
    // 300 мс хватает для прямого UART на 20 Гц. Для редкой телеметрии
    // по радиоканалу значение надо увеличивать - см. замечание в docs/.
    static constexpr uint64_t MAX_GAP_US = 300000;

    void push(const S& s) {
        std::lock_guard<std::mutex> lk(mu_);
        buf_[head_] = s;
        head_ = (head_ + 1) % N;
        if (count_ < N) ++count_;
    }

    std::optional<S> latest() const {
        std::lock_guard<std::mutex> lk(mu_);
        if (count_ == 0) return std::nullopt;
        return at(count_ - 1);
    }

    size_t size() const { std::lock_guard<std::mutex> lk(mu_); return count_; }

    // Средний период записей - диагностика: достаточно ли частая телеметрия.
    double mean_period_s() const {
        std::lock_guard<std::mutex> lk(mu_);
        if (count_ < 2) return 0.0;
        return double(at(count_ - 1).t_us - at(0).t_us) / 1e6 / double(count_ - 1);
    }

    // Куда попал запрошенный момент относительно буфера. Диагностика:
    // если он систематически Later, интерполировать не между чем, и метод
    // вырождается в "последний пакет".
    enum class Where { Empty, Before, Inside, Later };

    Where classify(uint64_t t_us) const {
        std::lock_guard<std::mutex> lk(mu_);
        if (count_ < 2) return Where::Empty;
        if (t_us < at(0).t_us) return Where::Before;
        if (t_us > at(count_ - 1).t_us) return Where::Later;
        return Where::Inside;
    }

    // Состояние на момент t_us. nullopt, если данных нет или дыра велика:
    // лучше отдать находку без координат, чем метку не в том месте.
    std::optional<S> interpolate(uint64_t t_us) const {
        std::lock_guard<std::mutex> lk(mu_);
        if (count_ < 2) return std::nullopt;

        const S& oldest = at(0);
        const S& newest = at(count_ - 1);
        if (t_us <= oldest.t_us)
            return (oldest.t_us - t_us <= MAX_GAP_US) ? std::optional<S>(oldest)
                                                      : std::nullopt;
        if (t_us >= newest.t_us)
            return (t_us - newest.t_us <= MAX_GAP_US) ? std::optional<S>(newest)
                                                      : std::nullopt;

        size_t lo = 0, hi = count_ - 1;
        while (hi - lo > 1) {
            const size_t mid = (lo + hi) / 2;
            if (at(mid).t_us <= t_us) lo = mid; else hi = mid;
        }
        const S& a = at(lo);
        const S& b = at(hi);
        const double dt = double(b.t_us - a.t_us);
        const double k  = (dt > 0.0) ? double(t_us - a.t_us) / dt : 0.0;
        S r = lerp_sample(a, b, k);
        r.t_us = t_us;
        return r;
    }

private:
    const S& at(size_t i) const {              // i = 0 - самая старая запись
        const size_t start = (count_ < N) ? 0 : head_;
        return buf_[(start + i) % N];
    }

    mutable std::mutex mu_;
    std::array<S, N> buf_{};
    size_t head_ = 0, count_ = 0;
};

using AttitudeBuffer = RingBuffer<AttitudeSample>;
using PositionBuffer = RingBuffer<PositionSample>;

// ---------------------------------------------------------------------------
// Сборка полного состояния на момент кадра
// ---------------------------------------------------------------------------

inline std::optional<Pose> pose_at(const AttitudeBuffer& att,
                                   const PositionBuffer& pos,
                                   uint64_t t_us) {
    const auto a = att.interpolate(t_us);
    const auto p = pos.interpolate(t_us);
    if (!a || !p) return std::nullopt;

    Pose r;
    r.t_us     = t_us;
    r.lat      = p->lat;
    r.lon      = p->lon;
    r.alt_rel  = p->alt_rel;
    r.alt_agl  = p->alt_agl;
    r.fix_type = p->fix_type;
    r.sats     = p->sats;
    r.roll     = a->roll;
    r.pitch    = a->pitch;
    r.yaw      = a->yaw;
    return r;
}

} // namespace sar
