#pragma once
// MspThread.hpp - телеметрия с Betaflight по протоколу MSP.
//
// ДЛЯ ЧЕГО: Foxeer F405 V2 и другие полётники на Betaflight. Подключение по
// USB (появляется как /dev/ttyACM*), скорость для USB значения не имеет.
//
// ЧЕМ ЭТО ОТЛИЧАЕТСЯ ОТ MAVLINK - три вещи, из-за которых код другой:
//
// 1. BETAFLIGHT НИЧЕГО НЕ ПРИСЫЛАЕТ САМ. MAVLink шлёт поток, на который
//    достаточно подписаться. MSP - строго "запрос-ответ": хочешь угол -
//    спроси. Поэтому здесь не "читающий" поток, а ОПРАШИВАЮЩИЙ: он сам
//    задаёт темп и сам отвечает за то, чтобы данные были свежими.
//
// 2. ВЫСОТА ЛЕЖИТ НЕ ТАМ, ГДЕ КООРДИНАТЫ. В MSP_RAW_GPS высота есть, но
//    она в ЦЕЛЫХ МЕТРАХ и отрицательные значения обрезаны в ноль - в
//    исходнике Betaflight буквально:
//        sbufWriteU16(dst, constrain(gpsSol.llh.altCm / 100, 0, UINT16_MAX));
//    Для геопривязки это негодно: метр высоты на 80 м даёт ошибку в метр на
//    земле. Поэтому высота берётся отдельным запросом MSP_ALTITUDE, где она
//    в сантиметрах. Две величины приходят разными ответами и сводятся здесь.
//
// 3. ВРЕМЕНИ В ОТВЕТЕ НЕТ. MAVLink присылает своё время, и его приходилось
//    сводить с часами платы. Здесь проще и честнее: метка ставится как
//    СЕРЕДИНА между моментом запроса и моментом ответа - измерение сделано
//    где-то внутри этого промежутка. Сводить нечего, поэтому clock_locked()
//    отдаёт true, как только пришёл первый ответ.
//
// ЧЕГО В BETAFLIGHT НЕТ, и это не чинится кодом:
//   - дальномера, то есть высоты над рельефом (alt_agl остаётся пустым);
//     высота - барометрическая, относительно точки взлёта;
//   - признака качества фикса: в MSP_RAW_GPS он ДВОИЧНЫЙ (есть/нет), а не
//     2D/3D, как в MAVLink. Поэтому качество оценивается по числу спутников;
//   - рыскание отдаётся в ЦЕЛЫХ ГРАДУСАХ (DECIDEGREES_TO_DEGREES в
//     исходнике), то есть с шагом 1°. На краю кадра с 80 м это ~0.8 м
//     дискретности, которой в MAVLink не было.
//
// ТРЕБОВАНИЕ К СБОРКЕ ДРОНА: к полётнику должен быть подключён и включён
// GPS-модуль (в Betaflight: feature GPS плюс UART с функцией GPS). Без него
// MSP_RAW_GPS отвечает нулями и фиксом 0 - код это увидит и честно откажет,
// но координат не появится.

#include "TelemetrySource.hpp"

#include <atomic>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <thread>

#include <fcntl.h>
#include <termios.h>
#include <unistd.h>

namespace sar {

class MspThread : public ITelemetrySource {
public:
    // Команды MSP. Номера - из Betaflight.
    enum : uint8_t {
        MSP_API_VERSION = 1,
        MSP_FC_VARIANT  = 2,
        MSP_FC_VERSION  = 3,
        MSP_RAW_GPS     = 106,
        MSP_ATTITUDE    = 108,
        MSP_ALTITUDE    = 109,
    };

    // Темп опроса. Углы - чаще всего: от них зависит точность интерполяции
    // на момент кадра. GPS в Betaflight обновляется 5-10 Гц, чаще спрашивать
    // бессмысленно - будут приходить те же значения.
    static constexpr int HZ_ATTITUDE = 25;
    static constexpr int HZ_ALTITUDE = 10;
    static constexpr int HZ_GPS      = 5;

    static constexpr uint8_t MIN_SATS = 6;      // ниже этого координате не верим

    MspThread(AttitudeBuffer& att, PositionBuffer& pos) : att_(att), pos_(pos) {}
    ~MspThread() override { stop(); }

    bool open(const std::string& uri) override {
        uri_ = uri;
        return reopen();
    }

    void start() override { run_ = true; th_ = std::thread([this] { loop(); }); }

    void stop() override {
        run_ = false;
        if (th_.joinable()) th_.join();
        close_fd();
    }

    bool link_ok() const override { return now_us() - last_ok_us_.load() < 2000000ull; }
    // Часы не сводятся: метку ставим сами. "Сведены" = был хотя бы один ответ.
    bool clock_locked() const override { return n_msgs_.load() > 0; }
    uint64_t messages() const override { return n_msgs_.load(); }
    const char* name() const override { return "MSP (Betaflight)"; }

    // Высота приходит своим ответом и не зависит от GPS - отдаём как есть.
    bool baro_alt_m(float& out) const override {
        if (!have_alt_.load()) return false;
        out = alt_rel_m_.load();
        return true;
    }

private:
    // ---------------- порт ----------------

    bool reopen() {
        close_fd();
        fd_ = ::open(uri_.c_str(), O_RDWR | O_NOCTTY);
        if (fd_ < 0) return false;

        termios t{};
        tcgetattr(fd_, &t);
        cfmakeraw(&t);                 // бинарный протокол, без обработки символов
        // Для USB CDC скорость игнорируется, но на обычном UART нужна.
        cfsetispeed(&t, B115200);
        cfsetospeed(&t, B115200);
        t.c_cflag |= (CLOCAL | CREAD);
        t.c_cflag &= ~CRTSCTS;
        t.c_cc[VMIN]  = 0;
        t.c_cc[VTIME] = 1;             // 100 мс: read не виснет навсегда
        tcsetattr(fd_, TCSANOW, &t);
        tcflush(fd_, TCIOFLUSH);
        identified_ = false;
        next_identify_us_ = 0;
        return true;
    }

    void close_fd() { if (fd_ >= 0) { ::close(fd_); fd_ = -1; } }

    // ---------------- кадр MSP v1 ----------------
    //
    //   '$' 'M' '<'  длина  команда  данные...  CRC
    // CRC = XOR длины, команды и всех байт данных.
    // Запрос без данных: длина 0, и тогда CRC равен номеру команды.

    bool request(uint8_t cmd) {
        const uint8_t pkt[6] = {'$', 'M', '<', 0, cmd, cmd};
        return ::write(fd_, pkt, sizeof(pkt)) == (ssize_t)sizeof(pkt);
    }

    // Ждёт ответ на конкретную команду. Чужие ответы пропускает: в порт
    // может прийти что-то от предыдущего запроса, если тот опоздал.
    // Возвращает длину данных или -1.
    int read_response(uint8_t want_cmd, uint8_t* out, size_t out_max,
                      uint64_t deadline_us) {
        enum { S_D, S_M, S_DIR, S_LEN, S_CMD, S_DATA, S_CRC } st = S_D;
        uint8_t len = 0, cmd = 0, crc = 0, idx = 0, buf[256];
        bool err_resp = false;

        while (now_us() < deadline_us) {
            uint8_t c;
            const ssize_t n = ::read(fd_, &c, 1);
            if (n < 0 && errno != EAGAIN && errno != EINTR) return -2;  // порт пропал
            if (n <= 0) continue;

            switch (st) {
            case S_D:   st = (c == '$') ? S_M : S_D; break;
            case S_M:   st = (c == 'M') ? S_DIR : S_D; break;
            case S_DIR:
                // '>' - нормальный ответ, '!' - полётник не знает такой команды
                err_resp = (c == '!');
                st = (c == '>' || c == '!') ? S_LEN : S_D;
                break;
            case S_LEN: len = c; crc = c; idx = 0; st = S_CMD; break;
            case S_CMD: cmd = c; crc ^= c; st = len ? S_DATA : S_CRC; break;
            case S_DATA:
                buf[idx++] = c; crc ^= c;
                if (idx >= len) st = S_CRC;
                break;
            case S_CRC:
                st = S_D;
                if (crc != c) break;                  // битый кадр - ждём следующий
                ++n_msgs_;
                last_ok_us_ = now_us();
                if (err_resp && cmd == want_cmd) return -1;
                if (cmd != want_cmd) break;           // чужой ответ - пропускаем
                if (len > out_max) return -1;
                std::memcpy(out, buf, len);
                return len;
            }
        }
        return -1;                                     // не дождались
    }

    // Запрос и ответ одной операцией. t_mid - середина между запросом и
    // ответом: измерение сделано где-то внутри этого промежутка, и середина
    // - лучшая оценка момента, к которому относятся данные.
    int ask(uint8_t cmd, uint8_t* out, size_t out_max, uint64_t& t_mid) {
        const uint64_t t0 = now_us();
        if (!request(cmd)) return -2;
        const int n = read_response(cmd, out, out_max, t0 + TIMEOUT_US);
        t_mid = (t0 + now_us()) / 2;
        return n;
    }

    // ---------------- чтение величин ----------------

    static int16_t  i16(const uint8_t* p) { return int16_t(uint16_t(p[0]) | (uint16_t(p[1]) << 8)); }
    static uint16_t u16(const uint8_t* p) { return uint16_t(p[0]) | (uint16_t(p[1]) << 8); }
    static int32_t  i32(const uint8_t* p) {
        return int32_t(uint32_t(p[0]) | (uint32_t(p[1]) << 8)
                     | (uint32_t(p[2]) << 16) | (uint32_t(p[3]) << 24));
    }

    // MSP_ATTITUDE: int16 roll (децидюймы градуса), int16 pitch (то же),
    //               int16 yaw (ЦЕЛЫЕ градусы)
    void handle_attitude(const uint8_t* p, int n, uint64_t t_us) {
        if (n < 6) return;
        AttitudeSample s{};
        s.t_us  = t_us;
        s.roll  = float(i16(p + 0)) * 0.1f * DEG2RAD;
        s.pitch = float(i16(p + 2)) * 0.1f * DEG2RAD;
        s.yaw   = wrap_pi(float(i16(p + 4)) * DEG2RAD);   // уже в градусах
        att_.push(s);
    }

    // MSP_ALTITUDE: int32 высота в САНТИМЕТРАХ от точки взлёта, int16 вариометр.
    void handle_altitude(const uint8_t* p, int n) {
        if (n < 4) return;
        alt_rel_m_ = float(i32(p)) * 0.01f;
        have_alt_ = true;
    }

    // MSP_RAW_GPS: u8 фикс, u8 спутники, i32 широта, i32 долгота (1e7),
    //              u16 высота (метры, негодна), u16 скорость, u16 курс, [u16 pdop]
    void handle_gps(const uint8_t* p, int n, uint64_t t_us) {
        if (n < 14) return;
        const uint8_t fix  = p[0];
        const uint8_t sats = p[1];
        fix_  = fix;
        sats_ = sats;

        PositionSample s{};
        s.t_us    = t_us;
        s.lat     = i32(p + 2) / 1e7;
        s.lon     = i32(p + 6) / 1e7;
        // Высота ТОЛЬКО из MSP_ALTITUDE. Поле высоты в этом сообщении
        // огрублено до метров и обрезано снизу нулём - не используем.
        s.alt_rel = have_alt_.load() ? alt_rel_m_.load() : NAN;
        s.alt_msl = NAN;
        s.alt_agl = NAN;                 // дальномера в Betaflight нет
        // Фикс в Betaflight двоичный. Приводим к шкале MAVLink, чтобы
        // проверка fix_type >= 3 выше по коду осталась без изменений:
        // "есть фикс и спутников достаточно" = 3, иначе 0.
        s.fix_type = (fix && sats >= MIN_SATS) ? 3 : 0;
        s.sats     = sats;
        pos_.push(s);
    }

    // Один раз спрашиваем, кто это вообще - в консоль, для диагностики.
    // Если не ответил - НЕ объявляем опознанным: порт мог открыться, а
    // полётник быть выключен. Пробуем снова раз в две секунды.
    void identify() {
        uint8_t buf[64]; uint64_t t;
        const int n = ask(MSP_FC_VARIANT, buf, sizeof(buf), t);
        char variant[5] = {0};
        if (n >= 4) std::memcpy(variant, buf, 4);
        const int nv = ask(MSP_FC_VERSION, buf, sizeof(buf), t);

        if (n < 0 && nv < 0) {
            next_identify_us_ = now_us() + 2000000ull;
            if (!warned_silent_) {
                warned_silent_ = true;
                std::fprintf(stderr,
                    "полётник: порт %s открыт, но на MSP никто не отвечает - "
                    "проверь, тот ли это порт и включён ли полётник\n", uri_.c_str());
            }
            return;
        }
        if (nv >= 3)
            std::fprintf(stderr, "полётник: %s %u.%u.%u по MSP\n",
                         n >= 4 ? variant : "?", buf[0], buf[1], buf[2]);
        else
            std::fprintf(stderr, "полётник: отвечает по MSP, версию не сообщил\n");
        identified_   = true;
        warned_silent_ = false;
    }

    // ---------------- цикл опроса ----------------

    void loop() {
        uint8_t buf[256];
        uint64_t next_att = 0, next_alt = 0, next_gps = 0;

        while (run_) {
            if (fd_ < 0) {
                if (!reopen()) { std::this_thread::sleep_for(std::chrono::seconds(2)); continue; }
                std::fprintf(stderr, "полётник: порт %s открыт\n", uri_.c_str());
            }
            if (!identified_ && now_us() >= next_identify_us_) identify();

            const uint64_t t = now_us();
            bool did = false;
            int  n   = 0;
            uint64_t t_mid = 0;

            // Высоту и GPS спрашиваем реже углов; очередь выбирается по тому,
            // чей срок наступил раньше - так опрос не толпится.
            if (t >= next_att) {
                n = ask(MSP_ATTITUDE, buf, sizeof(buf), t_mid);
                if (n >= 0) handle_attitude(buf, n, t_mid);
                next_att = t + 1000000ull / HZ_ATTITUDE;
                did = true;
            } else if (t >= next_alt) {
                n = ask(MSP_ALTITUDE, buf, sizeof(buf), t_mid);
                if (n >= 0) handle_altitude(buf, n);
                next_alt = t + 1000000ull / HZ_ALTITUDE;
                did = true;
            } else if (t >= next_gps) {
                n = ask(MSP_RAW_GPS, buf, sizeof(buf), t_mid);
                if (n >= 0) handle_gps(buf, n, t_mid);
                next_gps = t + 1000000ull / HZ_GPS;
                did = true;
            }

            if (n == -2) {              // порт отвалился: выдернули USB
                std::fprintf(stderr, "полётник: порт отвалился, переоткрываю\n");
                close_fd();
                continue;
            }
            if (!did) std::this_thread::sleep_for(std::chrono::milliseconds(2));
        }
    }

    // ---------------- поля ----------------

    static constexpr uint64_t TIMEOUT_US = 200000;   // ответ дольше 200 мс = потерян
    static constexpr float    DEG2RAD    = 3.14159265f / 180.f;

    AttitudeBuffer& att_;
    PositionBuffer& pos_;

    std::string uri_;
    int fd_ = -1;
    bool identified_ = false;
    bool warned_silent_ = false;
    uint64_t next_identify_us_ = 0;

    std::thread th_;
    std::atomic<bool> run_{false};
    std::atomic<uint64_t> last_ok_us_{0};
    std::atomic<uint64_t> n_msgs_{0};

    // Атомарные: пишет поток опроса, читает поток живой телеметрии.
    std::atomic<float> alt_rel_m_{NAN};
    std::atomic<bool>  have_alt_{false};
    uint8_t fix_ = 0, sats_ = 0;
};

} // namespace sar
