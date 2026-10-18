// MavlinkThread.hpp - отдельный поток: читает MAVLink с полётника,
// сводит его часы с часами платы, складывает телеметрию в два буфера.
//
// Ни с кем не синхронизируется и никого не ждёт: короткий мьютекс на push,
// и всё. Инференс на NPU не блокируется ни на миллисекунду.
//
// Требует заголовки mavlink/c_library_v2 (header-only):
//   cd third_party
//   git clone --depth 1 https://github.com/mavlink/c_library_v2.git mavlink
#pragma once

#include "TelemetrySource.hpp"

#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <string>
#include <thread>

#include <arpa/inet.h>
#include <fcntl.h>
#include <sys/socket.h>
#include <termios.h>
#include <unistd.h>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Waddress-of-packed-member"
#include <common/mavlink.h>
#pragma GCC diagnostic pop

namespace sar {

class MavlinkThread : public ITelemetrySource {
public:
    MavlinkThread(AttitudeBuffer& att, PositionBuffer& pos)
        : att_(att), pos_(pos) {}
    ~MavlinkThread() override { stop(); }

    void set_baud(speed_t b) { baud_ = b; }   // для USB (CDC-ACM) значения не имеет

    // "/dev/ttyS3"                 UART на плате
    // "/dev/serial/by-id/usb-..."  автопилот по USB
    // "udp:0.0.0.0:14550"          ArduPilot SITL при отладке на ноутбуке
    bool open(const std::string& uri) override {
        uri_ = uri;                            // запомнили: понадобится для переоткрытия
        return reopen();
    }

    void start() override { run_ = true; th_ = std::thread([this] { loop(); }); }

    void stop() override {
        run_ = false;
        if (th_.joinable()) th_.join();
        close_fd();
    }

    // --- состояние, которое стоит показывать оператору
    bool link_ok() const override { return now_us() - last_hb_us_.load() < 3000000ull; }
    bool clock_locked() const override { return have_offset_; }
    uint64_t messages() const override { return n_msgs_.load(); }
    const char* name() const override { return "MAVLink"; }

private:
    bool reopen() {
        close_fd();
        streams_requested_ = false;            // новому соединению - новый запрос частот
        if (uri_.rfind("udp:", 0) == 0) return open_udp(uri_);
        return open_serial(uri_.c_str(), baud_);
    }

    void close_fd() { if (fd_ >= 0) { ::close(fd_); fd_ = -1; } }

    // ---------------- транспорт ----------------

    bool open_serial(const char* dev, speed_t baud) {
        fd_ = ::open(dev, O_RDWR | O_NOCTTY);
        if (fd_ < 0) return false;

        termios t{};
        tcgetattr(fd_, &t);
        cfmakeraw(&t);                  // бинарный протокол, без обработки символов
        cfsetispeed(&t, baud);
        cfsetospeed(&t, baud);
        t.c_cflag |= (CLOCAL | CREAD);
        t.c_cflag &= ~CRTSCTS;          // аппаратного управления потоком на TELEM нет
        t.c_cc[VMIN]  = 0;
        t.c_cc[VTIME] = 1;              // 100 мс таймаут: read не виснет навсегда
        tcsetattr(fd_, TCSANOW, &t);
        tcflush(fd_, TCIOFLUSH);
        is_udp_ = false;
        return true;
    }

    bool open_udp(const std::string& uri) {
        const auto p1 = uri.find(':', 4);
        if (p1 == std::string::npos) return false;
        const std::string addr = uri.substr(4, p1 - 4);
        const int port = std::stoi(uri.substr(p1 + 1));

        fd_ = ::socket(AF_INET, SOCK_DGRAM, 0);
        if (fd_ < 0) return false;
        sockaddr_in sa{};
        sa.sin_family = AF_INET;
        sa.sin_port = htons(static_cast<uint16_t>(port));
        sa.sin_addr.s_addr = inet_addr(addr.c_str());
        if (::bind(fd_, reinterpret_cast<sockaddr*>(&sa), sizeof(sa)) < 0) {
            ::close(fd_); fd_ = -1; return false;
        }
        timeval tv{0, 100000};
        setsockopt(fd_, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
        is_udp_ = true;
        return true;
    }

    // ---------------- основной цикл ----------------

    void loop() {
        uint8_t buf[1024];
        mavlink_message_t msg{};
        mavlink_status_t  st{};

        while (run_) {
            // ПЕРЕОТКРЫТИЕ. По USB автопилот исчезает и появляется: дёрнули
            // разъём, автопилот перезагрузился - /dev/ttyACM* пропал. Без
            // этого программа молча читала бы из мёртвого дескриптора до
            // конца полёта.
            if (fd_ < 0) {
                if (!reopen()) {
                    std::this_thread::sleep_for(std::chrono::seconds(2));
                    continue;
                }
                std::fprintf(stderr, "полётник: порт %s открыт\n", uri_.c_str());
            }

            ssize_t n;
            if (is_udp_) {
                sockaddr_in from{}; socklen_t fl = sizeof(from);
                n = ::recvfrom(fd_, buf, sizeof(buf), 0,
                               reinterpret_cast<sockaddr*>(&from), &fl);
                if (n > 0 && !peer_known_) { peer_ = from; peer_known_ = true; }
            } else {
                // UART отдаёт ПОТОК БАЙТОВ: один read регулярно возвращает
                // половину пакета. Парсер MAVLink собирает их сам.
                n = ::read(fd_, buf, sizeof(buf));
            }

            if (n < 0 && errno != EAGAIN && errno != EINTR) {
                // Устройство пропало (ENODEV/EIO) - закрываем и ждём возврата.
                std::fprintf(stderr, "полётник: порт отвалился (%s), переоткрываю\n",
                             std::strerror(errno));
                close_fd();
                continue;
            }
            if (n <= 0) continue;

            for (ssize_t i = 0; i < n; ++i) {
                if (!mavlink_parse_char(MAVLINK_COMM_0, buf[i], &msg, &st)) continue;
                ++n_msgs_;

                switch (msg.msgid) {

                case MAVLINK_MSG_ID_HEARTBEAT: {
                    last_hb_us_ = now_us();
                    if (!streams_requested_) {
                        sys_ = msg.sysid; comp_ = msg.compid;
                        request_streams();
                        streams_requested_ = true;
                    }
                    break;
                }

                case MAVLINK_MSG_ID_ATTITUDE: {
                    mavlink_attitude_t a{};
                    mavlink_msg_attitude_decode(&msg, &a);
                    update_clock_offset(a.time_boot_ms);
                    AttitudeSample s{};
                    s.t_us  = fc_to_local(a.time_boot_ms);
                    s.roll  = a.roll;
                    s.pitch = a.pitch;
                    s.yaw   = a.yaw;
                    att_.push(s);
                    break;
                }

                case MAVLINK_MSG_ID_GLOBAL_POSITION_INT: {
                    mavlink_global_position_int_t p{};
                    mavlink_msg_global_position_int_decode(&msg, &p);
                    update_clock_offset(p.time_boot_ms);
                    PositionSample s{};
                    s.t_us     = fc_to_local(p.time_boot_ms);
                    s.lat      = p.lat / 1e7;
                    s.lon      = p.lon / 1e7;
                    s.alt_msl  = p.alt / 1000.f;
                    s.alt_rel  = p.relative_alt / 1000.f;
                    s.alt_agl  = agl_;            // из дальномера, если он есть
                    s.fix_type = fix_;
                    s.sats     = sats_;
                    pos_.push(s);
                    break;
                }

                case MAVLINK_MSG_ID_GPS_RAW_INT: {
                    mavlink_gps_raw_int_t g{};
                    mavlink_msg_gps_raw_int_decode(&msg, &g);
                    fix_  = g.fix_type;
                    sats_ = g.satellites_visible;
                    break;
                }

                case MAVLINK_MSG_ID_DISTANCE_SENSOR: {
                    mavlink_distance_sensor_t d{};
                    mavlink_msg_distance_sensor_decode(&msg, &d);
                    // на краях диапазона дальномер врёт - отбрасываем
                    agl_ = (d.current_distance > d.min_distance &&
                            d.current_distance < d.max_distance)
                           ? d.current_distance / 100.f : NAN;
                    break;
                }

                default: break;     // в потоке 30+ типов, нам нужны четыре
                }
            }
        }
    }

    // ---------------- передача ----------------

    void send(mavlink_message_t& m) {
        uint8_t b[MAVLINK_MAX_PACKET_LEN];
        const int len = mavlink_msg_to_send_buffer(b, &m);
        if (is_udp_) {
            if (peer_known_)
                ::sendto(fd_, b, len, 0,
                         reinterpret_cast<sockaddr*>(&peer_), sizeof(peer_));
        } else {
            ssize_t ignored = ::write(fd_, b, len); (void)ignored;
        }
    }

    // Частоты: подписываться на весь поток не надо, он забьёт канал.
    // 20 Гц по углам достаточно - см. требование к периоду в docs/.
    void request_streams() {
        const struct { uint32_t id; float hz; } want[] = {
            { MAVLINK_MSG_ID_ATTITUDE,            20.f },
            { MAVLINK_MSG_ID_GLOBAL_POSITION_INT, 10.f },
            { MAVLINK_MSG_ID_GPS_RAW_INT,          2.f },
            { MAVLINK_MSG_ID_DISTANCE_SENSOR,     10.f },
        };
        for (const auto& w : want) {
            mavlink_message_t m{};
            mavlink_msg_command_long_pack(255, 190, &m, sys_, comp_,
                MAV_CMD_SET_MESSAGE_INTERVAL, 0,
                static_cast<float>(w.id), 1e6f / w.hz, 0, 0, 0, 0, 0);
            send(m);
        }
    }

    // ---------------- сведение часов ----------------
    //
    // Разность (местное время - время полётника) = истинное смещение часов
    // плюс задержка доставки. Задержка всегда положительна и скачет, поэтому
    // МИНИМУМ разностей за окно - лучшая оценка истинного смещения.
    // Окно нужно, иначе один случайно "быстрый" пакет зафиксирует неверное
    // смещение на весь полёт.
    //
    // Штатный TIMESYNC делает то же по принципу NTP, но поддержан не всеми
    // прошивками. Эта оценка не требует от полётника ничего.

    void update_clock_offset(uint32_t fc_boot_ms) {
        const uint64_t local = now_us();
        const int64_t diff = static_cast<int64_t>(local)
                           - static_cast<int64_t>(uint64_t(fc_boot_ms) * 1000ull);

        const size_t bucket = (local / 1000000ull) % NBUCKETS;   // корзина = 1 с
        if (bucket != cur_bucket_) {
            cur_bucket_ = bucket;
            bmin_[bucket] = diff;
            bused_[bucket] = true;
        } else if (diff < bmin_[bucket]) {
            bmin_[bucket] = diff;
        }

        int64_t best = INT64_MAX;
        for (size_t i = 0; i < NBUCKETS; ++i)
            if (bused_[i] && bmin_[i] < best) best = bmin_[i];
        if (best != INT64_MAX) { offset_us_ = best; have_offset_ = true; }
    }

    uint64_t fc_to_local(uint32_t fc_boot_ms) const {
        if (!have_offset_) return now_us();
        return static_cast<uint64_t>(
            static_cast<int64_t>(uint64_t(fc_boot_ms) * 1000ull) + offset_us_);
    }

    // ---------------- поля ----------------

    static constexpr size_t NBUCKETS = 30;    // окно оценки смещения, секунды

    AttitudeBuffer& att_;
    PositionBuffer& pos_;

    std::thread th_;
    std::atomic<bool> run_{false};
    std::atomic<uint64_t> last_hb_us_{0};
    std::atomic<uint64_t> n_msgs_{0};

    std::string uri_;
    speed_t baud_ = B921600;
    int  fd_ = -1;
    bool is_udp_ = false;
    sockaddr_in peer_{};
    bool peer_known_ = false;

    uint8_t sys_ = 1, comp_ = 1;
    bool streams_requested_ = false;

    uint8_t fix_ = 0, sats_ = 0;
    float   agl_ = NAN;

    bool    have_offset_ = false;
    int64_t offset_us_ = 0;
    int64_t bmin_[NBUCKETS]{};
    bool    bused_[NBUCKETS]{};
    size_t  cur_bucket_ = SIZE_MAX;
};

} // namespace sar
