#pragma once
// ReliableLink.hpp - доставка находок на землю с гарантией.
//
// ПОЧЕМУ ОДНОГО TCP НЕ ХВАТАЕТ. TCP гарантирует доставку ВНУТРИ соединения:
// пока оно живо, ядро само переспрашивает потерянные пакеты. Но если канал
// оборвётся на середине кадра - а на дроне он будет рваться постоянно, -
// соединение закроется, и отправитель НЕ УЗНАЕТ, сколько успело дойти.
// send() давно вернул успех: данные ушли в буфер ядра, а не на землю.
// Поэтому поверх TCP нужны ещё три вещи:
//
//   1. ОЧЕРЕДЬ НА КАРТЕ. Находка сначала ложится файлом в queue/, и только
//      потом отправляется. Переживает обрыв связи, перезапуск программы и
//      снятие питания с платы.
//   2. ПОДТВЕРЖДЕНИЕ ОТ ЗЕМЛИ. Запись удаляется из очереди только когда
//      земля прислала ACK с её номером. Не когда send() вернул успех.
//   3. ПЕРЕОТПРАВКА. При переподключении всё неподтверждённое уходит заново,
//      в том же порядке.
//
// ПОЧЕМУ ОТДЕЛЬНЫЙ ПОТОК. send() по TCP на узком канале блокируется на
// секунды, когда буфер отправки полон. В потоке инференса это означало бы
// пропущенные кадры. Поэтому инференс только кладёт файл в очередь и идёт
// дальше, а отправкой занимается этот поток.
//
// ФОРМАТ ЗАПИСИ (TCP - это ПОТОК БАЙТОВ, границ сообщений в нём нет,
// поэтому длина пишется явно; та же причина, что и в разборе MAVLink с UART):
//   0..3   "SARL"
//   4      версия = 1
//   5      тип: 0 = проверка связи, 1 = кадр JPEG, 2 = вырезка JPEG, 3 = метка JSON
//   6..9   номер записи seq (little-endian uint32) - по нему приходит ACK
//   10..13 номер события event_id
//   14..17 длина данных
//   далее  данные
//
// ОТВЕТ ЗЕМЛИ: "SARA" + seq (4 байта), итого 8.

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <arpa/inet.h>
#include <dirent.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <unistd.h>

namespace sar {

enum : uint8_t { REC_PING = 0, REC_FRAME = 1, REC_CROP = 2, REC_META = 3,
                 REC_TELEMETRY = 4 };

class ReliableLink {
public:
    static constexpr size_t   HDR          = 18;
    static constexpr uint32_t ACK_WAIT_S   = 20;   // нет ACK за это время - считаем связь мёртвой
    static constexpr uint32_t RECONNECT_S  = 3;    // пауза между попытками подключения
    static constexpr uint32_t PING_S       = 10;   // проверка связи, когда отправлять нечего
    static constexpr size_t   QUEUE_MAX    = 2000; // записей; дальше предупреждаем и не копим
    static constexpr uint64_t LIVE_PERIOD_US = 1000000; // как часто слать живую телеметрию
    static constexpr uint64_t LIVE_STALE_US  = 3000000; // старее этого - не слать вовсе

    ReliableLink(std::string ip, uint16_t port, std::string queue_dir)
        : ip_(std::move(ip)), port_(port), dir_(std::move(queue_dir)) {
        ::mkdir(dir_.c_str(), 0755);
    }
    ~ReliableLink() { stop(); }

    void start() { run_ = true; th_ = std::thread([this] { loop(); }); }

    void stop() {
        run_ = false;
        cv_.notify_all();
        if (th_.joinable()) th_.join();
        close_sock();
    }

    // Вызывается из потока инференса. НЕ БЛОКИРУЕТСЯ: пишет файл и выходит.
    // false = не удалось записать в очередь (нет места на карте?) - это повод
    // сказать в лог громко, находка иначе потеряется совсем.
    bool enqueue(uint8_t type, uint32_t event_id,
                 const uint8_t* data, size_t len) {
        if (queued() >= QUEUE_MAX) {
            std::fprintf(stderr,
                "очередь находок переполнена (%zu записей) - связи нет слишком долго\n",
                queued());
            return false;
        }
        const uint32_t seq = next_seq_++;

        std::vector<uint8_t> rec(HDR + len);
        std::memcpy(rec.data(), "SARL", 4);
        rec[4] = 1;
        rec[5] = type;
        put32(&rec[6],  seq);
        put32(&rec[10], event_id);
        put32(&rec[14], static_cast<uint32_t>(len));
        if (len) std::memcpy(rec.data() + HDR, data, len);

        // Пишем во временный файл и переименовываем: если питание пропадёт
        // посреди записи, в очереди не окажется обрезанной записи.
        char tmp[256], fin[256];
        std::snprintf(tmp, sizeof(tmp), "%s/.tmp_%010u", dir_.c_str(), seq);
        std::snprintf(fin, sizeof(fin), "%s/%010u_%u_%u.rec",
                      dir_.c_str(), seq, unsigned(type), event_id);

        FILE* f = std::fopen(tmp, "wb");
        if (!f) return false;
        const bool ok = std::fwrite(rec.data(), 1, rec.size(), f) == rec.size();
        std::fflush(f);
        ::fsync(::fileno(f));            // иначе при снятии питания файл будет пуст
        std::fclose(f);
        if (!ok || std::rename(tmp, fin) != 0) { ::remove(tmp); return false; }

        cv_.notify_one();
        return true;
    }

    bool enqueue(uint8_t type, uint32_t event_id, const std::vector<uint8_t>& v) {
        return enqueue(type, event_id, v.data(), v.size());
    }
    bool enqueue(uint8_t type, uint32_t event_id, const std::string& s) {
        return enqueue(type, event_id,
                       reinterpret_cast<const uint8_t*>(s.data()), s.size());
    }

    // ЖИВАЯ ТЕЛЕМЕТРИЯ - НЕ через очередь, и это сознательно.
    //
    // Находка - ценность, терять её нельзя: отсюда очередь на карте,
    // подтверждения и переотправка. Телеметрия - ровно наоборот: важно
    // только ПОСЛЕДНЕЕ значение, позапрошлое положение дрона не нужно
    // никому. Если гнать её через очередь, при обрыве связи на карте
    // накопятся тысячи записей со старыми координатами, и они хлынут на
    // землю пачкой ровно тогда, когда канал нужен для находок.
    //
    // Поэтому: храним только последнее значение, шлём раз в секунду и
    // только при живой связи. Не дошло - забыли.
    void set_live(std::string json) {
        std::lock_guard<std::mutex> lk(live_mu_);
        live_ = std::move(json);
        live_set_us_ = now_us();
    }

    bool   connected() const { return connected_; }
    size_t queued()    const { return list_queue().size(); }
    uint64_t delivered() const { return delivered_; }

private:
    // ---------------- очередь на карте ----------------

    // Имена вида 0000000042_1_7.rec; сортировка по имени = порядок отправки,
    // потому что номер записи дополнен нулями слева.
    std::vector<std::string> list_queue() const {
        std::vector<std::string> names;
        DIR* d = ::opendir(dir_.c_str());
        if (!d) return names;
        while (dirent* e = ::readdir(d)) {
            const std::string n = e->d_name;
            if (n.size() > 4 && n.compare(n.size() - 4, 4, ".rec") == 0)
                names.push_back(n);
        }
        ::closedir(d);
        std::sort(names.begin(), names.end());
        return names;
    }

    // Чтобы после перезапуска программы новые записи не получили номера,
    // уже лежащие в очереди.
    void restore_seq() {
        uint32_t mx = 0;
        for (const auto& n : list_queue())
            mx = std::max(mx, uint32_t(std::strtoul(n.c_str(), nullptr, 10)));
        if (mx >= next_seq_) next_seq_ = mx + 1;
    }

    // ---------------- соединение ----------------

    bool connect_once() {
        close_sock();
        sock_ = ::socket(AF_INET, SOCK_STREAM, 0);
        if (sock_ < 0) return false;

        sockaddr_in a{};
        a.sin_family = AF_INET;
        a.sin_port   = htons(port_);
        if (::inet_pton(AF_INET, ip_.c_str(), &a.sin_addr) != 1) return false;

        // Таймаут на connect через неблокирующий режим был бы точнее, но на
        // практике хватает таймаутов на чтение: мёртвое соединение обнаружится
        // по отсутствию ACK.
        timeval tv{ACK_WAIT_S, 0};
        ::setsockopt(sock_, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
        ::setsockopt(sock_, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));
        int one = 1;
        ::setsockopt(sock_, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
        ::setsockopt(sock_, SOL_SOCKET, SO_KEEPALIVE, &one, sizeof(one));

        if (::connect(sock_, reinterpret_cast<sockaddr*>(&a), sizeof(a)) != 0) {
            close_sock();
            return false;
        }
        connected_ = true;
        return true;
    }

    void close_sock() {
        if (sock_ >= 0) { ::close(sock_); sock_ = -1; }
        connected_ = false;
    }

    bool send_all(const uint8_t* p, size_t n) {
        while (n) {
            const ssize_t k = ::send(sock_, p, n, MSG_NOSIGNAL);
            if (k <= 0) return false;
            p += k; n -= size_t(k);
        }
        return true;
    }

    // Ждёт ACK с нужным номером. Чужие номера (дубликаты после переотправки)
    // пропускает - земля могла подтвердить то, что мы уже удалили.
    bool wait_ack(uint32_t seq) {
        uint8_t buf[8];
        for (int attempts = 0; attempts < 8; ++attempts) {
            size_t got = 0;
            while (got < sizeof(buf)) {
                const ssize_t k = ::recv(sock_, buf + got, sizeof(buf) - got, 0);
                if (k <= 0) return false;         // таймаут или обрыв
                got += size_t(k);
            }
            if (std::memcmp(buf, "SARA", 4) != 0) return false;   // мусор в потоке
            if (get32(&buf[4]) == seq) return true;
        }
        return false;
    }

    // ---------------- основной цикл ----------------

    void loop() {
        restore_seq();
        while (run_) {
            if (!connected_) {
                if (!connect_once()) {
                    wait_for(RECONNECT_S);
                    continue;
                }
                std::fprintf(stderr, "связь с землёй установлена, в очереди %zu\n",
                             list_queue().size());
            }

            const auto names = list_queue();
            if (names.empty()) {
                // Находок нет - шлём живую телеметрию. Она же служит
                // проверкой связи: молчащее соединение может быть мёртвым
                // часами, ядро об этом не скажет.
                if (!send_live_or_ping()) close_sock();
                wait_ms(200);
                continue;
            }

            // Отправляем по одной записи и ждём подтверждения. Так проще
            // убедиться, что ничего не потеряно, а находки редки - выигрыш
            // от конвейера был бы невелик.
            const std::string path = dir_ + "/" + names.front();
            std::vector<uint8_t> rec;
            if (!read_file(path, rec) || rec.size() < HDR) {
                ::remove(path.c_str());           // битый файл - не зацикливаться
                continue;
            }
            const uint32_t seq = get32(&rec[6]);

            if (!send_all(rec.data(), rec.size()) || !wait_ack(seq)) {
                std::fprintf(stderr, "обрыв при отправке записи %u - переподключаюсь, "
                                     "запись остаётся в очереди\n", seq);
                close_sock();
                wait_for(RECONNECT_S);
                continue;
            }
            ::remove(path.c_str());               // подтверждено землёй - можно удалять
            ++delivered_;
        }
    }

    // Возвращает false только при обрыве связи.
    bool send_live_or_ping() {
        const uint64_t now = now_us();
        if (now - last_live_us_ < LIVE_PERIOD_US) return true;

        std::string payload;
        {
            std::lock_guard<std::mutex> lk(live_mu_);
            if (!live_.empty() && now - live_set_us_ < LIVE_STALE_US) payload = live_;
        }
        if (payload.empty()) {
            // Телеметрии нет (полётник молчит) - раз в PING_S проверяем связь.
            if (now - last_ping_us_ < uint64_t(PING_S) * 1000000ull) return true;
            last_ping_us_ = now;
            return send_ping();
        }
        last_live_us_ = now;
        last_ping_us_ = now;

        const uint32_t seq = next_seq_++;
        std::vector<uint8_t> rec(HDR + payload.size());
        std::memcpy(rec.data(), "SARL", 4);
        rec[4] = 1;
        rec[5] = REC_TELEMETRY;
        put32(&rec[6],  seq);
        put32(&rec[10], 0);                       // номера события у телеметрии нет
        put32(&rec[14], uint32_t(payload.size()));
        std::memcpy(rec.data() + HDR, payload.data(), payload.size());
        return send_all(rec.data(), rec.size()) && wait_ack(seq);
    }

    void wait_ms(int ms) {
        std::unique_lock<std::mutex> lk(m_);
        cv_.wait_for(lk, std::chrono::milliseconds(ms), [this] { return !run_; });
    }

    bool send_ping() {
        uint8_t h[HDR];
        std::memcpy(h, "SARL", 4);
        h[4] = 1; h[5] = REC_PING;
        const uint32_t seq = next_seq_++;
        put32(&h[6], seq); put32(&h[10], 0); put32(&h[14], 0);
        return send_all(h, HDR) && wait_ack(seq);
    }

    void wait_for(uint32_t sec) {
        std::unique_lock<std::mutex> lk(m_);
        cv_.wait_for(lk, std::chrono::seconds(sec), [this] { return !run_; });
    }

    static bool read_file(const std::string& p, std::vector<uint8_t>& out) {
        FILE* f = std::fopen(p.c_str(), "rb");
        if (!f) return false;
        std::fseek(f, 0, SEEK_END);
        const long n = std::ftell(f);
        std::fseek(f, 0, SEEK_SET);
        out.resize(n > 0 ? size_t(n) : 0);
        const bool ok = n > 0 && std::fread(out.data(), 1, out.size(), f) == out.size();
        std::fclose(f);
        return ok;
    }

    static void put32(uint8_t* p, uint32_t v) {
        p[0] = uint8_t(v); p[1] = uint8_t(v >> 8);
        p[2] = uint8_t(v >> 16); p[3] = uint8_t(v >> 24);
    }
    static uint32_t get32(const uint8_t* p) {
        return uint32_t(p[0]) | (uint32_t(p[1]) << 8)
             | (uint32_t(p[2]) << 16) | (uint32_t(p[3]) << 24);
    }

    std::string ip_;
    uint16_t    port_;
    std::string dir_;

    std::thread th_;
    std::atomic<bool> run_{false};
    std::atomic<bool> connected_{false};
    std::atomic<uint32_t> next_seq_{1};
    std::atomic<uint64_t> delivered_{0};
    int sock_ = -1;

    mutable std::mutex m_;
    std::condition_variable cv_;

    mutable std::mutex live_mu_;
    std::string live_;
    uint64_t live_set_us_ = 0;
    uint64_t last_live_us_ = 0;
    uint64_t last_ping_us_ = 0;
};

} // namespace sar
