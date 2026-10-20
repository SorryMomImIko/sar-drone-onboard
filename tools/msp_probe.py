"""
msp_probe.py - проверка полётника на Betaflight до запуска пайплайна.

Спрашивает у полётника версию, углы, высоту и GPS и печатает их раз в секунду.
Нужен, чтобы отделить "полётник не отвечает" от "мой код не работает" -
без этого разбираться пришлось бы в двух неизвестных сразу.

    python3 tools/msp_probe.py /dev/ttyACM0
    python3 tools/msp_probe.py /dev/serial/by-id/usb-...   # надёжнее: не зависит от порядка подключения

Только стандартная библиотека, pyserial не нужен.

ВАЖНО: сниффер (sniff_serial.py) с Betaflight покажет НОЛЬ БАЙТ, и это не
поломка. MSP - протокол "запрос-ответ": пока не спросишь, полётник молчит.

Что можно проверить ПРЯМО НА СТОЛЕ:
  - углы: наклони полётник - крен и тангаж должны меняться;
  - высоту: она барометрическая от включения, около нуля;
  - GPS: в помещении фикса не будет НИКОГДА, это нормально. Нужен модуль
    GPS, включённый в Betaflight (feature GPS + UART с функцией GPS),
    и открытое небо.
"""
import os
import struct
import sys
import termios
import time

MSP_API_VERSION = 1
MSP_FC_VARIANT  = 2
MSP_FC_VERSION  = 3
MSP_RAW_GPS     = 106
MSP_ATTITUDE    = 108
MSP_ALTITUDE    = 109


def open_port(path):
    fd = os.open(path, os.O_RDWR | os.O_NOCTTY)
    a = termios.tcgetattr(fd)
    a[0] = 0                                    # iflag: без обработки входа
    a[1] = 0                                    # oflag: без обработки выхода
    a[2] = termios.CS8 | termios.CREAD | termios.CLOCAL
    a[3] = 0                                    # lflag: без построчного режима
    a[6] = list(a[6])
    a[6][termios.VMIN] = 0
    a[6][termios.VTIME] = 2                     # 0.2 с - read не виснет
    termios.tcsetattr(fd, termios.TCSANOW, a)
    termios.tcflush(fd, termios.TCIOFLUSH)
    return fd


def ask(fd, cmd, timeout=0.5):
    """Запрос MSP v1 и ответ. None, если не ответил."""
    # '$' 'M' '<' длина команда CRC ; CRC = XOR длины, команды и данных
    os.write(fd, b"$M<" + bytes([0, cmd, cmd]))

    st, ln, got, crc, buf = 0, 0, 0, 0, bytearray()
    end = time.time() + timeout
    while time.time() < end:
        try:
            c = os.read(fd, 1)
        except OSError:
            return None
        if not c:
            continue
        b = c[0]
        if st == 0:  st = 1 if b == 0x24 else 0            # '$'
        elif st == 1: st = 2 if b == 0x4D else 0           # 'M'
        elif st == 2: st = 3 if b in (0x3E, 0x21) else 0   # '>' ответ, '!' не знает команды
        elif st == 3: ln, crc, got, buf, st = b, b, 0, bytearray(), 4
        elif st == 4: crc ^= b; st = 5 if ln else 6
        elif st == 5:
            buf.append(b); crc ^= b; got += 1
            if got >= ln: st = 6
        elif st == 6:
            return bytes(buf) if crc == b else None
    return None


def main():
    if len(sys.argv) < 2:
        print("укажи порт, например:  python3 tools/msp_probe.py /dev/ttyACM0")
        return 1
    path = sys.argv[1]
    try:
        fd = open_port(path)
    except OSError as e:
        print(f"не открылся {path}: {e}")
        print("проверь: тот ли порт (ls -l /dev/serial/by-id/), "
              "и состоишь ли в группе dialout (groups | grep dialout)")
        return 1

    # --- опознание, один раз
    var = ask(fd, MSP_FC_VARIANT)
    ver = ask(fd, MSP_FC_VERSION)
    api = ask(fd, MSP_API_VERSION)
    if var is None and ver is None:
        print("полётник МОЛЧИТ на MSP.")
        print("  - тот ли это порт? ls -l /dev/serial/by-id/")
        print("  - включён ли полётник? по USB он питается сам, светодиоды должны гореть")
        print("  - не занят ли порт? fuser -v " + path)
        print("  - не открыт ли Betaflight Configurator на другом компьютере?")
        return 2

    name = var.decode("ascii", "replace") if var else "?"
    vs = f"{ver[0]}.{ver[1]}.{ver[2]}" if ver and len(ver) >= 3 else "?"
    aps = f"{api[1]}.{api[2]}" if api and len(api) >= 3 else "?"
    print(f"полётник: {name} {vs}, MSP API {aps}\n")

    print(f"{'крен':>8}{'тангаж':>9}{'курс':>7}{'высота':>10}{'фикс':>6}{'спутн':>7}")
    try:
        while True:
            att = ask(fd, MSP_ATTITUDE)
            alt = ask(fd, MSP_ALTITUDE)
            gps = ask(fd, MSP_RAW_GPS)

            # MSP_ATTITUDE: int16 крен и тангаж в десятых долях градуса,
            #               int16 курс в ЦЕЛЫХ градусах
            if att and len(att) >= 6:
                r, p, y = struct.unpack("<hhh", att[:6])
                a_s = f"{r/10:>7.1f}°{p/10:>8.1f}°{y:>6}°"
            else:
                a_s = f"{'нет':>8}{'нет':>9}{'нет':>7}"

            # MSP_ALTITUDE: int32 высота в САНТИМЕТРАХ от включения, int16 вариометр
            alt_s = f"{struct.unpack('<i', alt[:4])[0]/100:>9.2f}м" if alt and len(alt) >= 4 \
                    else f"{'нет':>10}"

            # MSP_RAW_GPS: u8 фикс, u8 спутники, i32 широта, i32 долгота (1e7), ...
            if gps and len(gps) >= 14:
                fix, sats = gps[0], gps[1]
                lat, lon = struct.unpack("<ii", gps[2:10])
                g_s = f"{fix:>6}{sats:>7}"
                extra = f"   {lat/1e7:.6f}, {lon/1e7:.6f}" if fix and sats else ""
            else:
                g_s, extra = f"{'нет':>6}{'нет':>7}", ""

            print(a_s + alt_s + g_s + extra, flush=True)
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\nостановлено")
    finally:
        os.close(fd)
    return 0


if __name__ == "__main__":
    sys.exit(main())
