"""
sniff_serial.py - снять дамп с полётника и определить, какой это протокол.

Нужен, когда автопилот говорит НЕ на MAVLink, а на чём-то другом, и надо
понять, на чём именно, прежде чем писать разбор. Ничего не отправляет в порт,
только слушает - полёту не мешает.

    # снять 10 секунд с USB-порта автопилота
    python sniff_serial.py /dev/serial/by-id/usb-... --seconds 10

    # разобрать уже снятый дамп (или любой бинарный лог)
    python sniff_serial.py --file dump.bin

Порт ищется так:
    ls -l /dev/serial/by-id/      <- имя привязано к серийному номеру, не меняется
    dmesg | tail                  <- что появилось при подключении

pyserial не обязателен: на Linux порт читается как обычный файл, скорость
при этом остаётся той, что настроена в системе (для USB CDC-ACM она не важна
вовсе). Если скорость надо задать - поставь pyserial: pip install pyserial

Что печатает: таблицу кандидатов с долей байт, уложившихся в правильные
кадры. Больше 90 % - это он. Меньше 30 % - совпадение случайное.
"""
import argparse
import collections
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------- детекторы
#
# Каждый детектор идёт по дампу, пытается разобрать его на кадры своего
# протокола и возвращает, сколько байт уложилось в непротиворечивые кадры.
# Это надёжнее, чем просто искать стартовый байт: в случайных данных любой
# байт встречается, а вот чтобы длина, контрольная сумма и следующий стартовый
# байт сошлись подряд много раз - нужен настоящий протокол.


def scan_mavlink_v1(b):
    """0xFE, длина, ... Следующий кадр должен начинаться ровно там, где кончился."""
    ok = i = 0
    while i + 8 <= len(b):
        if b[i] == 0xFE:
            ln = b[i + 1]
            total = 8 + ln                      # заголовок 6 + данные + CRC 2
            if i + total <= len(b) and (i + total == len(b) or b[i + total] == 0xFE):
                ok += total
                i += total
                continue
        i += 1
    return ok


def scan_mavlink_v2(b):
    """0xFD, длина, флаги... заголовок 10 байт + данные + CRC 2."""
    ok = i = 0
    while i + 12 <= len(b):
        if b[i] == 0xFD:
            ln = b[i + 1]
            incompat = b[i + 2]
            total = 12 + ln + (13 if incompat & 0x01 else 0)   # 0x01 = подпись
            if i + total <= len(b) and (i + total == len(b) or b[i + total] == 0xFD):
                ok += total
                i += total
                continue
        i += 1
    return ok


def scan_msp(b):
    """MSP v1: '$M' + направление + длина + команда + данные + XOR-сумма."""
    ok = i = 0
    while i + 6 <= len(b):
        if b[i] == 0x24 and b[i + 1] == 0x4D and b[i + 2] in (0x3C, 0x3E, 0x21):
            ln = b[i + 3]
            total = 6 + ln
            if i + total <= len(b):
                chk = 0
                for c in b[i + 3:i + 5 + ln]:
                    chk ^= c
                if chk == b[i + 5 + ln]:        # контрольная сумма сошлась
                    ok += total
                    i += total
                    continue
        i += 1
    return ok


def scan_msp_v2(b):
    """MSP v2: '$X' + направление + флаги + команда(2) + длина(2) + данные + CRC8."""
    ok = i = 0
    while i + 9 <= len(b):
        if b[i] == 0x24 and b[i + 1] == 0x58 and b[i + 2] in (0x3C, 0x3E, 0x21):
            ln = b[i + 6] | (b[i + 7] << 8)
            total = 9 + ln
            if i + total <= len(b) and ln < 1024:
                ok += total
                i += total
                continue
        i += 1
    return ok


def scan_crsf(b):
    """CRSF (ELRS/Crossfire): адрес 0xC8/0xEE/0xEC, длина, тип, данные, CRC8."""
    ok = i = 0
    while i + 4 <= len(b):
        if b[i] in (0xC8, 0xEE, 0xEC, 0xEA):
            ln = b[i + 1]
            total = 2 + ln
            if 2 <= ln <= 62 and i + total <= len(b):
                if i + total == len(b) or b[i + total] in (0xC8, 0xEE, 0xEC, 0xEA):
                    ok += total
                    i += total
                    continue
        i += 1
    return ok


def scan_ubx(b):
    """u-blox UBX: 0xB5 0x62, класс, id, длина(2), данные, два байта суммы."""
    ok = i = 0
    while i + 8 <= len(b):
        if b[i] == 0xB5 and b[i + 1] == 0x62:
            ln = b[i + 4] | (b[i + 5] << 8)
            total = 8 + ln
            if i + total <= len(b) and ln < 2048:
                ok += total
                i += total
                continue
        i += 1
    return ok


def scan_sbus(b):
    """S.BUS: кадр ровно 25 байт, 0x0F в начале, 0x00 в конце."""
    ok = i = 0
    while i + 25 <= len(b):
        if b[i] == 0x0F and b[i + 24] == 0x00:
            ok += 25
            i += 25
            continue
        i += 1
    return ok


def scan_sport(b):
    """FrSky S.Port: 0x7E, физический id, затем 8 байт."""
    ok = i = 0
    while i + 10 <= len(b):
        if b[i] == 0x7E:
            ok += 10
            i += 10
            continue
        i += 1
    return ok


def scan_duml(b):
    """DJI DUML: 0x55, длина(10 бит), версия, CRC8 заголовка."""
    ok = i = 0
    while i + 4 <= len(b):
        if b[i] == 0x55:
            ln = b[i + 1] | ((b[i + 2] & 0x03) << 8)
            if 13 <= ln <= 1024 and i + ln <= len(b):
                if i + ln == len(b) or b[i + ln] == 0x55:
                    ok += ln
                    i += ln
                    continue
        i += 1
    return ok


def scan_nmea(b):
    """NMEA 0183: текстовые строки '$GPxxx,...*hh' с переводом строки."""
    ok = 0
    for line in b.split(b"\n"):
        if line.startswith(b"$G") and b"," in line and b"*" in line:
            ok += len(line) + 1
    return ok


DETECTORS = [
    ("MAVLink v1",  scan_mavlink_v1, "ArduPilot / PX4, старая версия протокола"),
    ("MAVLink v2",  scan_mavlink_v2, "ArduPilot / PX4, текущая версия"),
    ("MSP v1",      scan_msp,        "Betaflight / INAV / iNavFlight"),
    ("MSP v2",      scan_msp_v2,     "Betaflight / INAV, новая версия"),
    ("CRSF",        scan_crsf,       "ExpressLRS / TBS Crossfire"),
    ("UBX",         scan_ubx,        "приёмник u-blox напрямую, без автопилота"),
    ("S.BUS",       scan_sbus,       "каналы управления, телеметрии в нём нет"),
    ("FrSky S.Port", scan_sport,     "телеметрия FrSky"),
    ("DJI DUML",    scan_duml,       "полётники и камеры DJI"),
    ("NMEA 0183",   scan_nmea,       "текстовые строки GPS"),
]


def analyse(data):
    n = len(data)
    print(f"\nДАМП: {n} байт")
    if n < 200:
        print("Слишком мало данных. Проверь, тот ли порт и идёт ли вообще поток.")
        return

    printable = sum(1 for c in data if 32 <= c < 127 or c in (10, 13))
    print(f"печатаемых символов: {100 * printable / n:.1f} %"
          f"{'  <- похоже на текстовый протокол' if printable / n > 0.85 else ''}")

    top = collections.Counter(data).most_common(5)
    print("самые частые байты: " +
          ", ".join(f"0x{v:02X} ({100 * c / n:.1f} %)" for v, c in top))

    print("\nКАНДИДАТЫ (доля байт, уложившихся в правильные кадры)")
    print(f"  {'протокол':<14} {'сходится':>9}   что это")
    results = []
    for name, fn, note in DETECTORS:
        try:
            share = fn(data) / n
        except (IndexError, ZeroDivisionError):
            share = 0.0
        results.append((share, name, note))
    for share, name, note in sorted(results, reverse=True):
        mark = "  <<<" if share > 0.9 else ""
        print(f"  {name:<14} {100 * share:>8.1f} %   {note}{mark}")

    best_share, best_name, _ = max(results)
    print()
    if best_share > 0.9:
        print(f"ВЫВОД: это {best_name}.")
    elif best_share > 0.3:
        print(f"ВЫВОД: похоже на {best_name}, но сходится не всё "
              f"({100 * best_share:.0f} %). Возможно, в потоке два протокола "
              f"вперемешку или дамп начат с середины кадра.")
    else:
        print("ВЫВОД: ни один из известных протоколов не подошёл. Протокол "
              "нестандартный - пришли дамп и первые байты ниже, разберём вручную.")

    print("\nПЕРВЫЕ 96 БАЙТ")
    for off in range(0, min(96, n), 16):
        chunk = data[off:off + 16]
        hexs = " ".join(f"{c:02X}" for c in chunk)
        text = "".join(chr(c) if 32 <= c < 127 else "." for c in chunk)
        print(f"  {off:04X}  {hexs:<47}  {text}")


def capture(port, seconds, baud):
    try:
        import serial                                    # pyserial, если есть
        with serial.Serial(port, baud, timeout=0.2) as s:
            print(f"слушаю {port} на {baud} бод, {seconds} с…")
            end = time.time() + seconds
            buf = bytearray()
            while time.time() < end:
                buf += s.read(4096)
            return bytes(buf)
    except ImportError:
        print(f"pyserial не установлен - читаю {port} как файл "
              f"(скорость порта берётся системная)")
        buf = bytearray()
        end = time.time() + seconds
        with open(port, "rb", buffering=0) as f:
            while time.time() < end:
                chunk = f.read(4096)
                if chunk:
                    buf += chunk
        return bytes(buf)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("port", nargs="?", help="/dev/serial/by-id/... или /dev/ttyACM0")
    ap.add_argument("--file", help="разобрать готовый дамп вместо порта")
    ap.add_argument("--seconds", type=int, default=10)
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--save", default="dump.bin")
    a = ap.parse_args()

    if a.file:
        data = Path(a.file).read_bytes()
        print(f"разбираю {a.file}")
    elif a.port:
        data = capture(a.port, a.seconds, a.baud)
        Path(a.save).write_bytes(data)
        print(f"дамп сохранён: {a.save}")
    else:
        ap.error("укажи порт или --file")

    analyse(data)


if __name__ == "__main__":
    sys.exit(main())
