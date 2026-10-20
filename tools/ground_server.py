"""
ground_server.py - наземная станция: принимает находки по TCP и подтверждает.

Дрон подключается сюда сам и переподключается после каждого обрыва, поэтому
сервер можно запустить один раз и забыть. Подтверждение (ACK) отправляется
ТОЛЬКО после того, как данные записаны на диск и сброшены на него: иначе
дрон удалит находку из очереди, а она пропадёт при падении сервера.

    python ground_server.py                       # порт 5700, папка finds_received
    python ground_server.py --dir rescue_17oct --port 5700

Что печатает:

    [10.0.0.9] подключился
    [событие 4] метка: человек, 0.82, 59.938601 30.314102, фикс 3/11
    [событие 5] метка: маркер ArUco №7, 59.938612 30.314090, фикс 3/11
    [событие 4] frame: 11061 байт -> rescue_17oct/event_000004_frame.jpg
    [10.0.0.9] связь разорвана, принято записей: 2
"""
import argparse
import json
import os
import socket
import struct
import sys
import threading
from pathlib import Path

MAGIC = b"SARL"
ACK_MAGIC = b"SARA"
HDR = 18
REC_PING, REC_FRAME, REC_CROP, REC_META, REC_TELEMETRY = 0, 1, 2, 3, 4
KIND = {REC_FRAME: "frame", REC_CROP: "crop"}
CLASS_NAMES = {0: "человек", 1: "машина", 2: "задымление"}
KIND_NAMES = {"person": "человек", "smoke": "задымление", "aruco": "маркер ArUco"}

MAX_PAYLOAD = 8 << 20   # 8 МБ: больше ни один кадр быть не может, а мусор в
                        # потоке иначе заставит сервер ждать несуществующие данные


def recv_exactly(conn, n):
    """TCP - поток байтов: один recv почти никогда не отдаёт всё сразу."""
    buf = b""
    while len(buf) < n:
        part = conn.recv(n - len(buf))
        if not part:
            return None
        buf += part
    return buf


def load_seen(out_dir):
    """Что уже принято раньше - чтобы не задвоить после перезапуска сервера."""
    seen = set()
    path = out_dir / "events.jsonl"
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                seen.add((json.loads(line).get("event_id"), REC_META))
            except json.JSONDecodeError:
                pass
    for f in out_dir.glob("event_*_*.jpg"):
        try:
            num = int(f.stem.split("_")[1])
        except (IndexError, ValueError):
            continue
        for t, k in KIND.items():
            if f.stem.endswith(k):
                seen.add((num, t))
    return seen


def handle(conn, addr, out_dir, lock, seen):
    who = f"[{addr[0]}]"
    print(f"{who} подключился", flush=True)
    count = 0
    try:
        while True:
            head = recv_exactly(conn, HDR)
            if head is None:
                break
            if head[:4] != MAGIC:
                print(f"{who} мусор в потоке, рву соединение", flush=True)
                break
            ver, typ = head[4], head[5]
            seq, event_id, length = struct.unpack_from("<III", head, 6)
            if ver != 1 or length > MAX_PAYLOAD:
                print(f"{who} непонятная запись (версия {ver}, длина {length})",
                      flush=True)
                break

            payload = recv_exactly(conn, length) if length else b""
            if payload is None:
                print(f"{who} обрыв на середине записи {seq} - "
                      f"НЕ подтверждаю, дрон пришлёт заново", flush=True)
                break

            # Телеметрия идёт потоком, по ней нет номера события и
            # защищаться от повторов не нужно - важно последнее значение.
            if typ == REC_TELEMETRY:
                with lock:
                    store_telemetry(payload, out_dir)
                count += 1
            elif typ != REC_PING:
                # Подтверждение могло не дойти до дрона, и он прислал запись
                # заново. Данные те же, но в журнал её второй раз писать
                # нельзя - на карте появится вторая метка о том же человеке.
                with lock:
                    if (event_id, typ) in seen:
                        print(f"[событие {event_id}] повтор - уже принято, "
                              f"подтверждаю ещё раз", flush=True)
                    else:
                        store(typ, event_id, payload, out_dir)
                        seen.add((event_id, typ))
                count += 1

            # ACK только теперь, когда данные уже на диске.
            conn.sendall(ACK_MAGIC + struct.pack("<I", seq))
    except (ConnectionResetError, socket.timeout, OSError) as e:
        print(f"{who} ошибка связи: {e}", flush=True)
    finally:
        conn.close()
        print(f"{who} связь разорвана, принято записей: {count}", flush=True)


def store_telemetry(payload, out_dir):
    """Живая телеметрия: дописываем в журнал и показываем одной строкой."""
    try:
        t = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return
    with (out_dir / "telemetry.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(t, ensure_ascii=False) + "\n")

    # Три разных неисправности выглядят по-разному: порт не открылся,
    # порт открыт но полётник молчит, полётник отвечает но без GPS.
    # Для оператора это три разных действия, поэтому и строки разные.
    if not t.get("fc_open"):
        print("  телеметрия: ПОЛЁТНИК НЕ ПОДКЛЮЧЁН (порт не открылся)", flush=True)
        return
    if not t.get("fc_link"):
        print(f"  телеметрия: полётник МОЛЧИТ по MSP "
              f"(ответов всего: {t.get('msp_messages', 0)})", flush=True)
        return

    if "roll_deg" in t:
        angles = (f"крен {t['roll_deg']:+.1f}° тангаж {t['pitch_deg']:+.1f}° "
                  f"курс {t['yaw_deg']:.0f}°")
    else:
        angles = "углов нет"
    alt = f"высота {t['alt_m']:.1f} м" if "alt_m" in t else "высоты нет"
    mark = "ТЕСТОВАЯ ПОЗА: " if t.get("fake_pose") else ""
    if t.get("geo_valid"):
        where = f"{t['lat']:.6f} {t['lon']:.6f}"
    elif "fix_type" in t:
        where = f"координат нет (фикс {t['fix_type']}/{t.get('sats', 0)})"
    else:
        where = "GPS не отвечает"
    print(f"  телеметрия: {mark}{where}, {alt}, {angles}", flush=True)


def store(typ, event_id, payload, out_dir):
    if typ == REC_META:
        try:
            ev = json.loads(payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            print(f"[событие {event_id}] метка не разобралась", flush=True)
            return
        with (out_dir / "events.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(ev, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
        kind = ev.get("kind")
        name = KIND_NAMES.get(kind) or CLASS_NAMES.get(
            ev.get("class_id"), f"класс {ev.get('class_id')}")
        if ev.get("aruco_id", -1) >= 0:
            name += f" №{ev['aruco_id']}"
            conf = ""                      # у маркера уверенности нет, код проверен
        else:
            conf = f", {ev.get('confidence', 0):.2f}"
        where = (f"{ev['lat']:.6f} {ev['lon']:.6f}" if ev.get("geo_valid")
                 else "БЕЗ КООРДИНАТ")
        print(f"[событие {event_id}] метка: {name}{conf}, "
              f"{where}, фикс {ev.get('fix_type')}/{ev.get('sats')}", flush=True)
        return

    kind = KIND.get(typ, f"type{typ}")
    path = out_dir / f"event_{event_id:06d}_{kind}.jpg"
    # Запись через временный файл: при падении сервера на диске не окажется
    # половины картинки, которую дрон уже считает доставленной.
    tmp = path.with_suffix(".part")
    with tmp.open("wb") as f:
        f.write(payload)
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)
    print(f"[событие {event_id}] {kind}: {len(payload)} байт -> {path}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=5700)
    ap.add_argument("--dir", default="finds_received")
    a = ap.parse_args()

    out_dir = Path(a.dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    lock = threading.Lock()
    seen = load_seen(out_dir)
    if seen:
        print(f"уже принято раньше: {len(seen)} записей - повторы отброшу")

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("0.0.0.0", a.port))
    srv.listen(4)
    print(f"жду дрон на порту {a.port}, пишу в {out_dir.resolve()}")
    print("Ctrl+C чтобы остановить\n", flush=True)

    try:
        while True:
            conn, addr = srv.accept()
            threading.Thread(target=handle, args=(conn, addr, out_dir, lock, seen),
                             daemon=True).start()
    except KeyboardInterrupt:
        print("\nостановлено")
    finally:
        srv.close()


if __name__ == "__main__":
    sys.exit(main())
