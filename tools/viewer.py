#!/usr/bin/env python3
"""
viewer.py - наземная станция с окном: принимает находки с дрона и показывает
их на экране - кадр с рамкой и всё, что пришло в JSON-метке.

Протокол тот же, что у tools/ground_server.py и ReliableLink.hpp на борту,
поэтому дрону всё равно, какая из двух программ его принимает. Запускать
надо что-то одно: обе слушают один и тот же порт.

    python3 viewer.py                               # порт 5700, папка finds_viewer/
    python3 viewer.py --port 5700 --dir rescue_17oct
    python3 viewer.py --image find.jpg --json find.json --no-listen   # просто показать файлы

Нужны tkinter и Pillow (Ubuntu: sudo apt install python3-tk python3-pil).
ImageTk не требуется: кадр передаётся в Tk как PNG, его Tk 8.6 читает сам.
"""
import argparse
import base64
import hashlib
import io
import json
import os
import queue
import re
import socket
import struct
import subprocess
import threading
import time
import webbrowser
from datetime import datetime, timezone
from pathlib import Path

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from tkinter import font as tkfont

try:
    from PIL import Image
except ImportError:
    raise SystemExit("нужен Pillow: sudo apt install python3-pil  (или pip install pillow)")

MAGIC = b"SARL"
ACK_MAGIC = b"SARA"
HDR = 18
REC_PING, REC_FRAME, REC_CROP, REC_META, REC_TELEMETRY = 0, 1, 2, 3, 4
IMG_KIND = {REC_FRAME: "frame", REC_CROP: "crop"}
KIND_TYPE = {v: k for k, v in IMG_KIND.items()}

MAX_PAYLOAD = 8 << 20   # больше ни один кадр быть не может, а мусор в потоке
                        # иначе заставит ждать несуществующие данные
IDLE_S = 45             # дрон шлёт проверку связи раз в 10 с; молчание дольше -
                        # соединение мёртвое, хотя ядро об этом не скажет
ONLINE_S = 25           # "на связи", если последний пакет пришёл не раньше

KIND_NAMES = {"person": "человек", "smoke": "задымление", "aruco": "маркер ArUco"}
CLASS_NAMES = {0: "человек", 1: "машина", 2: "задымление"}
KIND_COLORS = {"person": "#c62828", "smoke": "#e65100", "aruco": "#1565c0"}

# Имена кадров на диске: <время съёмки, мкс>_<номер события>_<frame|crop>.jpg
IMAGE_NAME = re.compile(r"^(\d+_\d+)_(frame|crop)$")

LANCZOS = getattr(Image, "Resampling", Image).LANCZOS


# ---------------------------------------------------------------- мелочи

def recv_exactly(conn, n):
    """TCP - поток байтов: один recv почти никогда не отдаёт всё сразу."""
    buf = b""
    while len(buf) < n:
        part = conn.recv(n - len(buf))
        if not part:
            return None
        buf += part
    return buf


def digest(typ, event_id, payload):
    return hashlib.sha1(struct.pack("<BI", typ, event_id) + payload).hexdigest()


def write_atomic(path, data):
    """Через временный файл: при падении на диске не останется половины
    кадра, который дрон уже считает доставленным."""
    tmp = path.with_suffix(".part")
    with tmp.open("wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


def parse_meta(text):
    """(словарь, None) или (None, исходный текст), если это не JSON-объект."""
    try:
        meta = json.loads(text)
    except json.JSONDecodeError:
        return None, text
    return (meta, None) if isinstance(meta, dict) else (None, text)


def describe(ev):
    meta = ev["meta"]
    if not meta:
        return "кадр без метки" if ev["raw"] is None else "метка не разобралась"
    kind = meta.get("kind")
    name = KIND_NAMES.get(kind) or CLASS_NAMES.get(meta.get("class_id"), str(kind or "?"))
    aid = meta.get("aruco_id", -1)
    if isinstance(aid, int) and aid >= 0:
        name += f" №{aid}"
    return name


def geo(meta):
    """(широта, долгота), только если борт сказал, что им можно верить."""
    if not meta or meta.get("geo_valid") is not True:
        return None
    try:
        return float(meta["lat"]), float(meta["lon"])
    except (KeyError, TypeError, ValueError):
        return None


def capture_time(ev):
    """Момент съёмки по часам борта (UTC), или None."""
    ts = (ev["meta"] or {}).get("timestamp_us")
    if not isinstance(ts, (int, float)):
        return None
    try:
        return datetime.fromtimestamp(ts / 1e6, timezone.utc)
    except (OverflowError, OSError, ValueError):
        return None


def hms(dt):
    return f"{dt:%H:%M:%S}.{dt.microsecond // 1000:03d}"


# ---------------------------------------------------------------- хранилище

class Store:
    """Принятые находки - на диске и в памяти.

    ПОВТОР ОПРЕДЕЛЯЕТСЯ ПО СОДЕРЖИМОМУ, А НЕ ПО НОМЕРУ СОБЫТИЯ. На борту
    номер события начинается с 1 при каждом запуске программы. Если отсеивать
    повторы по номеру, как ground_server.py, то после перезапуска борта новая
    находка №1 будет подтверждена и выброшена как "уже принятая". Переотправка
    после обрыва приходит байт в байт той же, а новая находка отличается хотя
    бы временем съёмки - поэтому сравнивается хеш записи.

    Кадр привязывается к последней метке с тем же номером: борт шлёт записи
    строго по очереди, метка события всегда идёт раньше его кадра.
    """

    def __init__(self, out_dir):
        self.dir = Path(out_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.journal = self.dir / "events.jsonl"
        self.lock = threading.Lock()
        self.events = {}     # ключ -> находка
        self.order = []      # ключи в порядке поступления
        self.by_eid = {}     # номер события -> ключ последней находки с ним
        self.seen = set()    # хеши уже принятых записей
        self._local = 0
        self._load()

    def _new(self, key, eid, received, source="drone"):
        ev = {"key": key, "event_id": eid, "meta": None, "raw": None,
              "frame": None, "crop": None, "received": received, "source": source}
        self.events[key] = ev
        self.order.append(key)
        return ev

    def _unique_key(self, ts_us, eid):
        key = f"{ts_us}_{eid:06d}"
        while key in self.events:
            ts_us += 1
            key = f"{ts_us}_{eid:06d}"
        return key

    def _load(self):
        """То, что принято в прошлые запуски: окно открывается с историей,
        а повторы старых записей не задваиваются."""
        if self.journal.exists():
            for line in self.journal.read_text(encoding="utf-8").splitlines():
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                key = rec.pop("_key", None)
                if not key or key in self.events:
                    continue
                sha = rec.pop("_sha1", None)
                received = rec.pop("_received", 0)
                ev = self._new(key, int(key.split("_")[1]), received)
                if "_raw" in rec:
                    ev["raw"] = rec["_raw"]
                else:
                    ev["meta"] = rec
                if sha:
                    self.seen.add(sha)

        for f in self.dir.glob("*.jpg"):
            m = IMAGE_NAME.match(f.stem)
            if not m:
                continue
            key, kind = m.groups()
            eid = int(key.split("_")[1])
            if key not in self.events:
                self._new(key, eid, f.stat().st_mtime)
            self.events[key][kind] = f
            self.seen.add(digest(KIND_TYPE[kind], eid, f.read_bytes()))

        self.order.sort(key=lambda k: int(k.split("_")[0]))
        for key in self.order:
            self.by_eid[self.events[key]["event_id"]] = key

    def add(self, typ, eid, payload):
        """Записывает на диск. Ключ находки - если данные новые, None - повтор.
        OSError наружу: значит, НЕ записано, и подтверждать нельзя."""
        sha = digest(typ, eid, payload)
        with self.lock:
            if sha in self.seen:
                return None
            now = time.time()
            if typ == REC_META:
                meta, raw = parse_meta(payload.decode("utf-8", "replace"))
                ts = (meta or {}).get("timestamp_us")
                ts = int(ts) if isinstance(ts, (int, float)) and ts > 0 else int(now * 1e6)
                key = self._unique_key(ts, eid)
                rec = dict(meta) if meta is not None else {"_raw": raw}
                rec.update(_key=key, _sha1=sha, _received=now)
                with self.journal.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    f.flush()
                    os.fsync(f.fileno())
                ev = self._new(key, eid, now)
                ev["meta"], ev["raw"] = meta, raw
                self.by_eid[eid] = key
            else:
                kind = IMG_KIND[typ]
                key = self.by_eid.get(eid)
                # Метки нет (не попала в очередь на борту) или у неё уже есть
                # такой кадр - это отдельная находка.
                if key is None or self.events[key][kind] is not None:
                    key = self._unique_key(int(now * 1e6), eid)
                    self._new(key, eid, now)
                    self.by_eid[eid] = key
                path = self.dir / f"{key}_{kind}.jpg"
                write_atomic(path, payload)
                self.events[key][kind] = path
            self.seen.add(sha)
            return key

    def add_local(self, image, meta, raw, eid=-1):
        """Файлы, открытые вручную: только показать, на диск не пишем."""
        with self.lock:
            self._local += 1
            key = f"file:{self._local}"
            ev = self._new(key, eid, time.time(), source="file")
            ev["meta"], ev["raw"] = meta, raw
            ev["frame"] = Path(image) if image else None
            return key

    def get(self, key):
        with self.lock:
            return dict(self.events[key])

    def count(self):
        with self.lock:
            return sum(1 for e in self.events.values() if e["source"] == "drone")


# ---------------------------------------------------------------- приём по сети

class Server:
    def __init__(self, port, store, notify):
        self.port, self.store, self.notify = port, store, notify
        self.lock = threading.Lock()
        self.peers = {}      # номер соединения -> [адрес, время последнего пакета]
        # Последняя телеметрия: важно ТОЛЬКО последнее значение, поэтому
        # не очередь, а одна ячейка. Пишет поток приёма, читает окно.
        self.telemetry = None
        self.telemetry_t = 0.0
        self.sock = None
        self._next = 0

    def start(self):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(("0.0.0.0", self.port))
        s.listen(4)
        self.sock = s
        threading.Thread(target=self._accept, daemon=True).start()

    def peer_status(self):
        with self.lock:
            return [tuple(v) for v in self.peers.values()]

    def _log(self, text):
        print(text, flush=True)
        self.notify(("log", text))

    def _telemetry(self, payload):
        try:
            t = json.loads(payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return
        with self.lock:
            self.telemetry = t
            self.telemetry_t = time.time()
        try:
            with (self.store.dir / "telemetry.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps(t, ensure_ascii=False) + "\n")
        except OSError:
            pass            # журнал телеметрии не критичен, окно важнее

    def live_telemetry(self):
        with self.lock:
            return self.telemetry, self.telemetry_t

    def _touch(self, cid, who):
        with self.lock:
            self.peers[cid] = [who, time.time()]

    def _accept(self):
        while True:
            try:
                conn, addr = self.sock.accept()
            except OSError:
                return
            self._next += 1
            threading.Thread(target=self._handle, args=(conn, addr[0], self._next),
                             daemon=True).start()

    def _handle(self, conn, who, cid):
        conn.settimeout(IDLE_S)
        self._touch(cid, who)
        self._log(f"[{who}] подключился")
        count = 0
        try:
            while True:
                head = recv_exactly(conn, HDR)
                if head is None:
                    break
                if head[:4] != MAGIC:
                    self._log(f"[{who}] мусор в потоке, рву соединение")
                    break
                ver, typ = head[4], head[5]
                seq, eid, length = struct.unpack_from("<III", head, 6)
                if ver != 1 or length > MAX_PAYLOAD:
                    self._log(f"[{who}] непонятная запись (версия {ver}, длина {length})")
                    break
                payload = recv_exactly(conn, length) if length else b""
                if payload is None:
                    self._log(f"[{who}] обрыв на середине записи {seq} - "
                              f"не подтверждаю, дрон пришлёт заново")
                    break
                self._touch(cid, who)

                if typ in (REC_META, REC_FRAME, REC_CROP):
                    try:
                        key = self.store.add(typ, eid, payload)
                    except OSError as e:
                        self._log(f"[событие {eid}] не записалось на диск ({e}) - НЕ подтверждаю")
                        break
                    count += 1
                    if key:
                        self.notify(("event", key))
                    else:
                        self._log(f"[событие {eid}] повтор - уже принято, подтверждаю ещё раз")
                elif typ == REC_TELEMETRY:
                    # Не в лог: телеметрия идёт раз в секунду и забила бы его.
                    # Показывается отдельной строкой в окне, пишется в журнал.
                    self._telemetry(payload)
                elif typ != REC_PING:
                    self._log(f"[{who}] запись неизвестного типа {typ} - пропускаю")

                # ACK только теперь, когда данные уже на диске.
                conn.sendall(ACK_MAGIC + struct.pack("<I", seq))
        except (socket.timeout, OSError) as e:
            self._log(f"[{who}] ошибка связи: {e}")
        finally:
            conn.close()
            with self.lock:
                self.peers.pop(cid, None)
            self._log(f"[{who}] связь разорвана, принято записей: {count}")


# ---------------------------------------------------------------- окно

class App:
    def __init__(self, root, store, server, port, events):
        self.root, self.store, self.server, self.port, self.q = root, store, server, port, events
        self.current = None
        self.pil = None
        self.photo = None          # ссылку держим, иначе Tk покажет пустоту
        self.placeholder = "Находок пока нет.\nЖду дрон…"
        self._resize_job = None
        self.follow = tk.BooleanVar(value=True)
        self.status = tk.StringVar()
        self.telemetry_line = tk.StringVar(value="телеметрия: нет данных")
        self.last_log = tk.StringVar()

        base = tkfont.nametofont("TkDefaultFont")
        self.bold = base.copy()
        self.bold.configure(weight="bold")
        self.value_font = base.copy()
        self.value_font.configure(size=base.cget("size") + 1)

        self._build()
        for key in store.order:
            self._add_row(key, unseen=False)
        if store.order:
            self._select(store.order[-1])
        self.root.after(100, self._poll)
        self._tick()

    # ---- раскладка

    def _build(self):
        root = self.root
        root.title("SAR — находки с дрона")
        root.geometry("1360x820")
        root.minsize(900, 560)

        top = ttk.Frame(root, padding=(8, 6))
        top.pack(fill="x")
        ttk.Label(top, textvariable=self.status, font=self.bold).pack(side="left")
        ttk.Button(top, text="Открыть файлы…", command=self.open_files).pack(side="right")
        ttk.Button(top, text="Папка с находками", command=self.open_dir).pack(side="right", padx=6)
        ttk.Checkbutton(top, text="Сразу показывать новые",
                        variable=self.follow).pack(side="right", padx=6)

        # Вторая строка - живая телеметрия с полётника. Отдельно от строки
        # состояния: она меняется раз в секунду, а состояние связи - редко.
        tel = ttk.Frame(root, padding=(8, 0))
        tel.pack(fill="x")
        ttk.Label(tel, textvariable=self.telemetry_line,
                  foreground="#6aa84f").pack(side="left")

        pane = ttk.PanedWindow(root, orient="horizontal")
        pane.pack(fill="both", expand=True, padx=8)

        left = ttk.Frame(pane)
        pane.add(left, weight=1)
        cols = (("time", "Время", 95, "center"), ("what", "Что", 160, "w"),
                ("conf", "Увер.", 55, "center"), ("coords", "Координаты", 180, "w"))
        self.tree = ttk.Treeview(left, columns=[c[0] for c in cols],
                                 show="headings", selectmode="browse")
        for name, title, width, anchor in cols:
            self.tree.heading(name, text=title)
            self.tree.column(name, width=width, anchor=anchor, stretch=(name == "coords"))
        sb = ttk.Scrollbar(left, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=sb.set)
        self.tree.pack(side="left", fill="both", expand=True)
        sb.pack(side="left", fill="y")
        for kind, color in KIND_COLORS.items():
            self.tree.tag_configure(kind, foreground=color)
        self.tree.tag_configure("unseen", font=self.bold)
        self.tree.bind("<<TreeviewSelect>>", self._on_select)

        right = ttk.Frame(pane, padding=(8, 0, 0, 0))
        pane.add(right, weight=3)
        self.img_box = tk.Frame(right, bg="#1e1e1e")
        self.img_box.pack(fill="both", expand=True)
        self.img_label = tk.Label(self.img_box, bg="#1e1e1e", fg="#bbbbbb",
                                  text=self.placeholder, font=self.value_font)
        self.img_label.place(relx=0.5, rely=0.5, anchor="center")
        self.img_box.bind("<Configure>", self._on_resize)

        info = ttk.Frame(right, padding=(0, 8, 0, 0))
        info.pack(fill="x")
        self.fields = {}
        rows = (("what", "Что"), ("conf", "Уверенность"), ("coords", "Координаты"),
                ("gps", "GPS"), ("time", "Снято"), ("event", "Событие"), ("recv", "Принято"))
        for i, (name, title) in enumerate(rows):
            r, c = i % 4, (i // 4) * 2
            ttk.Label(info, text=title + ":", foreground="#777").grid(
                row=r, column=c, sticky="w", padx=(0 if c == 0 else 28, 6), pady=1)
            var = tk.StringVar(value="—")
            self.fields[name] = var
            ttk.Label(info, textvariable=var, font=self.value_font).grid(
                row=r, column=c + 1, sticky="w")
        info.columnconfigure(4, weight=1)
        btns = ttk.Frame(info)
        btns.grid(row=0, column=4, rowspan=4, sticky="ne")
        self.btn_map = ttk.Button(btns, text="Открыть на карте", command=self.open_map)
        self.btn_map.pack(fill="x")
        self.btn_copy = ttk.Button(btns, text="Копировать", command=self.copy_coords)
        self.btn_copy.pack(fill="x", pady=(4, 0))

        ttk.Label(right, text="JSON-метка:", foreground="#777").pack(anchor="w", pady=(8, 2))
        self.json_text = tk.Text(right, height=8, wrap="word", relief="flat",
                                 font=tkfont.nametofont("TkFixedFont"))
        self.json_text.pack(fill="x")
        self.json_text.configure(state="disabled")

        bottom = ttk.Frame(root, padding=(8, 4, 8, 6))
        bottom.pack(fill="x")
        ttk.Label(bottom, textvariable=self.last_log, foreground="#777").pack(side="left")

    # ---- список

    def _row_values(self, ev):
        dt = capture_time(ev)
        t = hms(dt.astimezone()) if dt else datetime.fromtimestamp(ev["received"]).strftime("%H:%M:%S")
        meta = ev["meta"] or {}
        if ev["source"] == "file":
            conf = ""
        elif meta.get("aruco_id", -1) not in (-1, None):
            conf = "код"
        elif isinstance(meta.get("confidence"), (int, float)):
            conf = f"{meta['confidence']:.2f}"
        else:
            conf = ""
        g = geo(meta)
        coords = f"{g[0]:.6f}, {g[1]:.6f}" if g else ("нет" if meta else "")
        return t, describe(ev), conf, coords

    def _add_row(self, key, unseen):
        ev = self.store.get(key)
        kind = (ev["meta"] or {}).get("kind")
        tags = [kind] if kind in KIND_COLORS else []
        if self.tree.exists(key):
            if "unseen" in self.tree.item(key, "tags"):
                tags.append("unseen")
            self.tree.item(key, values=self._row_values(ev), tags=tags)
        else:
            if unseen:
                tags.append("unseen")
            self.tree.insert("", 0, iid=key, values=self._row_values(ev), tags=tags)

    def _select(self, key):
        self.tree.selection_set(key)
        self.tree.see(key)

    def _on_select(self, _event=None):
        sel = self.tree.selection()
        if not sel:
            return
        key = sel[0]
        tags = [t for t in self.tree.item(key, "tags") if t != "unseen"]
        self.tree.item(key, tags=tags)
        self.show(key)

    # ---- карточка находки

    def show(self, key):
        ev = self.store.get(key)
        self.current = key
        self.pil = None
        path = ev["frame"] or ev["crop"]
        if path:
            try:
                with Image.open(path) as im:
                    self.pil = im.convert("RGB")
            except OSError as e:
                self.placeholder = f"Кадр не открылся:\n{e}"
        elif ev["source"] == "drone":
            self.placeholder = "Кадра пока нет —\nещё в пути или не отправлялся"
        else:
            self.placeholder = "Кадра нет"
        self._render()

        meta = ev["meta"] or {}
        f = self.fields
        f["what"].set(describe(ev))
        if meta.get("aruco_id", -1) not in (-1, None):
            f["conf"].set("— (код маркера проверен)")
        elif isinstance(meta.get("confidence"), (int, float)):
            f["conf"].set(f"{meta['confidence']:.2f}")
        else:
            f["conf"].set("—")
        g = geo(meta)
        if g:
            f["coords"].set(f"{g[0]:.6f}, {g[1]:.6f}")
        elif meta.get("geo_valid") is False:
            f["coords"].set("НЕТ — геопривязка не удалась")
        else:
            f["coords"].set("—")
        if "fix_type" in meta or "sats" in meta:
            f["gps"].set(f"фикс {meta.get('fix_type', '?')}, спутников {meta.get('sats', '?')}")
        else:
            f["gps"].set("—")
        dt = capture_time(ev)
        f["time"].set(f"{dt.astimezone():%d.%m.%Y} {hms(dt.astimezone())}  ({hms(dt)} UTC)"
                      if dt else "—")
        if ev["source"] == "file":
            f["event"].set("открыто из файла")
        else:
            f["event"].set(f"№{ev['event_id']}")
        f["recv"].set(datetime.fromtimestamp(ev["received"]).strftime("%d.%m.%Y %H:%M:%S"))

        if ev["meta"] is not None:
            text = json.dumps(ev["meta"], ensure_ascii=False, indent=2)
        else:
            text = ev["raw"] if ev["raw"] is not None else "(метки нет)"
        self.json_text.configure(state="normal")
        self.json_text.delete("1.0", "end")
        self.json_text.insert("1.0", text)
        self.json_text.configure(state="disabled")

        state = "normal" if g else "disabled"
        self.btn_map.configure(state=state)
        self.btn_copy.configure(state=state)

    def _on_resize(self, _event=None):
        if self._resize_job:
            self.root.after_cancel(self._resize_job)
        self._resize_job = self.root.after(60, self._render)

    def _render(self):
        self._resize_job = None
        if self.pil is None:
            self.photo = None
            self.img_label.configure(image="", text=self.placeholder)
            return
        w, h = self.img_box.winfo_width(), self.img_box.winfo_height()
        if w < 20 or h < 20:
            return
        k = min(w / self.pil.width, h / self.pil.height)
        im = self.pil.resize((max(1, int(self.pil.width * k)),
                              max(1, int(self.pil.height * k))), LANCZOS)
        buf = io.BytesIO()
        im.save(buf, format="PNG", compress_level=1)
        self.photo = tk.PhotoImage(data=base64.b64encode(buf.getvalue()).decode("ascii"))
        self.img_label.configure(image=self.photo, text="")

    # ---- кнопки

    def _current_geo(self):
        return geo(self.store.get(self.current)["meta"]) if self.current else None

    def open_map(self):
        g = self._current_geo()
        if g:
            webbrowser.open(f"https://www.openstreetmap.org/?mlat={g[0]:.7f}&mlon={g[1]:.7f}"
                            f"#map=18/{g[0]:.7f}/{g[1]:.7f}")

    def copy_coords(self):
        g = self._current_geo()
        if g:
            self.root.clipboard_clear()
            self.root.clipboard_append(f"{g[0]:.7f}, {g[1]:.7f}")
            self.last_log.set("координаты скопированы")

    def open_dir(self):
        try:
            subprocess.Popen(["xdg-open", str(self.store.dir.resolve())])
        except OSError as e:
            messagebox.showerror("Папка с находками", str(e))

    def open_files(self):
        image = filedialog.askopenfilename(
            title="Кадр находки",
            filetypes=[("Изображения", "*.jpg *.jpeg *.png *.bmp"), ("Все файлы", "*")])
        if not image:
            return
        js = filedialog.askopenfilename(
            title="JSON-метка (можно отменить)",
            initialdir=str(Path(image).parent),
            filetypes=[("JSON", "*.json *.jsonl"), ("Все файлы", "*")])
        self.load_files(image, js or None)

    def load_files(self, image, json_path):
        meta = raw = None
        if json_path:
            try:
                meta, raw = parse_meta(Path(json_path).read_text(encoding="utf-8"))
            except OSError as e:
                messagebox.showerror("JSON-метка", f"Не читается {json_path}:\n{e}")
                return
        if image and not Path(image).exists():
            messagebox.showerror("Кадр", f"Нет файла {image}")
            return
        eid = meta.get("event_id", -1) if meta else -1
        key = self.store.add_local(image, meta, raw, eid if isinstance(eid, int) else -1)
        self._add_row(key, unseen=False)
        self._select(key)

    # ---- события из потоков приёма

    def _poll(self):
        try:
            while True:
                kind, data = self.q.get_nowait()
                if kind == "event":
                    new = not self.tree.exists(data)
                    self._add_row(data, unseen=new and not self.follow.get())
                    if new:
                        self.root.bell()
                        if self.follow.get():
                            self._select(data)
                    elif data == self.current:
                        self.show(data)       # пришёл кадр к уже показанной метке
                elif kind == "log":
                    self.last_log.set(time.strftime("%H:%M:%S  ") + data)
        except queue.Empty:
            pass
        self.root.after(100, self._poll)

    def _tick(self):
        if self.server is None:
            s = "Приём выключен — только просмотр"
        else:
            now = time.time()
            peers = [(who, now - last) for who, last in self.server.peer_status()]
            if peers:
                who, ago = min(peers, key=lambda p: p[1])
                s = (f"Дрон на связи: {who}" if ago < ONLINE_S
                     else f"Дрон {who} молчит {ago:.0f} с")
            else:
                s = f"Жду дрон на порту {self.port}"
        self.status.set(f"{s}    ·    находок: {self.store.count()}")
        self.telemetry_line.set(self._telemetry_text())
        self.root.after(1000, self._tick)

    def _telemetry_text(self):
        """Строка живой телеметрии. Три неисправности выглядят по-разному:
        порт не открылся, полётник молчит, полётник отвечает без GPS -
        на земле это три разных действия."""
        if self.server is None:
            return ""
        t, when = self.server.live_telemetry()
        if t is None:
            return "телеметрия: нет данных"
        ago = time.time() - when
        if ago > 5:
            return f"телеметрия: молчит {ago:.0f} с"
        if not t.get("fc_open"):
            return "телеметрия: ПОЛЁТНИК НЕ ПОДКЛЮЧЁН (порт не открылся)"
        if not t.get("fc_link"):
            return (f"телеметрия: полётник МОЛЧИТ по MSP "
                    f"(ответов: {t.get('msp_messages', 0)})")
        if "roll_deg" in t:
            angles = (f"крен {t['roll_deg']:+.1f}°  тангаж {t['pitch_deg']:+.1f}°  "
                      f"курс {t['yaw_deg']:.0f}°")
        else:
            angles = "углов нет"
        alt = f"высота {t['alt_m']:.1f} м" if "alt_m" in t else "высоты нет"
        # Подставная поза должна быть видна сразу: иначе тестовую координату
        # легко принять за настоящую и поехать не туда.
        mark = "⚠ ТЕСТОВАЯ ПОЗА  ·  " if t.get("fake_pose") else ""
        if t.get("geo_valid"):
            where = f"{t['lat']:.6f} {t['lon']:.6f}"
        elif "fix_type" in t:
            where = f"координат нет (фикс {t['fix_type']}/{t.get('sats', 0)})"
        else:
            where = "GPS не отвечает"
        return f"телеметрия: {mark}{where}   ·   {alt}   ·   {angles}"



def main():
    ap = argparse.ArgumentParser(description="Наземная станция с окном: приём и показ находок")
    ap.add_argument("--port", type=int, default=5700)
    ap.add_argument("--dir", default="finds_viewer", help="куда сохранять принятое")
    ap.add_argument("--image", help="сразу показать этот кадр")
    ap.add_argument("--json", help="и эту JSON-метку к нему")
    ap.add_argument("--no-listen", action="store_true", help="не принимать по сети, только показать")
    a = ap.parse_args()

    store = Store(a.dir)
    events = queue.Queue()
    root = tk.Tk()

    server = None
    if not a.no_listen:
        server = Server(a.port, store, events.put)
        try:
            server.start()
            print(f"жду дрон на порту {a.port}, пишу в {store.dir.resolve()}", flush=True)
        except OSError as e:
            server = None
            messagebox.showerror(
                "Приём не запущен",
                f"Порт {a.port} занят или недоступен: {e}\n\n"
                f"Возможно, уже запущен ground_server.py или второе окно.\n"
                f"Окно откроется только для просмотра.")

    app = App(root, store, server, a.port, events)
    if a.image or a.json:
        app.load_files(a.image, a.json)
    root.mainloop()


if __name__ == "__main__":
    main()
