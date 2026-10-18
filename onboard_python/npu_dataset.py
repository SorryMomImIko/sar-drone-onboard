"""Прогон RKNN-модели по всей калибровочной выборке на NPU Orange Pi.
Пишет: out_calib/ - кадры с рамками, detections.csv - все детекции, sheets/ - контактные листы.
predict.py и npu_detect.py не затрагивает."""
import csv, time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from rknnlite.api import RKNNLite

ROOT  = Path(__file__).parent
MODEL = ROOT / 'rknn' / 'best_i8_split.rknn'
SRC   = ROOT / 'calib'
OUT   = ROOT / 'out_calib'
SHEET = ROOT / 'sheets'
SIZE  = 640
LOW_CONF = 0.10      # в CSV пишем всё от 0.10, чтобы потом подбирать порог без перезапуска
DRAW_CONF = 0.25     # на картинках рисуем от 0.25
IOU = 0.45

LABEL = {0: 'person', 1: 'car', 2: 'smoke'}
COLOR = {0: (0, 200, 0), 1: (0, 140, 255), 2: (255, 120, 0)}


def letterbox(img, s=SIZE):
    h, w = img.shape[:2]
    r = min(s / h, s / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    canvas = np.full((s, s, 3), 114, np.uint8)
    dy, dx = (s - nh) // 2, (s - nw) // 2
    canvas[dy:dy + nh, dx:dx + nw] = cv2.resize(img, (nw, nh))
    return canvas, r, dx, dy


def decode(outs, r, dx, dy, conf_th):
    box = cls = None
    for o in outs:
        o = np.squeeze(o, 0) if o.ndim == 3 else o
        if o.shape[0] == 4 and box is None:
            box = o
        else:
            cls = o
    conf, kind = cls.max(0), cls.argmax(0)
    keep = conf > conf_th
    if not keep.any():
        return []
    b, conf, kind = box[:, keep], conf[keep], kind[keep]
    x0 = (b[0] - b[2] / 2 - dx) / r
    y0 = (b[1] - b[3] / 2 - dy) / r
    rects = np.stack([x0, y0, b[2] / r, b[3] / r], 1)
    idx = cv2.dnn.NMSBoxes(rects.tolist(), conf.tolist(), conf_th, IOU)
    idx = np.array(idx).flatten() if len(idx) else []
    return [(int(kind[i]), float(conf[i]), rects[i]) for i in idx]


rknn = RKNNLite()
assert rknn.load_rknn(str(MODEL)) == 0
assert rknn.init_runtime() == 0
OUT.mkdir(exist_ok=True)
SHEET.mkdir(exist_ok=True)

images = sorted(SRC.glob('*.jpg'))
print(f'кадров в выборке: {len(images)}')

rows = []
per_src = defaultdict(lambda: {'n': 0, 'person': 0, 'car': 0, 'smoke': 0, 'empty': 0, 'conf': []})
drawn = defaultdict(list)
times = []

for p in images:
    img = cv2.imread(str(p))
    if img is None:
        continue
    source = p.name.split('__')[0]
    padded, r, dx, dy = letterbox(img)
    rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)

    t0 = time.perf_counter()
    outs = rknn.inference(inputs=[np.expand_dims(rgb, 0)], data_format=['nhwc'])
    times.append(time.perf_counter() - t0)

    dets = decode(outs, r, dx, dy, LOW_CONF)
    st = per_src[source]
    st['n'] += 1
    shown = 0
    vis = img.copy()
    for c, s, (x, y, w, h) in dets:
        rows.append([p.name, source, LABEL[c], round(s, 4),
                     int(x), int(y), int(w), int(h)])
        if s >= DRAW_CONF:
            st[LABEL[c]] += 1
            st['conf'].append(s)
            shown += 1
            x, y, w, h = int(x), int(y), int(w), int(h)
            cv2.rectangle(vis, (x, y), (x + w, y + h), COLOR[c], 2)
            cv2.putText(vis, f'{LABEL[c]} {s:.2f}', (x, max(12, y - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR[c], 1, cv2.LINE_AA)
    if shown == 0:
        st['empty'] += 1
    cv2.imwrite(str(OUT / p.name), vis)
    if len(drawn[source]) < 12:
        drawn[source].append(vis)

with open(ROOT / 'detections.csv', 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['image', 'source', 'class', 'conf', 'x', 'y', 'w', 'h'])
    w.writerows(rows)

print(f'\nсредний инференс: {np.mean(times)*1000:.1f} мс  ({1/np.mean(times):.1f} fps)')
print(f'\n{"источник":15} {"кадров":>7} {"person":>7} {"car":>6} {"smoke":>7} {"пустых":>7} {"ср.conf":>8}')
for s in sorted(per_src):
    d = per_src[s]
    mc = np.mean(d['conf']) if d['conf'] else 0
    print(f'{s:15} {d["n"]:7} {d["person"]:7} {d["car"]:6} {d["smoke"]:7} {d["empty"]:7} {mc:8.2f}')

for s, ims in drawn.items():
    cell = 320
    cols = 4
    rowsn = (len(ims) + cols - 1) // cols
    sheet = np.full((rowsn * cell, cols * cell, 3), 30, np.uint8)
    for i, im in enumerate(ims):
        rr, cc = divmod(i, cols)
        sheet[rr*cell:(rr+1)*cell, cc*cell:(cc+1)*cell] = cv2.resize(im, (cell, cell))
    cv2.imwrite(str(SHEET / f'{s}.jpg'), sheet, [cv2.IMWRITE_JPEG_QUALITY, 85])

print('\nкадры с рамками:', OUT)
print('все детекции (порог 0.10):', ROOT / 'detections.csv')
print('контактные листы:', SHEET)
rknn.release()
