"""Инференс YOLO на NPU RK3566 через RKNN. Отдельный скрипт, predict.py не затрагивает.

Запуск:
    cd ~/model_detection && source .venv_npu/bin/activate
    LD_LIBRARY_PATH=$PWD/rknn python npu_detect.py

Снимки берутся из photos/, результаты кладутся в out_npu/.
"""
import time
from pathlib import Path

import cv2
import numpy as np
from rknnlite.api import RKNNLite

ROOT   = Path(__file__).parent
MODEL  = ROOT / 'rknn' / 'best_i8_split.rknn'
FOLDER = ROOT / 'photos'
OUT    = ROOT / 'out_npu'
SIZE   = 640
CONF, IOU = 0.25, 0.45
BENCH_ITERS = 30

LABEL = {0: 'person', 1: 'car', 2: 'smoke'}
COLOR = {0: (0, 200, 0), 1: (0, 140, 255), 2: (255, 120, 0)}


def letterbox(img, s=SIZE):
    h, w = img.shape[:2]
    r = min(s / h, s / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    rs = cv2.resize(img, (nw, nh))
    canvas = np.full((s, s, 3), 114, np.uint8)
    dy, dx = (s - nh) // 2, (s - nw) // 2
    canvas[dy:dy + nh, dx:dx + nw] = rs
    return canvas, r, dx, dy


def decode(outs, r, dx, dy):
    """Два выхода: [1,4,A] рамки и [1,nc,A] вероятности. Порядок определяем по форме."""
    box = cls = None
    for o in outs:
        o = np.squeeze(o, 0) if o.ndim == 3 else o
        if o.shape[0] == 4 and box is None:
            box = o
        else:
            cls = o
    if box is None or cls is None:
        raise RuntimeError(f'неожиданные выходы модели: {[o.shape for o in outs]}')

    conf = cls.max(0)
    kind = cls.argmax(0)
    keep = conf > CONF
    if not keep.any():
        return []
    b, conf, kind = box[:, keep], conf[keep], kind[keep]

    x0 = (b[0] - b[2] / 2 - dx) / r
    y0 = (b[1] - b[3] / 2 - dy) / r
    ww, hh = b[2] / r, b[3] / r
    rects = np.stack([x0, y0, ww, hh], 1)

    idx = cv2.dnn.NMSBoxes(rects.tolist(), conf.tolist(), CONF, IOU)
    idx = np.array(idx).flatten() if len(idx) else []
    return [(int(kind[i]), float(conf[i]), rects[i]) for i in idx]


assert MODEL.exists(), f'нет модели: {MODEL}'
rknn = RKNNLite()
assert rknn.load_rknn(str(MODEL)) == 0, 'load_rknn провалился'
# NPU_CORE_AUTO: у RK3566 ядро одно, параметр оставлен для совместимости
assert rknn.init_runtime() == 0, (
    'init_runtime провалился - обычно это несовпадение версий librknnrt.so и драйвера NPU')
print('модель загружена:', MODEL.name)

FOLDER.mkdir(exist_ok=True)
OUT.mkdir(exist_ok=True)
images = [p for p in FOLDER.rglob('*') if p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}]
if not images:
    print(f'\nПоложите снимки в {FOLDER} и запустите снова.')
    raise SystemExit

print(f'найдено файлов: {len(images)}\n')
times = []
for path in images:
    img = cv2.imread(str(path))
    if img is None:
        print('не читается:', path.name)
        continue
    padded, r, dx, dy = letterbox(img)
    rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)

    t0 = time.perf_counter()
    outs = rknn.inference(inputs=[np.expand_dims(rgb, 0)], data_format=['nhwc'])
    dt = time.perf_counter() - t0
    times.append(dt)

    dets = decode(outs, r, dx, dy)
    tally = {}
    width = max(1, round(min(img.shape[:2]) / 400))
    for c, s, (x, y, w, h) in dets:
        x, y, w, h = int(x), int(y), int(w), int(h)
        cv2.rectangle(img, (x, y), (x + w, y + h), COLOR[c], width)
        cv2.putText(img, f'{LABEL[c]} {s:.2f}', (x, max(14, y - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR[c], width, cv2.LINE_AA)
        tally[LABEL[c]] = tally.get(LABEL[c], 0) + 1
    cv2.imwrite(str(OUT / path.name), img)
    print(f'{path.name[:46]:48} {str(tally) if tally else "объектов не найдено":34} [{dt*1000:6.1f} мс]')

# чистый замер инференса на одном кадре
padded, r, dx, dy = letterbox(cv2.imread(str(images[0])))
rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
for _ in range(5):
    rknn.inference(inputs=[np.expand_dims(rgb, 0)], data_format=['nhwc'])
t0 = time.perf_counter()
for _ in range(BENCH_ITERS):
    rknn.inference(inputs=[np.expand_dims(rgb, 0)], data_format=['nhwc'])
pure = (time.perf_counter() - t0) / BENCH_ITERS

print(f'\nинференс по кадрам:  средн {np.mean(times)*1000:.1f} мс  ->  {1/np.mean(times):.1f} fps')
print(f'чистый инференс NPU: {pure*1000:.1f} мс  ->  {1/pure:.1f} fps  (по {BENCH_ITERS} прогонам)')
print('результаты в', OUT)
rknn.release()
