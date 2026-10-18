"""Одновременный поиск людей (YOLO на NPU) и маркеров ArUco (CPU).

Два детектора работают параллельно: пока NPU считает сеть, процессор
ищет маркеры. И rknn, и cv2.aruco отпускают GIL, поэтому это настоящая
параллельность, а не чередование.

Запуск:
    cd ~/model_detection && source .venv_npu/bin/activate
    LD_LIBRARY_PATH=$PWD/rknn python detect_both.py aruco/images
    LD_LIBRARY_PATH=$PWD/rknn python detect_both.py /path/video.mp4
    LD_LIBRARY_PATH=$PWD/rknn python detect_both.py 0          # камера
"""
import sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from rknnlite.api import RKNNLite

sys.path.insert(0, str(Path(__file__).parent / 'aruco'))
from aruco_detect import build_detectors, detect as aruco_detect, DICTS

ROOT  = Path(__file__).parent
MODEL = ROOT / 'rknn' / 'best_i8_split.rknn'
OUT   = ROOT / 'out_both'
SIZE, CONF, IOU = 640, 0.25, 0.45
LABEL = {0: 'person', 1: 'car', 2: 'smoke'}
COLOR = {0: (0, 200, 0), 1: (0, 140, 255), 2: (255, 120, 0)}
ARUCO_COLOR = (0, 230, 255)


def letterbox(img, s=SIZE):
    h, w = img.shape[:2]
    r = min(s / h, s / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    c = np.full((s, s, 3), 114, np.uint8)
    dy, dx = (s - nh) // 2, (s - nw) // 2
    c[dy:dy + nh, dx:dx + nw] = cv2.resize(img, (nw, nh))
    return c, r, dx, dy


def decode(outs, r, dx, dy):
    box = cls = None
    for o in outs:
        o = np.squeeze(o, 0) if o.ndim == 3 else o
        if o.shape[0] == 4 and box is None:
            box = o
        else:
            cls = o
    conf, kind = cls.max(0), cls.argmax(0)
    keep = conf > CONF
    if not keep.any():
        return []
    b, conf, kind = box[:, keep], conf[keep], kind[keep]
    x0 = (b[0] - b[2] / 2 - dx) / r
    y0 = (b[1] - b[3] / 2 - dy) / r
    rects = np.stack([x0, y0, b[2] / r, b[3] / r], 1)
    idx = cv2.dnn.NMSBoxes(rects.tolist(), conf.tolist(), CONF, IOU)
    idx = np.array(idx).flatten() if len(idx) else []
    return [(int(kind[i]), float(conf[i]), rects[i]) for i in idx]


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else 'aruco/images'
    rknn = RKNNLite()
    assert rknn.load_rknn(str(MODEL)) == 0, 'не загрузилась модель'
    assert rknn.init_runtime() == 0, 'не поднялся рантайм NPU'
    markers = build_detectors(DICTS)
    OUT.mkdir(exist_ok=True)
    pool = ThreadPoolExecutor(max_workers=1)

    t_yolo, t_aruco, t_frame = [], [], []
    n_people = n_mark = frames = 0

    def process(img):
        nonlocal n_people, n_mark
        t0 = time.perf_counter()
        fut = pool.submit(lambda: (lambda s: (aruco_detect(img, markers),
                                              time.perf_counter() - s))(time.perf_counter()))
        pad, r, dx, dy = letterbox(img)
        rgb = cv2.cvtColor(pad, cv2.COLOR_BGR2RGB)
        ty = time.perf_counter()
        outs = rknn.inference(inputs=[np.expand_dims(rgb, 0)], data_format=['nhwc'])
        dets = decode(outs, r, dx, dy)
        t_yolo.append(time.perf_counter() - ty)
        found, ta = fut.result()
        t_aruco.append(ta)
        t_frame.append(time.perf_counter() - t0)
        n_people += sum(1 for c, _, _ in dets if c == 0)
        n_mark += len(found)
        return dets, found

    def draw(img, dets, found):
        for c, s, (x, y, w, h) in dets:
            x, y, w, h = int(x), int(y), int(w), int(h)
            cv2.rectangle(img, (x, y), (x + w, y + h), COLOR[c], 2)
            cv2.putText(img, f'{LABEL[c]} {s:.2f}', (x, max(12, y - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR[c], 1, cv2.LINE_AA)
        for f in found:
            pts = f['corners'].astype(int)
            cv2.polylines(img, [pts], True, ARUCO_COLOR, 2, cv2.LINE_AA)
            c = pts.mean(0).astype(int)
            cv2.putText(img, f"ArUco id{f['id']}", (c[0] - 28, c[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, ARUCO_COLOR, 2, cv2.LINE_AA)
        return img

    p = Path(src)
    if p.is_dir():
        files = sorted(x for x in p.iterdir() if x.suffix.lower() in {'.jpg', '.jpeg', '.png'})
        print('кадров:', len(files))
        for f in files:
            img = cv2.imread(str(f))
            if img is None:
                continue
            dets, found = process(img)
            frames += 1
            if dets or found:
                cv2.imwrite(str(OUT / f.name), draw(img, dets, found))
    else:
        cap = cv2.VideoCapture(int(src) if src.isdigit() else src)
        if not cap.isOpened():
            print('не открылся источник:', src)
            return
        while True:
            ok, img = cap.read()
            if not ok:
                break
            dets, found = process(img)
            frames += 1
            if frames % 30 == 0:
                print(f'  кадр {frames} | люди {n_people} | маркеры {n_mark} | '
                      f'{np.mean(t_frame)*1000:.1f} мс')
        cap.release()

    pool.shutdown()
    if frames:
        y, a, f = np.mean(t_yolo), np.mean(t_aruco), np.mean(t_frame)
        print(f'\nкадров: {frames} | людей: {n_people} | маркеров: {n_mark}')
        print(f'YOLO на NPU:   {y*1000:7.1f} мс')
        print(f'ArUco на CPU:  {a*1000:7.1f} мс  (идёт параллельно)')
        print(f'кадр целиком:  {f*1000:7.1f} мс  ->  {1/f:.1f} fps')
        print(f'накладные от совмещения: {(f-max(y,a))*1000:.1f} мс')
        print('кадры с находками:', OUT)
    rknn.release()


if __name__ == '__main__':
    main()
