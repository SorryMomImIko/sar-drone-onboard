"""Детектор ArUco для Orange Pi. Работает на CPU, нейросеть не нужна.

Запуск по папке:   python aruco_detect.py images
Запуск по видео:   python aruco_detect.py /path/video.mp4
С камеры (V4L2):   python aruco_detect.py 0

Результаты с рамками кладутся в out_aruco/, сводка печатается в консоль.
"""
import sys, time, json
from pathlib import Path
import cv2, numpy as np

ROOT = Path(__file__).parent
OUT  = ROOT / 'out_aruco'
DICTS = ['DICT_4X4_50']          # сюда добавить реальные словари, если другие

# Профиль параметров. Замер на 400 кадрах (profiles.json):
#   быстрый          6.7 мс  | 28px: 84%  | ложных 13
#   сбалансированный 22.9 мс | 28px: 90%  | ложных 19
#   агрессивный      76.7 мс | 28px: 92%  | ложных 40  - не окупается
# Время указано для ноутбука; на плате примерно втрое больше.
PROFILE = 'быстрый'


def build_detectors(names, profile=PROFILE):
    p = cv2.aruco.DetectorParameters()
    if profile == 'агрессивный':
        p.adaptiveThreshWinSizeMin, p.adaptiveThreshWinSizeMax, p.adaptiveThreshWinSizeStep = 3, 23, 4
        p.minMarkerPerimeterRate = 0.01
        p.errorCorrectionRate = 0.7
        p.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    elif profile == 'сбалансированный':
        p.adaptiveThreshWinSizeMin, p.adaptiveThreshWinSizeMax, p.adaptiveThreshWinSizeStep = 5, 17, 6
        p.minMarkerPerimeterRate = 0.015
        p.errorCorrectionRate = 0.45
        p.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    else:
        p.adaptiveThreshWinSizeMin, p.adaptiveThreshWinSizeMax, p.adaptiveThreshWinSizeStep = 7, 7, 1
        p.minMarkerPerimeterRate = 0.02
        p.errorCorrectionRate = 0.3
        p.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_NONE
    p.maxMarkerPerimeterRate = 4.0
    p.polygonalApproxAccuracyRate = 0.05
    p.minCornerDistanceRate = 0.04
    out = []
    for n in names:
        d = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, n))
        out.append((n, cv2.aruco.ArucoDetector(d, p)))
    return out


def detect(frame, detectors):
    g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    found = []
    for name, det in detectors:
        corners, ids, _ = det.detectMarkers(g)
        if ids is not None:
            for c, i in zip(corners, ids.flatten()):
                found.append({'dict': name, 'id': int(i), 'corners': c.reshape(4, 2)})
    return found

def draw(frame, found):
    for f in found:
        pts = f['corners'].astype(int)
        cv2.polylines(frame, [pts], True, (0, 230, 255), 2, cv2.LINE_AA)
        c = pts.mean(0).astype(int)
        cv2.putText(frame, f"id {f['id']}", (c[0] - 18, c[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 230, 255), 2, cv2.LINE_AA)
    return frame

def main():
    src = sys.argv[1] if len(sys.argv) > 1 else 'images'
    dets = build_detectors(DICTS)
    OUT.mkdir(exist_ok=True)
    times, total, frames = [], 0, 0

    p = Path(src)
    if p.is_dir():
        files = sorted([x for x in p.iterdir() if x.suffix.lower() in {'.jpg', '.jpeg', '.png'}])
        print(f'кадров: {len(files)}')
        for f in files:
            img = cv2.imread(str(f))
            if img is None: continue
            t0 = time.perf_counter(); found = detect(img, dets); times.append(time.perf_counter() - t0)
            frames += 1; total += len(found)
            if found:
                cv2.imwrite(str(OUT / f.name), draw(img, found))
    else:
        cap = cv2.VideoCapture(int(src) if src.isdigit() else src)
        if not cap.isOpened():
            print('не открылся источник:', src); return
        while True:
            ok, img = cap.read()
            if not ok: break
            t0 = time.perf_counter(); found = detect(img, dets); times.append(time.perf_counter() - t0)
            frames += 1; total += len(found)
            if frames % 30 == 0:
                print(f'  кадр {frames}, маркеров {total}, {np.mean(times)*1000:.1f} мс')
        cap.release()

    if times:
        print(f'\nкадров: {frames} | маркеров найдено: {total}')
        print(f'время детекции: средн {np.mean(times)*1000:.2f} мс, макс {max(times)*1000:.2f} мс '
              f'-> {1/np.mean(times):.0f} fps по этому этапу')
        print('кадры с находками:', OUT)

if __name__ == '__main__':
    main()
