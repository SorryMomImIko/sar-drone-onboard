"""Замер: с какого размера маркера детектор начинает его видеть.
Сравнивает детекции с gt.json - по ID и по точности углов."""
import json, time
from collections import defaultdict
from pathlib import Path
import cv2, numpy as np
from aruco_detect import build_detectors, detect, DICTS

ROOT = Path(__file__).parent
gt = json.load(open(ROOT / 'gt.json'))
dets = build_detectors(DICTS)
print('словарь набора:', gt['dictionary'], '| кадров:', len(gt['items']))

def bucket(v):
    for b in (8, 12, 16, 20, 26, 34, 46, 70):
        if v < b: return b
    return 999

by_size = defaultdict(lambda: {'n': 0, 'hit': 0, 'wrong_id': 0, 'err': []})
times = []
false_pos = 0
for it in gt['items']:
    img = cv2.imread(str(ROOT / 'images' / it['image']))
    if img is None: continue
    t0 = time.perf_counter(); found = detect(img, dets); times.append(time.perf_counter() - t0)
    truth0 = np.array(it['corners'], np.float32)
    # эффективный размер: короткая сторона четырёхугольника на кадре
    sides = [float(np.linalg.norm(truth0[k] - truth0[(k+1) % 4])) for k in range(4)]
    eff = min(sides)
    s = by_size[bucket(eff)]; s['n'] += 1
    truth = np.array(it['corners'], np.float32)
    center = truth.mean(0)
    near = [f for f in found if np.linalg.norm(f['corners'].mean(0) - center) < it['side_px'] * 1.5]
    false_pos += len(found) - len(near)
    if near:
        f = near[0]
        if f['id'] == it['id']:
            s['hit'] += 1
            d = np.linalg.norm(np.sort(f['corners'], 0) - np.sort(truth, 0), axis=1)
            s['err'].append(float(d.mean()))
        else:
            s['wrong_id'] += 1

print(f'\n{"высота, px":>12} {"кадров":>7} {"найдено":>8} {"доля":>7} {"ошибка ID":>10} {"углы, px":>9}')
for k in sorted(by_size):
    v = by_size[k]
    rate = v['hit'] / v['n'] if v['n'] else 0
    err = np.mean(v['err']) if v['err'] else float('nan')
    label = f'<{k}' if k != 999 else '>=70'
    print(f'{label:>12} {v["n"]:>7} {v["hit"]:>8} {rate:>6.0%} {v["wrong_id"]:>10} {err:>9.2f}')
tot = sum(v['n'] for v in by_size.values()); hit = sum(v['hit'] for v in by_size.values())
print(f'\nвсего: {hit}/{tot} ({hit/tot:.0%}) | ложных срабатываний: {false_pos}')
print(f'время детекции: средн {np.mean(times)*1000:.2f} мс на кадр 640x640')
