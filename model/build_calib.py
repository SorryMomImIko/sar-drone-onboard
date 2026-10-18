"""Сборка калибровочной выборки для квантизации RKNN.
Letterbox 640x640, серая заливка 114 - как в RknnYolo/ultralytics."""
import os, random, cv2, numpy as np
from pathlib import Path

ROOT = Path(__file__).parent
SRC  = [ROOT/'calib_raw', ROOT/'calib_kaggle']
OUT  = ROOT/'calib'
SIZE = 640
random.seed(0)
OUT.mkdir(exist_ok=True)

def letterbox(img, s=SIZE):
    h, w = img.shape[:2]
    r = min(s/h, s/w)
    nh, nw = int(round(h*r)), int(round(w*r))
    im = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR)
    canvas = np.full((s, s, 3), 114, np.uint8)
    top, left = (s-nh)//2, (s-nw)//2
    canvas[top:top+nh, left:left+nw] = im
    return canvas

def source_of(name):
    n = name.lower()
    for tag in ('sard', 'ladd', 'visdrone', 'vaid'):
        if n.startswith(tag):
            return tag
    for tag in ('evo', 'heinola', 'karkkila', 'ruokolahti'):
        if n.startswith(tag):
            return 'boreal_smoke'
    if n.startswith('empty'):
        return 'forest_empty'
    return 'other'

files = []
for d in SRC:
    if d.exists():
        files += [p for p in d.iterdir() if p.suffix.lower() in {'.jpg', '.jpeg', '.png'}]

groups = {}
for p in files:
    groups.setdefault(source_of(p.name), []).append(p)

written, stats = [], {}
for g, ps in sorted(groups.items()):
    ps.sort()
    for p in ps:
        img = cv2.imread(str(p))
        if img is None:
            continue
        dst = OUT/f'{g}__{p.stem}.jpg'
        cv2.imwrite(str(dst), letterbox(img), [cv2.IMWRITE_JPEG_QUALITY, 95])
        written.append(dst)
        stats[g] = stats.get(g, 0) + 1

random.shuffle(written)
with open(ROOT/'dataset.txt', 'w') as f:
    for p in written:
        f.write(f'./calib/{p.name}\n')

print('собрано кадров:', len(written))
for g, n in sorted(stats.items()):
    print(f'  {g:16} {n:4}')
print('список для квантизации: dataset.txt')
