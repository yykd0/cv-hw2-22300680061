
from __future__ import annotations

import csv
import os
import shutil
import subprocess
import time
from pathlib import Path

import cv2
from PIL import Image
import yaml

ROOT = Path('/root/autodl-tmp/cv_hw2_submit')
PY = '/root/miniconda3/bin/python'
LOG = ROOT / 'runs' / 'logs' / 'quick_task2_pipeline.log'
LOG.parent.mkdir(parents=True, exist_ok=True)

NAMES = [
    'pedestrian', 'people', 'bicycle', 'car', 'van', 'truck',
    'tricycle', 'awning-tricycle', 'bus', 'motor'
]


def log(msg: str) -> None:
    line = f'{time.strftime("%F %T")} {msg}'
    print(line, flush=True)
    with LOG.open('a', encoding='utf-8') as f:
        f.write(line + '\n')


def run(cmd: list[str], timeout: int | None = None) -> int:
    log('RUN ' + ' '.join(cmd))
    with LOG.open('a', encoding='utf-8') as f:
        p = subprocess.run(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT, timeout=timeout)
    log(f'RC={p.returncode}')
    return p.returncode


def convert_annotation(src: Path, dst: Path, size: tuple[int, int]) -> int:
    width, height = size
    rows = []
    if src.exists():
        for line in src.read_text(encoding='utf-8', errors='ignore').splitlines():
            if not line.strip():
                continue
            parts = [float(x) for x in line.split(',')[:8]]
            x, y, w, h, score, category, truncation, occlusion = parts
            category = int(category)
            if category < 1 or category > 10 or w <= 1 or h <= 1:
                continue
            cx = (x + w / 2) / width
            cy = (y + h / 2) / height
            rows.append(f'{category - 1} {cx:.6f} {cy:.6f} {w / width:.6f} {h / height:.6f}')
    dst.write_text('\n'.join(rows) + ('\n' if rows else ''), encoding='utf-8')
    return len(rows)


def build_quick_dataset() -> Path:
    src = ROOT / 'datasets' / 'VisDrone' / 'VisDrone2019-DET-val'
    img_dir = src / 'images'
    ann_dir = src / 'annotations'
    if not img_dir.exists():
        raise FileNotFoundError(f'VisDrone val images not found: {img_dir}')
    images = sorted(img_dir.glob('*.jpg'))
    if len(images) < 120:
        raise RuntimeError(f'not enough val images yet: {len(images)}')
    train_imgs = images[: min(480, max(80, int(len(images) * 0.82)))]
    val_imgs = images[len(train_imgs):] or images[-80:]
    out = ROOT / 'data' / 'visdrone_quick_yolo'
    for split, items in [('train', train_imgs), ('val', val_imgs)]:
        (out / 'images' / split).mkdir(parents=True, exist_ok=True)
        (out / 'labels' / split).mkdir(parents=True, exist_ok=True)
        boxes = 0
        for img in items:
            target = out / 'images' / split / img.name
            if not target.exists():
                try:
                    os.link(img, target)
                except OSError:
                    shutil.copy2(img, target)
            with Image.open(img) as im:
                size = im.size
            boxes += convert_annotation(ann_dir / f'{img.stem}.txt', out / 'labels' / split / f'{img.stem}.txt', size)
        log(f'{split}: images={len(items)} boxes={boxes}')
    data = {'path': str(out), 'train': 'images/train', 'val': 'images/val', 'names': {i: n for i, n in enumerate(NAMES)}}
    yaml_path = out / 'visdrone.yaml'
    yaml_path.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding='utf-8')
    return yaml_path


def make_video() -> Path:
    img_dir = ROOT / 'datasets' / 'VisDrone' / 'VisDrone2019-DET-val' / 'images'
    images = sorted(img_dir.glob('*.jpg'))[:300]
    if not images:
        raise RuntimeError('no images to make video')
    out = ROOT / 'data' / 'visdrone_mot_val_15s.mp4'
    first = cv2.imread(str(images[0]))
    h, w = first.shape[:2]
    target_w = 960
    scale = min(1.0, target_w / w)
    out_w, out_h = int(w * scale), int(h * scale)
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*'mp4v'), 20, (out_w, out_h))
    for p in images:
        frame = cv2.imread(str(p))
        if frame is None:
            continue
        if (frame.shape[1], frame.shape[0]) != (out_w, out_h):
            frame = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA)
        writer.write(frame)
    writer.release()
    (ROOT / 'data' / 'visdrone_mot_video_source.txt').write_text(
        'source=VisDrone2019-DET validation frames quick video\nframes=300\nfps=20\nduration_sec=15.00\nvideo=data/visdrone_mot_val_15s.mp4\n',
        encoding='utf-8',
    )
    log(f'video={out} size={out.stat().st_size}')
    return out


def train_yolo(data_yaml: Path) -> Path:
    from ultralytics import YOLO
    model = YOLO('yolov8n.pt')
    log('YOLO quick train start')
    model.train(
        data=str(data_yaml), epochs=3, imgsz=640, batch=32, device=0, workers=8,
        project=str(ROOT / 'runs'), name='final_task2_yolov8n_visdrone', exist_ok=True,
        patience=2, verbose=True, plots=True,
    )
    candidates = sorted((ROOT / 'runs').glob('**/final_task2_yolov8n_visdrone*/weights/best.pt'))
    if not candidates:
        raise FileNotFoundError('best.pt not found after quick YOLO train')
    log(f'best={candidates[0]}')
    return candidates[0]


def main():
    log('quick task2 pipeline start')
    subprocess.run(['pkill', '-f', 'yolo detect train'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    subprocess.run(['pkill', '-f', 'mot_video_track_worker.py'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    data_yaml = build_quick_dataset()
    video = make_video()
    weights = train_yolo(data_yaml)
    cap = cv2.VideoCapture(str(video))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    line = [str(width // 2), str(max(1, int(height * 0.15))), str(width // 2), str(min(height - 1, int(height * 0.9)))]
    run([PY, 'scripts/track_count.py', '--weights', str(weights), '--source', str(video), '--out', 'runs/final_task2_tracking', '--line', *line], timeout=1800)
    run([PY, 'tools/finalize_report.py'], timeout=600)
    log('quick task2 pipeline done')

if __name__ == '__main__':
    main()
