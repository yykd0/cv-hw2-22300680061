
from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

import yaml

ROOT = Path('/root/autodl-tmp/cv_hw2_submit')
PY = '/root/miniconda3/bin/python'
LOG_DIR = ROOT / 'runs' / 'logs'
LOG_DIR.mkdir(parents=True, exist_ok=True)
SUMMARY = ROOT / 'runs' / 'final_summary.json'

os.environ.setdefault('PYTHONUNBUFFERED', '1')
os.environ.setdefault('MPLBACKEND', 'Agg')
os.environ.setdefault('WANDB_MODE', 'offline')
os.environ.setdefault('ULTRALYTICS_SETTINGS', str(ROOT / 'ultralytics_settings.json'))

results: dict[str, dict] = {}


def save_summary():
    SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding='utf-8')


def run(name: str, cmd: list[str], timeout: int | None = None) -> int:
    print(f'\n===== {name} =====', flush=True)
    print(' '.join(cmd), flush=True)
    start = time.time()
    log_path = LOG_DIR / f'{name}.log'
    with log_path.open('w', encoding='utf-8') as log:
        log.write(' '.join(cmd) + '\n\n')
        proc = subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, timeout=timeout)
    elapsed = time.time() - start
    results[name] = {'returncode': proc.returncode, 'seconds': round(elapsed, 2), 'log': str(log_path)}
    save_summary()
    print(f'{name}: rc={proc.returncode}, {elapsed/60:.1f} min', flush=True)
    if proc.returncode != 0:
        print(log_path.read_text(encoding='utf-8', errors='replace')[-3000:], flush=True)
    return proc.returncode


def load_yaml(path: str | Path) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding='utf-8'))


def write_yaml(path: str | Path, data: dict) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding='utf-8')
    return path


def make_cls_config(src: str, name: str, epochs=8, batch=128, pretrained=None) -> Path:
    cfg = load_yaml(ROOT / 'configs' / src)
    cfg['output_dir'] = f'runs/{name}'
    cfg['data']['root'] = 'data/oxford_iiit_pet'
    cfg['data']['download'] = True
    cfg['data']['num_workers'] = 8
    cfg['train']['epochs'] = epochs
    cfg['train']['batch_size'] = batch
    cfg['logging']['backend'] = 'csv'
    cfg['logging']['run_name'] = name
    if pretrained is not None:
        cfg['model']['pretrained'] = pretrained
    return write_yaml(ROOT / 'configs' / f'{name}.yaml', cfg)


def make_seg_config(src: str, name: str, epochs=10, batch=64) -> Path:
    cfg = load_yaml(ROOT / 'configs' / src)
    cfg['output_dir'] = f'runs/{name}'
    cfg['data']['root'] = 'data/oxford_iiit_pet'
    cfg['data']['download'] = True
    cfg['data']['num_workers'] = 8
    cfg['train']['epochs'] = epochs
    cfg['train']['batch_size'] = batch
    cfg['logging']['backend'] = 'csv'
    cfg['logging']['run_name'] = name
    return write_yaml(ROOT / 'configs' / f'{name}.yaml', cfg)

cls_jobs = [
    ('task1_resnet18_imagenet.yaml', 'final_task1_resnet18_imagenet', None),
    ('task1_resnet18_scratch.yaml', 'final_task1_resnet18_scratch', None),
    ('task1_resnet18_se.yaml', 'final_task1_resnet18_se', None),
    ('task1_resnet18_cbam.yaml', 'final_task1_resnet18_cbam', None),
    ('task1_vit_tiny.yaml', 'final_task1_vit_tiny', False),
]
for src, name, pre in cls_jobs:
    cfg = make_cls_config(src, name, pretrained=pre)
    run(name, [PY, 'scripts/train_pet_cls.py', '--config', str(cfg)])

seg_jobs = [
    ('task3_unet_ce.yaml', 'final_task3_unet_ce'),
    ('task3_unet_dice.yaml', 'final_task3_unet_dice'),
    ('task3_unet_ce_dice.yaml', 'final_task3_unet_ce_dice'),
]
for src, name in seg_jobs:
    cfg = make_seg_config(src, name)
    run(name, [PY, 'scripts/train_pet_seg.py', '--config', str(cfg)])

yolo_cmd = [
    'yolo', 'detect', 'train', 'model=yolov8n.pt', 'data=VisDrone.yaml', 'epochs=5',
    'imgsz=640', 'batch=32', 'device=0', 'workers=8', 'project=runs',
    'name=final_task2_yolov8n_visdrone', 'exist_ok=True', 'patience=3'
]
run('final_task2_yolov8n_visdrone', yolo_cmd)

def make_demo_video():
    import cv2
    img_dir = ROOT / 'datasets' / 'VisDrone' / 'images' / 'val'
    images = sorted(img_dir.glob('*.jpg'))[:300]
    out_dir = ROOT / 'data'
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / 'visdrone_demo_video.mp4'
    if not images:
        return None
    first = cv2.imread(str(images[0]))
    h, w = first.shape[:2]
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*'mp4v'), 25, (w, h))
    for img in images:
        frame = cv2.imread(str(img))
        if frame is None:
            continue
        if frame.shape[:2] != (h, w):
            frame = cv2.resize(frame, (w, h))
        writer.write(frame)
    writer.release()
    return out_path

try:
    video = make_demo_video()
    weights = ROOT / 'runs' / 'final_task2_yolov8n_visdrone' / 'weights' / 'best.pt'
    if video and weights.exists():
        run('final_task2_tracking', [PY, 'scripts/track_count.py', '--weights', str(weights), '--source', str(video), '--out', 'runs/final_task2_tracking', '--line', '320', '120', '320', '620'])
    else:
        results['final_task2_tracking'] = {'returncode': 1, 'reason': f'video={video}, weights_exists={weights.exists()}'}
        save_summary()
except Exception as exc:
    results['final_task2_tracking'] = {'returncode': 1, 'reason': repr(exc)}
    save_summary()

run('finalize_report', [PY, 'tools/finalize_report.py'])
print('ALL_DONE', flush=True)
