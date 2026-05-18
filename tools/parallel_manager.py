
from __future__ import annotations

import json
import os
import signal
import subprocess
import time
from pathlib import Path

import yaml

ROOT = Path('/root/autodl-tmp/cv_hw2_submit')
PY = '/root/miniconda3/bin/python'
LOG_DIR = ROOT / 'runs' / 'logs'
LOG_DIR.mkdir(parents=True, exist_ok=True)
STATE = ROOT / 'runs' / 'parallel_manager_state.json'

ENV = os.environ.copy()
ENV.setdefault('PYTHONUNBUFFERED', '1')
ENV.setdefault('MPLBACKEND', 'Agg')
ENV.setdefault('WANDB_MODE', 'offline')
ENV.setdefault('ULTRALYTICS_SETTINGS', str(ROOT / 'ultralytics_settings.json'))

state = {'started': [], 'events': []}

def save():
    STATE.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding='utf-8')

def event(msg: str):
    line = f'{time.strftime("%F %T")} {msg}'
    print(line, flush=True)
    state['events'].append(line)
    save()

def load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding='utf-8'))

def write_yaml(path: Path, data: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding='utf-8')
    return path

def make_cls_config(src: str, name: str, epochs=8, batch=96, pretrained=None) -> Path:
    cfg = load_yaml(ROOT / 'configs' / src)
    cfg['output_dir'] = f'runs/{name}'
    cfg['data']['root'] = 'data/oxford_iiit_pet'
    cfg['data']['download'] = False
    cfg['data']['num_workers'] = 6
    cfg['train']['epochs'] = epochs
    cfg['train']['batch_size'] = batch
    cfg['logging']['backend'] = 'csv'
    cfg['logging']['run_name'] = name
    if pretrained is not None:
        cfg['model']['pretrained'] = pretrained
    return write_yaml(ROOT / 'configs' / f'{name}.yaml', cfg)

def make_seg_config(src: str, name: str, epochs=10, batch=48) -> Path:
    cfg = load_yaml(ROOT / 'configs' / src)
    cfg['output_dir'] = f'runs/{name}'
    cfg['data']['root'] = 'data/oxford_iiit_pet'
    cfg['data']['download'] = False
    cfg['data']['num_workers'] = 6
    cfg['train']['epochs'] = epochs
    cfg['train']['batch_size'] = batch
    cfg['logging']['backend'] = 'csv'
    cfg['logging']['run_name'] = name
    return write_yaml(ROOT / 'configs' / f'{name}.yaml', cfg)

def is_running(pattern: str) -> bool:
    res = subprocess.run(['pgrep', '-f', pattern], cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
    pids = [p for p in res.stdout.split() if p and p != str(os.getpid())]
    return bool(pids)

def launch(name: str, cmd: list[str], done_path: Path | None = None):
    if done_path and done_path.exists():
        event(f'skip {name}: done file exists')
        return None
    if is_running(name):
        event(f'skip {name}: process already running')
        return None
    log_path = LOG_DIR / f'{name}.parallel.log'
    log = log_path.open('a', encoding='utf-8')
    log.write('\n===== launch ' + time.strftime('%F %T') + ' =====\n')
    log.write(' '.join(cmd) + '\n\n')
    log.flush()
    proc = subprocess.Popen(cmd, cwd=ROOT, env=ENV, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    state['started'].append({'name': name, 'pid': proc.pid, 'log': str(log_path), 'cmd': cmd})
    save()
    event(f'launched {name} pid={proc.pid}')
    return proc.pid

def pkill(pattern: str):
    subprocess.run(['pkill', '-f', pattern], cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

# Start segmentation experiments immediately; Oxford data is already available.
seg_jobs = [
    ('task3_unet_ce.yaml', 'final_task3_unet_ce'),
    ('task3_unet_dice.yaml', 'final_task3_unet_dice'),
    ('task3_unet_ce_dice.yaml', 'final_task3_unet_ce_dice'),
]
for src, name in seg_jobs:
    cfg = make_seg_config(src, name)
    launch(name, [PY, 'scripts/train_pet_seg.py', '--config', str(cfg)], ROOT / 'runs' / name / 'metrics.csv')

# Start YOLO/VisDrone line in parallel.
launch(
    'final_task2_yolov8n_visdrone',
    ['/usr/local/bin/yolo', 'detect', 'train', 'model=yolov8n.pt', 'data=VisDrone.yaml', 'epochs=5', 'imgsz=640', 'batch=32', 'device=0', 'workers=8', 'project=runs', 'name=final_task2_yolov8n_visdrone', 'exist_ok=True', 'patience=3'],
    ROOT / 'runs' / 'final_task2_yolov8n_visdrone' / 'results.csv',
)

# Wait for the serial ImageNet baseline to finish. It is currently responsible for the slow pretrained checkpoint download.
event('waiting for serial ImageNet baseline to finish before launching remaining classification jobs')
while is_running('final_task1_resnet18_imagenet.yaml'):
    time.sleep(30)

event('serial ImageNet baseline process finished; stopping serial orchestrator and launching remaining classification jobs')
pkill('tools/run_full_hw2.py')
for pat in ['final_task1_resnet18_scratch.yaml', 'final_task1_resnet18_se.yaml', 'final_task1_resnet18_cbam.yaml', 'final_task1_vit_tiny.yaml']:
    pkill(pat)

cls_jobs = [
    ('task1_resnet18_scratch.yaml', 'final_task1_resnet18_scratch', None),
    ('task1_resnet18_se.yaml', 'final_task1_resnet18_se', None),
    ('task1_resnet18_cbam.yaml', 'final_task1_resnet18_cbam', None),
    ('task1_vit_tiny.yaml', 'final_task1_vit_tiny', False),
]
for src, name, pre in cls_jobs:
    cfg = make_cls_config(src, name, pretrained=pre)
    launch(name, [PY, 'scripts/train_pet_cls.py', '--config', str(cfg)], ROOT / 'runs' / name / 'metrics.csv')

event('parallel manager launch phase complete')
