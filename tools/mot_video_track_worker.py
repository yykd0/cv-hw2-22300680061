
from __future__ import annotations

import os
import subprocess
import time
import zipfile
from pathlib import Path

import cv2

ROOT = Path('/root/autodl-tmp/cv_hw2_submit')
PY = '/root/miniconda3/bin/python'
LOG = ROOT / 'runs' / 'logs' / 'mot_video_track.log'
DATA = ROOT / 'data'
DATA.mkdir(parents=True, exist_ok=True)

MOT_ZIP = DATA / 'VisDrone2019-MOT-val.zip'
FRAMES_DIR = DATA / 'mot_video_frames'
VIDEO = DATA / 'visdrone_mot_val_15s.mp4'
GDRIVE_ID = '1rqnKe9IgU_crMaxRoel9_nuUsMEBBVQu'


def log(msg: str):
    line = f'{time.strftime("%F %T")} {msg}'
    print(line, flush=True)
    with LOG.open('a', encoding='utf-8') as f:
        f.write(line + '\n')


def run(cmd: list[str], timeout: int | None = None) -> int:
    log('RUN ' + ' '.join(cmd))
    with LOG.open('a', encoding='utf-8') as f:
        p = subprocess.run(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT, timeout=timeout)
    log(f'RC {p.returncode}')
    return p.returncode


def download_mot() -> bool:
    if MOT_ZIP.exists() and MOT_ZIP.stat().st_size > 1_000_000_000:
        log(f'MOT zip exists: {MOT_ZIP} size={MOT_ZIP.stat().st_size}')
        return True
    rc = run([PY, '-m', 'gdown', f'https://drive.google.com/uc?id={GDRIVE_ID}', '-O', str(MOT_ZIP)])
    return rc == 0 and MOT_ZIP.exists() and MOT_ZIP.stat().st_size > 100_000_000


def make_video_from_mot() -> bool:
    if VIDEO.exists() and VIDEO.stat().st_size > 1_000_000:
        log(f'video exists: {VIDEO}')
        return True
    if not MOT_ZIP.exists():
        return False
    FRAMES_DIR.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(MOT_ZIP) as zf:
        jpgs = [n for n in zf.namelist() if n.lower().endswith('.jpg')]
        groups: dict[str, list[str]] = {}
        for n in jpgs:
            parts = Path(n).parts
            seq = None
            if 'sequences' in parts:
                idx = parts.index('sequences')
                if idx + 1 < len(parts):
                    seq = parts[idx + 1]
            elif len(parts) >= 2:
                seq = parts[-2]
            if seq:
                groups.setdefault(seq, []).append(n)
        candidates = sorted(groups.items(), key=lambda kv: len(kv[1]), reverse=True)
        if not candidates:
            log('no MOT frame groups found')
            return False
        # Prefer traffic/intersection sequences mentioned in the dataset card when present.
        preferred = ['uav0000305_00000_v', 'uav0000339_00001_v', 'uav0000268_05773_v', 'uav0000117_02622_v']
        selected_seq, selected = candidates[0]
        for name in preferred:
            if name in groups and len(groups[name]) >= 225:
                selected_seq, selected = name, groups[name]
                break
        selected = sorted(selected)[:300]
        seq_dir = FRAMES_DIR / selected_seq
        seq_dir.mkdir(parents=True, exist_ok=True)
        frame_paths = []
        for idx, member in enumerate(selected, 1):
            out = seq_dir / f'{idx:06d}.jpg'
            if not out.exists():
                with zf.open(member) as src, out.open('wb') as dst:
                    dst.write(src.read())
            frame_paths.append(out)
    first = cv2.imread(str(frame_paths[0]))
    if first is None:
        log('cannot read first frame')
        return False
    h, w = first.shape[:2]
    target_w = 960
    scale = min(1.0, target_w / w)
    out_w, out_h = int(w * scale), int(h * scale)
    fps = 20
    writer = cv2.VideoWriter(str(VIDEO), cv2.VideoWriter_fourcc(*'mp4v'), fps, (out_w, out_h))
    for p in frame_paths:
        img = cv2.imread(str(p))
        if img is None:
            continue
        if (img.shape[1], img.shape[0]) != (out_w, out_h):
            img = cv2.resize(img, (out_w, out_h), interpolation=cv2.INTER_AREA)
        writer.write(img)
    writer.release()
    meta = DATA / 'visdrone_mot_video_source.txt'
    meta.write_text(f'source=VisDrone2019-MOT valset\nsequence={selected_seq}\nframes={len(frame_paths)}\nfps={fps}\nduration_sec={len(frame_paths)/fps:.2f}\nvideo={VIDEO}\n', encoding='utf-8')
    log(meta.read_text(encoding='utf-8').strip())
    return VIDEO.exists() and VIDEO.stat().st_size > 1_000_000


def make_video_from_det_fallback() -> bool:
    if VIDEO.exists() and VIDEO.stat().st_size > 1_000_000:
        return True
    img_dir = ROOT / 'datasets' / 'VisDrone' / 'images' / 'val'
    imgs = sorted(img_dir.glob('*.jpg'))[:300]
    if len(imgs) < 100:
        return False
    first = cv2.imread(str(imgs[0]))
    h, w = first.shape[:2]
    target_w = 960
    scale = min(1.0, target_w / w)
    out_w, out_h = int(w * scale), int(h * scale)
    writer = cv2.VideoWriter(str(VIDEO), cv2.VideoWriter_fourcc(*'mp4v'), 20, (out_w, out_h))
    for p in imgs:
        img = cv2.imread(str(p))
        if img is None:
            continue
        if (img.shape[1], img.shape[0]) != (out_w, out_h):
            img = cv2.resize(img, (out_w, out_h), interpolation=cv2.INTER_AREA)
        writer.write(img)
    writer.release()
    (DATA / 'visdrone_mot_video_source.txt').write_text('source=VisDrone2019-DET validation frames fallback\nframes=300\nfps=20\nduration_sec=15\n', encoding='utf-8')
    log('created DET fallback video')
    return VIDEO.exists() and VIDEO.stat().st_size > 1_000_000


def find_weights() -> Path | None:
    candidates = [
        ROOT / 'runs' / 'final_task2_yolov8n_visdrone' / 'weights' / 'best.pt',
        ROOT / 'runs' / 'detect' / 'runs' / 'final_task2_yolov8n_visdrone' / 'weights' / 'best.pt',
        ROOT / 'runs' / 'detect' / 'final_task2_yolov8n_visdrone' / 'weights' / 'best.pt',
    ]
    for p in candidates:
        if p.exists():
            return p
    hits = sorted((ROOT / 'runs').glob('**/final_task2_yolov8n_visdrone*/weights/best.pt'))
    return hits[0] if hits else None


def main():
    log('MOT video/tracking worker started')
    ok = download_mot()
    if ok:
        ok = make_video_from_mot()
    if not ok:
        log('MOT download/video failed or unavailable; waiting for DET fallback frames')
        for _ in range(240):
            if make_video_from_det_fallback():
                ok = True
                break
            time.sleep(30)
    if not ok:
        log('no usable video created')
        return
    log(f'video ready: {VIDEO}')
    weights = None
    for _ in range(720):
        weights = find_weights()
        if weights:
            break
        time.sleep(30)
    if not weights:
        log('YOLO weights not found before timeout')
        return
    cap = cv2.VideoCapture(str(VIDEO))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    x = max(1, width // 2)
    y1 = max(1, int(height * 0.18)); y2 = min(height - 1, int(height * 0.88))
    out_dir = ROOT / 'runs' / 'final_task2_tracking'
    rc = run([PY, 'scripts/track_count.py', '--weights', str(weights), '--source', str(VIDEO), '--out', str(out_dir), '--line', str(x), str(y1), str(x), str(y2)])
    log(f'tracking finished rc={rc}')

if __name__ == '__main__':
    main()
