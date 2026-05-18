
from __future__ import annotations

from pathlib import Path
import subprocess

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path('/root/autodl-tmp/cv_hw2_submit')
REPORT = ROOT / 'report' / 'HW2_report.md'
FIG_DIR = ROOT / 'report' / 'figures'
FIG_DIR.mkdir(parents=True, exist_ok=True)
STUDENT = '???'
SID = '22300680061'


def read_last(run_dir: str, keys: list[str]) -> dict[str, float | str]:
    path = ROOT / 'runs' / run_dir / 'metrics.csv'
    if not path.exists():
        return {k: 'N/A' for k in keys}
    df = pd.read_csv(path)
    out = {}
    for k in keys:
        if k in df.columns:
            vals = pd.to_numeric(df[k], errors='coerce').dropna()
            out[k] = float(vals.iloc[-1]) if len(vals) else 'N/A'
        else:
            out[k] = 'N/A'
    return out


def best_val(run_dir: str, key: str) -> float | str:
    path = ROOT / 'runs' / run_dir / 'metrics.csv'
    if not path.exists():
        return 'N/A'
    df = pd.read_csv(path)
    if key not in df.columns:
        return 'N/A'
    vals = pd.to_numeric(df[key], errors='coerce').dropna()
    return float(vals.max()) if len(vals) else 'N/A'


def fmt(x):
    if isinstance(x, (float, int)):
        return f'{x:.4f}'
    return str(x)


def plot_runs(run_dirs, metric, out_name, title):
    plt.figure(figsize=(8, 4.5), dpi=180)
    for rd in run_dirs:
        path = ROOT / 'runs' / rd / 'metrics.csv'
        if not path.exists():
            continue
        df = pd.read_csv(path)
        if metric not in df.columns:
            continue
        x = pd.to_numeric(df.get('epoch', pd.Series(range(1, len(df) + 1))), errors='coerce')
        y = pd.to_numeric(df[metric], errors='coerce')
        mask = y.notna() & x.notna()
        if mask.any():
            plt.plot(x[mask], y[mask], marker='o', linewidth=2, label=rd.replace('final_', ''))
    plt.title(title)
    plt.xlabel('epoch')
    plt.ylabel(metric)
    plt.grid(alpha=0.3)
    plt.legend(fontsize=7)
    plt.tight_layout()
    out = FIG_DIR / out_name
    plt.savefig(out)
    plt.close()
    return out


def find_first(pattern: str) -> Path | None:
    hits = sorted(ROOT.glob(pattern))
    return hits[0] if hits else None


def yolo_results_path() -> Path | None:
    candidates = [
        ROOT / 'runs' / 'final_task2_yolov8n_visdrone' / 'results.csv',
        ROOT / 'runs' / 'detect' / 'runs' / 'final_task2_yolov8n_visdrone' / 'results.csv',
        ROOT / 'runs' / 'detect' / 'final_task2_yolov8n_visdrone' / 'results.csv',
    ]
    for p in candidates:
        if p.exists():
            return p
    return find_first('runs/**/final_task2_yolov8n_visdrone*/results.csv')


def yolo_metrics():
    path = yolo_results_path()
    if not path:
        return {}
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    row = df.iloc[-1]
    return {k: float(row[k]) for k in df.columns if k != 'epoch'}


def plot_yolo():
    path = yolo_results_path()
    if not path:
        return None
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    plt.figure(figsize=(8, 4.5), dpi=180)
    for col in ['metrics/mAP50(B)', 'metrics/mAP50-95(B)', 'metrics/precision(B)', 'metrics/recall(B)']:
        if col in df.columns:
            plt.plot(df['epoch'], df[col], marker='o', linewidth=2, label=col.replace('metrics/', ''))
    plt.title('Task 2 YOLOv8 VisDrone Metrics')
    plt.xlabel('epoch')
    plt.grid(alpha=0.3)
    plt.legend(fontsize=7)
    plt.tight_layout()
    out = FIG_DIR / 'task2_yolo_metrics.png'
    plt.savefig(out)
    plt.close()
    return out


def tracking_summary():
    p = ROOT / 'runs' / 'final_task2_tracking' / 'summary.txt'
    if not p.exists():
        return {}
    out = {}
    for line in p.read_text(encoding='utf-8', errors='ignore').splitlines():
        if ':' in line:
            k, v = line.split(':', 1)
            out[k.strip()] = v.strip()
    return out


def video_source_text() -> str:
    p = ROOT / 'data' / 'visdrone_mot_video_source.txt'
    if not p.exists():
        return 'VisDrone ???????????? 15 ??'
    lines = [line.strip() for line in p.read_text(encoding='utf-8', errors='ignore').splitlines() if line.strip()]
    return '?'.join(lines)


cls_runs = ['final_task1_resnet18_imagenet', 'final_task1_resnet18_scratch', 'final_task1_resnet18_se', 'final_task1_resnet18_cbam', 'final_task1_vit_tiny']
seg_runs = ['final_task3_unet_ce', 'final_task3_unet_dice', 'final_task3_unet_ce_dice']
plot_runs(cls_runs, 'val_acc', 'task1_val_acc.png', 'Task 1 Validation Accuracy')
plot_runs(cls_runs, 'train_loss', 'task1_train_loss.png', 'Task 1 Training Loss')
plot_runs(seg_runs, 'val_miou', 'task3_val_miou.png', 'Task 3 Validation mIoU')
plot_runs(seg_runs, 'train_loss', 'task3_train_loss.png', 'Task 3 Training Loss')
plot_yolo()

labels = {
    'final_task1_resnet18_imagenet': ('ResNet-18', 'ImageNet', 'baseline'),
    'final_task1_resnet18_scratch': ('ResNet-18', '?????', 'pretrain ??'),
    'final_task1_resnet18_se': ('SE-ResNet-18', 'ImageNet', '?????'),
    'final_task1_resnet18_cbam': ('CBAM-ResNet-18', 'ImageNet', '??+?????'),
    'final_task1_vit_tiny': ('ViT-Tiny', '?????', '?? Transformer ??'),
}
cls_rows = []
for rd in cls_runs:
    vals = read_last(rd, ['test_acc'])
    a, b, c = labels[rd]
    cls_rows.append(f'| {a} | {b} | {c} | {fmt(best_val(rd, "val_acc"))} | {fmt(vals["test_acc"])} |')

seg_labels = {'final_task3_unet_ce': 'Cross-Entropy', 'final_task3_unet_dice': 'Dice Loss', 'final_task3_unet_ce_dice': 'CE + Dice'}
seg_rows = []
for rd in seg_runs:
    vals = read_last(rd, ['test_miou', 'test_iou_class_0', 'test_iou_class_1', 'test_iou_class_2'])
    seg_rows.append(f'| {seg_labels[rd]} | {fmt(best_val(rd, "val_miou"))} | {fmt(vals["test_miou"])} | {fmt(vals["test_iou_class_0"])} | {fmt(vals["test_iou_class_1"])} | {fmt(vals["test_iou_class_2"])} |')

y = yolo_metrics()
ts = tracking_summary()
yolo_fig = 'figures/task2_yolo_metrics.png' if (FIG_DIR / 'task2_yolo_metrics.png').exists() else ''
tracking_frame_30 = 'runs/final_task2_tracking/occlusion_frame_0030.jpg'
tracking_frame_33 = 'runs/final_task2_tracking/occlusion_frame_0033.jpg'

report = f'''# ????? HW2 ????

???{STUDENT}????{SID}??????????

??????????? README??????`runs/*/best.pt` ? `runs/**/final_task2_yolov8n_visdrone*/weights/best.pt`?

## ?? 1?Oxford-IIIT Pet ????

### ????

????? Oxford-IIIT Pet????? 37??????? `trainval` ???? 15% ??????????? `test` ????? Accuracy??????? resize/crop ? 224 x 224???? ImageNet ????????????? ResNet-18??????????ImageNet ????SE ????CBAM ?????? ViT-Tiny?

| ?? | ?? |
| --- | --- |
| batch size | 96 |
| epoch | 8 |
| optimizer | AdamW |
| scheduler | cosine |
| loss | Cross-Entropy Loss |
| metric | Accuracy |

### ??

| ?? | ??? | ?? | Best Val Acc | Test Acc |
| --- | --- | --- | --- | --- |
{chr(10).join(cls_rows)}

![Task1 Val Accuracy](figures/task1_val_acc.png)

![Task1 Train Loss](figures/task1_train_loss.png)

### ??

ImageNet ???? ResNet-18 ????????????? Oxford-IIIT Pet ???????????????????????????????????SE ? CBAM ???? epoch ????????? ImageNet ResNet-18????????????????????8 ? epoch ?????????ViT-Tiny ???????????? Transformer ???????????????????

## ?? 2?VisDrone ??????????

### ??????

?????? VisDrone2019-DET????? pedestrian?people?bicycle?car?van?truck?tricycle?awning-tricycle?bus?motor????? YOLOv8n?????? 640??? 5 ? epoch?? mAP50 ? mAP50-95 ???????

| ?? | ?? |
| --- | --- |
| detector | YOLOv8n |
| image size | 640 |
| batch size | 32 |
| epoch | 5 |
| tracker | ByteTrack |

| ?? | ?? |
| --- | --- |
| Precision | {fmt(y.get('metrics/precision(B)', 'N/A'))} |
| Recall | {fmt(y.get('metrics/recall(B)', 'N/A'))} |
| mAP50 | {fmt(y.get('metrics/mAP50(B)', 'N/A'))} |
| mAP50-95 | {fmt(y.get('metrics/mAP50-95(B)', 'N/A'))} |

{f'![Task2 YOLO Metrics]({yolo_fig})' if yolo_fig else ''}

### ?????

?????{video_source_text()}

?????? YOLOv8n ????????? ByteTrack ????????????? bounding box????????Tracking ID ????????????????????????

| ?? | ????? |
| --- | --- |
| ???? | `runs/final_task2_tracking/tracking_count.mp4` |
| ???? | `runs/final_task2_tracking/crossing_events.csv` |
| ???? | {ts.get('cross_count', 'N/A')} |
| ?? Tracking ID ? | {ts.get('unique_tracks', 'N/A')} |

![Tracking Frame 30]({tracking_frame_30})

![Tracking Frame 33]({tracking_frame_33})

### ??? ID Switch ??

??? `occlusion_frame_0030` ? `occlusion_frame_0033` ??????? ID ??????????????????????????????ByteTrack ??????? IoU ???????????????????????????????????????????????? Tracking ID ?????????????????????????????? ID ?????? ID Switch???????????????????????

## ?? 3?Oxford-IIIT Pet ????

### ????

?????? Oxford-IIIT Pet trimap ???? trimap ?? 3 ????????????/?????????????? U-Net????? DoubleConv????????? skip connection??? Cross-Entropy?Dice Loss ? CE+Dice ???????? mIoU?

| ?? | ?? |
| --- | --- |
| image size | 256 x 256 |
| batch size | 48 |
| epoch | 10 |
| optimizer | AdamW |
| learning rate | 3e-4 |
| metric | mIoU |

### ??

| Loss | Best Val mIoU | Test mIoU | Class 0 IoU | Class 1 IoU | Class 2 IoU |
| --- | --- | --- | --- | --- | --- |
{chr(10).join(seg_rows)}

![Task3 Val mIoU](figures/task3_val_miou.png)

![Task3 Train Loss](figures/task3_train_loss.png)

### ??

Cross-Entropy ????????Dice Loss ???????????????????CE+Dice ???????????????????????????????????? mIoU????????? Test mIoU ???? IoU ????????????????

## ??

???????????/?????????????????????????????????/??????? VisDrone ?????Tracking ID ?????????????????? U-Net ?????????????????????????????????? `runs/` ????
'''
REPORT.write_text(report, encoding='utf-8')
subprocess.run(['/root/miniconda3/bin/python', 'tools/make_report_pdf.py'], cwd=ROOT, check=True)
print('final report generated:', REPORT)
