# 计算机视觉 HW2

姓名：邓凯源  
学号：22300680061

本仓库整理了计算机视觉 HW2 的实验代码、配置文件、训练脚本、结果文件和实验报告，内容覆盖图像分类、目标检测与多目标跟踪、语义分割三个任务。

## 代码与结果链接

- GitHub 仓库：https://github.com/yykd0/cv-hw2-22300680061
- 模型权重与任务二视频：https://drive.google.com/file/d/1nmT0xaLfUJzoOaF6HLzBV9tynE4U50vw/view?usp=drive_link

## 目录结构

```text
configs/                 实验配置文件
scripts/                 训练、数据转换、跟踪计数脚本
src/cvhw2/               数据集、模型、损失函数和指标代码
tools/                   批量运行、绘图和报告辅助脚本
report/                  实验报告
weights_parts/           权重与任务二视频压缩包的 GitHub 分卷备份
```

## 环境配置

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

GPU 环境下根据 CUDA 版本安装对应的 PyTorch wheel。

## 任务 1：图像分类

```powershell
python scripts/train_pet_cls.py --config configs/task1_resnet18_imagenet.yaml
python scripts/train_pet_cls.py --config configs/task1_resnet18_scratch.yaml
python scripts/train_pet_cls.py --config configs/task1_resnet18_se.yaml
python scripts/train_pet_cls.py --config configs/task1_resnet18_cbam.yaml
python scripts/train_pet_cls.py --config configs/task1_vit_tiny.yaml
```

## 任务 2：检测与跟踪

```powershell
python scripts/visdrone_to_yolo.py --visdrone-root D:\datasets\VisDrone --out data\visdrone_yolo
yolo detect train model=yolov8n.pt data=data\visdrone_yolo\visdrone.yaml epochs=80 imgsz=640 batch=16 project=runs name=task2_yolov8n_visdrone
python scripts/track_count.py --weights runs\task2_yolov8n_visdrone\weights\best.pt --source data\demo_video.mp4 --out runs\task2_tracking --line 320 120 320 620
```

## 任务 3：语义分割

```powershell
python scripts/train_pet_seg.py --config configs/task3_unet_ce.yaml
python scripts/train_pet_seg.py --config configs/task3_unet_dice.yaml
python scripts/train_pet_seg.py --config configs/task3_unet_ce_dice.yaml
```

## 模型权重

完整模型权重和任务二视频结果已上传至 Google Drive：

https://drive.google.com/file/d/1nmT0xaLfUJzoOaF6HLzBV9tynE4U50vw/view?usp=drive_link

`weights_parts/` 中保留了同一压缩包的 GitHub 分卷备份。
