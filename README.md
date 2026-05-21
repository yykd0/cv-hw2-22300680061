# 计算机视觉 HW2

姓名：邓凯源  
学号：22300680061

本仓库为计算机视觉 HW2 的实验代码与结果整理，包含图像分类、目标检测与多目标跟踪、语义分割三个部分。

## 实验任务

1. Oxford-IIIT Pet 图像分类：基于 ResNet-18 进行 ImageNet 预训练微调，并与随机初始化、SE、CBAM、ViT-Tiny 等设置对比。
2. VisDrone 目标检测与多目标跟踪：使用 YOLOv8 训练检测模型，并在 10-30 秒视频上输出 Bounding Box、类别、Tracking ID、遮挡片段分析和虚拟线计数。
3. Oxford-IIIT Pet 语义分割：从零实现 U-Net，在 trimap 三分类任务上比较 Cross-Entropy、Dice Loss 和 CE+Dice。

## 目录结构

```text
configs/                 实验配置文件
scripts/                 训练、数据转换、跟踪计数脚本
src/cvhw2/               数据集、模型、损失函数和指标代码
tools/                   批量运行、绘图和报告辅助脚本
report/                  实验报告
weights_parts/           模型权重与任务二视频结果分卷
```

## 环境配置

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

GPU 环境下可根据 CUDA 版本安装对应的 PyTorch wheel。

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

模型权重和任务二跟踪视频保存在 `weights_parts/` 目录中。下载全部分卷后，按照 `weights_parts/README.md` 合并为完整压缩包。
