# 计算机视觉期中作业 HW2

姓名：邓凯源  
学号：22300680061

本仓库对应三项实验任务：

1. 在 Oxford-IIIT Pet Dataset 上微调 ImageNet 预训练分类模型，并比较 Baseline、超参数、从零训练、注意力模块和轻量级 Transformer。
2. 在 VisDrone 数据集上训练 YOLOv8 检测模型，并完成视频多目标跟踪、遮挡/ID 跳变分析和越线计数。
3. 从零搭建 U-Net，在 Oxford-IIIT Pet 三分类 trimap 分割任务上比较 Cross-Entropy、Dice Loss 和组合损失。

模型权重与任务二跟踪视频按作业要求通过网盘提交，不放入 GitHub 仓库。报告中填写本仓库链接和网盘下载链接。

## 目录结构

```text
cv_hw2_submit/
  configs/                 # 三个任务的实验配置
  scripts/                 # 训练、数据转换、跟踪计数脚本
  src/cvhw2/               # 可复用数据集、模型、损失、指标代码
  tools/                   # 批量运行、绘图和报告辅助脚本
  docs/                    # 作业要求摘要
  SUBMISSION_CHECKLIST.md  # 提交前核对清单
```

## 环境配置

推荐使用 Python 3.10+ 和独立虚拟环境：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

如果使用 CUDA，请按本机 CUDA 版本从 PyTorch 官网选择对应安装命令。`requirements.txt` 中的 `torch` / `torchvision` 是通用入口，GPU 环境建议替换为官方 CUDA wheel。

## 任务 1：宠物识别

ResNet-18 ImageNet 预训练 Baseline：

```powershell
python scripts/train_pet_cls.py --config configs/task1_resnet18_imagenet.yaml
```

从零训练对照：

```powershell
python scripts/train_pet_cls.py --config configs/task1_resnet18_scratch.yaml
```

SE/CBAM 注意力对照：

```powershell
python scripts/train_pet_cls.py --config configs/task1_resnet18_se.yaml
python scripts/train_pet_cls.py --config configs/task1_resnet18_cbam.yaml
```

轻量级 Transformer：

```powershell
python scripts/train_pet_cls.py --config configs/task1_vit_tiny.yaml
```

训练日志保存在 `runs/task1_*`，其中 `metrics.csv` 可用于画 loss/accuracy 曲线；若配置了 `wandb` 或 `swanlab`，脚本也会同步记录。

## 任务 2：VisDrone 检测与跟踪

先将 VisDrone DET 数据转换为 YOLO 格式：

```powershell
python scripts/visdrone_to_yolo.py `
  --visdrone-root D:\datasets\VisDrone `
  --out data\visdrone_yolo
```

训练 YOLOv8：

```powershell
yolo detect train model=yolov8n.pt data=data\visdrone_yolo\visdrone.yaml epochs=80 imgsz=640 batch=16 project=runs name=task2_yolov8n_visdrone
```

对 10-30 秒视频进行跟踪、遮挡分析和越线计数：

```powershell
python scripts/track_count.py `
  --weights runs\task2_yolov8n_visdrone\weights\best.pt `
  --source data\demo_video.mp4 `
  --out runs\task2_tracking `
  --line 320 120 320 620
```

输出包括带检测框、类别、Tracking ID 和计数叠加的视频、越线事件 CSV，以及连续若干帧截图，方便写报告中的遮挡与 ID 跳变分析。

## 任务 3：U-Net 分割

三种损失配置分别运行：

```powershell
python scripts/train_pet_seg.py --config configs/task3_unet_ce.yaml
python scripts/train_pet_seg.py --config configs/task3_unet_dice.yaml
python scripts/train_pet_seg.py --config configs/task3_unet_ce_dice.yaml
```

脚本会输出验证集 mIoU、每类 IoU、loss 曲线 CSV 和最优权重。

## 提交说明

- GitHub 仓库：用于提交代码、配置和 README。
- 模型权重：上传 `HW2_model_weights_and_video_22300680061.zip` 到百度云或 Google Drive。
- 实验报告：将中文 DOCX 填入 GitHub 和网盘链接后另存为 PDF 提交。
