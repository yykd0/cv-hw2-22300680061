# 实验资源说明

## 基本信息

- 姓名：邓凯源
- 学号：22300680061
- 课程作业：计算机视觉 HW2

## 实验内容

本仓库包含三个实验任务的代码与配置：

- 任务 1：Oxford-IIIT Pet 图像分类，比较 ImageNet 预训练、随机初始化、SE、CBAM 和 ViT-Tiny。
- 任务 2：VisDrone 目标检测与多目标跟踪，使用 YOLOv8 完成检测，并输出 Tracking ID、遮挡片段分析和虚拟线计数。
- 任务 3：Oxford-IIIT Pet trimap 三分类分割，从零实现 U-Net，并比较 Cross-Entropy、Dice Loss 和 CE+Dice。

## 主要文件

- `configs/`：实验配置文件。
- `scripts/`：训练、数据转换和跟踪计数脚本。
- `src/cvhw2/`：模型、数据集、损失函数和指标实现。
- `tools/`：批量运行、绘图和报告生成辅助脚本。
- `report/HW2_report_CHINESE_22300680061.docx`：实验报告。
- `weights_parts/`：模型权重和任务二视频结果的分卷文件。

## 模型权重

模型权重与任务二视频结果保存在 `weights_parts/` 目录中。下载全部 `.partXX` 文件后，按照 `weights_parts/README.md` 中的命令合并为完整压缩包。
