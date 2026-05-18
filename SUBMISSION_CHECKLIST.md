# HW2 提交核对清单

## 必交材料

- `HW2_report.pdf`：仅提交 PDF 格式实验报告。
- Public GitHub repo：包含完整代码、配置文件、README.md、训练/测试说明。
- 模型权重网盘链接：百度云或 Google Drive 均可。

## 报告必须包含

- 小组成员姓名、学号、具体分工。
- 模型结构、数据集和实验结果的基本介绍。
- 训练/测试集划分、网络结构、batch size、learning rate、优化器、iteration、epoch、loss function、评价指标等详细设置。
- wandb 或 swanlab 可视化截图：训练集/验证集 loss 曲线，验证集 Accuracy / mAP 曲线。
- GitHub repo 链接和模型权重下载链接。

## 任务 1 检查点

- ResNet-18 或 ResNet-34 ImageNet 预训练 Baseline。
- 从零训练对照。
- 至少一组超参数分析，例如学习率、训练步数、batch size。
- 注意力机制或轻量级 Transformer 对照，例如 SE-block、CBAM、ViT-Tiny、Swin-T。
- 使用 Accuracy 对比模型性能。

## 任务 2 检查点

- VisDrone 数据集训练 YOLOv8 或同类现代单阶段检测模型。
- 10-30 秒测试视频，逐帧输出 Bounding Box、类别和稳定 Tracking ID。
- 选取遮挡或密集交汇片段，截取连续 3-4 帧并分析 ID 是否保持。
- 在视频中设置虚拟线，统计跨线物体总数。
- 报告 mAP、跟踪展示截图和越线计数结果。

## 任务 3 检查点

- 从零搭建 U-Net，不使用任何预训练权重。
- 包含下采样编码器、上采样解码器和 Skip Connection。
- Oxford-IIIT Pet 三分类 trimap 分割。
- 对比 Cross-Entropy、Dice Loss、Cross-Entropy + Dice Loss。
- 使用验证集 mIoU 对比结果。

## 最终提交前需要你补齐

- 报告首页姓名、学号、分工。
- 实际完整训练结果和 wandb/swanlab 截图。
- 公开 GitHub repo 链接。
- 训练权重网盘下载地址。

