# 作业要求摘要

## 任务 1：微调 ImageNet 预训练 CNN 实现宠物识别

- 数据集：Oxford-IIIT Pet Dataset。
- Baseline：修改 ResNet-18 或 ResNet-34 输出层，使用 ImageNet 预训练参数初始化，其余参数用较小学习率微调。
- 超参数分析：训练步数、学习率及组合影响。
- 预训练消融：与随机初始化从零训练对比。
- 注意力/Transformer：在 Baseline 上加入 SE-block、CBAM，或使用 ViT-Tiny、Swin-T，并与 Baseline 比较 Accuracy。

## 任务 2：场景目标检测与视频多目标跟踪

- 数据集：VisDrone。
- 检测模型：YOLOv8 或同类现代单阶段检测模型。
- 测试视频：10-30 秒，可用校园/路口自拍视频。
- 输出：Bounding Box、类别、稳定 Tracking ID。
- 遮挡分析：选取遮挡或密集交汇片段，截取连续 3-4 帧，分析 ID 是否保持、目标丢失或 ID 跳变原因。
- 越线计数：设定虚拟线，依据检测框中心点与 Tracking ID 连续性统计跨线物体总数。

## 任务 3：从零搭建 U-Net 分割模型

- 不使用任何预训练权重。
- 使用框架基础 API 从零实现经典 U-Net。
- 必须包含下采样编码器、上采样解码器和 Skip Connection。
- 数据集：Oxford-IIIT Pet Dataset 三分类分割。
- 损失函数对比：Cross-Entropy Loss、Dice Loss、Cross-Entropy Loss + Dice Loss。
- 指标：验证集 mIoU。

## 提交要求

- 只提交 PDF 格式实验报告。
- 报告必须包含模型结构、数据集、实验结果、详细训练设置、wandb/swanlab 曲线截图。
- 代码提交到 public GitHub repo，README.md 中说明环境配置、训练和测试方法。
- 训练好的模型权重上传到百度云或 Google Drive。
- 报告中必须包含代码 repo 链接和模型权重网盘下载地址。
- 团队人数少于或等于 2 人；同质量下 1 人完成有额外加分。
- 小组只需 1 人提交；报告首页必须写明全部成员姓名、学号和分工。

