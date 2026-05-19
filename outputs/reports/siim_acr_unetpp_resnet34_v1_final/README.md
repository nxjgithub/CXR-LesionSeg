# SIIM-ACR Unet++-ResNet34 最终说明

## 目录用途

本目录整理 `siim_acr_unetpp_resnet34_v1` 实验的轻量结果文件，用于归档 SIIM-ACR 气胸分割任务上的基线模型训练、阈值选择和简单后处理分析。原始训练输出位于：

```text
outputs/experiments/siim_acr_unetpp_resnet34_v1/20260518_180917
```

本目录保留配置、训练指标、阈值扫描、后处理参数扫描和最终逐样本统计结果，不包含模型权重和大体积可视化图片。

## 文件说明

| 文件 | 说明 |
|---|---|
| `config.json` | 本次 Unet++-ResNet34 训练配置 |
| `metrics.csv` | 50 轮训练与验证指标 |
| `threshold_sweep_val_*.csv` | 验证集二值化阈值扫描结果 |
| `postprocess_sweep_val*.csv` | 验证集后处理参数组合搜索结果 |
| `final_summary.json` | 最终后处理配置下的验证集汇总统计 |
| `final_per_image_metrics.csv` | 最终后处理配置下的逐样本指标 |
| `final_selected_cases.json` | 定性分析中选出的代表性样本 |

## 实验设置

本实验使用 `segmentation_models_pytorch` 中的 Unet++，编码器为 ResNet34，编码器初始化为 ImageNet 预训练权重。输入图像尺寸为 `512 x 512`，训练 50 轮，使用 BCE + Dice 组合损失，其中 BCE 权重为 `0.3`，Dice 权重为 `0.7`，BCE `pos_weight=2.0`。训练阶段启用 balanced sampling，正样本采样比例为 `0.35`。

训练日志显示，模型在前 20 轮内收敛较快，验证集 Dice 在第 24 轮达到最高值 `0.5146`。之后训练集 Dice 继续升高，但验证集 Dice 未继续改善，说明模型开始出现过拟合。

## 阈值选择

在 `best_model.pth` 上进行验证集 threshold sweep 后，单纯阈值扫描的最佳点为：

| Threshold | Dice | IoU | Precision | Recall |
|---:|---:|---:|---:|---:|
| 0.04 | 0.5198 | 0.3512 | 0.5720 | 0.4764 |

与默认阈值 `0.50` 相比，降低阈值可以提升 Recall，并略微提高整体 Dice；但 Precision 会下降，阴性样本上的误报也会增加。因此，阈值调整只能改变召回与误报之间的权衡，不能根本解决困难阳性样本漏检问题。

## 最终后处理配置

综合阈值扫描和简单连通域后处理分析，本目录采用以下配置作为最终整理结果：

| 参数 | 数值 |
|---|---:|
| Threshold | 0.08 |
| Min component area | 150 |
| Keep largest component | False |

该配置对应的验证集统计如下：

| 指标 | 数值 |
|---|---:|
| 验证集样本数 | 2135 |
| 阳性样本数 | 476 |
| 阴性样本数 | 1659 |
| Mean Dice all | 0.7576 |
| Mean Dice positive | 0.3980 |
| Mean Dice negative | 0.8608 |
| Mean IoU positive | 0.3038 |
| False positive negative | 231 / 1659 |
| Negative FP rate | 13.92% |

后处理能够有效降低阴性样本上的误报，使整体 Dice 和负样本 Dice 得到提升。与不做连通域后处理的 `threshold=0.04` 可视化统计相比，阴性样本假阳性数从 `345 / 1659` 降至 `231 / 1659`，阴性假阳性率从 `20.80%` 降至 `13.92%`。

## 定性分析结论

从最终可视化和逐样本统计看，模型已经能够在部分典型阳性样本上定位气胸区域，说明 Unet++-ResNet34 具备一定的病灶区域学习能力。

但在阳性样本上，模型对小面积、边界模糊或低对比度气胸区域仍存在漏检和边界偏差，`mean_dice_positive` 仍处于约 `0.40` 的水平。后处理虽然可以减少阴性胸片误报，但也可能删除一部分小病灶或边缘区域，因此无法替代模型本身对困难阳性样本的学习能力。

因此，当前模型更适合作为后续多模型、多数据集实验的基线，而不是最终性能上限。

## 后续建议

1. 将 `mean_dice_positive`、阴性假阳性率和整体 Dice 同时作为模型选择指标，避免只优化整体 Dice。
2. 继续比较 DeepLabV3+、Attention U-Net、Swin-Unet 等模型，观察是否能提升困难阳性样本召回。
3. 尝试 Focal Loss、Tversky Loss 或 Focal Tversky Loss，缓解气胸区域小和正负像素不均衡问题。
4. 加强胸片灰度、对比度、局部遮挡和尺度变化相关的数据增强。
5. 对低 Dice 阳性样本和假阳性阴性样本进行集中回看，区分模型能力不足、预处理问题和标注差异。
