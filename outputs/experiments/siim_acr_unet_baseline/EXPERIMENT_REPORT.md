# SIIM-ACR U-Net Baseline 实验报告

## 实验概览

- 实验名称：`siim_acr_unet_baseline`
- 运行目录：`outputs/experiments/siim_acr_unet_baseline/20260515_152651`
- 数据集：SIIM-ACR Pneumothorax Segmentation
- 任务：气胸二值分割
- 模型：基础 U-Net
- 输入：单通道 512 x 512 胸片
- 训练设备：CUDA
- AMP：启用
- 随机种子：42

## 数据划分

| split | 数量 |
| --- | ---: |
| train | 8540 |
| val | 2135 |
| test | 1377 |

训练/验证集合的图像级 mask 分布：

| has_mask | 数量 |
| --- | ---: |
| 0 | 8296 |
| 1 | 2379 |

该分布显示阳性气胸样本占比较低，且前景 mask 像素更稀疏，因此 Dice、recall 和 foreground precision 对采样策略与损失权重较敏感。

## 训练配置

配置文件：`outputs/experiments/siim_acr_unet_baseline/20260515_152651/config.json`

| 参数 | 值 |
| --- | --- |
| image_size | 512 |
| in_channels | 1 |
| base_channels | 32 |
| epochs | 20 |
| batch_size | 32 |
| num_workers | 8 |
| lr | 0.001 |
| weight_decay | 0.0001 |
| loss | BCE + Dice |
| bce_weight | 0.5 |
| dice_weight | 0.5 |
| threshold | 0.5 |

本实验未启用 balanced sampling，也未设置 BCE `pos_weight`。

## 最佳结果

验证集最佳 Dice 出现在第 20 个 epoch。

| 指标 | 值 |
| --- | ---: |
| val_loss | 0.4654 |
| val_dice | 0.3632 |
| val_iou | 0.2300 |
| val_precision | 0.3484 |
| val_recall | 0.4162 |
| val_specificity | 0.9979 |

对应训练指标：

| 指标 | 值 |
| --- | ---: |
| train_loss | 0.4509 |
| train_dice | 0.4477 |
| train_iou | 0.2985 |
| train_precision | 0.4142 |
| train_recall | 0.5323 |
| train_specificity | 0.9979 |

## 训练过程观察

- 第 1 个 epoch 的验证 Dice 接近 0，说明模型初期基本预测为空 mask。
- 随训练推进，验证 Dice 逐步提升，最终达到 0.3632。
- 训练 Dice 为 0.4477，验证 Dice 为 0.3632，存在一定泛化差距，但尚未出现特别严重的训练集记忆现象。
- recall 为 0.4162，precision 为 0.3484，说明模型仍有较多漏检和误检空间。
- specificity 接近 0.998，这在强背景占比任务中较常见，不能单独作为分割质量判断依据。

## 产物

- 最佳模型：`outputs/experiments/siim_acr_unet_baseline/20260515_152651/checkpoints/best_model.pth`
- 最后模型：`outputs/experiments/siim_acr_unet_baseline/20260515_152651/checkpoints/last_model.pth`
- 指标日志：`outputs/experiments/siim_acr_unet_baseline/20260515_152651/logs/metrics.csv`

## 结论

该实验可以作为 SIIM-ACR 气胸分割的基础 U-Net baseline。模型已经能学习到一定的气胸区域定位能力，但验证 Dice 和 IoU 偏低，尤其 recall 不足。考虑到数据存在明显阳性/阴性不均衡，后续应优先尝试阳性采样、前景像素加权、阈值调优和更强的数据增强策略。

本报告只基于验证集日志生成，尚未包含独立 test 集评估。
