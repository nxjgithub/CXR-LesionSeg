# SIIM-ACR U-Net Baseline v2 实验报告

## 实验概览

- 实验名称：`siim_acr_unet_baseline_v2_bs32`
- 运行目录：`outputs/experiments/siim_acr_unet_baseline_v2_bs32/20260515_160529`
- 数据集：SIIM-ACR Pneumothorax Segmentation
- 任务：气胸二值分割
- 模型：基础 U-Net
- 输入：单通道 512 x 512 胸片
- 训练设备：CUDA
- AMP：启用
- 随机种子：42

该实验是在 baseline 基础上加入类别不均衡处理的版本。

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

## 训练配置

配置文件：`outputs/experiments/siim_acr_unet_baseline_v2_bs32/20260515_160529/config.json`

| 参数 | 值 |
| --- | --- |
| image_size | 512 |
| in_channels | 1 |
| base_channels | 32 |
| configured_epochs | 60 |
| recorded_epochs | 42 |
| batch_size | 32 |
| num_workers | 8 |
| lr | 0.0005 |
| weight_decay | 0.0001 |
| loss | BCE + Dice |
| bce_weight | 0.5 |
| dice_weight | 0.5 |
| threshold | 0.5 |
| balanced_sampling | true |
| positive_sample_ratio | 0.5 |
| pos_weight | 5.0 |

说明：`config.json` 中计划训练 60 个 epoch，但 `metrics.csv` 实际记录到第 42 个 epoch。本报告以已有日志中的 42 个 epoch 为准。

## 最佳结果

验证集最佳 Dice 出现在第 26 个 epoch。

| 指标 | 值 |
| --- | ---: |
| val_loss | 0.4941 |
| val_dice | 0.4250 |
| val_iou | 0.2822 |
| val_precision | 0.4346 |
| val_recall | 0.4408 |
| val_specificity | 0.9985 |

对应训练指标：

| 指标 | 值 |
| --- | ---: |
| train_loss | 0.2867 |
| train_dice | 0.8893 |
| train_iou | 0.8018 |
| train_precision | 0.8548 |
| train_recall | 0.9287 |
| train_specificity | 0.9989 |

## 最后一轮结果

第 42 个 epoch 的验证指标如下：

| 指标 | 值 |
| --- | ---: |
| val_loss | 0.5161 |
| val_dice | 0.4098 |
| val_iou | 0.2708 |
| val_precision | 0.5292 |
| val_recall | 0.3491 |
| val_specificity | 0.9992 |

第 42 个 epoch 的训练 Dice 达到 0.9633，明显高于验证 Dice 0.4098。

## 与 baseline 对比

| 实验 | 最佳 epoch | 最佳 val Dice | 最佳 val IoU | val precision | val recall |
| --- | ---: | ---: | ---: | ---: | ---: |
| `siim_acr_unet_baseline` | 20 | 0.3632 | 0.2300 | 0.3484 | 0.4162 |
| `siim_acr_unet_baseline_v2_bs32` | 26 | 0.4250 | 0.2822 | 0.4346 | 0.4408 |

相较 baseline，v2 的最佳验证 Dice 提升约 0.0618，IoU 提升约 0.0522。balanced sampling 与 `pos_weight=5.0` 对前景学习有正向作用。

## 训练过程观察

- 第 1 个 epoch 的验证 Dice 仍接近 0，模型初期同样偏向空 mask。
- 第 26 个 epoch 达到最佳验证 Dice 0.4250。
- 第 26 个 epoch 时训练 Dice 已达到 0.8893，而验证 Dice 为 0.4250，训练/验证差距很大。
- 第 42 个 epoch 训练 Dice 继续升至 0.9633，但验证 Dice 降至 0.4098，说明继续训练主要提升了训练集拟合，泛化收益有限。
- 最后一轮 precision 提升到 0.5292，但 recall 降至 0.3491，模型预测更保守，漏检风险增加。

## 产物

- 最佳模型：`outputs/experiments/siim_acr_unet_baseline_v2_bs32/20260515_160529/checkpoints/best_model.pth`
- 最后模型：`outputs/experiments/siim_acr_unet_baseline_v2_bs32/20260515_160529/checkpoints/last_model.pth`
- 指标日志：`outputs/experiments/siim_acr_unet_baseline_v2_bs32/20260515_160529/logs/metrics.csv`

## 结论

v2 是当前两个实验中验证集 Dice 更高的版本。类别不均衡处理提升了气胸前景学习能力，但也暴露出明显过拟合。后续建议优先基于第 26 个 epoch 的 best checkpoint 做验证集可视化和阈值扫描，再决定是否进入 test 集评估。

建议下一步：

- 对 best checkpoint 做定性可视化，分别查看小 mask、中等 mask、大 mask 和空 mask 样本。
- 在验证集上扫描 threshold，例如 0.2 到 0.7，观察 Dice、precision、recall 权衡。
- 加入数据增强，尤其是轻量几何增强和强度扰动。
- 尝试早停策略，以验证 Dice 为主保存最佳模型。
- 在完成阈值选择后，再做独立 test 集评估。

本报告只基于验证集日志生成，尚未包含独立 test 集评估。
