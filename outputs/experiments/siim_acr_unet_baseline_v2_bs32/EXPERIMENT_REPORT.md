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

## 阈值扫描结果

阈值扫描文件：`outputs/experiments/siim_acr_unet_baseline_v2_bs32/20260515_160529/logs/threshold_sweep.csv`

该扫描基于验证集，共包含 476 张阳性图像和 1659 张阴性图像。由于 `threshold_sweep.csv` 使用的是扫描脚本中的聚合口径，本节主要比较同一份扫描结果内部不同 threshold 的相对变化。

| threshold | global Dice | global IoU | precision | recall | negative FP rate | positive mean Dice | all mean Dice |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.50 | 0.4602 | 0.2988 | 0.4718 | 0.4491 | 0.4250 | 0.3928 | 0.4712 |
| 0.80 | 0.4612 | 0.2997 | 0.4964 | 0.4307 | 0.3924 | 0.3850 | 0.5041 |
| 0.95 | 0.4593 | 0.2981 | 0.5317 | 0.4042 | 0.3436 | 0.3712 | 0.5432 |

按 global Dice 选择，最佳阈值为 0.80，global Dice 为 0.4612，相比 threshold=0.50 的 0.4602 仅提升约 0.0011。提高 threshold 可以提升 precision、specificity，并降低阴性样本误检率，但同时会降低 recall 和阳性图像上的 mean Dice。

因此，阈值搜索只能带来很小的指标收益，当前主要瓶颈不是 threshold 选择，而是模型泛化能力和阴性样本误检。即使 threshold 提高到 0.80，negative FP rate 仍有 0.3924；提高到 0.95 后 negative FP rate 仍为 0.3436，同时 recall 下降到 0.4042，说明仅靠后处理阈值无法解决负样本误检问题。

## 验证集可视化观察

可视化目录：`outputs/experiments/siim_acr_unet_baseline_v2_bs32/20260515_160529/visualizations/val_predictions_thr0p80`

该批可视化基于验证集和 threshold=0.80 生成，共覆盖 2135 张验证图像，其中阳性 476 张、阴性 1659 张。目录中同时保存了 `summary.json`、`per_image_metrics.csv` 和 `selected_cases.json`，便于后续按单图指标追踪具体病例。

| 子目录 | 含义 | 样本数 |
| --- | --- | ---: |
| `high_dice_positive` | 正样本中预测较好的例子 | 48 |
| `low_dice_positive` | 正样本中预测较差的例子 | 48 |
| `false_positive_negative` | 负样本误检明显的例子 | 48 |
| `true_negative` | 负样本预测干净的例子 | 48 |

`summary.json` 显示，在 threshold=0.80 下，1659 张阴性图像中有 651 张存在预测前景，negative FP rate 为 0.3924；1008 张为 true negative。阳性图像 mean Dice 为 0.3849，负样本 mean Dice 为 0.5389，全部图像 mean Dice 为 0.5045。

从可视化组织方式看，当前错误分析应重点比较两类问题：一是 `low_dice_positive` 中真实气胸区域漏检、边界偏移或预测面积不足的问题；二是 `false_positive_negative` 中负样本出现局灶性误检的问题。`high_dice_positive` 和 `true_negative` 可作为模型已能处理的参考样本，用来对照失败样本中影像质量、胸膜边缘、肺尖/肋膈角和高亮伪影等因素。

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
- 阈值扫描：`outputs/experiments/siim_acr_unet_baseline_v2_bs32/20260515_160529/logs/threshold_sweep.csv`
- 验证集可视化：`outputs/experiments/siim_acr_unet_baseline_v2_bs32/20260515_160529/visualizations/val_predictions_thr0p80`

## 结论

v2 是当前两个实验中验证集 Dice 更高的版本。类别不均衡处理提升了气胸前景学习能力，但也暴露出明显过拟合。阈值扫描显示，最佳 global Dice 出现在 threshold=0.80，但相比 threshold=0.50 只提升约 0.0011；因此当前提升空间主要不在 threshold，而在模型泛化和负样本误检控制。threshold=0.80 的验证集可视化进一步显示，负样本误检仍较明显，应优先做错误类型归因，再决定是否进入 test 集评估。

建议下一步：

- 继续细看 `low_dice_positive` 和 `false_positive_negative`，整理常见失败模式。
- 重点分析阴性样本误检案例，区分边缘伪影、肺野外响应、低置信度散点和疑似标注噪声。
- 加入数据增强，尤其是轻量几何增强和强度扰动。
- 尝试早停策略，以验证 Dice 为主保存最佳模型。
- 在验证集错误分析和阈值选择固定后，再做独立 test 集评估。

本报告只基于验证集日志生成，尚未包含独立 test 集评估。
