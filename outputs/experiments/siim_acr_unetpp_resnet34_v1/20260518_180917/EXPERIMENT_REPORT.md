# SIIM-ACR Unet++-ResNet34 实验报告

## 1. 实验目的

本实验在 SIIM-ACR Pneumothorax Segmentation 数据集上评估基于 `segmentation_models_pytorch` 的 Unet++-ResNet34 分割模型，目标是验证相较于当前 U-Net baseline，更强的编码器和嵌套解码结构是否能提升气胸区域定位与分割能力。

本报告仅基于验证集结果进行分析。当前实验未包含独立 test 集评估，因此以下结果不能表述为最终泛化性能。

## 2. 实验配置

实验输出目录：

```text
outputs/experiments/siim_acr_unetpp_resnet34_v1/20260518_180917
```

主要配置来自 `config.json`：

| 配置项 | 数值 |
|---|---:|
| 数据集 | SIIM-ACR Pneumothorax Segmentation |
| 模型 | Unet++ |
| Encoder | ResNet34 |
| Encoder 预训练 | ImageNet |
| 输入尺寸 | 512 x 512 |
| 输入通道 | 3 |
| 训练轮数 | 50 |
| Batch size | 32 |
| 初始学习率 | 0.0003 |
| Weight decay | 0.0003 |
| Loss | BCE + Dice |
| BCE 权重 | 0.3 |
| Dice 权重 | 0.7 |
| BCE `pos_weight` | 2.0 |
| 默认二值化阈值 | 0.50 |
| Balanced sampling | 是 |
| 正样本采样比例 | 0.35 |
| 随机种子 | 42 |
| 设备 | CUDA |
| AMP | 开启 |

本实验对应的代码变更包括 SMP 模型封装、SMP 训练入口、后处理参数搜索脚本和验证集预测可视化脚本：

```text
src/cxr_lesionseg/models/smp_models.py
scripts/train/train_siim_smp.py
scripts/evaluate/threshold_sweep_siim_smp.py
scripts/evaluate/postprocess_sweep_siim_smp.py
scripts/evaluate/visualize_siim_predictions_smp.py
scripts/evaluate/visualize_siim_predictions_smp_postprocess.py
```

其中 `threshold_sweep_siim_smp.py` 和 `postprocess_sweep_siim_smp.py` 用于在验证集上搜索最优阈值和后处理参数组合，对应结果保存在 `logs/` 目录；`visualize_siim_predictions_smp.py` 和 `visualize_siim_predictions_smp_postprocess.py` 用于可视化模型预测结果并生成逐样本定性分析统计，对应结果保存在 `visualizations/` 目录。`visualizations/` 中包含大量图片产物，已通过 `.gitignore` 排除，不作为轻量实验记录提交。

## 3. 训练结果

训练日志来自 `logs/metrics.csv`。模型在前 20 轮内收敛较快，验证集 Dice 在训练早期整体上升，并在第 24 轮达到最高值。第 24 轮后，训练集 Dice 继续升高，但验证集 Dice 未能继续改善，说明模型开始对训练集产生过拟合。

关键 epoch 指标如下：

| Epoch | Train Dice | Val Dice | Val IoU | Val Precision | Val Recall |
|---:|---:|---:|---:|---:|---:|
| 10 | 0.6547 | 0.3938 | - | - | - |
| 15 | 0.7621 | 0.4845 | - | - | - |
| 24 | 0.8929 | 0.5146 | 0.3464 | 0.6161 | 0.4418 |
| 30 | 0.9400 | 0.4921 | - | - | - |
| 40 | 0.9689 | 0.4792 | - | - | - |
| 50 | 0.9753 | 0.4847 | 0.3199 | 0.6695 | 0.3798 |

最佳验证集结果：

| 指标 | 数值 |
|---|---:|
| Best epoch | 24 |
| Train loss | 0.4851 |
| Train Dice | 0.8929 |
| Train IoU | 0.8065 |
| Train Precision | 0.8844 |
| Train Recall | 0.9015 |
| Val loss | 0.6492 |
| Val Dice | 0.5146 |
| Val IoU | 0.3464 |
| Val Precision | 0.6161 |
| Val Recall | 0.4418 |
| Val Specificity | 0.9992 |

最终第 50 轮结果：

| 指标 | 数值 |
|---|---:|
| Train loss | 0.4672 |
| Train Dice | 0.9753 |
| Train IoU | 0.9518 |
| Train Precision | 0.9753 |
| Train Recall | 0.9753 |
| Val loss | 0.6610 |
| Val Dice | 0.4847 |
| Val IoU | 0.3199 |
| Val Precision | 0.6695 |
| Val Recall | 0.3798 |
| Val Specificity | 0.9994 |

从最佳 epoch 与末轮结果对比可以看到，训练集 Dice 从 0.8929 继续升至 0.9753，但验证集 Dice 从 0.5146 下降至 0.4847，验证集 Recall 也从 0.4418 下降至 0.3798。该现象表明继续训练主要提升了训练集拟合程度，并未带来验证集收益。

## 4. Threshold sweep 结果

为分析二值化阈值对分割性能的影响，本实验在验证集上对 `checkpoints/best_model.pth` 进行了 threshold sweep。结果显示，在已评估阈值中，当阈值设为 0.04 时模型取得最高 Dice。

| Threshold | Dice | IoU | Precision | Recall | Specificity |
|---:|---:|---:|---:|---:|---:|
| 0.04 | 0.5198 | 0.3512 | 0.5720 | 0.4764 | 0.9989 |
| 0.05 | 0.5198 | 0.3511 | 0.5765 | 0.4732 | 0.9990 |
| 0.10 | 0.5190 | 0.3504 | 0.5884 | 0.4642 | 0.9990 |
| 0.25 | 0.5171 | 0.3487 | 0.6029 | 0.4526 | 0.9991 |
| 0.50 | 0.5146 | 0.3464 | 0.6160 | 0.4418 | 0.9992 |

与默认阈值 0.50 相比，阈值 0.04 使 Dice 从 0.5146 提升到 0.5198，Recall 从 0.4418 提升到 0.4764，但 Precision 从 0.6160 下降到 0.5720。这说明降低阈值可以在一定程度上缓解漏检问题，但整体提升幅度较小，模型性能瓶颈仍主要来自病灶区域学习能力和泛化能力，而不只是后处理阈值设置。

## 5. 后处理参数搜索结果

为进一步分析简单连通域后处理对验证集表现的影响，本实验使用 `postprocess_sweep_siim_smp.py` 搜索了 `threshold`、`min_component_area` 和 `keep_largest_component` 的组合，结果保存在：

```text
logs/postprocess_sweep_val.csv
logs/postprocess_sweep_val_fine.csv
```

本轮后处理分析选取如下参数组合作为后处理 baseline：

| 参数 | 数值 |
|---|---:|
| 模型 | Unet++-ResNet34 |
| Threshold | 0.08 |
| Min component area | 150 |
| Keep largest component | False |

对应验证集结果如下：

| 指标 | 数值 |
|---|---:|
| Positive Dice | 0.3980 |
| Positive IoU | 0.3038 |
| Negative FP rate | 13.92% |
| False positive negative | 231 / 1659 |
| Mean Dice all | 0.7576 |

与 `threshold=0.04` 且不做连通域后处理的可视化统计相比，该后处理配置将阴性样本假阳性比例从 20.80% 降低到 13.92%，假阳性阴性样本数从 345 降低到 231，说明去除小连通域可以有效减少阴性胸片上的误检。但 positive-only Dice 从 0.4080 下降到 0.3980，说明该后处理也会删除一部分真实阳性区域或削弱小病灶召回。因此，后处理主要改善阴性样本误检和整体平均 Dice，对阳性病灶分割能力的提升有限。

从 `postprocess_sweep_val_fine.csv` 看，若仅优化 `mean_dice_all`，较大的 `min_component_area` 会进一步降低阴性假阳性率，例如 `threshold=0.25, min_component_area=300, keep_largest_component=False` 可达到 `mean_dice_all=0.7822`、`negative_fp_rate=10.25%`，但 positive-only Dice 降至 0.3804。该结果提示：在包含大量阴性样本的验证集中，`mean_dice_all` 容易偏向保守后处理；后续选择后处理参数时应同时报告 positive-only Dice、阴性假阳性率和整体 Dice，避免只按单一整体指标选型。

## 6. 可视化与逐样本分析

可视化结果位于：

```text
outputs/experiments/siim_acr_unetpp_resnet34_v1/20260518_180917/visualizations/val_predictions_thr0p04
```

该目录按 `high_dice_positive`、`low_dice_positive`、`false_positive_negative` 和 `true_negative` 分类保存验证集样本可视化。每类保存 24 个样本，并同时包含对应的预测叠加图和对比图。

逐样本统计来自 `visualizations/val_predictions_thr0p04/summary.json` 和 `per_image_metrics.csv`：

| 指标 | 数值 |
|---|---:|
| 验证集样本数 | 2135 |
| 阳性样本数 | 476 |
| 阴性样本数 | 1659 |
| 阴性样本假阳性数 | 345 |
| 阴性样本假阳性比例 | 20.80% |
| 阴性真阴性数 | 1314 |
| Positive-only Dice 均值 | 0.4080 |
| Positive-only IoU 均值 | 0.3100 |
| 阴性样本 Dice 均值 | 0.7523 |
| 阳性样本中预测为空的数量 | 87 |
| 阳性样本中 Dice < 0.01 的数量 | 117 |
| 阳性样本中 Dice >= 0.80 的数量 | 61 |

从 `threshold=0.04` 的可视化结果看，模型能够在部分典型阳性样本上获得较好的分割效果。例如高 Dice 样本中，预测区域与人工标注区域基本重合，说明 Unet++-ResNet34 已经学习到一定的气胸区域定位能力。

但模型仍存在两类主要错误：

1. 阴性样本上产生较大面积假阳性预测。阈值 0.04 下，1659 个阴性样本中有 345 个样本出现假阳性，比例约为 20.80%。
2. 部分阳性样本完全未能激活。476 个阳性样本中有 87 个样本预测区域为空，117 个样本 Dice 低于 0.01，说明模型对困难阳性样本仍存在明显漏检。

因此，单纯降低二值化阈值虽然可以略微提升整体 Dice，但不能根本解决困难阳性样本漏检和复杂阴性样本误检的问题。后处理可进一步降低阴性样本误检，但也会带来一定阳性 Dice 损失，因此需要结合任务目标选择参数。

## 7. 结论

本实验验证了 Unet++-ResNet34 在 SIIM-ACR 气胸分割任务上的基本有效性。模型在验证集上取得最佳 Dice 0.5146，经过阈值扫描后最佳验证 Dice 可提升至 0.5198。模型在部分典型阳性样本上可以得到较准确的气胸区域分割，说明网络具备一定病灶定位能力。

同时，实验也暴露出当前方案的主要问题：训练后期过拟合明显，验证集 Precision 高于 Recall，预测结果偏保守；降低阈值后 Recall 有所提升，但阴性样本假阳性增多，且 positive-only Dice 仍只有 0.4080。后处理 baseline 可以将阴性假阳性比例降至 13.92%，但 positive-only Dice 降至 0.3980。这表明模型对困难阳性样本的识别能力和整体泛化能力仍不足，后处理只能调整误检/漏检权衡，不能替代模型本身的病灶学习能力。

## 8. 后续改进方向

后续实验建议从以下方向继续改进：

1. 阈值选择：保留 threshold sweep 作为固定评估流程，并在报告中同时记录默认阈值和验证集最优阈值结果。
2. 后处理选择：保留 postprocess sweep 作为固定评估流程，同时记录 `threshold`、`min_component_area`、`keep_largest_component`、positive-only Dice、阴性假阳性率和整体 Dice。
3. 数据增强：加强与胸片灰度、对比度、局部遮挡、尺度变化相关的数据增强，重点提升困难阳性样本泛化能力。
4. 损失函数：尝试 Focal Loss、Tversky Loss、Focal Tversky Loss，缓解气胸区域小、正负像素极度不均衡带来的优化问题。
5. 正样本评估：将 positive-only Dice、阳性漏检率、阴性假阳性率纳入标准实验报告，避免整体 Dice 被大量阴性样本掩盖。
6. 训练策略：考虑早停或保存多个候选 checkpoint，避免训练后期仅提升训练集拟合而损害验证集表现。
7. 错误样本回看：重点检查低 Dice 阳性样本和假阳性阴性样本，判断是否存在标注质量、图像预处理或分布差异问题。
