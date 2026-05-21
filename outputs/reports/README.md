# SIIM-ACR 结果归档说明

## 1. 目录用途

`outputs/reports/` 用于保存可提交、可复盘的轻量实验结果归档。该目录不保存模型权重和大体积可视化图片，主要保留配置、指标 CSV、summary JSON 和 Markdown 说明。

当前最重要的归档结果包括：

| 文件或目录 | 说明 |
|---|---|
| `siim_acr_unetpp_resnet34_v1_final/` | 单模型 U-Net++ ResNet34 baseline 的最终整理结果 |
| `final_ensemble_results.csv` | 2/3/4 模型 ensemble 的验证集候选结果 |
| `model_comparison_postprocess_val.csv` | 部分单模型后处理对比结果 |

本文仅基于验证集结果总结当前阶段模型选择。尚未进行独立 test 集最终评估。

## 2. 单模型主线结果

单模型阶段，当前主线 baseline 为：

```text
outputs/experiments/siim_acr_unetpp_resnet34_v1/20260518_180917
```

最终整理目录：

```text
outputs/reports/siim_acr_unetpp_resnet34_v1_final
```

推荐后处理配置：

| 参数 | 数值 |
|---|---:|
| Threshold | 0.08 |
| Min component area | 150 |
| Keep largest component | False |

验证集结果：

| 指标 | 数值 |
|---|---:|
| Mean Dice all | 0.7576 |
| Mean Dice positive | 0.3980 |
| Mean IoU positive | 0.3038 |
| Negative FP rate | 13.92% |
| False positive negative | 231 / 1659 |

该模型在正样本 Dice 与阴性误报率之间表现较为均衡，是后续模型融合前的主要 baseline。

## 3. Ensemble 结果归档

2 模型融合实验目录：

```text
outputs/experiments/ensemble_unetpp_r34_r50_recall_v1
```

融合模型：

| 模型 | 来源 |
|---|---|
| U-Net++ ResNet34 recall v2 | `outputs/experiments/siim_acr_unetpp_resnet34_recall_v2/20260519_160852` |
| U-Net++ ResNet50 recall v1 | `outputs/experiments/siim_acr_unetpp_resnet50_recall_v1/20260519_171710` |

最终主配置：

| 参数 | 数值 |
|---|---:|
| Threshold | 0.25 |
| Min component area | 500 |
| Min pred area | 0 |
| Keep largest component | False |
| TTA | hflip |
| Num models | 2 |

验证集结果来自 `final_ensemble_results.csv` 中的 `main` 行：

| 指标 | 数值 |
|---|---:|
| Mean Dice all | 0.7798 |
| Mean Dice positive | 0.4365 |
| Mean IoU positive | 0.3407 |
| Mean Dice negative | 0.8782 |
| Negative FP rate | 12.18% |
| False positive negative | 202 / 1659 |
| Positive image recall | 72.90% |
| False negative positive | 129 / 476 |

低误报对照配置来自 `final_ensemble_results.csv` 中的 `low_fp` 行：

| 指标 | 数值 |
|---|---:|
| Threshold | 0.30 |
| Min component area | 200 |
| Mean Dice all | 0.7802 |
| Mean Dice positive | 0.4239 |
| Negative FP rate | 11.75% |
| False positive negative | 195 / 1659 |
| Positive image recall | 74.16% |

新增 3 模型融合实验目录：

```text
outputs/experiments/ensemble_unetpp_r34_r50_deeplab_r34_v1
```

融合模型在 2 模型基础上加入 DeepLabV3+ ResNet34。当前主可视化配置来自 `visual_main` 行：

| 指标 | 数值 |
|---|---:|
| Threshold | 0.30 |
| Min component area | 500 |
| Mean Dice all | 0.7961 |
| Mean Dice positive | 0.4301 |
| Mean IoU positive | 0.3380 |
| Mean Dice negative | 0.9011 |
| Negative FP rate | 9.89% |
| False positive negative | 164 / 1659 |
| Positive image recall | 70.38% |
| False negative positive | 141 / 476 |

同一实验的 `balanced_candidate` 行为 `threshold=0.30, min_component_area=300`，positive Dice 为 0.4595，negative FP rate 为 12.48%。该配置更偏召回，但尚需结合可视化复查。

新增 4 模型融合实验目录：

```text
outputs/experiments/ensemble_unetpp_r34_r50_deeplab_unet_v1
```

融合模型在 3 模型基础上加入 SMP U-Net ResNet34。当前主可视化配置来自 `visual_main` 行：

| 指标 | 数值 |
|---|---:|
| Threshold | 0.30 |
| Min component area | 500 |
| Mean Dice all | 0.8115 |
| Mean Dice positive | 0.4255 |
| Mean IoU positive | 0.3346 |
| Mean Dice negative | 0.9222 |
| Negative FP rate | 7.78% |
| False positive negative | 129 / 1659 |
| Positive image recall | 69.12% |
| False negative positive | 147 / 476 |

同一实验的 `balanced_candidate` 行为 `threshold=0.25, min_component_area=300`，positive Dice 为 0.4759，negative FP rate 为 12.12%。该配置是当前 CSV 中正样本 Dice 较高且误报仍接近 2 模型主配置的候选，但尚需补充同等可视化复查。

## 4. 与单模型 baseline 对比

| 模型与配置 | Positive Dice | Negative FP rate | Mean Dice all | False positive negative |
|---|---:|---:|---:|---:|
| U-Net++ ResNet34 baseline | 0.3980 | 13.92% | 0.7576 | 231 / 1659 |
| 2-model Ensemble main | 0.4365 | 12.18% | 0.7798 | 202 / 1659 |
| 2-model Ensemble low-FP | 0.4239 | 11.75% | 0.7802 | 195 / 1659 |
| 3-model Ensemble visual main | 0.4301 | 9.89% | 0.7961 | 164 / 1659 |
| 4-model Ensemble visual main | 0.4255 | 7.78% | 0.8115 | 129 / 1659 |
| 4-model Ensemble balanced candidate | 0.4759 | 12.12% | 0.7890 | 201 / 1659 |

Ensemble main 相比单模型 baseline：

| 指标 | 变化 |
|---|---:|
| Positive Dice | +0.0385 |
| Negative FP rate | -1.74 pp |
| Mean Dice all | +0.0222 |
| False positive negative | -29 |

该结果说明，多模型融合结合水平翻转 TTA 与连通域面积过滤后，可以在不同配置下形成两类候选：2 模型主配置更均衡；3/4 模型主可视化配置更偏低误报。

## 5. 当前推荐

当前验证集上正样本 Dice 与误报平衡较清晰的候选方案为：

```text
Ensemble U-Net++ ResNet34 + ResNet50
threshold=0.25
min_component_area=500
min_pred_area=0
tta=hflip
```

推荐理由：

1. 相比单模型 U-Net++ ResNet34 baseline，positive-only Dice 从约 0.3980 提升至 0.4365。
2. 阴性样本误报率从约 13.92% 降至 12.18%。
3. mean Dice all 从 0.7576 提升至 0.7798。
4. 该配置比单纯追求最高 positive-only Dice 的低阈值配置更可控。

当前验证集上低误报候选方案为：

```text
Ensemble U-Net++ ResNet34 + ResNet50 + DeepLabV3+ ResNet34 + SMP U-Net ResNet34
threshold=0.30
min_component_area=500
min_pred_area=0
tta=hflip
```

该配置把 negative FP rate 降至 7.78%，但 positive-only Dice 为 0.4255，低于 2 模型主配置。后续若进行独立 test 集评估，应先固定候选配置，不应再根据 test 集调整阈值或后处理参数。
