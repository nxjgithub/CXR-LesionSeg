# SIIM-ACR 4 模型融合实验报告

## 1. 实验目的

本实验在 SIIM-ACR Pneumothorax Segmentation 验证集上评估 4 模型融合方案。融合对象按实验目录和当前模型清单记录为：

| 模型 | 来源实验 |
|---|---|
| U-Net++ ResNet34 recall v2 | `outputs/experiments/siim_acr_unetpp_resnet34_recall_v2/20260519_160852` |
| U-Net++ ResNet50 recall v1 | `outputs/experiments/siim_acr_unetpp_resnet50_recall_v1/20260519_171710` |
| DeepLabV3+ ResNet34 v1 | `outputs/experiments/siim_acr_deeplabv3plus_resnet34_v1/20260519_115843` |
| SMP U-Net ResNet34 v1 | `outputs/experiments/siim_acr_smp_unet_resnet34_v1/20260519_141839` |

实验目标是观察在 3 模型融合基础上继续加入 SMP U-Net ResNet34 后，是否能进一步改善正样本分割能力、阴性样本误报控制和整体 Dice 的平衡。

本报告仅基于验证集结果进行分析，尚未包含独立 test 集评估，因此不能表述为最终泛化性能。

## 2. 实验配置

实验输出目录：

```text
outputs/experiments/ensemble_unetpp_r34_r50_deeplab_unet_v1
```

融合与评估设置：

| 配置项 | 数值 |
|---|---:|
| Split | val |
| 输入尺寸 | 512 x 512 |
| 模型数 | 4 |
| 融合方式 | 预测概率平均 |
| TTA | 水平翻转 `hflip` |
| 后处理 | 阈值二值化 + 连通域面积过滤 |
| `keep_largest_component` | False |

主要结果文件：

```text
logs/ensemble_postprocess_sweep_val.csv
visualizations/val_ensemble4_thr0p3_mincomp500_minpred0/summary.json
```

## 3. 主可视化配置结果

当前可视化目录对应的主配置为：

| 参数 | 数值 |
|---|---:|
| Threshold | 0.30 |
| Min component area | 500 |
| Min pred area | 0 |
| Keep largest component | False |
| TTA | hflip |
| Num models | 4 |

验证集结果如下：

| 指标 | 数值 |
|---|---:|
| 验证集样本数 | 2135 |
| 阳性样本数 | 476 |
| 阴性样本数 | 1659 |
| Mean Dice all | 0.8115 |
| Mean Dice positive | 0.4255 |
| Mean IoU positive | 0.3346 |
| Mean Dice negative | 0.9222 |
| Negative FP rate | 7.78% |
| False positive negative | 129 / 1659 |
| Positive image recall | 69.12% |
| False negative positive | 147 / 476 |
| Pixel Dice | 0.5694 |
| Pixel Precision | 0.6487 |
| Pixel Recall | 0.5075 |

与 2 模型和 3 模型主可视化配置相比：

| 模型与配置 | Positive Dice | Negative FP rate | Mean Dice all | False positive negative |
|---|---:|---:|---:|---:|
| 2 模型融合：R34+R50，thr=0.25, area=500 | 0.4365 | 12.18% | 0.7798 | 202 / 1659 |
| 3 模型融合：+ DeepLabV3+ R34，thr=0.30, area=500 | 0.4301 | 9.89% | 0.7961 | 164 / 1659 |
| 4 模型融合：+ SMP U-Net R34，thr=0.30, area=500 | 0.4255 | 7.78% | 0.8115 | 129 / 1659 |

4 模型主可视化配置进一步降低阴性误报，并提升 mean Dice all，但 positive-only Dice 与阳性图像召回继续下降。该配置更适合作为低误报候选，而不是单纯追求正样本 Dice 的候选。

## 4. 后处理搜索观察

后处理搜索结果来自：

```text
outputs/experiments/ensemble_unetpp_r34_r50_deeplab_unet_v1/logs/ensemble_postprocess_sweep_val.csv
```

按 `mean_dice_all` 排序，最高配置为：

| Threshold | Min component area | Min pred area | Positive Dice | Negative FP rate | Positive recall | Mean Dice all |
|---:|---:|---:|---:|---:|---:|---:|
| 0.35 | 800 | 0 | 0.3817 | 3.98% | 60.08% | 0.8312 |

该配置整体 Dice 最高，但正样本 Dice 和召回仍偏保守，不宜作为病灶分割主配置。

在 `negative_fp_rate <= 13%` 的约束下，正样本 Dice 较高的候选配置为：

| Threshold | Min component area | Min pred area | Positive Dice | Negative FP rate | Positive recall | Mean Dice all |
|---:|---:|---:|---:|---:|---:|---:|
| 0.25 | 300 | 0 | 0.4759 | 12.12% | 80.04% | 0.7890 |

该候选比当前可视化主配置更偏召回：positive-only Dice 从 0.4255 提升到 0.4759，阳性图像召回从 69.12% 提升到 80.04%，但阴性误报率从 7.78% 升至 12.12%。如果后续目标是延续 2 模型融合的误报水平并提高正样本 Dice，应优先复查该配置。

## 5. 可视化与错误分析

可视化结果保存于：

```text
outputs/experiments/ensemble_unetpp_r34_r50_deeplab_unet_v1/visualizations/val_ensemble4_thr0p3_mincomp500_minpred0
```

目录中按以下类别保存代表性样本：

```text
high_dice_positive
low_dice_positive
false_positive_negative
false_negative_positive
true_negative
```

从统计结果看，4 模型可视化配置将阴性误报压到 129 / 1659，但仍有 147 个阳性样本漏检。说明该融合方案在当前阈值和面积过滤下更偏保守，后续需要根据应用目标决定是优先低误报还是提高阳性召回。

## 6. 结论

4 模型融合在验证集上进一步增强了误报控制能力。当前可视化配置相比 3 模型融合主配置，negative FP rate 从 9.89% 降至 7.78%，mean Dice all 从 0.7961 提升至 0.8115；代价是 positive-only Dice 从 0.4301 降至 0.4255，positive image recall 从 70.38% 降至 69.12%。

当前建议：

1. 将 `threshold=0.30, min_component_area=500` 作为低误报主可视化配置归档。
2. 将 `threshold=0.25, min_component_area=300` 作为更均衡的候选配置重点复查。
3. 当前 4 模型融合不应简单按最高 `mean_dice_all` 选型，因为最高 overall Dice 配置对阳性病灶分割偏保守。
4. 若进入 test 集评估，应先在验证集上固定一个配置，不再根据 test 结果调参。

## 图表解读

后处理 trade-off 图以阴性误报率为横轴、positive-only Dice 为纵轴，适合判断候选配置是否只是用误报换召回。

![ensemble postprocess sweep val tradeoff clear](report_assets/ensemble_postprocess_sweep_val_tradeoff_clear.png)

定性示例 contact sheet 汇总了典型高 Dice、低 Dice、误报和漏检样本，用于复查错误模式。

![val ensemble4 thr0p3 mincomp500 minpred0 contact sheet](report_assets/val_ensemble4_thr0p3_mincomp500_minpred0_contact_sheet.jpg)
