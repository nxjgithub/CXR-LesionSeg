# SIIM-ACR 气胸分割实验总说明

## 1. 总体目标

当前阶段围绕 `SIIM-ACR Pneumothorax Segmentation` 数据集开展胸片气胸二值分割实验。实验主线不是单纯追求验证集全局 Dice，而是同时关注：

1. 阳性样本上的病灶分割能力，即 positive-only Dice。
2. 阴性样本上的误报控制，即 negative false positive rate。
3. 全体验证集 Dice，即 mean Dice all 或全局像素级 Dice。
4. 训练配置、阈值选择、后处理参数和实验结论是否可复现。

目前所有结论均基于验证集分析，尚未进行独立 test 集最终评估。因此，本文中的结果只能作为模型选择和后续实验设计依据，不能表述为最终泛化性能。

## 2. 当前实验目录

已有实验记录位于 `outputs/experiments/`：

| 实验 | 主要模型/策略 | 角色 |
|---|---|---|
| `siim_acr_unet_baseline` | 原始 U-Net baseline | 早期基础对照 |
| `siim_acr_unet_baseline_v2_bs32` | U-Net + balanced sampling + 更强正类权重 | 早期类别不均衡对照 |
| `siim_acr_unet_baseline_v3` | U-Net + 更稳定的采样与损失配置 | 早期 U-Net 最佳对照 |
| `siim_acr_unet_cls_v2` | U-Net + image-level classification head | 分类辅助 gate 早期探索 |
| `siim_acr_unetpp_resnet34_v1` | Unet++-ResNet34 | 当前主线 baseline |
| `siim_acr_deeplabv3plus_resnet34_v1` | DeepLabV3+-ResNet34 | SMP 横向架构对照 |
| `siim_acr_smp_unet_resnet34_v1` | SMP U-Net-ResNet34 | SMP 横向架构对照 |
| `siim_acr_unetpp_resnet34_cls_gate_v1` | Unet++-ResNet34 + classification gate | 分类 gate 探索 |
| `siim_acr_unetpp_resnet34_recall_v2` | Unet++-ResNet34 recall-oriented | 提高正样本召回探索 |
| `siim_acr_unetpp_resnet50_recall_v1` | Unet++-ResNet50 recall-oriented | 更强 encoder + recall 探索 |
| `ensemble_unetpp_r34_r50_recall_v1` | Unet++ ResNet34 + ResNet50 ensemble | 当前验证集最佳候选 |
| `ensemble_unetpp_r34_r50_deeplab_r34_v1` | 3 模型 ensemble：R34 + R50 + DeepLabV3+ R34 | 低误报融合候选 |
| `ensemble_unetpp_r34_r50_deeplab_unet_v1` | 4 模型 ensemble：R34 + R50 + DeepLabV3+ R34 + SMP U-Net R34 | 更低误报融合候选 |

## 3. 实验演进脉络

### 3.1 U-Net baseline 阶段

最早的 U-Net baseline 用于验证数据预处理、数据加载、训练循环和基础指标计算是否可用。后续 `v2_bs32` 和 `v3` 逐步加入 balanced sampling、BCE `pos_weight` 和 BCE/Dice 权重调整，以缓解 SIIM-ACR 中大量阴性图像和前景像素稀少带来的类别不均衡。

主要结论：

| 实验 | 最佳验证 Dice | 结论 |
|---|---:|---|
| `siim_acr_unet_baseline` | 0.3632 | 基础流程可用，但分割能力较弱 |
| `siim_acr_unet_baseline_v2_bs32` | 0.4250 | 类别不均衡处理带来明显提升 |
| `siim_acr_unet_baseline_v3` | 0.4289 | 训练更稳定，但普通 U-Net 瓶颈明显 |

U-Net baseline 阶段的价值是建立可复现训练流程和初始对照。后续主线转向带 ImageNet encoder 的 SMP 模型。

### 3.2 分类辅助 gate 早期探索

`siim_acr_unet_cls_v2` 在普通 U-Net 上加入图像级分类头，尝试用 classification gate 清空阴性样本预测。该实验的最佳 gated Dice 为 0.4120，低于 `siim_acr_unet_baseline_v3` 的 0.4289。

结论：简单分类辅助头可以在个别 epoch 提高 precision，但会牺牲 recall，整体没有稳定超过 U-Net v3 baseline。

### 3.3 SMP ResNet34 架构横向对比

SMP 阶段引入 ImageNet 预训练 encoder，并比较 Unet++、DeepLabV3+ 和普通 U-Net decoder。

| 实验 | 训练期最佳 Val Dice | Threshold sweep 最佳 Dice | 后处理观察 |
|---|---:|---:|---|
| `siim_acr_unetpp_resnet34_v1` | 0.5146 | 0.5198 | 正样本 Dice 与阴性误报较均衡 |
| `siim_acr_deeplabv3plus_resnet34_v1` | 0.5218 | 0.5241 | 全局 Dice 略高，但后处理平衡不优于 Unet++ |
| `siim_acr_smp_unet_resnet34_v1` | 0.5195 | 0.5220 | 可提高正样本 Dice，但误报控制与正样本 Dice 难兼顾 |

虽然 DeepLabV3+ 和 SMP U-Net 的全局像素级 Dice 接近或略高，但后处理后的 positive-only Dice、negative FP rate 和 mean Dice all 之间没有形成明显综合优势。

当前更推荐将 `siim_acr_unetpp_resnet34_v1` 作为主线 baseline，原因是它在阳性分割和阴性误报之间更均衡。

### 3.4 Unet++-ResNet34 主线 baseline

`siim_acr_unetpp_resnet34_v1` 是当前最重要的基础模型：

| 指标 | 数值 |
|---|---:|
| 训练期最佳 Val Dice | 0.5146 |
| Threshold sweep 最佳 Dice | 0.5198 |
| 后处理 baseline threshold | 0.08 |
| 后处理 baseline min component area | 150 |
| 后处理 baseline positive Dice | 0.3980 |
| 后处理 baseline negative FP rate | 13.92% |
| 后处理 baseline mean Dice all | 0.7576 |

该模型已经明显超过普通 U-Net baseline，且在后处理后能够把阴性样本误报控制在约 14% 的水平，同时保留约 0.40 的 positive-only Dice。

当前主线结论：`siim_acr_unetpp_resnet34_v1` 是后续实验对比的主要 baseline。

### 3.5 Unet++ 分类 gate 探索

`siim_acr_unetpp_resnet34_cls_gate_v1` 在 Unet++-ResNet34 上加入图像级分类 gate，试图进一步降低阴性误报。

关键结果：

| 指标 | 数值 |
|---|---:|
| 最佳 raw Dice | 0.5097 |
| 最佳 gated Dice | 0.4993 |
| best checkpoint epoch | 20 |

结论：classification gate 收益不稳定。早期 epoch 可以提升 Dice，但在最佳 raw epoch 和训练后期会降低 Dice。该实验没有超过普通 Unet++-ResNet34，不作为主模型。

### 3.6 Recall-oriented 实验

Recall-oriented 实验提高正样本采样比例和 BCE 正类权重，目标是提升阳性样本分割能力。主要实验包括 ResNet34 recall v2 和 ResNet50 recall v1。

| 实验与设置 | Positive Dice | Negative FP rate | Mean Dice all | 结论 |
|---|---:|---:|---:|---|
| ResNet34 recall v2 best positive：threshold=0.02, area=0 | 0.3946 | 31.65% | 0.4393 | 正样本未超过旧模型，误报过高 |
| ResNet34 recall v2 balanced：threshold=0.02, area=200 | 0.3856 | 14.23% | 0.7525 | 误报接近旧模型，但正样本 Dice 下降 |
| ResNet34 recall v2 low-FP：threshold=0.05, area=200 | 0.3797 | 11.63% | 0.7713 | 可作为低误报对照，但分割能力下降 |
| ResNet50 recall v1 best positive：threshold=0.02, area=0 | 0.4213 | 29.05% | 0.6092 | 正样本 Dice 提升，但误报过高 |
| ResNet50 recall v1 balanced：threshold=0.02, area=500 | 0.3822 | 13.74% | 0.7555 | 误报可控，但正样本 Dice 低于 ResNet34 baseline |

结论：

1. ResNet34 recall v2 没有提升 positive-only Dice，不采用。
2. ResNet50 recall v1 显示更强 encoder 有提升正样本分割的潜力，但该收益依赖低阈值和弱后处理，阴性误报过高。
3. 一旦通过较强后处理控制误报，ResNet50 recall v1 的 positive-only Dice 低于旧 Unet++-ResNet34 baseline。

因此，recall-oriented 系列目前不作为主模型。

### 3.7 多模型融合实验

在单模型实验基础上，进一步融合 `U-Net++ ResNet34 recall v2` 与 `U-Net++ ResNet50 recall v1`，并使用水平翻转 TTA 与连通域面积过滤。

最终主配置为：

| 参数 | 数值 |
|---|---:|
| Threshold | 0.25 |
| Min component area | 500 |
| Min pred area | 0 |
| TTA | hflip |
| Num models | 2 |

与单模型 `U-Net++ ResNet34` baseline 相比：

| 模型与配置 | Positive Dice | Negative FP rate | Mean Dice all | False positive negative |
|---|---:|---:|---:|---:|
| U-Net++ ResNet34 baseline | 0.3980 | 13.92% | 0.7576 | 231 / 1659 |
| Ensemble R34+R50 main | 0.4365 | 12.18% | 0.7798 | 202 / 1659 |

随后继续评估了 3 模型与 4 模型融合方案：

| 融合方案与配置 | Positive Dice | Negative FP rate | Positive recall | Mean Dice all | False positive negative |
|---|---:|---:|---:|---:|---:|
| 2 模型 R34+R50：thr=0.25, area=500 | 0.4365 | 12.18% | 72.90% | 0.7798 | 202 / 1659 |
| 3 模型 + DeepLabV3+ R34：thr=0.30, area=500 | 0.4301 | 9.89% | 70.38% | 0.7961 | 164 / 1659 |
| 4 模型 + SMP U-Net R34：thr=0.30, area=500 | 0.4255 | 7.78% | 69.12% | 0.8115 | 129 / 1659 |

3 模型和 4 模型主可视化配置持续降低阴性误报，并提高 mean Dice all，但会牺牲一部分 positive-only Dice 和阳性图像召回。因此，当前结论不是简单替换 2 模型融合，而是形成两类候选：

1. 若更重视正样本 Dice 与误报平衡，2 模型 R34+R50 仍是主候选。
2. 若更重视低误报，3 模型和 4 模型主可视化配置更有价值。
3. 4 模型还有一个更均衡的验证集候选：`threshold=0.25, min_component_area=300`，positive Dice 为 0.4759，negative FP rate 为 12.12%，但该配置尚未生成同等可视化复查。

## 4. 当前模型选择结论

### 4.1 推荐主线 baseline

当前推荐主线模型仍保留为单模型 baseline：

```text
outputs/experiments/siim_acr_unetpp_resnet34_v1/20260518_180917
```

推荐后处理配置：

| 参数 | 数值 |
|---|---:|
| Threshold | 0.08 |
| Min component area | 150 |
| Keep largest component | False |

推荐理由：

1. 相比普通 U-Net baseline，Unet++-ResNet34 有明显提升。
2. 相比 DeepLabV3+ 和 SMP U-Net，Unet++-ResNet34 在 positive-only Dice 与 negative FP rate 之间更均衡。
3. 相比 classification gate 和 recall-oriented 实验，Unet++-ResNet34 baseline 更稳定，不依赖高误报换取正样本 Dice。

当前推荐验证集最佳候选方案：

```text
outputs/experiments/ensemble_unetpp_r34_r50_recall_v1
```

推荐配置：

| 参数 | 数值 |
|---|---:|
| Threshold | 0.25 |
| Min component area | 500 |
| Min pred area | 0 |
| TTA | hflip |

推荐理由：相比单模型 baseline，ensemble 同时提升 positive-only Dice、降低阴性误报率并提高 mean Dice all。

当前低误报候选方案：

```text
outputs/experiments/ensemble_unetpp_r34_r50_deeplab_unet_v1
```

推荐低误报配置：

| 参数 | 数值 |
|---|---:|
| Threshold | 0.30 |
| Min component area | 500 |
| Min pred area | 0 |
| TTA | hflip |

推荐理由：4 模型融合主可视化配置将 negative FP rate 降至 7.78%，false positive negative 降至 129 / 1659，适合作为低误报场景候选。但其 positive-only Dice 为 0.4255，低于 2 模型融合主配置的 0.4365。

### 4.2 暂不采用的实验

| 实验 | 不采用原因 |
|---|---|
| `siim_acr_unet_cls_v2` | 分类 gate 未稳定超过 U-Net v3 |
| `siim_acr_unetpp_resnet34_cls_gate_v1` | gated Dice 低于普通 Unet++，且 recall 受损 |
| `siim_acr_unetpp_resnet34_recall_v2` | positive-only Dice 未提升，低阈值误报过高 |
| `siim_acr_unetpp_resnet50_recall_v1` | best positive Dice 提升但 negative FP rate 过高；balanced setting 又低于 ResNet34 baseline |

### 4.3 保留为横向对照的实验

| 实验 | 保留价值 |
|---|---|
| `siim_acr_deeplabv3plus_resnet34_v1` | 说明 DeepLabV3+ 全局 Dice 接近，但后处理平衡未明显更优 |
| `siim_acr_smp_unet_resnet34_v1` | 说明普通 U-Net decoder 在 SMP 框架下可作架构对照 |
| `siim_acr_unetpp_resnet50_recall_v1` | 说明更强 encoder 有正样本潜力，但需要更好的误报控制机制 |
| `ensemble_unetpp_r34_r50_recall_v1` | 说明多模型融合可以改善单模型召回/误报折中 |
| `ensemble_unetpp_r34_r50_deeplab_r34_v1` | 说明加入 DeepLabV3+ 后可以降低误报，但主可视化配置略降 positive Dice |
| `ensemble_unetpp_r34_r50_deeplab_unet_v1` | 说明 4 模型融合可进一步降低误报，并提供一个高 positive Dice 的平衡候选 |

## 5. 当前主要问题

### 5.1 Positive-only Dice 仍偏低

当前主线后处理 baseline 的 positive-only Dice 约为 0.3980。即使 ResNet50 recall v1 在低阈值下可达到 0.4213，也伴随 29.05% 的阴性误报率。说明模型对困难阳性样本、小气胸区域、低对比度边界和胸膜附近细小区域仍学习不足。

### 5.2 阴性误报与阳性召回存在明显冲突

降低阈值通常能提高 positive-only Dice 或 Recall，但阴性样本误报显著增加。增大 `min_component_area` 能降低阴性误报，但会删除部分小病灶，导致 positive-only Dice 下降。

因此，当前简单阈值和连通域后处理只能调节误报/漏检权衡，不能根本突破模型能力。

### 5.3 训练后期普遍过拟合

多数实验在中前期达到最佳验证 Dice，后续训练集 Dice 继续升高，但验证集 Dice 和 Recall 下降。例如：

| 实验 | 最佳 epoch | 后续现象 |
|---|---:|---|
| Unet++-ResNet34 | 24 | 训练集 Dice 继续升高，验证 Dice 下降 |
| DeepLabV3+-ResNet34 | 25 | 后期 recall 下降 |
| SMP U-Net-ResNet34 | 25 | 后期验证 Dice 下降 |
| ResNet50 recall v1 | 19 | 后期预测更保守，验证 Recall 下降 |

后续训练策略应考虑 early stopping、多 checkpoint 选择和更强泛化约束。

## 6. 后续实验建议

1. 以 `siim_acr_unetpp_resnet34_v1` 作为单模型主线 baseline，以 `ensemble_unetpp_r34_r50_recall_v1` 作为正样本 Dice 与误报平衡的验证集主候选。
2. 固定报告指标：训练期最佳全局 Dice、threshold sweep 最佳 Dice、positive-only Dice、negative FP rate、mean Dice all。
3. 对低 Dice 阳性样本和高误报阴性样本做系统可视化复查，判断错误来源是小病灶、边界模糊、肺尖/胸膜线误判，还是标注质量问题。
4. 尝试 Focal Loss、Tversky Loss、Focal Tversky Loss 或 Dice/Tversky 混合损失，重点改善小病灶和类别极不均衡问题。
5. 加强胸片灰度、对比度、局部遮挡、尺度变化和轻微几何扰动相关的数据增强。
6. 若继续探索 ResNet50，应引入更强的误报抑制机制，而不是只依赖更低阈值。
7. 对 3 模型和 4 模型融合保留两类配置：低误报主可视化配置，以及 `negative_fp_rate <= 13%` 约束下的 balanced candidate。
8. 若进入独立 test 集评估，应先固定候选配置，避免将 test 集结果用于再次调参。

## 7. 当前阶段结论

当前 SIIM-ACR 气胸分割实验已经完成从普通 U-Net baseline 到 SMP 系列模型、classification gate 和 recall-oriented 配置的多轮探索。

综合验证集结果，`Unet++-ResNet34` 仍是当前最合适的单模型 baseline；`U-Net++ ResNet34 + ResNet50 ensemble` 是当前正样本 Dice 与误报平衡最清晰的候选方案。3 模型和 4 模型融合进一步证明加入横向架构可以降低阴性误报，其中 4 模型主可视化配置把 negative FP rate 降至 7.78%，但 positive-only Dice 和阳性图像召回略有下降。

后续应根据评估目标固定候选：若优先平衡正样本分割与误报，使用 2 模型融合主配置；若优先低误报，使用 4 模型融合主可视化配置；若希望继续提高 positive-only Dice，可复查 4 模型 `threshold=0.25, min_component_area=300` 的 balanced candidate。所有这些结论仍只来自验证集。
