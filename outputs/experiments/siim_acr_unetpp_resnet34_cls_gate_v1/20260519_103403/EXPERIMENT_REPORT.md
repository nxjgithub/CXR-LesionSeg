# SIIM-ACR Unet++-ResNet34 + Classification Gate 实验报告

## 1. 实验目的

本实验在 SIIM-ACR Pneumothorax Segmentation 数据集上评估 `Unet++-ResNet34` 分割模型加入图像级分类辅助任务后的表现。实验目标是观察分类分支能否作为 image-level gate 抑制阴性胸片上的误检，并在不明显牺牲阳性召回的前提下提升验证集分割 Dice。

本报告仅基于训练集和验证集日志进行分析。当前实验未包含独立 test 集评估，因此以下结果不能表述为最终泛化性能。

## 2. 实验配置

实验输出目录：

```text
outputs/experiments/siim_acr_unetpp_resnet34_cls_gate_v1/20260519_103403
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
| 计划训练轮数 | 50 |
| 日志记录轮数 | 39 |
| Batch size | 32 |
| 初始学习率 | 0.0003 |
| Weight decay | 0.0003 |
| 分割 Loss | BCE + Dice |
| BCE 权重 | 0.3 |
| Dice 权重 | 0.7 |
| BCE `pos_weight` | 2.0 |
| 分类 loss 权重 `cls_weight` | 0.3 |
| 分类 BCE `cls_pos_weight` | 2.0 |
| 分类 dropout | 0.2 |
| 分割阈值 `threshold` | 0.08 |
| 分类 gate 阈值 `cls_threshold` | 0.50 |
| Balanced sampling | 是 |
| 正样本采样比例 | 0.35 |
| 随机种子 | 42 |
| 设备 | CUDA |
| AMP | 开启 |

数据划分文件来自 `data/splits/siim_acr_pneumothorax/`。其中训练集和验证集的图像级标签分布如下：

| Split | 总数 | 阳性 | 阴性 |
|---|---:|---:|---:|
| train | 8540 | 1903 | 6637 |
| val | 2135 | 476 | 1659 |

说明：`config.json` 中计划训练 50 个 epoch，但 `logs/metrics.csv` 实际记录到第 39 个 epoch。本报告以已有 39 个 epoch 日志为准。

## 3. 训练产物

本实验目录包含以下核心产物：

```text
config.json
logs/metrics.csv
checkpoints/best_model.pth
checkpoints/last_model.pth
```

从 checkpoint 元信息看，`best_model.pth` 保存于第 20 轮，`best_score=0.499331`，对应 `metrics.csv` 中的最佳 `val_gated_dice`。因此本实验的默认最佳模型是 gated Dice 最佳模型，而不是 raw Dice 最佳模型。

## 4. 训练结果

训练日志来自 `logs/metrics.csv`。模型在前 10 到 20 轮内快速收敛；之后训练集 Dice 持续升高，但验证集 Dice 没有继续改善，训练后期出现明显过拟合。

关键 epoch 指标如下：

| Epoch | Train raw Dice | Train gated Dice | Val raw Dice | Val gated Dice | Val gated Precision | Val gated Recall | Val cls Acc | Val cls Recall |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.0128 | 0.0249 | 0.1017 | 0.1542 | 0.0873 | 0.6601 | 0.8042 | 0.8529 |
| 5 | 0.5463 | 0.6151 | 0.4427 | 0.4443 | 0.5786 | 0.3606 | 0.8581 | 0.8130 |
| 10 | 0.6707 | 0.7162 | 0.4828 | 0.4977 | 0.4924 | 0.5032 | 0.8576 | 0.8172 |
| 14 | 0.7538 | 0.7769 | 0.5097 | 0.4941 | 0.6067 | 0.4167 | 0.8763 | 0.6618 |
| 20 | 0.8329 | 0.8458 | 0.4975 | 0.4993 | 0.5799 | 0.4384 | 0.8796 | 0.6996 |
| 30 | 0.9228 | 0.9250 | 0.5000 | 0.4779 | 0.6229 | 0.3876 | 0.8763 | 0.6282 |
| 39 | 0.9584 | 0.9600 | 0.4773 | 0.4644 | 0.6349 | 0.3661 | 0.8754 | 0.6450 |

## 5. 最佳 Raw 分割结果

不使用分类 gate 时，验证集最高 `val_raw_dice` 出现在第 14 轮：

| 指标 | 数值 |
|---|---:|
| Best raw epoch | 14 |
| Train loss | 0.5349 |
| Train raw Dice | 0.7538 |
| Train gated Dice | 0.7769 |
| Train cls Acc | 0.9972 |
| Train cls Recall | 0.9970 |
| Train cls Specificity | 0.9973 |
| Val loss | 0.9255 |
| Val raw Dice | 0.5097 |
| Val gated Dice | 0.4941 |
| Val gated IoU | 0.3281 |
| Val gated Precision | 0.6067 |
| Val gated Recall | 0.4167 |
| Val gated Specificity | 0.9992 |
| Val cls Acc | 0.8763 |
| Val cls Precision | 0.7536 |
| Val cls Recall | 0.6618 |
| Val cls Specificity | 0.9379 |

第 14 轮中，分类 gate 将验证集 Dice 从 raw 0.5097 降至 gated 0.4941。该结果说明，当分割分支达到本实验最佳 raw Dice 时，固定 `cls_threshold=0.50` 的 gate 更偏向抑制预测区域，带来了额外漏检风险。

## 6. 最佳 Gated 分割结果

使用分类 gate 后，验证集最高 `val_gated_dice` 出现在第 20 轮，也是 `best_model.pth` 对应的 checkpoint：

| 指标 | 数值 |
|---|---:|
| Best gated epoch | 20 |
| Train loss | 0.5090 |
| Train raw Dice | 0.8329 |
| Train gated Dice | 0.8458 |
| Train cls Acc | 0.9996 |
| Train cls Recall | 0.9997 |
| Train cls Specificity | 0.9996 |
| Val loss | 0.9300 |
| Val raw Dice | 0.4975 |
| Val gated Dice | 0.4993 |
| Val gated IoU | 0.3327 |
| Val gated Precision | 0.5799 |
| Val gated Recall | 0.4384 |
| Val gated Specificity | 0.9990 |
| Val cls Acc | 0.8796 |
| Val cls Precision | 0.7450 |
| Val cls Recall | 0.6996 |
| Val cls Specificity | 0.9313 |

第 20 轮中，分类 gate 将 Dice 从 raw 0.4975 小幅提升到 gated 0.4993，提升幅度只有 0.0018。该 checkpoint 的 gated Dice 低于本实验第 14 轮 raw Dice 0.5097，也低于无分类 gate 的 `siim_acr_unetpp_resnet34_v1` 训练期最佳验证 Dice 0.5146。

## 7. 最后一轮结果

第 39 轮为当前日志中的最后一轮：

| 指标 | 数值 |
|---|---:|
| Epoch | 39 |
| Train loss | 0.4661 |
| Train raw Dice | 0.9584 |
| Train gated Dice | 0.9600 |
| Train cls Acc | 1.0000 |
| Train cls Recall | 1.0000 |
| Train cls Specificity | 1.0000 |
| Val loss | 1.0398 |
| Val raw Dice | 0.4773 |
| Val gated Dice | 0.4644 |
| Val gated IoU | 0.3024 |
| Val gated Precision | 0.6349 |
| Val gated Recall | 0.3661 |
| Val gated Specificity | 0.9994 |
| Val cls Acc | 0.8754 |
| Val cls Precision | 0.7599 |
| Val cls Recall | 0.6450 |
| Val cls Specificity | 0.9415 |

从第 20 轮到第 39 轮，训练集 gated Dice 从 0.8458 升至 0.9600，但验证集 gated Dice 从 0.4993 降至 0.4644，验证集 gated Recall 从 0.4384 降至 0.3661。该现象说明继续训练主要提升了训练集拟合程度，并未带来验证集收益。

## 8. Classification Gate 分析

分类分支在训练集上很快达到接近满分的分类指标，例如第 20 轮训练集 `train_cls_acc=0.9996`，第 39 轮达到 `1.0000`。但验证集分类指标没有同步提升，验证集 `val_cls_acc` 最高约为 0.8820，说明分类辅助任务本身也存在泛化瓶颈。

分类 gate 对分割结果的影响不稳定：

| 对比点 | Raw Dice | Gated Dice | Gate 影响 |
|---|---:|---:|---:|
| Epoch 8 | 0.4210 | 0.4976 | +0.0766 |
| Epoch 10 | 0.4828 | 0.4977 | +0.0150 |
| Epoch 14 | 0.5097 | 0.4941 | -0.0156 |
| Epoch 20 | 0.4975 | 0.4993 | +0.0018 |
| Epoch 39 | 0.4773 | 0.4644 | -0.0129 |

早期阶段 gate 可以明显改善 Dice，主要原因可能是分类分支抑制了一部分阴性样本误检。但当分割分支本身变强后，固定阈值 gate 的收益迅速缩小，并且在 raw Dice 最佳点和末轮都带来了负收益。后期 gated Precision 提高、gated Recall 降低，说明 gate 更偏保守，容易牺牲阳性召回。

## 9. 与已有实验对比

与同一数据集上的已有实验相比，本实验没有超过普通 Unet++-ResNet34：

| 实验 | 关键设置 | 最佳 epoch | 最佳验证 Dice | 备注 |
|---|---|---:|---:|---|
| `siim_acr_unetpp_resnet34_v1` | Unet++-ResNet34，默认训练期评估 | 24 | 0.5146 | 无分类 gate |
| `siim_acr_unetpp_resnet34_v1` | best model threshold sweep | - | 0.5198 | 验证集最佳阈值 0.04 |
| `siim_acr_unetpp_resnet34_cls_gate_v1` raw | 加分类辅助任务，不使用 gate | 14 | 0.5097 | 本实验最高 raw Dice |
| `siim_acr_unetpp_resnet34_cls_gate_v1` gated | 加分类辅助任务，`cls_threshold=0.50` gate | 20 | 0.4993 | `best_model.pth` |

本实验的分类 gate 最佳 Dice 为 0.4993，低于普通 Unet++-ResNet34 的训练期最佳 Dice 0.5146。即使不使用 gate，仅看 raw 分割输出，本实验最高 raw Dice 0.5097 也仍低于普通 Unet++ 实验。这说明当前分类辅助损失和固定 gate 策略没有带来稳定收益。

需要注意，本实验分割阈值为 `0.08`，普通 Unet++ 训练期报告中的默认阈值为 `0.50`，两者不是完全相同的阈值设置。更严格的横向比较应对本实验也补充 threshold sweep、classification threshold sweep 和后处理搜索。

## 10. 结论

本实验完成了 Unet++-ResNet34 加图像级分类 gate 的训练记录分析。当前最佳 gated checkpoint 为第 20 轮，验证集 `val_gated_dice=0.4993`；最佳 raw 分割结果出现在第 14 轮，`val_raw_dice=0.5097`。分类 gate 在早期 epoch 可以提升 Dice，但在最佳 raw epoch 和训练后期都会降低 Dice，整体收益不稳定。

分类分支在训练集上快速接近满分，但验证集分类指标停留在约 0.88 accuracy，且固定 `cls_threshold=0.50` 容易在 precision 和 recall 之间产生偏移。当前结果表明，单纯加入图像级分类头并用固定阈值清空分割 mask，不能稳定改善 SIIM-ACR 气胸分割表现。

本实验不建议作为优于普通 Unet++-ResNet34 的主结果。它更适合作为分类辅助 gate 的探索性对照实验，用于说明该策略在当前设置下存在不稳定收益和阳性漏检风险。

## 11. 后续建议

1. 对本实验补充 `threshold` sweep，分别评估 raw segmentation 输出在不同分割阈值下的 Dice、Precision 和 Recall。
2. 对 `cls_threshold` 做单独扫描，观察 classification gate 对 Dice、阳性漏检率和阴性假阳性率的影响。
3. 增加逐样本可视化，分别查看 gate 改善、gate 变差、假阳性阴性样本和漏检阳性样本。
4. 若继续探索分类辅助任务，建议降低 `cls_weight` 或采用 warm-up 后再加入分类 loss，避免分类任务过早主导共享表征。
5. 不应只按 gated Dice 选模型；后续可同时保存最佳 raw checkpoint 和最佳 gated checkpoint，方便比较 gate 本身的净收益。
6. 在没有完成 threshold sweep、gate threshold sweep 和验证集错误分析前，不建议进入独立 test 集评估。
