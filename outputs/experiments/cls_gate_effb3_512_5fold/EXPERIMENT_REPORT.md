# SIIM-ACR 分类门控实验说明

## 实验目的

本实验训练一个独立的图像级二分类模型，用于判断胸片是否存在气胸病灶。该模型不直接输出分割 mask，而是作为分割结果的后处理门控：当测试图像的分类概率低于门控阈值时，将对应提交行的 RLE 置为 `-1`，以减少明显阴性图像上的假阳性分割。

本实验采用 5 折交叉验证。每一折模型在 4/5 训练集上训练，在剩余 1/5 验证集上输出 OOF 分类概率；最终将 5 折模型在官方 test 图像上的概率取平均，作为提交阶段的分类门控概率。

## 实验目录

- 实验目录：`outputs/experiments/cls_gate_effb3_512_5fold`
- 每折输出：`fold_0` 至 `fold_4`
- 每折配置：`fold_*/config.json`
- 每折训练日志：`fold_*/metrics.csv`
- 每折最佳验证预测：`fold_*/val_predictions_best.csv`
- 合并 OOF 预测：`oof_cls_predictions.csv`
- OOF 阈值扫描：`oof_gate_threshold_sweep.csv`
- 5 折 test 平均概率：`test_cls_predictions_5fold_avg.csv`

## 数据与划分

数据集为 `SIIM-ACR Pneumothorax Segmentation` 的预处理结果：

- 数据根目录：`data/processed/siim_acr_pneumothorax`
- 图像目录：`data/processed/siim_acr_pneumothorax/train`
- 标签来源：`data/processed/siim_acr_pneumothorax/mask_summary.csv`
- 5 折划分目录：`data/splits/siim_acr_pneumothorax_5fold`

OOF 汇总样本数为 10675，其中阳性 2379，阴性 8296，阳性比例为 22.29%。各折验证集规模基本一致：

| Fold | 验证样本数 | 阳性数 | 阳性比例 |
|---|---:|---:|---:|
| fold_0 | 2136 | 476 | 22.28% |
| fold_1 | 2135 | 476 | 22.30% |
| fold_2 | 2135 | 476 | 22.30% |
| fold_3 | 2135 | 476 | 22.30% |
| fold_4 | 2134 | 475 | 22.26% |

## 模型与训练配置

每折使用相同训练配置，仅训练/验证 split 文件不同。关键配置如下：

| 项目 | 值 |
|---|---|
| 模型 | `timm-efficientnet-b3` encoder classifier |
| 预训练权重 | `imagenet` |
| 输入尺寸 | 512 |
| 输入通道 | 3 |
| Epochs | 30 |
| Batch size | 64 |
| Optimizer LR | 0.0002 |
| Weight decay | 0.0003 |
| Dropout | 0.2 |
| AMP | 启用 |
| Balanced sampling | 启用 |
| Positive sample ratio | 0.5 |
| BCE pos_weight | 未设置 |
| 数据增强 | 启用 |
| Early stopping patience | 8 |
| 训练设备 | `cuda` |
| 记录阈值 | 0.5 |
| 随机种子 | 42 |

分类门控显式处理了类别不均衡：训练阶段启用了 balanced sampling，并将采样中的阳性比例设为 0.5；未额外设置 BCE `pos_weight`。

## 每折验证结果

下表记录每折按最高 `val_auc` 选出的最佳 epoch。表中 Accuracy、Recall、Specificity、Negative FP rate 均来自该 epoch 的验证集日志，计算阈值为训练配置中的 `metric_threshold=0.5`。

| Fold | 实际训练 epoch 数 | 最佳 epoch | Val AUC | Val AP | Val Acc | Val Recall | Val Specificity | Negative FP rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| fold_0 | 29 | 21 | 0.9328 | 0.8002 | 0.8886 | 0.7479 | 0.9289 | 0.0711 |
| fold_1 | 11 | 3 | 0.9367 | 0.7923 | 0.8684 | 0.9013 | 0.8590 | 0.1410 |
| fold_2 | 14 | 6 | 0.9336 | 0.7885 | 0.8824 | 0.8046 | 0.9048 | 0.0952 |
| fold_3 | 13 | 5 | 0.9392 | 0.7882 | 0.8656 | 0.8929 | 0.8577 | 0.1423 |
| fold_4 | 26 | 18 | 0.9392 | 0.8014 | 0.8861 | 0.8295 | 0.9024 | 0.0976 |
| 平均 | - | - | 0.9363 | 0.7941 | 0.8782 | 0.8352 | 0.8905 | 0.1095 |

5 折验证 AUC 较稳定，范围为 0.9328 至 0.9392。不同折在 Recall 和 Specificity 上存在更明显差异，说明固定阈值 0.5 并不一定是最佳门控阈值，后续提交应优先参考 OOF 阈值扫描。

## OOF 阈值扫描

`oof_gate_threshold_sweep.csv` 基于 5 折 OOF 预测统一扫描分类门控阈值。OOF 整体排序指标如下：

- OOF AUC：0.9263
- OOF AP：0.7849

按 F2 最大化选择阈值时，最佳门控阈值为 0.19：

| Threshold | Acc | Precision | Recall | Specificity | Negative FP rate | Predicted positive rate | F2 | TP | FP | TN | FN |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.19 | 0.8556 | 0.6247 | 0.8810 | 0.8482 | 0.1518 | 0.3143 | 0.8142 | 2096 | 1259 | 7037 | 283 |

如果门控目标更偏向“尽量不误删阳性分割”，可选择满足高 Recall 的阈值；如果目标更偏向“减少阴性图像假阳性”，可选择更高阈值。OOF 上几个可参考的 operating points：

| 选择原则 | Threshold | Acc | Precision | Recall | Specificity | Negative FP rate | Predicted positive rate | F2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Recall >= 0.90 下 specificity 最高 | 0.09 | 0.8331 | 0.5808 | 0.9016 | 0.8134 | 0.1866 | 0.3459 | 0.8119 |
| F2 最高，偏召回 | 0.19 | 0.8556 | 0.6247 | 0.8810 | 0.8482 | 0.1518 | 0.3143 | 0.8142 |
| Recall >= 0.85 下 specificity 最高 | 0.38 | 0.8720 | 0.6670 | 0.8504 | 0.8783 | 0.1217 | 0.2841 | 0.8060 |
| Recall >= 0.80 下 specificity 最高 | 0.70 | 0.8858 | 0.7187 | 0.8012 | 0.9101 | 0.0899 | 0.2484 | 0.7832 |
| Accuracy 最高 | 0.84 | 0.8902 | 0.7518 | 0.7575 | 0.9283 | 0.0717 | 0.2245 | 0.7563 |

建议默认从阈值 `0.19` 开始用于分割提交门控，因为该点在 OOF 上取得最高 F2，保留较高召回，同时能过滤一部分阴性假阳性。若后续分割模型本身假阳性较多，可以对 `0.38`、`0.70` 等更严格阈值做提交对比；若担心误删阳性病例，应优先使用 `0.09` 或 `0.19`。

## Test 概率文件

`test_cls_predictions_5fold_avg.csv` 是 5 折分类模型在官方 test 图像上的平均概率，包含 3205 张图像。该文件没有真实标签，因此不能计算 test AUC、AP、Recall 或 Specificity。

5 折平均 test 概率的摘要：

| 指标 | 值 |
|---|---:|
| 样本数 | 3205 |
| 平均概率 | 0.1894 |
| 中位数概率 | 0.0049 |
| 最小概率 | 0.0000 |
| 最大概率 | 1.0000 |

不同阈值下，test 集会被分类门控保留为阳性的图像数量如下：

| Threshold | 保留阳性数 | 保留比例 |
|---:|---:|---:|
| 0.10 | 972 | 30.33% |
| 0.20 | 817 | 25.49% |
| 0.30 | 722 | 22.53% |
| 0.40 | 662 | 20.66% |
| 0.50 | 582 | 18.16% |
| 0.60 | 507 | 15.82% |
| 0.70 | 430 | 13.42% |
| 0.80 | 358 | 11.17% |
| 0.90 | 280 | 8.74% |

## 与分割提交的结合方式

分类门控应作用在已有分割提交文件之后，而不是替代分割模型。处理逻辑为：

1. 先由分割模型生成 Kaggle submission CSV。
2. 读取 `test_cls_predictions_5fold_avg.csv` 中每张 test 图像的平均分类概率。
3. 若 `cls_prob < gate_threshold`，则将该图像的 `EncodedPixels` 置为 `-1`。
4. 若 `cls_prob >= gate_threshold`，则保留原分割模型输出。

相关脚本：

```powershell
python scripts/evaluate/apply_cls_gate_to_submission.py `
  --input-submission <segmentation_submission.csv> `
  --cls-probs-csv outputs/experiments/cls_gate_effb3_512_5fold/test_cls_predictions_5fold_avg.csv `
  --output-submission <gated_submission.csv> `
  --gate-threshold 0.19
```

也可以使用 `--keep-top-k` 按 test 分类概率排序保留固定数量的图像，但该策略不直接对应 OOF 阈值，需要单独记录和对比。

## 注意事项

- 本报告中的 OOF 指标只评估图像级分类门控，不是 mask Dice，也不是最终分割性能。
- 官方 test 集没有标签，`test_cls_predictions_5fold_avg.csv` 只能作为提交后处理输入，不能证明泛化性能。
- 分类门控会降低阴性图像上的假阳性分割，但也可能把真实阳性病例的分割结果误删；阈值越高，误删风险越大。
- 后续比较 gated submission 与原始 segmentation submission 时，应分别记录分割模型配置、门控阈值、最终提交文件名和线上/线下指标。
- 若后续引入新的分割模型，建议复用同一份 OOF 分类门控结果做初始对比，避免同时改变分割模型和门控策略导致归因不清。

## 后续建议

1. 使用 `0.09`、`0.19`、`0.38`、`0.70` 四个代表性阈值分别生成 gated submission，与未门控提交做对比。
2. 在 OOF 层面把分类门控和分割 OOF mask 预测结合，直接评估门控前后的 Dice 变化，而不是只看分类 F2。
3. 记录每个 gated submission 对应的分割实验目录、分类门控目录、阈值和输出文件，避免提交结果无法追溯。
