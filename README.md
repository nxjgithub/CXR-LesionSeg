# CXR-LesionSeg

胸部 X 光多病灶识别与分割实验项目。

本项目面向胸片病灶分割与识别研究，目标是建立一个可扩展、可复现的实验框架，用于支持多个数据集、多个模型和多组训练策略的系统性比较。当前阶段主要围绕 `SIIM-ACR Pneumothorax Segmentation` 数据集开展气胸分割实验，后续会逐步扩展到更多胸片病灶任务。

## 项目目标

- 构建统一的数据预处理、数据加载、训练和评估流程。
- 支持不同胸片数据集之间的实验复用。
- 支持传统 CNN 分割模型、Transformer 分割模型和医学基础模型相关方法的横向比较。
- 保留完整实验配置、指标日志和实验报告，方便复盘和后续论文实验整理。

## 当前进展

当前已完成 SIIM-ACR 气胸分割任务的基础工程搭建，包括：

- SIIM-ACR 数据预处理脚本。
- train/val/test 数据划分。
- PyTorch 数据集封装。
- 基础 U-Net 模型。
- BCE + Dice 分割损失。
- Dice、IoU、precision、recall、specificity 等指标。
- U-Net baseline 训练脚本。
- 三组 U-Net baseline 实验记录和报告。
- 针对 v2 best checkpoint 的验证集阈值扫描和可视化错误分析。
- 一组 U-Net + classification head 辅助分类实验记录。
- 基于 `segmentation_models_pytorch` 的 Unet++-ResNet34 实验记录。
- 面向 SMP 模型的 threshold sweep、postprocess sweep 和验证集预测可视化工具。

已有实验报告位于：

- `outputs/experiments/siim_acr_unet_baseline/EXPERIMENT_REPORT.md`
- `outputs/experiments/siim_acr_unet_baseline_v2_bs32/EXPERIMENT_REPORT.md`
- `outputs/experiments/siim_acr_unet_baseline_v3/EXPERIMENT_REPORT.md`
- `outputs/experiments/siim_acr_unet_cls_v2/EXPERIMENT_REPORT.md`
- `outputs/experiments/siim_acr_unetpp_resnet34_v1/20260518_180917/EXPERIMENT_REPORT.md`

当前 SIIM-ACR U-Net 系列实验结果概览：

| 实验 | 主要设置 | 最佳 val Dice | 观察 |
| --- | --- | ---: | --- |
| `siim_acr_unet_baseline` | 原始 U-Net baseline | 0.3632 | 可作为基础对照，但前景学习和 recall 偏弱。 |
| `siim_acr_unet_baseline_v2_bs32` | balanced sampling、`pos_weight=5.0`、BCE/Dice=0.5/0.5 | 0.4250 | 类别不均衡处理明显提升验证 Dice，但训练/验证差距较大；阈值扫描显示 threshold 只能带来很小提升，负样本误检仍明显。 |
| `siim_acr_unet_baseline_v3` | 降低正样本采样比例、`pos_weight=2.0`、BCE/Dice=0.3/0.7 | 0.4289 | 训练过程相比 v2 更稳定，过拟合有所缓解，但验证 Dice 提升有限，普通 U-Net 瓶颈仍未明显突破。 |
| `siim_acr_unet_cls_v2` | U-Net + classification head，cls gate | 0.4120 | 分类门控在部分 epoch 能提高 gated Dice，但整体低于 v3 baseline，简单分类辅助头未能稳定提升分割性能。 |
| `siim_acr_unetpp_resnet34_v1` | Unet++-ResNet34，ImageNet encoder，balanced sampling、`pos_weight=2.0`、BCE/Dice=0.3/0.7 | 0.5146 | SMP Unet++ 明显高于 U-Net baseline；threshold sweep 最佳 Dice 为 0.5198。训练后期过拟合，后处理可降低阴性假阳性，但会牺牲部分阳性 Dice。 |

## 项目结构

```text
CXR-LesionSeg/
├── configs/                  # 实验配置
├── data/                     # 数据目录
├── docs/                     # 项目文档
├── notebooks/                # 探索性分析
├── outputs/                  # 实验输出
├── scripts/                  # 预处理、训练、可视化脚本
├── src/cxr_lesionseg/        # 项目核心代码
└── tests/                    # 测试代码
```

核心代码按功能划分：

```text
src/cxr_lesionseg/
├── datasets/                 # 数据集封装
├── losses/                   # 损失函数
├── metrics/                  # 评价指标
└── models/                   # 模型结构
```

## 快速开始

建议使用独立虚拟环境。

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements-preprocess.txt
```

训练环境还需要安装与本机 CUDA 版本匹配的 PyTorch。

## 常用命令

数据预处理：

```powershell
python scripts/preprocess/prepare_siim_acr.py
```

数据加载检查：

```powershell
python scripts/preprocess/check_siim_dataloader.py
```

训练 SIIM-ACR U-Net baseline：

```powershell
python scripts/train/train_siim_unet.py --device cuda --amp
```

训练 SIIM-ACR SMP 模型，例如 Unet++-ResNet34：

```powershell
python scripts/train/train_siim_smp.py --output-root outputs/experiments/siim_acr_unetpp_resnet34_v1 --model-name unetplusplus --encoder-name resnet34 --encoder-weights imagenet --image-size 512 --batch-size 32 --epochs 50 --lr 0.0003 --weight-decay 0.0003 --bce-weight 0.3 --dice-weight 0.7 --pos-weight 2.0 --balanced-sampling --positive-sample-ratio 0.35 --device cuda --amp
```

对 SMP best checkpoint 进行阈值扫描：

```powershell
python scripts/evaluate/threshold_sweep_siim_smp.py --config outputs/experiments/siim_acr_unetpp_resnet34_v1/20260518_180917/config.json --checkpoint outputs/experiments/siim_acr_unetpp_resnet34_v1/20260518_180917/checkpoints/best_model.pth --split val --output-csv outputs/experiments/siim_acr_unetpp_resnet34_v1/20260518_180917/logs/threshold_sweep_val.csv --device cuda
```

搜索 SMP 后处理参数组合：

```powershell
python scripts/evaluate/postprocess_sweep_siim_smp.py --config outputs/experiments/siim_acr_unetpp_resnet34_v1/20260518_180917/config.json --checkpoint outputs/experiments/siim_acr_unetpp_resnet34_v1/20260518_180917/checkpoints/best_model.pth --split val --output-csv outputs/experiments/siim_acr_unetpp_resnet34_v1/20260518_180917/logs/postprocess_sweep_val.csv --device cuda
```

可视化 SMP 验证集预测并生成定性分析统计：

```powershell
python scripts/evaluate/visualize_siim_predictions_smp.py --checkpoint outputs/experiments/siim_acr_unetpp_resnet34_v1/20260518_180917/checkpoints/best_model.pth --split val --threshold 0.04 --output-dir outputs/experiments/siim_acr_unetpp_resnet34_v1/20260518_180917/visualizations/val_predictions_thr0p04 --device cuda --amp
```

使用后处理参数可视化 SMP 验证集预测：

```powershell
python scripts/evaluate/visualize_siim_predictions_smp_postprocess.py --checkpoint outputs/experiments/siim_acr_unetpp_resnet34_v1/20260518_180917/checkpoints/best_model.pth --split val --threshold 0.08 --min-component-area 150 --output-dir outputs/experiments/siim_acr_unetpp_resnet34_v1/20260518_180917/visualizations/val_predictions_thr0p08_area150 --device cuda --amp
```

## 实验管理

实验输出默认保存在 `outputs/experiments/`。每次训练会生成独立运行目录，并保存：

- `config.json`
- `logs/metrics.csv`
- `checkpoints/best_model.pth`
- `checkpoints/last_model.pth`

其中模型权重和大体积数据不纳入版本管理。`outputs/experiments/**/visualizations/` 下的预测可视化图片也默认忽略；推荐保留轻量的配置、指标日志、阈值/后处理搜索 CSV 和实验报告，用于追踪实验过程。

## 后续计划

- 扩展更多胸片病灶数据集。
- 增加 UNet++、DeepLabV3+、Attention U-Net、Swin-Unet 等模型。
- 增加数据增强、阈值扫描、负样本误检分析和独立 test 集评估流程。
- 尝试图像级分类辅助分支，增强模型对正常/异常胸片的整体判别能力。
- 统一配置管理和实验汇总脚本。
- 建立多模型、多数据集、多指标的对比实验表。
