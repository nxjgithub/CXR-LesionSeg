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
- 两组 baseline 实验记录和报告。

已有实验报告位于：

- `outputs/experiments/siim_acr_unet_baseline/EXPERIMENT_REPORT.md`
- `outputs/experiments/siim_acr_unet_baseline_v2_bs32/EXPERIMENT_REPORT.md`

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

## 实验管理

实验输出默认保存在 `outputs/experiments/`。每次训练会生成独立运行目录，并保存：

- `config.json`
- `logs/metrics.csv`
- `checkpoints/best_model.pth`
- `checkpoints/last_model.pth`

其中模型权重和大体积数据不纳入版本管理。推荐保留轻量的配置、指标日志和实验报告，用于追踪实验过程。

## 后续计划

- 扩展更多胸片病灶数据集。
- 增加 UNet++、DeepLabV3+、Attention U-Net、Swin-Unet 等模型。
- 增加数据增强、类别不均衡处理、阈值扫描和独立 test 集评估流程。
- 统一配置管理和实验汇总脚本。
- 建立多模型、多数据集、多指标的对比实验表。
