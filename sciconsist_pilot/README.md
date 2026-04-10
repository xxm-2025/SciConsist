# SciConsist Pilot: Cross-Model Factual Entailment Head

> Week 1 Pilot 实验框架 — 验证 FEH 作为 GRPO reward signal 的可行性

## 概述

本项目实现 SciConsist v3.1 研究计划的 Week 1 Pilot 实验，包含：

- **Factual Entailment Head (FEH)**：在 InternVL2.5-8B 隐空间上训练的轻量级事实蕴含三分类头
- **5 个 Go/No-Go 验证实验**：
  - **P1** — FEH 三分类基础能力 (accuracy > 75%)
  - **P2** — 数值粒度敏感度 (±5% 检出率 > 60%)
  - **P3** — Many-to-One 解决 (ENTAILS+NEUTRAL > 80%)
  - **P4** — Full figure vs cropped region (gap < 5%)
  - **P5** — Cross-model vs same-model (gap < 10%)

## 快速开始

```bash
# 1. 安装依赖
uv sync

# 2. 准备数据（先用 placeholder 跑通 pipeline）
python scripts/prepare_data.py

# 3. 训练 FEH（自动生成 placeholder 特征 + 训练 + P1 评估）
python scripts/train_feh.py

# 4. 运行全部 pilot 实验
python scripts/run_pilot.py
```

## 使用真实数据

```bash
# 下载 S1-MMAlign + 提取 InternVL2.5 特征（需 GPU）
python scripts/prepare_data.py --real-data --model OpenGVLab/InternVL2_5-8B

# 训练 FEH
python scripts/train_feh.py

# 运行 pilot
python scripts/run_pilot.py
```

## 项目结构

```
sciconsist_pilot/
├── src/
│   ├── models/
│   │   └── feh.py              # FEH 模型 + FEHReward
│   ├── data/
│   │   ├── prepare.py          # 数据下载与构造
│   │   ├── dataset.py          # FEH 训练数据集
│   │   └── perturbation.py     # 数值篡改 (hard negatives)
│   ├── features/
│   │   └── extract.py          # InternVL/Qwen 特征提取
│   └── evaluate/
│       └── metrics.py          # P1-P5 评估指标
├── scripts/
│   ├── prepare_data.py         # Step 1: 数据准备
│   ├── train_feh.py            # Step 2: FEH 训练 (Hydra)
│   └── run_pilot.py            # Step 3: P1-P5 验证
├── run/conf/
│   └── pilot.yaml              # Hydra 配置
├── data/                       # 数据目录
├── outputs/                    # 输出 (checkpoints, figures, logs)
└── pyproject.toml
```

## Go/No-Go 判定逻辑

| 实验 | 条件 | 判定 |
|------|------|------|
| P1 | accuracy ≥ 0.75 | 🟢 GO → 继续 P2-P5 |
| P1 | 0.65 ≤ accuracy < 0.75 | 🟡 调整超参后重试 |
| P1 | accuracy < 0.65 | 🔴 NO-GO → 需改 FEH 架构/数据 |
| P2 | ±5% 检出率 ≥ 0.60 | ✅ 数值敏感度足够 |
| P2 | ±5% 检出率 < 0.60 | 需加 numerical token injection |
| P3 | ENTAILS+NEUTRAL ≥ 0.80 | ✅ Many-to-One 已解决 |
| P4 | gap < 0.05 | ✅ 用 full figure 即可 |
| P5 | gap < 0.10 | ✅ Cross-model 可行 |
| P5 | gap ≥ 0.10 | 退回 same-model + 加大 KL penalty |

## 配置覆盖 (Hydra)

```bash
# 缩小数据量快速迭代
python scripts/train_feh.py data.num_entails=1000 data.num_neutral=1000 data.num_contradicts=1000

# 调整学习率
python scripts/train_feh.py training.lr=5e-4

# 改变 latent dim
python scripts/train_feh.py model.latent_dim=256
```
