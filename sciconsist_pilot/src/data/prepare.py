"""
数据下载与准备模块 — FEH pilot 实验的数据 pipeline 入口。

下载 S1-MMAlign / MuSciClaims / SciClaimEval 子集，
构造三类训练数据 (ENTAILS / NEUTRAL / CONTRADICTS)，
包含梯度化 hard negatives。

典型用法:
    python scripts/prepare_data.py --config-path ../run/conf --config-name pilot
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from datasets import load_dataset

from src.data.perturbation import NumericalPerturber

logger = logging.getLogger(__name__)


@dataclass
class RawSample:
    """从 HuggingFace 加载的原始样本。

    Attributes:
        figure_path: 图像文件路径 (或 PIL.Image)
        text: 对应文本 (caption / claim)
        source: 数据来源 ("s1mmalign" / "musciclaims" / "sciclaimeval")
        label: 原始标签 (如有)
        metadata: 额外元信息
    """

    figure_path: str = ""
    text: str = ""
    source: str = ""
    label: str | None = None
    metadata: dict = field(default_factory=dict)


def download_s1mmalign(
    raw_dir: str | Path,
    max_samples: int = 60000,
    seed: int = 42,
) -> list[RawSample]:
    """下载 S1-MMAlign 数据集子集 (figure-text 对)。

    S1-MMAlign 包含 15.5M 科研 figure-text 对。
    我们只下载一个子集用于 FEH 训练。

    Args:
        raw_dir: 原始数据存放目录
        max_samples: 最大下载样本数
        seed: 随机种子

    Returns:
        RawSample 列表
    """
    raw_dir = Path(raw_dir)
    cache_file = raw_dir / "s1mmalign_samples.json"

    if cache_file.exists():
        logger.info(f"Loading cached S1-MMAlign from {cache_file}")
        with open(cache_file) as f:
            data = json.load(f)
        return [RawSample(**d) for d in data]

    logger.info(f"Downloading S1-MMAlign (max {max_samples} samples)...")
    # S1-MMAlign 在 HuggingFace 上的 repo: Yuxiang-Luo/S1-MMAlign
    # 如果 HuggingFace 上不可用，回退到手动下载
    try:
        ds = load_dataset(
            "Yuxiang-Luo/S1-MMAlign",
            split="train",
            streaming=True,
        )
        samples = []
        for i, item in enumerate(ds):
            if i >= max_samples:
                break
            samples.append(
                RawSample(
                    figure_path=str(item.get("image", "")),
                    text=item.get("caption", item.get("text", "")),
                    source="s1mmalign",
                    metadata={"index": i},
                )
            )
    except Exception as e:
        logger.warning(f"Failed to load S1-MMAlign from HuggingFace: {e}")
        logger.info("Creating synthetic placeholder data for development...")
        samples = _create_placeholder_data(max_samples, seed)

    raw_dir.mkdir(parents=True, exist_ok=True)
    with open(cache_file, "w") as f:
        json.dump([{"figure_path": s.figure_path, "text": s.text, "source": s.source} for s in samples], f)

    logger.info(f"Downloaded {len(samples)} S1-MMAlign samples")
    return samples


def _create_placeholder_data(n: int, seed: int) -> list[RawSample]:
    """生成占位数据，用于离线开发和调试 pipeline。

    实际实验时替换为真实数据。
    """
    rng = np.random.RandomState(seed)
    templates = [
        "Our model achieves {val:.1f}% accuracy on the {dataset} benchmark.",
        "The BLEU score of {val:.2f} demonstrates significant improvement.",
        "Method A outperforms Method B by {val:.1f} points on F1.",
        "The training loss converges to {val:.4f} after {epoch} epochs.",
        "We observe a {val:.1f}% reduction in error rate.",
    ]
    datasets_names = ["MMLU", "GSM8K", "HumanEval", "ChartQA", "ScienceQA"]
    samples = []
    for i in range(n):
        tmpl = templates[i % len(templates)]
        text = tmpl.format(
            val=rng.uniform(50, 99),
            dataset=datasets_names[i % len(datasets_names)],
            epoch=rng.randint(5, 50),
        )
        samples.append(RawSample(figure_path=f"placeholder_{i}.png", text=text, source="placeholder"))
    return samples


def construct_feh_training_data(
    entails_samples: list[RawSample],
    num_entails: int = 50000,
    num_neutral: int = 50000,
    num_contradicts: int = 50000,
    perturbation_distribution: dict[str, float] | None = None,
    seed: int = 42,
) -> tuple[list[RawSample], list[int], list[float]]:
    """从原始样本构造 FEH 三分类训练数据。

    ENTAILS: 使用原始 figure-text 配对
    NEUTRAL: 随机打乱 figure-text 配对
    CONTRADICTS: 对 text 做数值篡改 (梯度化 hard negatives)

    Args:
        entails_samples: 原始 figure-text 对 (作为 ENTAILS 正例)
        num_entails: ENTAILS 样本数
        num_neutral: NEUTRAL 样本数
        num_contradicts: CONTRADICTS 样本数
        perturbation_distribution: 各篡改幅度的占比, e.g. {"0.01": 0.15, "0.05": 0.30, ...}
        seed: 随机种子

    Returns:
        (samples, labels, perturbation_ratios):
            samples: 构造后的样本列表
            labels: 对应标签 (0/1/2)
            perturbation_ratios: 篡改幅度 (CONTRADICTS 有值，其余为 0)
    """
    rng = np.random.RandomState(seed)
    perturber = NumericalPerturber(seed=seed)

    if perturbation_distribution is None:
        perturbation_distribution = {"0.01": 0.15, "0.02": 0.15, "0.05": 0.30, "0.10": 0.20, "0.20": 0.20}

    all_samples: list[RawSample] = []
    all_labels: list[int] = []
    all_ratios: list[float] = []

    # --- ENTAILS ---
    logger.info(f"Constructing {num_entails} ENTAILS samples...")
    indices = rng.choice(len(entails_samples), num_entails, replace=num_entails > len(entails_samples))
    for idx in indices:
        all_samples.append(entails_samples[idx])
        all_labels.append(0)
        all_ratios.append(0.0)

    # --- NEUTRAL (随机打乱配对) ---
    logger.info(f"Constructing {num_neutral} NEUTRAL samples...")
    for _ in range(num_neutral):
        fig_idx, txt_idx = rng.choice(len(entails_samples), 2, replace=False)
        sample = RawSample(
            figure_path=entails_samples[fig_idx].figure_path,
            text=entails_samples[txt_idx].text,
            source="neutral_shuffle",
        )
        all_samples.append(sample)
        all_labels.append(1)
        all_ratios.append(0.0)

    # --- CONTRADICTS (梯度化数值篡改) ---
    logger.info(f"Constructing {num_contradicts} CONTRADICTS samples (gradient hard negatives)...")
    for ratio_str, proportion in perturbation_distribution.items():
        ratio = float(ratio_str)
        n_this = int(num_contradicts * proportion)
        logger.info(f"  ±{ratio*100:.0f}%: {n_this} samples")
        for _ in range(n_this):
            idx = rng.randint(0, len(entails_samples))
            original = entails_samples[idx]
            result = perturber.perturb_text(original.text, ratio=ratio)
            if result.num_values_changed == 0:
                # 文本中没有数值，跳过
                idx = rng.randint(0, len(entails_samples))
                original = entails_samples[idx]
                result = perturber.perturb_text(original.text, ratio=ratio)
            sample = RawSample(
                figure_path=original.figure_path,
                text=result.perturbed_text,
                source=f"contradicts_perturb_{ratio}",
            )
            all_samples.append(sample)
            all_labels.append(2)
            all_ratios.append(ratio)

    # Shuffle
    perm = rng.permutation(len(all_samples))
    all_samples = [all_samples[i] for i in perm]
    all_labels = [all_labels[i] for i in perm]
    all_ratios = [all_ratios[i] for i in perm]

    logger.info(
        f"Total: {len(all_samples)} samples "
        f"(E={sum(1 for l in all_labels if l==0)}, "
        f"N={sum(1 for l in all_labels if l==1)}, "
        f"C={sum(1 for l in all_labels if l==2)})"
    )
    return all_samples, all_labels, all_ratios
