"""
FEH 训练数据集 — 管理三分类 (ENTAILS/NEUTRAL/CONTRADICTS) 训练样本。

样本来源:
  - ENTAILS: S1-MMAlign 中 figure-text 自然配对
  - NEUTRAL: 随机 figure + 不相关 text
  - CONTRADICTS: S1-MMAlign + 数值篡改 (梯度化 hard negatives)

数据集直接操作预提取的特征向量，无需在训练时重新跑 VLM inference。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class FEHSample:
    """单个 FEH 训练样本。

    Attributes:
        visual_feature: 视觉特征向量 (hidden_dim,)
        text_feature: 文本特征向量 (hidden_dim,)
        label: 0=ENTAILS, 1=NEUTRAL, 2=CONTRADICTS
        perturbation_ratio: 若为 CONTRADICTS，记录篡改幅度 (0.01~0.20)；否则为 None
        source: 数据来源标识
    """

    visual_feature: np.ndarray
    text_feature: np.ndarray
    label: int
    perturbation_ratio: float | None = None
    source: str = ""


class FEHDataset(Dataset):
    """FEH 训练/验证数据集。

    从预提取的特征 .npz 文件加载数据，支持按标签过滤和采样。

    Args:
        features_dir: 预提取特征目录，包含 visual_features.npy, text_features.npy, labels.npy
        split: 数据集划分 ("train" / "val" / "test")
        max_samples: 最大样本数（None 表示不限制）
        seed: 随机种子
    """

    def __init__(
        self,
        features_dir: str | Path,
        split: str = "train",
        max_samples: int | None = None,
        seed: int = 42,
    ) -> None:
        self.features_dir = Path(features_dir)
        self.split = split

        split_dir = self.features_dir / split
        self.visual_features = np.load(split_dir / "visual_features.npy", mmap_mode="r")
        self.text_features = np.load(split_dir / "text_features.npy", mmap_mode="r")
        self.labels = np.load(split_dir / "labels.npy")

        if (split_dir / "perturbation_ratios.npy").exists():
            self.perturbation_ratios = np.load(split_dir / "perturbation_ratios.npy")
        else:
            self.perturbation_ratios = np.zeros(len(self.labels), dtype=np.float32)

        if max_samples is not None and max_samples < len(self.labels):
            rng = np.random.RandomState(seed)
            indices = rng.choice(len(self.labels), max_samples, replace=False)
            indices.sort()
            self.indices = indices
        else:
            self.indices = np.arange(len(self.labels))

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """获取单个样本。

        Returns:
            字典，包含:
                visual: 视觉特征 (hidden_dim,)
                text: 文本特征 (hidden_dim,)
                label: 标签 (标量)
                perturbation_ratio: 篡改幅度 (标量)
        """
        real_idx = self.indices[idx]
        return {
            "visual": torch.from_numpy(np.array(self.visual_features[real_idx])).float(),
            "text": torch.from_numpy(np.array(self.text_features[real_idx])).float(),
            "label": torch.tensor(self.labels[real_idx], dtype=torch.long),
            "perturbation_ratio": torch.tensor(self.perturbation_ratios[real_idx], dtype=torch.float32),
        }

    @property
    def label_counts(self) -> dict[int, int]:
        """各标签的样本数量。"""
        labels = self.labels[self.indices]
        unique, counts = np.unique(labels, return_counts=True)
        return dict(zip(unique.tolist(), counts.tolist()))

    @staticmethod
    def create_from_features(
        visual_features: np.ndarray,
        text_features: np.ndarray,
        labels: np.ndarray,
        output_dir: str | Path,
        split: str = "train",
        perturbation_ratios: np.ndarray | None = None,
    ) -> None:
        """从特征数组创建数据集文件。

        Args:
            visual_features: 视觉特征 (N, hidden_dim)
            text_features: 文本特征 (N, hidden_dim)
            labels: 标签 (N,)
            output_dir: 输出目录
            split: 数据集划分名称
            perturbation_ratios: 篡改幅度 (N,)，仅 CONTRADICTS 有值
        """
        split_dir = Path(output_dir) / split
        split_dir.mkdir(parents=True, exist_ok=True)
        np.save(split_dir / "visual_features.npy", visual_features)
        np.save(split_dir / "text_features.npy", text_features)
        np.save(split_dir / "labels.npy", labels)
        if perturbation_ratios is not None:
            np.save(split_dir / "perturbation_ratios.npy", perturbation_ratios)
