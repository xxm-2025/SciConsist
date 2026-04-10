"""
数值篡改模块 — 用于构造 FEH 训练的 hard negatives。

提供梯度化数值篡改策略：从 ±1% (极难) 到 ±20% (简单)，
模拟科研文档中不同程度的数值不一致。

典型用法:
    perturber = NumericalPerturber(seed=42)
    perturbed_text = perturber.perturb_text("accuracy is 85.2%", ratio=0.05)
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field


@dataclass
class PerturbationResult:
    """数值篡改结果。

    Attributes:
        original_text: 原始文本
        perturbed_text: 篡改后文本
        original_values: 原始数值列表
        perturbed_values: 篡改后数值列表
        perturbation_ratio: 篡改幅度
        num_values_changed: 被篡改的数值数量
    """

    original_text: str
    perturbed_text: str
    original_values: list[float] = field(default_factory=list)
    perturbed_values: list[float] = field(default_factory=list)
    perturbation_ratio: float = 0.0
    num_values_changed: int = 0


# 匹配科研文档中常见的数值模式：整数、小数、百分数
_NUMBER_PATTERN = re.compile(
    r"(?<!\w)"  # 不能紧跟字母
    r"(\d+\.?\d*)"  # 数字（整数或小数）
    r"(%?)"  # 可选的百分号
    r"(?!\w)"  # 不能紧接字母
)


class NumericalPerturber:
    """科研文档数值篡改器。

    按指定幅度随机篡改文本中的数值，保持格式一致。

    Args:
        seed: 随机种子
        min_value: 只篡改大于此值的数字（过滤年份、引用编号等）
        max_change_per_text: 每段文本最多篡改多少个数值
    """

    def __init__(
        self,
        seed: int = 42,
        min_value: float = 0.0,
        max_change_per_text: int = 3,
    ) -> None:
        self.rng = random.Random(seed)
        self.min_value = min_value
        self.max_change_per_text = max_change_per_text

    def _perturb_value(self, value: float, ratio: float) -> float:
        """对单个数值做随机篡改。

        Args:
            value: 原始数值
            ratio: 篡改幅度 (e.g. 0.05 = ±5%)

        Returns:
            篡改后的数值
        """
        direction = self.rng.choice([-1, 1])
        actual_ratio = self.rng.uniform(ratio * 0.5, ratio * 1.5)
        return value * (1 + direction * actual_ratio)

    def perturb_text(self, text: str, ratio: float = 0.05) -> PerturbationResult:
        """对文本中的数值进行篡改。

        Args:
            text: 原始文本 (e.g. "Our model achieves 85.2% accuracy on BLEU")
            ratio: 篡改幅度 (e.g. 0.05 = ±5%)

        Returns:
            PerturbationResult 包含原始和篡改后的文本及数值
        """
        matches = list(_NUMBER_PATTERN.finditer(text))
        candidates = []
        for m in matches:
            val = float(m.group(1))
            if val >= self.min_value and val < 2100:  # 过滤年份
                candidates.append(m)

        if not candidates:
            return PerturbationResult(
                original_text=text,
                perturbed_text=text,
                perturbation_ratio=ratio,
            )

        n_change = min(len(candidates), self.max_change_per_text)
        to_change = self.rng.sample(candidates, n_change)
        to_change.sort(key=lambda m: m.start(), reverse=True)

        perturbed = text
        original_values = []
        perturbed_values = []

        for m in to_change:
            orig_val = float(m.group(1))
            new_val = self._perturb_value(orig_val, ratio)
            original_values.append(orig_val)
            perturbed_values.append(new_val)

            # 保持原始数值格式
            orig_str = m.group(1)
            if "." in orig_str:
                decimal_places = len(orig_str.split(".")[1])
                new_str = f"{new_val:.{decimal_places}f}"
            else:
                new_str = str(int(round(new_val)))

            pct = m.group(2)
            perturbed = perturbed[: m.start()] + new_str + pct + perturbed[m.end() :]

        return PerturbationResult(
            original_text=text,
            perturbed_text=perturbed,
            original_values=original_values,
            perturbed_values=perturbed_values,
            perturbation_ratio=ratio,
            num_values_changed=n_change,
        )

    def generate_hard_negatives(
        self,
        text: str,
        ratios: list[float] | None = None,
    ) -> list[PerturbationResult]:
        """生成多个梯度化 hard negatives。

        Args:
            text: 原始文本
            ratios: 篡改幅度列表，默认 [0.01, 0.02, 0.05, 0.10, 0.20]

        Returns:
            每种幅度对应的 PerturbationResult 列表
        """
        if ratios is None:
            ratios = [0.01, 0.02, 0.05, 0.10, 0.20]
        return [self.perturb_text(text, r) for r in ratios]
