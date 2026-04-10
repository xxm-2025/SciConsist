"""
FEH 评估指标模块 — 支持 5 个 pilot 实验的核心指标计算。

P1: 三分类 accuracy, per-class precision/recall/F1, confusion matrix
P2: 数值粒度敏感度曲线 (CONTRADICTS prob vs perturbation ratio)
P3: Many-to-One 解决率 (ENTAILS+NEUTRAL 占比)
P4: Full figure vs cropped region accuracy gap
P5: Cross-model vs same-model accuracy gap
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)


@dataclass
class P1Result:
    """P1 实验结果：FEH 三分类基础能力。

    Attributes:
        accuracy: 总体准确率
        per_class_f1: 各类别 F1 分数 {label_name: f1}
        per_class_precision: 各类别 Precision
        per_class_recall: 各类别 Recall
        confusion: 混淆矩阵 (3x3)
        cohens_kappa: Cohen's κ (与人类标注的一致性, 如有)
        go: 是否满足 Go 条件 (accuracy > threshold)
    """

    accuracy: float = 0.0
    per_class_f1: dict[str, float] = field(default_factory=dict)
    per_class_precision: dict[str, float] = field(default_factory=dict)
    per_class_recall: dict[str, float] = field(default_factory=dict)
    confusion: np.ndarray = field(default_factory=lambda: np.zeros((3, 3)))
    cohens_kappa: float | None = None
    go: bool = False


@dataclass
class P2Result:
    """P2 实验结果：数值粒度敏感度。

    Attributes:
        perturbation_levels: 篡改幅度列表 (e.g. [0.01, 0.02, 0.05, ...])
        detection_rates: 各幅度下 CONTRADICTS 的检出率
        avg_contradict_prob: 各幅度下 CONTRADICTS 概率的平均值
        target_met: ±5% 处检出率是否 > 60%
    """

    perturbation_levels: list[float] = field(default_factory=list)
    detection_rates: list[float] = field(default_factory=list)
    avg_contradict_prob: list[float] = field(default_factory=list)
    target_met: bool = False


@dataclass
class P3Result:
    """P3 实验结果：Many-to-One 解决。

    Attributes:
        non_contradict_ratio: ENTAILS+NEUTRAL 占比 (目标 > 80%)
        entails_ratio: ENTAILS 占比
        neutral_ratio: NEUTRAL 占比
        contradict_ratio: CONTRADICTS 占比 (越低越好, 表示 Many-to-One 解决)
        target_met: non_contradict_ratio > 80%
    """

    non_contradict_ratio: float = 0.0
    entails_ratio: float = 0.0
    neutral_ratio: float = 0.0
    contradict_ratio: float = 0.0
    target_met: bool = False


LABEL_NAMES = ["ENTAILS", "NEUTRAL", "CONTRADICTS"]


class FEHEvaluator:
    """FEH Pilot 实验评估器。

    封装了 P1-P5 五个实验的指标计算逻辑。
    """

    def evaluate_p1(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        go_threshold: float = 0.75,
        human_labels: np.ndarray | None = None,
    ) -> P1Result:
        """P1: 评估 FEH 三分类基础能力。

        Args:
            y_true: 真实标签 (N,)
            y_pred: 预测标签 (N,)
            go_threshold: Go/No-Go 准确率阈值
            human_labels: 人类标注标签 (N,)，用于计算 Cohen's κ

        Returns:
            P1Result 包含准确率、per-class 指标、混淆矩阵等
        """
        acc = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1, 2], zero_division=0)
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])

        result = P1Result(
            accuracy=acc,
            per_class_f1={name: float(f) for name, f in zip(LABEL_NAMES, f1)},
            per_class_precision={name: float(p) for name, p in zip(LABEL_NAMES, precision)},
            per_class_recall={name: float(r) for name, r in zip(LABEL_NAMES, recall)},
            confusion=cm,
            go=acc >= go_threshold,
        )

        if human_labels is not None:
            result.cohens_kappa = cohen_kappa_score(human_labels, y_pred)

        return result

    def evaluate_p2(
        self,
        perturbation_levels: list[float],
        true_labels_per_level: list[np.ndarray],
        pred_labels_per_level: list[np.ndarray],
        pred_probs_per_level: list[np.ndarray],
        target_detection_at_5pct: float = 0.60,
    ) -> P2Result:
        """P2: 评估数值粒度敏感度。

        对每种篡改幅度，计算 CONTRADICTS 的检出率和平均概率。

        Args:
            perturbation_levels: 篡改幅度列表 (e.g. [0.01, 0.02, 0.05, ...])
            true_labels_per_level: 各幅度下真实标签 (每个都应全是 CONTRADICTS=2)
            pred_labels_per_level: 各幅度下预测标签
            pred_probs_per_level: 各幅度下预测概率 (N, 3)
            target_detection_at_5pct: ±5% 处的目标检出率

        Returns:
            P2Result 包含检出率曲线和目标达成情况
        """
        detection_rates = []
        avg_probs = []

        for preds, probs in zip(pred_labels_per_level, pred_probs_per_level):
            det_rate = np.mean(preds == 2)  # CONTRADICTS = 2
            avg_prob = np.mean(probs[:, 2])  # CONTRADICTS prob
            detection_rates.append(float(det_rate))
            avg_probs.append(float(avg_prob))

        # 检查 ±5% 处的检出率
        target_met = False
        for level, rate in zip(perturbation_levels, detection_rates):
            if abs(level - 0.05) < 1e-6:
                target_met = rate >= target_detection_at_5pct
                break

        return P2Result(
            perturbation_levels=perturbation_levels,
            detection_rates=detection_rates,
            avg_contradict_prob=avg_probs,
            target_met=target_met,
        )

    def evaluate_p3(
        self,
        pred_labels: np.ndarray,
        target_non_contradict: float = 0.80,
    ) -> P3Result:
        """P3: 评估 Many-to-One 解决能力。

        对"说对了但换了说法"的 claim，检查 FEH 是否将其判为 ENTAILS 或 NEUTRAL
        （而非错误地判为 CONTRADICTS）。

        Args:
            pred_labels: 对 Many-to-One 样本的预测标签 (N,)
            target_non_contradict: 目标 ENTAILS+NEUTRAL 占比

        Returns:
            P3Result 包含各类别占比和目标达成情况
        """
        total = len(pred_labels)
        if total == 0:
            return P3Result()
        n_entails = np.sum(pred_labels == 0)
        n_neutral = np.sum(pred_labels == 1)
        n_contradict = np.sum(pred_labels == 2)

        non_contradict = (n_entails + n_neutral) / total

        return P3Result(
            non_contradict_ratio=float(non_contradict),
            entails_ratio=float(n_entails / total),
            neutral_ratio=float(n_neutral / total),
            contradict_ratio=float(n_contradict / total),
            target_met=non_contradict >= target_non_contradict,
        )

    def evaluate_accuracy_gap(
        self,
        accuracy_a: float,
        accuracy_b: float,
        max_gap: float,
    ) -> tuple[float, bool]:
        """P4/P5: 计算两种配置的 accuracy 差距。

        Args:
            accuracy_a: 配置 A 的准确率 (e.g. full_figure / cross-model)
            accuracy_b: 配置 B 的准确率 (e.g. cropped / same-model)
            max_gap: 允许的最大差距

        Returns:
            (gap, acceptable): 差距值 和 是否在可接受范围内
        """
        gap = abs(accuracy_a - accuracy_b)
        return gap, gap <= max_gap

    def format_p1_report(self, result: P1Result) -> str:
        """格式化 P1 结果为可读报告。"""
        lines = [
            "=" * 60,
            "P1: FEH 三分类基础能力",
            "=" * 60,
            f"Overall Accuracy: {result.accuracy:.4f}  {'✅ GO' if result.go else '❌ NO-GO'}",
            "",
            "Per-class metrics:",
        ]
        for name in LABEL_NAMES:
            lines.append(
                f"  {name:>12s}: P={result.per_class_precision.get(name, 0):.3f} "
                f"R={result.per_class_recall.get(name, 0):.3f} "
                f"F1={result.per_class_f1.get(name, 0):.3f}"
            )
        if result.cohens_kappa is not None:
            lines.append(f"\nCohen's κ (vs human): {result.cohens_kappa:.3f}")
        lines.append(f"\nConfusion Matrix:\n{result.confusion}")
        return "\n".join(lines)

    def format_p2_report(self, result: P2Result) -> str:
        """格式化 P2 结果为可读报告。"""
        lines = [
            "=" * 60,
            "P2: 数值粒度敏感度",
            "=" * 60,
            f"Target (detection@±5%): {'✅ MET' if result.target_met else '❌ NOT MET'}",
            "",
            f"{'Perturbation':>12s} | {'Detection Rate':>14s} | {'Avg CONTRADICTS Prob':>20s}",
            "-" * 55,
        ]
        for level, rate, prob in zip(result.perturbation_levels, result.detection_rates, result.avg_contradict_prob):
            marker = " ◄" if abs(level - 0.05) < 1e-6 else ""
            lines.append(f"  ±{level*100:5.1f}%     |   {rate:12.3f}   |   {prob:18.3f}{marker}")
        return "\n".join(lines)
