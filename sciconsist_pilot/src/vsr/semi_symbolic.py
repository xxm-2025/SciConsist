"""
Layer 1: Semi-Symbolic Verifier — 关系/趋势验证

验证比较关系 (A > B) 和趋势方向 (increases/decreases) 的正确性。

比 Layer 0 宽容:
  - 不要求精确数值，只要求方向/排序正确
  - 允许 tolerance (视觉估算有误差)

验证逻辑:
  比较关系: 从表中提取双方数值，检查排序方向是否一致
  趋势: 从表中提取序列数据，检查单调性方向

Reward:
  - 关系与数据一致 → r = +0.8
  - 关系正确但仅单源验证 → r = +0.5
  - 关系与数据矛盾 → r = -0.8
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from sciconsist_pilot.src.vsr.types import (
    AtomicClaim,
    RelationAssertion,
    TrendAssertion,
    StructuredTable,
    VerificationResult,
    VerificationLayer,
    _levenshtein_similarity,
)


@dataclass
class SemiSymbolicVerifierConfig:
    """Layer 1 配置

    Attributes:
        entity_match_threshold: 实体模糊匹配阈值
        relation_consistent_reward: 关系一致 reward
        relation_partial_reward: 仅单源验证 reward
        relation_inconsistent_reward: 关系矛盾 reward
        trend_consistent_reward: 趋势一致 reward
        trend_inconsistent_reward: 趋势矛盾 reward
        confidence_gate: 置信度门限
    """
    entity_match_threshold: float = 0.6
    relation_consistent_reward: float = 0.8
    relation_partial_reward: float = 0.5
    relation_inconsistent_reward: float = -0.8
    trend_consistent_reward: float = 0.8
    trend_inconsistent_reward: float = -0.8
    confidence_gate: float = 0.7


class SemiSymbolicVerifier:
    """Layer 1 Semi-Symbolic Verifier

    用法:
        verifier = SemiSymbolicVerifier()
        result = verifier.verify(claim, tables)
    """

    def __init__(self, config: SemiSymbolicVerifierConfig | None = None) -> None:
        self.config = config or SemiSymbolicVerifierConfig()

    def verify(
        self,
        claim: AtomicClaim,
        tables: list[StructuredTable],
    ) -> VerificationResult:
        """验证 claim 中的关系/趋势断言

        Args:
            claim: 含比较关系或趋势的 atomic claim
            tables: 该论文的结构化表格列表

        Returns:
            VerificationResult (layer=SEMI_SYMBOLIC)
        """
        results: list[VerificationResult] = []

        for relation in claim.relations:
            r = self._verify_relation(relation, tables)
            results.append(r)

        for trend in claim.trends:
            r = self._verify_trend(trend, claim, tables)
            results.append(r)

        if not results:
            return self._fallback("no relations or trends to verify")

        # 取最高置信度的结果
        best = max(results, key=lambda r: r.confidence)
        return best

    def verify_batch(
        self,
        claims: list[AtomicClaim],
        tables: list[StructuredTable],
    ) -> list[VerificationResult]:
        """批量验证"""
        return [self.verify(c, tables) for c in claims]

    def _verify_relation(
        self,
        relation: RelationAssertion,
        tables: list[StructuredTable],
    ) -> VerificationResult:
        """验证比较关系 A > B"""
        val_a = self._find_entity_value(relation.entity_a, relation.metric, tables)
        val_b = self._find_entity_value(relation.entity_b, relation.metric, tables)

        if val_a is None and val_b is None:
            return self._fallback(
                f"neither {relation.entity_a} nor {relation.entity_b} found"
            )

        if val_a is None or val_b is None:
            # 只找到一方，无法完整验证
            found = relation.entity_a if val_a is not None else relation.entity_b
            return VerificationResult(
                layer=VerificationLayer.SEMI_SYMBOLIC,
                reward=self.config.relation_partial_reward,
                confidence=0.5,
                matched_evidence=f"partial: only {found} found",
                details={
                    "match_type": "partial",
                    "found_entity": found,
                    "found_value": val_a if val_a is not None else val_b,
                },
            )

        # 双方都找到，验证关系方向
        actual_relation = self._compare_values(val_a, val_b)
        expected_relation = relation.relation

        consistent = self._relation_consistent(expected_relation, actual_relation)

        if consistent:
            reward = self.config.relation_consistent_reward
            match_type = "consistent"
        else:
            reward = self.config.relation_inconsistent_reward
            match_type = "inconsistent"

        return VerificationResult(
            layer=VerificationLayer.SEMI_SYMBOLIC,
            reward=reward,
            confidence=0.9,
            matched_evidence=f"{relation.entity_a}={val_a} {actual_relation} {relation.entity_b}={val_b}",
            details={
                "match_type": match_type,
                "entity_a": relation.entity_a,
                "entity_b": relation.entity_b,
                "value_a": val_a,
                "value_b": val_b,
                "expected_relation": expected_relation,
                "actual_relation": actual_relation,
            },
        )

    def _verify_trend(
        self,
        trend: TrendAssertion,
        claim: AtomicClaim,
        tables: list[StructuredTable],
    ) -> VerificationResult:
        """验证趋势断言

        从表格中寻找同一 metric 下不同条件的值序列，
        检查方向是否与声称的趋势一致。
        """
        # 尝试找到与趋势相关的数值序列
        sequence = self._find_trend_sequence(trend, claim, tables)

        if not sequence or len(sequence) < 2:
            return self._fallback(f"insufficient data to verify trend: {trend.direction}")

        # 判断实际趋势方向
        actual_direction = self._detect_direction(sequence)
        expected = trend.direction

        consistent = (
            (expected in ('increase',) and actual_direction == 'increase') or
            (expected in ('decrease',) and actual_direction == 'decrease') or
            (expected in ('stable', 'saturate') and actual_direction in ('stable', 'saturate'))
        )

        if consistent:
            reward = self.config.trend_consistent_reward
        else:
            reward = self.config.trend_inconsistent_reward

        return VerificationResult(
            layer=VerificationLayer.SEMI_SYMBOLIC,
            reward=reward,
            confidence=0.75,
            matched_evidence=f"sequence={sequence}, actual={actual_direction}, expected={expected}",
            details={
                "match_type": "consistent" if consistent else "inconsistent",
                "sequence": sequence,
                "actual_direction": actual_direction,
                "expected_direction": expected,
            },
        )

    def _find_entity_value(
        self,
        entity_name: str,
        metric_name: str,
        tables: list[StructuredTable],
    ) -> float | None:
        """在表格中查找实体的数值"""
        for table in tables:
            record = table.lookup(
                entity_name,
                metric_name,
                fuzzy_threshold=self.config.entity_match_threshold,
            )
            if record and record.value is not None:
                return record.value
        return None

    def _find_trend_sequence(
        self,
        trend: TrendAssertion,
        claim: AtomicClaim,
        tables: list[StructuredTable],
    ) -> list[float]:
        """从表格中提取与趋势相关的数值序列

        例如 "accuracy increases as model size grows" →
        找到同一列中按行排列的一组数值。
        """
        metric_name = claim.metrics[0] if claim.metrics else ""

        for table in tables:
            # 找到含对应指标的列
            for col_idx, header in enumerate(table.headers):
                header_norm = re.sub(r'[\s\-_]+', '', header.lower())
                metric_norm = re.sub(r'[\s\-_]+', '', metric_name.lower()) if metric_name else ""

                if metric_norm and metric_norm not in header_norm:
                    continue

                # 提取该列的所有数值
                col_values = [
                    rec.value for rec in table.records
                    if rec.col_idx == col_idx and rec.value is not None
                ]
                if len(col_values) >= 2:
                    return col_values

        # 退回: 如果有数值 claim，直接按行顺序取
        for table in tables:
            if len(table.records) >= 2:
                vals = [r.value for r in table.records if r.value is not None]
                if len(vals) >= 2:
                    return vals[:10]

        return []

    @staticmethod
    def _compare_values(a: float, b: float) -> str:
        """比较两个值，返回关系字符串"""
        if abs(a - b) < 1e-8:
            return "eq"
        return "gt" if a > b else "lt"

    @staticmethod
    def _relation_consistent(expected: str, actual: str) -> bool:
        """检查期望关系与实际关系是否一致"""
        if expected == actual:
            return True
        if expected == "gte" and actual in ("gt", "eq"):
            return True
        if expected == "lte" and actual in ("lt", "eq"):
            return True
        return False

    @staticmethod
    def _detect_direction(values: list[float]) -> str:
        """检测数值序列的方向"""
        if len(values) < 2:
            return "unknown"

        increases = sum(1 for i in range(len(values) - 1) if values[i + 1] > values[i])
        decreases = sum(1 for i in range(len(values) - 1) if values[i + 1] < values[i])
        total = len(values) - 1

        # 80% 以上一致则判定单调
        if increases / total >= 0.8:
            return "increase"
        if decreases / total >= 0.8:
            return "decrease"

        # 后半段变化幅度很小 → saturate
        second_half = values[len(values) // 2:]
        if len(second_half) >= 2:
            max_val = max(second_half)
            min_val = min(second_half)
            if max_val > 0 and (max_val - min_val) / max_val < 0.05:
                return "saturate"

        return "stable"

    def _fallback(self, reason: str) -> VerificationResult:
        """回退到 Layer 2"""
        return VerificationResult(
            layer=VerificationLayer.SEMI_SYMBOLIC,
            reward=0.0,
            confidence=0.0,
            details={"fallback": True, "reason": reason},
        )
