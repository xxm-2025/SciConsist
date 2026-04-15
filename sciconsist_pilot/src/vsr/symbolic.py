"""
Layer 0: Symbolic Verifier — 精确事实验证

利用预提取的结构化表格数据，对 claim 中的数值进行符号验证。

核心特性:
  - Deterministic: 不依赖任何 learned model
  - Unhackable: policy model 无法通过参数更新影响符号比较规则
  - Confidence-gated: 仅在高置信匹配时生效，否则回退 Layer 2

验证逻辑:
  1. 从 claim 解析 (entity, metric, value)
  2. 在结构化表格中查找对应 cell
  3. 符号比较:
     - Exact match (table): value == extracted → r = +1.0
     - Approximate match (tolerance): |value - extracted| < ε → r = +0.8
     - Mismatch → r = -1.0
  4. 匹配置信度低于阈值时，返回 FALLBACK 让 Layer 2 接管
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from sciconsist_pilot.src.vsr.types import (
    AtomicClaim,
    StructuredTable,
    TableRecord,
    VerificationResult,
    VerificationLayer,
    NumericValue,
    normalize_entity,
)


@dataclass
class SymbolicVerifierConfig:
    """Layer 0 配置

    Attributes:
        entity_match_threshold: 实体匹配最低相似度
        value_exact_tolerance: 精确匹配容差 (绝对值)
        value_approx_tolerance_pct: 近似匹配容差 (相对百分比)
        confidence_gate: 整体置信度门限, 低于此值回退 Layer 2
        exact_match_reward: 精确匹配 reward
        approx_match_reward: 近似匹配 reward
        mismatch_reward: 不匹配 reward
    """
    entity_match_threshold: float = 0.7
    value_exact_tolerance: float = 0.05
    value_approx_tolerance_pct: float = 1.0
    confidence_gate: float = 0.9
    exact_match_reward: float = 1.0
    approx_match_reward: float = 0.8
    mismatch_reward: float = -1.0


class SymbolicVerifier:
    """Layer 0 Symbolic Verifier

    用法:
        verifier = SymbolicVerifier()
        result = verifier.verify(claim, tables)
    """

    def __init__(self, config: SymbolicVerifierConfig | None = None) -> None:
        self.config = config or SymbolicVerifierConfig()

    def verify(
        self,
        claim: AtomicClaim,
        tables: list[StructuredTable],
    ) -> VerificationResult:
        """对单条 claim 执行符号验证

        Args:
            claim: 含数值+实体/指标的 atomic claim
            tables: 该论文的结构化表格列表

        Returns:
            VerificationResult (layer=SYMBOLIC)
            如果置信度不足, confidence < gate, 提示调用方 fallback
        """
        if not claim.numeric_values or not claim.entities:
            return self._fallback("no numeric values or entities in claim")

        best_result = None
        best_confidence = 0.0

        for nv in claim.numeric_values:
            for entity in claim.entities:
                metrics = claim.metrics if claim.metrics else [""]
                for metric in metrics:
                    result = self._verify_single(nv, entity.name, metric, tables)
                    if result.confidence > best_confidence:
                        best_confidence = result.confidence
                        best_result = result

        if best_result is None or best_confidence < self.config.confidence_gate:
            return self._fallback(
                f"best confidence {best_confidence:.2f} < gate {self.config.confidence_gate}"
            )

        return best_result

    def verify_batch(
        self,
        claims: list[AtomicClaim],
        tables: list[StructuredTable],
    ) -> list[VerificationResult]:
        """批量验证

        Args:
            claims: claim 列表
            tables: 表格列表

        Returns:
            对应的 VerificationResult 列表
        """
        return [self.verify(c, tables) for c in claims]

    def _verify_single(
        self,
        numeric: NumericValue,
        entity_name: str,
        metric_name: str,
        tables: list[StructuredTable],
    ) -> VerificationResult:
        """验证单个 (entity, metric, value) 三元组"""
        best_record: TableRecord | None = None
        best_table_id: str = ""
        best_match_score: float = 0.0

        for table in tables:
            record = table.lookup(
                entity_name,
                metric_name,
                fuzzy_threshold=self.config.entity_match_threshold,
            )
            if record and record.value is not None:
                from sciconsist_pilot.src.vsr.types import _levenshtein_similarity
                import re as _re
                e_norm = normalize_entity(entity_name)
                m_norm = _re.sub(r'[\s\-_]+', '', metric_name.lower()) if metric_name else ""
                e_sim = _levenshtein_similarity(e_norm, record.entity_normalized)
                m_sim = _levenshtein_similarity(m_norm, record.metric_normalized) if m_norm else 0.5
                match_score = e_sim * 0.7 + m_sim * 0.3

                if match_score > best_match_score:
                    best_match_score = match_score
                    best_record = record
                    best_table_id = table.table_id

        if best_record is None or best_record.value is None:
            return VerificationResult(
                layer=VerificationLayer.SYMBOLIC,
                reward=0.0,
                confidence=0.0,
                details={"reason": "no matching table record found"},
            )

        claim_val = numeric.value
        table_val = best_record.value

        abs_diff = abs(claim_val - table_val)
        rel_diff_pct = (abs_diff / max(abs(table_val), 1e-8)) * 100

        if abs_diff <= self.config.value_exact_tolerance:
            reward = self.config.exact_match_reward
            match_type = "exact"
        elif rel_diff_pct <= self.config.value_approx_tolerance_pct:
            reward = self.config.approx_match_reward
            match_type = "approximate"
        else:
            reward = self.config.mismatch_reward
            match_type = "mismatch"

        confidence = best_match_score

        return VerificationResult(
            layer=VerificationLayer.SYMBOLIC,
            reward=reward,
            confidence=confidence,
            matched_evidence=f"Table {best_table_id}: {best_record.entity}|{best_record.metric}={best_record.value}",
            details={
                "match_type": match_type,
                "claim_value": claim_val,
                "table_value": table_val,
                "abs_diff": round(abs_diff, 4),
                "rel_diff_pct": round(rel_diff_pct, 2),
                "entity_match_score": round(best_match_score, 3),
                "table_id": best_table_id,
            },
        )

    def _fallback(self, reason: str) -> VerificationResult:
        """生成 fallback 结果，提示调用方使用 Layer 2"""
        return VerificationResult(
            layer=VerificationLayer.SYMBOLIC,
            reward=0.0,
            confidence=0.0,
            details={"fallback": True, "reason": reason},
        )
