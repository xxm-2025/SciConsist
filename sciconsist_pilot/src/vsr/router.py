"""
Verifiability Router — 将 atomic claims 路由到对应验证层

路由规则:
  - Layer 0 (Symbolic): claim 含数值 + 实体/指标 → 精确验证
  - Layer 1 (Semi-Symbolic): claim 含比较关系或趋势 → 逻辑验证
  - Layer 2 (Learned): 纯定性或无可验维度 → 语义判断

一个 claim 可同时被路由到多个层 (例如 "A achieves 85.3%, outperforming B"
同时触发 Layer 0 和 Layer 1)。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from sciconsist_pilot.src.vsr.types import (
    AtomicClaim,
    VerifiabilityProfile,
    VerificationLayer,
)


@dataclass
class RoutingDecision:
    """路由决策

    Attributes:
        claim: 原始 claim
        profile: 可验证性特征
        layers: 路由到的验证层列表 (按优先级排序)
    """
    claim: AtomicClaim
    profile: VerifiabilityProfile
    layers: list[VerificationLayer] = field(default_factory=list)


class VerifiabilityRouter:
    """Verifiability Router — 无需训练的规则系统

    用法:
        router = VerifiabilityRouter()
        decisions = router.route(claims)
    """

    def route(self, claims: list[AtomicClaim]) -> list[RoutingDecision]:
        """对一组 claims 执行路由

        Args:
            claims: ClaimExtractor 输出的 AtomicClaim 列表

        Returns:
            RoutingDecision 列表, 每条对应一个 claim 的路由决策
        """
        return [self._route_single(claim) for claim in claims]

    def _route_single(self, claim: AtomicClaim) -> RoutingDecision:
        """路由单条 claim"""
        profile = self._build_profile(claim)
        layers = self._decide_layers(profile)

        profile.assigned_layers = layers

        return RoutingDecision(
            claim=claim,
            profile=profile,
            layers=layers,
        )

    def _build_profile(self, claim: AtomicClaim) -> VerifiabilityProfile:
        """从 claim 结构构建 verifiability profile"""
        has_numeric = len(claim.numeric_values) > 0
        has_entity_metric = (
            len(claim.entities) > 0 and len(claim.metrics) > 0
        )
        has_comparison = len(claim.relations) > 0
        has_trend = len(claim.trends) > 0
        is_qualitative = claim.is_qualitative

        return VerifiabilityProfile(
            has_numeric=has_numeric,
            has_entity_metric=has_entity_metric,
            has_comparison=has_comparison,
            has_trend=has_trend,
            is_qualitative_only=is_qualitative,
        )

    def _decide_layers(self, profile: VerifiabilityProfile) -> list[VerificationLayer]:
        """根据 profile 决定路由层

        一个 claim 可路由到多层:
          - 有数值+实体/指标 → 加入 Layer 0
          - 有比较/趋势 → 加入 Layer 1
          - 纯定性或以上均无 → Layer 2

        Layer 0/1 都触发时，两层都执行，取更严格的 reward。
        """
        layers: list[VerificationLayer] = []

        if profile.layer0_eligible:
            layers.append(VerificationLayer.SYMBOLIC)

        if profile.layer1_eligible:
            layers.append(VerificationLayer.SEMI_SYMBOLIC)

        if not layers or profile.is_qualitative_only:
            layers.append(VerificationLayer.LEARNED)

        return layers

    def get_layer_distribution(
        self, decisions: list[RoutingDecision]
    ) -> dict[str, float]:
        """统计路由分布

        Args:
            decisions: 路由决策列表

        Returns:
            各层覆盖率百分比
        """
        total = len(decisions)
        if total == 0:
            return {"layer_0": 0.0, "layer_1": 0.0, "layer_2": 0.0}

        l0 = sum(1 for d in decisions if VerificationLayer.SYMBOLIC in d.layers)
        l1 = sum(1 for d in decisions if VerificationLayer.SEMI_SYMBOLIC in d.layers)
        l2 = sum(1 for d in decisions if VerificationLayer.LEARNED in d.layers)

        return {
            "layer_0": l0 / total * 100,
            "layer_1": l1 / total * 100,
            "layer_2": l2 / total * 100,
            "layer_01_combined": (l0 + l1 - sum(
                1 for d in decisions
                if VerificationLayer.SYMBOLIC in d.layers
                and VerificationLayer.SEMI_SYMBOLIC in d.layers
            )) / total * 100,
        }
