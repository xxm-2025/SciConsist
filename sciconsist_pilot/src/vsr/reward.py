"""
VSR Reward Aggregator — 核心 reward function

将 Layer 0/1/2 的验证结果聚合为最终 reward，
同时附加 coverage bonus 和 specificity bonus。

公式:
  R_VSR(τ) = (1/K) Σᵢ R(cᵢ)
           + λ_coverage  * coverage_bonus(τ)
           + λ_specific  * specificity_bonus(τ)

其中:
  R(cᵢ) = Σⱼ wⱼ * rⱼ(cᵢ)   (j ∈ {sym, semi, learned})

关键设计:
  - Layer 0/1 高置信时覆盖 Layer 2 (符号锚定)
  - 低置信 Layer 0/1 fallback 到 Layer 2
  - Specificity bonus 鼓励输出可验证的具体 claim
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Callable

from sciconsist_pilot.src.vsr.types import (
    AtomicClaim,
    StructuredTable,
    VerificationResult,
    VerificationLayer,
)
from sciconsist_pilot.src.vsr.router import RoutingDecision
from sciconsist_pilot.src.vsr.symbolic import SymbolicVerifier, SymbolicVerifierConfig
from sciconsist_pilot.src.vsr.semi_symbolic import SemiSymbolicVerifier, SemiSymbolicVerifierConfig
from sciconsist_pilot.src.vsr.claim_extractor import ClaimExtractor
from sciconsist_pilot.src.vsr.router import VerifiabilityRouter
from sciconsist_pilot.src.vsr.learned import FEHVerifier, FEHVerifierConfig


@dataclass
class VSRConfig:
    """VSR Reward 聚合配置

    Attributes:
        lambda_coverage: coverage bonus 权重
        lambda_specificity: specificity bonus 权重
        layer_weights: 各层在多层共触发时的权重
        fallback_confidence_gate: Layer 0/1 低于此置信度时 fallback
        symbolic_config: Layer 0 配置
        semi_symbolic_config: Layer 1 配置
        feh_config: Layer 2 FEH 配置 (None 时使用 text_only 模式)
    """
    lambda_coverage: float = 0.1
    lambda_specificity: float = 0.05
    layer_weights: dict[int, float] = field(default_factory=lambda: {0: 0.5, 1: 0.3, 2: 0.2})
    fallback_confidence_gate: float = 0.7
    symbolic_config: SymbolicVerifierConfig = field(default_factory=SymbolicVerifierConfig)
    semi_symbolic_config: SemiSymbolicVerifierConfig = field(default_factory=SemiSymbolicVerifierConfig)
    feh_config: FEHVerifierConfig = field(default_factory=FEHVerifierConfig)


@dataclass
class VSRRewardOutput:
    """VSR Reward 完整输出

    Attributes:
        total_reward: 最终 reward 分数
        base_reward: 基础分 (claims 平均)
        coverage_bonus: 覆盖度奖励
        specificity_bonus: 可验证性奖励
        claim_rewards: 每条 claim 的详细验证结果
        layer_distribution: 各层覆盖率
        num_claims: claim 总数
    """
    total_reward: float
    base_reward: float
    coverage_bonus: float = 0.0
    specificity_bonus: float = 0.0
    claim_rewards: list[dict] = field(default_factory=list)
    layer_distribution: dict[str, float] = field(default_factory=dict)
    num_claims: int = 0


class VSRReward:
    """Verification-Stratified Reward — 主入口

    完整流程:
      response → ClaimExtractor → Router → Layer 0/1/2 → Aggregator

    用法:
        vsr = VSRReward()
        output = vsr.compute(response, tables)
        reward = output.total_reward
    """

    def __init__(
        self,
        config: VSRConfig | None = None,
        learned_verifier: Optional[Callable] = None,
    ) -> None:
        """
        Args:
            config: VSR 配置
            learned_verifier: Layer 2 learned verifier callable
                签名: (claim_text: str, evidence: str) -> float (reward)
                不提供时自动根据 config.feh_config 创建 FEHVerifier
        """
        self.config = config or VSRConfig()
        self.extractor = ClaimExtractor()
        self.router = VerifiabilityRouter()
        self.symbolic = SymbolicVerifier(self.config.symbolic_config)
        self.semi_symbolic = SemiSymbolicVerifier(self.config.semi_symbolic_config)

        if learned_verifier is not None:
            self.learned_verifier = learned_verifier
        else:
            self.learned_verifier = FEHVerifier(self.config.feh_config)

    def compute(
        self,
        response: str,
        tables: list[StructuredTable],
        evidence_text: str = "",
        paper_id: str = "",
        image_path: str = "",
    ) -> VSRRewardOutput:
        """计算一条 response 的 VSR reward

        Args:
            response: 模型生成的完整回答
            tables: 该论文的结构化表格
            evidence_text: 额外证据文本 (caption + references, Layer 2 用)
            paper_id: 论文 ID (Layer 2 cached 模式用于检索视觉缓存)
            image_path: 图像路径 (Layer 2 full 模式可选)

        Returns:
            VSRRewardOutput 包含总 reward 和各层详情
        """
        claims = self.extractor.extract(response)
        if not claims:
            return VSRRewardOutput(
                total_reward=0.0,
                base_reward=0.0,
                num_claims=0,
            )

        decisions = self.router.route(claims)
        layer_dist = self.router.get_layer_distribution(decisions)

        claim_rewards: list[dict] = []
        total_claim_reward = 0.0

        for decision in decisions:
            claim_reward, details = self._compute_claim_reward(
                decision, tables, evidence_text, paper_id, image_path
            )
            claim_rewards.append(details)
            total_claim_reward += claim_reward

        num_claims = len(claims)
        base_reward = total_claim_reward / num_claims

        cov_bonus = self._coverage_bonus(claims, tables) * self.config.lambda_coverage
        spec_bonus = self._specificity_bonus(decisions) * self.config.lambda_specificity

        total = base_reward + cov_bonus + spec_bonus

        return VSRRewardOutput(
            total_reward=total,
            base_reward=base_reward,
            coverage_bonus=cov_bonus,
            specificity_bonus=spec_bonus,
            claim_rewards=claim_rewards,
            layer_distribution=layer_dist,
            num_claims=num_claims,
        )

    def compute_batch(
        self,
        responses: list[str],
        tables_list: list[list[StructuredTable]],
        evidence_texts: list[str] | None = None,
        paper_ids: list[str] | None = None,
        image_paths: list[str] | None = None,
    ) -> list[VSRRewardOutput]:
        """批量计算

        Args:
            responses: response 列表
            tables_list: 每条 response 对应的表格列表
            evidence_texts: 每条 response 的证据文本
            paper_ids: 每条 response 对应的 paper_id
            image_paths: 每条 response 对应的 image_path

        Returns:
            VSRRewardOutput 列表
        """
        if evidence_texts is None:
            evidence_texts = [""] * len(responses)
        if paper_ids is None:
            paper_ids = [""] * len(responses)
        if image_paths is None:
            image_paths = [""] * len(responses)
        return [
            self.compute(resp, tables, ev, pid, img)
            for resp, tables, ev, pid, img in zip(
                responses, tables_list, evidence_texts, paper_ids, image_paths
            )
        ]

    def _compute_claim_reward(
        self,
        decision: RoutingDecision,
        tables: list[StructuredTable],
        evidence_text: str,
        paper_id: str = "",
        image_path: str = "",
    ) -> tuple[float, dict]:
        """计算单条 claim 的聚合 reward

        对路由到的每一层执行验证，按层权重加权聚合。
        Layer 0/1 低置信度时自动 fallback 到 Layer 2。

        Returns:
            (reward_score, details_dict)
        """
        claim = decision.claim
        layers = decision.layers
        layer_results: dict[int, VerificationResult] = {}

        # 执行各层验证
        if VerificationLayer.SYMBOLIC in layers:
            r = self.symbolic.verify(claim, tables)
            if r.confidence >= self.config.fallback_confidence_gate:
                layer_results[0] = r
            else:
                # fallback 到 Layer 2
                layers = [l for l in layers if l != VerificationLayer.SYMBOLIC]
                if VerificationLayer.LEARNED not in layers:
                    layers.append(VerificationLayer.LEARNED)

        if VerificationLayer.SEMI_SYMBOLIC in layers:
            r = self.semi_symbolic.verify(claim, tables)
            if r.confidence >= self.config.fallback_confidence_gate:
                layer_results[1] = r
            else:
                if VerificationLayer.LEARNED not in layers:
                    layers.append(VerificationLayer.LEARNED)

        if VerificationLayer.LEARNED in layers:
            r = self._run_learned(
                claim.text,
                evidence_text,
                paper_id=paper_id,
                image_path=image_path,
            )
            layer_results[2] = r

        if not layer_results:
            return 0.0, {"claim": claim.text[:100], "layers": [], "reward": 0.0}

        # 加权聚合
        weighted_sum = 0.0
        weight_total = 0.0
        for layer_id, result in layer_results.items():
            w = self.config.layer_weights.get(layer_id, 0.2)
            weighted_sum += w * result.reward
            weight_total += w

        reward = weighted_sum / weight_total if weight_total > 0 else 0.0

        details = {
            "claim": claim.text[:150],
            "layers": [
                {
                    "layer": r.layer.name,
                    "reward": round(r.reward, 3),
                    "confidence": round(r.confidence, 3),
                    "evidence": r.matched_evidence[:100],
                }
                for r in layer_results.values()
            ],
            "reward": round(reward, 4),
        }

        return reward, details

    def _run_learned(
        self,
        claim_text: str,
        evidence_text: str,
        paper_id: str = "",
        image_path: str = "",
    ) -> VerificationResult:
        """Layer 2: learned verifier 或 neutral fallback"""
        if self.learned_verifier:
            try:
                try:
                    reward = self.learned_verifier(
                        claim_text,
                        evidence_text,
                        paper_id=paper_id,
                        image_path=image_path,
                    )
                except TypeError:
                    # 兼容仅支持 (claim_text, evidence_text) 的旧版 verifier
                    reward = self.learned_verifier(claim_text, evidence_text)
                return VerificationResult(
                    layer=VerificationLayer.LEARNED,
                    reward=float(reward),
                    confidence=0.7,
                    matched_evidence="learned_verifier",
                )
            except Exception:
                pass

        return VerificationResult(
            layer=VerificationLayer.LEARNED,
            reward=0.0,
            confidence=0.5,
            details={"note": "no learned verifier, neutral score"},
        )

    def _coverage_bonus(
        self,
        claims: list[AtomicClaim],
        tables: list[StructuredTable],
    ) -> float:
        """Coverage bonus: response 覆盖了多少比例的 table 数据点

        鼓励模型引用更多的表格数据，而非只挑选部分。
        """
        if not tables:
            return 0.0

        total_records = sum(len(t.records) for t in tables)
        if total_records == 0:
            return 0.0

        # 统计 claims 中提到的 entity + metric 对
        mentioned_pairs: set[tuple[str, str]] = set()
        for claim in claims:
            for entity in claim.entities:
                for metric in claim.metrics:
                    mentioned_pairs.add((entity.normalized, metric.lower()))

        # 统计表格中被覆盖的记录
        covered = 0
        for table in tables:
            for rec in table.records:
                for ent_norm, met in mentioned_pairs:
                    if (ent_norm in rec.entity_normalized or
                            rec.entity_normalized in ent_norm):
                        covered += 1
                        break

        return min(covered / total_records, 1.0)

    def _specificity_bonus(
        self,
        decisions: list[RoutingDecision],
    ) -> float:
        """Specificity bonus: 可符号验证 claim 的占比

        鼓励模型输出具体、可核查的声明而非模糊废话。
        """
        if not decisions:
            return 0.0

        symbolic_count = sum(
            1 for d in decisions
            if VerificationLayer.SYMBOLIC in d.layers
            or VerificationLayer.SEMI_SYMBOLIC in d.layers
        )
        return symbolic_count / len(decisions)
