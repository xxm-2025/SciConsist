"""
Baseline Reward Functions — 用于与 VSR 对比的各类 reward 实现

所有 baseline 遵循统一接口:
  reward_fn(response: str, tables: list[StructuredTable],
            evidence_text: str, **kwargs) -> (float, dict)
  返回 (reward_score, metadata_dict)

实现的 baselines:
  1. SurfaceSimilarityReward — CycleReward-style 表面文本相似度
  2. LLMJudgeReward — GPT-4o / Qwen2.5-72B 打分 (需 API)
  3. GPT4oMultiAspectReward — 精细多维度 judge (strong baseline)
  4. HEROStyleReward — Binary verifier + learned RM 融合 (HERO-style)
  5. FEHHolisticReward — 原始 FEH holistic learned reward
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional, Callable

from sciconsist_pilot.src.vsr.types import StructuredTable, VerificationLayer
from sciconsist_pilot.src.vsr.claim_extractor import ClaimExtractor
from sciconsist_pilot.src.vsr.router import VerifiabilityRouter
from sciconsist_pilot.src.vsr.symbolic import SymbolicVerifier


# ── 1. Surface Similarity Reward (CycleReward-style) ─────────────


@dataclass
class SurfaceSimilarityConfig:
    """CycleReward-style 表面相似度 reward 配置

    Attributes:
        use_rouge: 是否用 ROUGE-L (否则用 token overlap)
        alpha: ROUGE-L 的 F-score α
    """
    use_rouge: bool = False
    alpha: float = 0.5


class SurfaceSimilarityReward:
    """CycleReward-style: response 与 evidence 的表面文本相似度

    最简单的 baseline — 直接计算 response 和 evidence_text 的 token 重叠度。
    缺陷: 语义等价但表述不同会被惩罚 (Many-to-One 退化)。
    """

    def __init__(self, config: SurfaceSimilarityConfig | None = None) -> None:
        self.config = config or SurfaceSimilarityConfig()

    def compute(
        self,
        response: str,
        tables: list[StructuredTable],
        evidence_text: str = "",
        **kwargs,
    ) -> tuple[float, dict]:
        """计算表面相似度 reward

        Args:
            response: 模型回答
            tables: (unused, 保持接口一致)
            evidence_text: 参考文本 (caption + context)

        Returns:
            (reward, metadata)
        """
        if not evidence_text:
            return 0.0, {"reason": "no evidence text"}

        resp_tokens = _tokenize(response)
        ev_tokens = _tokenize(evidence_text)

        if not resp_tokens or not ev_tokens:
            return 0.0, {"reason": "empty tokens"}

        # Token overlap F1
        resp_set = set(resp_tokens)
        ev_set = set(ev_tokens)
        overlap = resp_set & ev_set

        precision = len(overlap) / len(resp_set) if resp_set else 0
        recall = len(overlap) / len(ev_set) if ev_set else 0

        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)

        # 映射到 [-1, 1]: f1=0 → -0.5, f1=0.5 → 0.5, f1=1 → 1.0
        reward = f1 * 2 - 0.5

        return reward, {
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "f1": round(f1, 3),
            "overlap_tokens": len(overlap),
        }


# ── 2. LLM Judge Reward ──────────────────────────────────────────


LLM_JUDGE_PROMPT = """You are evaluating the factual accuracy of a response about a scientific paper.

Evidence (from the paper):
{evidence}

Model Response:
{response}

Rate the factual accuracy of the response on a scale of 0-10:
- 10: Completely accurate, all facts verified
- 7-9: Mostly accurate, minor issues
- 4-6: Partially accurate, some errors
- 1-3: Mostly inaccurate
- 0: Completely wrong or hallucinated

Output ONLY a JSON: {{"score": <0-10>, "reasoning": "<brief explanation>"}}"""


class LLMJudgeReward:
    """LLM-as-Judge reward — 调用外部 LLM API 打分

    需要提供一个 judge_fn callable 来实际调用 API。
    适用于 GPT-4o / Qwen2.5-72B 等大模型。

    Attributes:
        judge_fn: 调用 LLM 的函数, 签名: (prompt: str) -> str
        model_name: judge 模型名 (用于日志)
    """

    def __init__(
        self,
        judge_fn: Callable[[str], str],
        model_name: str = "gpt-4o",
    ) -> None:
        self.judge_fn = judge_fn
        self.model_name = model_name

    def compute(
        self,
        response: str,
        tables: list[StructuredTable],
        evidence_text: str = "",
        **kwargs,
    ) -> tuple[float, dict]:
        """调用 LLM judge 评分

        Returns:
            (reward, metadata) — reward 映射到 [-1, 1]
        """
        prompt = LLM_JUDGE_PROMPT.format(
            evidence=evidence_text[:2000],
            response=response[:1500],
        )

        try:
            raw_output = self.judge_fn(prompt)
            score, reasoning = _parse_judge_output(raw_output)
            reward = (score / 10.0) * 2 - 1  # [0,10] → [-1,1]
            return reward, {
                "judge_model": self.model_name,
                "raw_score": score,
                "reasoning": reasoning,
            }
        except Exception as e:
            return 0.0, {"error": str(e), "judge_model": self.model_name}


# ── 3. GPT-4o Multi-Aspect Judge ─────────────────────────────────


MULTI_ASPECT_PROMPT = """Evaluate this scientific claim on multiple dimensions.

Evidence: {evidence}
Response: {response}

Rate EACH dimension 0-10:
1. Numeric Precision: Are numbers/percentages exact?
2. Entity Accuracy: Are method/dataset names correct?
3. Relation Correctness: Are comparisons (A > B) correct?
4. Trend Accuracy: Are trends (increase/decrease) correct?
5. Completeness: Does the response cover key findings?

Output JSON: {{"numeric": <0-10>, "entity": <0-10>, "relation": <0-10>, "trend": <0-10>, "completeness": <0-10>, "reasoning": "<brief>"}}"""


class GPT4oMultiAspectReward:
    """GPT-4o Multi-Aspect Judge — 精细多维度评估 (strong baseline)

    跟 VSR 的三层结构对应, 但用 GPT-4o 替代符号验证。
    这是 VSR 需要打败的 strong baseline。

    Attributes:
        judge_fn: 调用 GPT-4o 的函数
        dimension_weights: 各维度权重
    """

    def __init__(
        self,
        judge_fn: Callable[[str], str],
        dimension_weights: dict[str, float] | None = None,
    ) -> None:
        self.judge_fn = judge_fn
        self.weights = dimension_weights or {
            "numeric": 0.3,
            "entity": 0.2,
            "relation": 0.2,
            "trend": 0.15,
            "completeness": 0.15,
        }

    def compute(
        self,
        response: str,
        tables: list[StructuredTable],
        evidence_text: str = "",
        **kwargs,
    ) -> tuple[float, dict]:
        """多维度评估

        Returns:
            (weighted_reward, per_dimension_scores)
        """
        prompt = MULTI_ASPECT_PROMPT.format(
            evidence=evidence_text[:2000],
            response=response[:1500],
        )

        try:
            raw = self.judge_fn(prompt)
            scores = _parse_multi_aspect_output(raw)
            weighted = sum(
                scores.get(dim, 5) * w
                for dim, w in self.weights.items()
            )
            reward = (weighted / 10.0) * 2 - 1
            return reward, {
                "judge": "gpt-4o-multi-aspect",
                "scores": scores,
                "weighted_raw": round(weighted, 2),
            }
        except Exception as e:
            return 0.0, {"error": str(e)}


# ── 4. HERO-style Hybrid Reward ──────────────────────────────────


class HEROStyleReward:
    """HERO-style: Binary symbolic verifier + dense learned RM

    HERO (ICLR 2026) 的多模态适配版:
    - 对可验证 claim 使用 binary symbolic verifier (correct/wrong)
    - 对不可验证 claim 使用 learned RM
    - 与 VSR 的区别: 只有二元 (可验/不可验), 没有三级 hierarchy

    Attributes:
        learned_rm: learned reward model callable
    """

    def __init__(
        self,
        learned_rm: Callable[[str, str], float] | None = None,
    ) -> None:
        self.extractor = ClaimExtractor()
        self.router = VerifiabilityRouter()
        self.symbolic = SymbolicVerifier()
        self.learned_rm = learned_rm

    def compute(
        self,
        response: str,
        tables: list[StructuredTable],
        evidence_text: str = "",
        **kwargs,
    ) -> tuple[float, dict]:
        """HERO-style 二元 hybrid reward

        Returns:
            (reward, metadata)
        """
        claims = self.extractor.extract(response)
        if not claims:
            return 0.0, {"n_claims": 0}

        decisions = self.router.route(claims)

        verifiable_rewards = []
        unverifiable_rewards = []

        for dec in decisions:
            is_verifiable = (
                VerificationLayer.SYMBOLIC in dec.layers
                or VerificationLayer.SEMI_SYMBOLIC in dec.layers
            )

            if is_verifiable and tables:
                result = self.symbolic.verify(dec.claim, tables)
                # HERO-style: binary (correct=+1, wrong=-1)
                if result.confidence > 0.5:
                    r = 1.0 if result.reward > 0 else -1.0
                    verifiable_rewards.append(r)
                else:
                    unverifiable_rewards.append(
                        self._learned_score(dec.claim.text, evidence_text)
                    )
            else:
                unverifiable_rewards.append(
                    self._learned_score(dec.claim.text, evidence_text)
                )

        # HERO aggregation: separate normalization then merge
        v_mean = (
            sum(verifiable_rewards) / len(verifiable_rewards)
            if verifiable_rewards else 0.0
        )
        u_mean = (
            sum(unverifiable_rewards) / len(unverifiable_rewards)
            if unverifiable_rewards else 0.0
        )

        # 按可验证 claim 占比加权
        total = len(verifiable_rewards) + len(unverifiable_rewards)
        v_ratio = len(verifiable_rewards) / total if total > 0 else 0
        reward = v_mean * v_ratio + u_mean * (1 - v_ratio)

        return reward, {
            "n_verifiable": len(verifiable_rewards),
            "n_unverifiable": len(unverifiable_rewards),
            "v_mean": round(v_mean, 3),
            "u_mean": round(u_mean, 3),
            "v_ratio": round(v_ratio, 3),
        }

    def _learned_score(self, claim_text: str, evidence: str) -> float:
        """调用 learned RM 或返回中性分"""
        if self.learned_rm:
            try:
                return float(self.learned_rm(claim_text, evidence))
            except Exception:
                pass
        return 0.0


# ── 5. FEH Holistic Reward ────────────────────────────────────────


class FEHHolisticReward:
    """原始 FEH holistic learned reward — SciConsist v1 的设计

    用 FactualEntailmentHead 对所有 claim 统一打分,
    不区分 claim 的 verifiability level。

    这是 "one-size-fits-all" 的典型代表, 用于证明 VSR 的分层优势。

    Attributes:
        feh_fn: FEH predict 函数, 签名:
            (claim_texts: list[str], evidence: str) -> list[float]
    """

    def __init__(
        self,
        feh_fn: Callable[[list[str], str], list[float]] | None = None,
    ) -> None:
        self.extractor = ClaimExtractor()
        self.feh_fn = feh_fn

    def compute(
        self,
        response: str,
        tables: list[StructuredTable],
        evidence_text: str = "",
        **kwargs,
    ) -> tuple[float, dict]:
        """FEH holistic reward

        Returns:
            (reward, metadata)
        """
        claims = self.extractor.extract(response)
        if not claims:
            return 0.0, {"n_claims": 0}

        claim_texts = [c.text for c in claims]

        if self.feh_fn:
            try:
                scores = self.feh_fn(claim_texts, evidence_text)
                reward = sum(scores) / len(scores)
                return reward, {
                    "n_claims": len(claims),
                    "claim_scores": [round(s, 3) for s in scores],
                }
            except Exception as e:
                return 0.0, {"error": str(e)}

        return 0.0, {"n_claims": len(claims), "note": "no FEH model loaded"}


# ── 工具函数 ──────────────────────────────────────────────────────

_TOKEN_SPLIT = re.compile(r'\W+')


def _tokenize(text: str) -> list[str]:
    """简单 whitespace + 标点 tokenizer"""
    return [t.lower() for t in _TOKEN_SPLIT.split(text) if len(t) > 1]


def _parse_judge_output(raw: str) -> tuple[int, str]:
    """解析 LLM judge 的 JSON 输出"""
    import json
    try:
        obj = json.loads(raw)
        return int(obj.get("score", 5)), str(obj.get("reasoning", ""))
    except (json.JSONDecodeError, ValueError):
        # fallback: 从文本中提取数字
        nums = re.findall(r'\b(\d+)\b', raw)
        score = int(nums[0]) if nums else 5
        return min(max(score, 0), 10), raw[:200]


def _parse_multi_aspect_output(raw: str) -> dict[str, float]:
    """解析 multi-aspect judge 的 JSON 输出"""
    import json
    try:
        obj = json.loads(raw)
        return {
            k: float(obj.get(k, 5))
            for k in ["numeric", "entity", "relation", "trend", "completeness"]
        }
    except (json.JSONDecodeError, ValueError):
        return {k: 5.0 for k in ["numeric", "entity", "relation", "trend", "completeness"]}
