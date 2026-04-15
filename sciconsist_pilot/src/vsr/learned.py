"""
VSR Layer 2: Learned Verifier — FEH 集成封装

将 FactualEntailmentHead + FeatureExtractor 封装为 VSRReward 可用的 callable。
支持三种运行模式:
  1. full: 加载 InternVL2.5 实时提取特征 + FEH 分类 (最准, 最慢)
  2. cached: 加载预提取的视觉特征 + 实时提取文本特征 + FEH 分类 (训练时用)
  3. text_only: 基于 token overlap + 语义相似度的轻量 NLI (无需 GPU)

当 FEH checkpoint 不存在时自动降级为 text_only 模式。

用法:
    # 最简: text_only 模式 (不依赖任何模型)
    verifier = FEHVerifier.create(mode="text_only")
    reward = verifier("The method achieves 89.4% accuracy", "table evidence...")

    # 完整: cached 模式 (GRPO 训练时)
    verifier = FEHVerifier.create(
        mode="cached",
        feh_checkpoint="/path/to/feh.pt",
        visual_cache_dir="/path/to/visual_features/",
    )
    reward = verifier("The method achieves 89.4% accuracy", "table evidence...",
                       paper_id="2405.02951v1")
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class FEHVerifierConfig:
    """FEH Verifier 配置

    Attributes:
        mode: 运行模式 ("full" / "cached" / "text_only")
        feh_checkpoint: FEH 模型 checkpoint 路径
        extractor_model: InternVL 模型名 (full 模式用)
        visual_cache_dir: 预提取视觉特征目录 (cached 模式用)
        device: 计算设备
        reward_entails: ENTAILS 的 reward
        reward_neutral: NEUTRAL 的 reward
        reward_contradicts: CONTRADICTS 的 reward
        text_similarity_weight: text_only 模式中 token overlap 的权重
        nli_weight: text_only 模式中 NLI 打分的权重
    """
    mode: str = "text_only"
    feh_checkpoint: str = ""
    extractor_model: str = "OpenGVLab/InternVL2_5-8B"
    visual_cache_dir: str = ""
    device: str = "cuda"
    reward_entails: float = 1.0
    reward_neutral: float = 0.0
    reward_contradicts: float = -0.8
    text_similarity_weight: float = 0.6
    nli_weight: float = 0.4


class FEHVerifier:
    """Layer 2 Learned Verifier — 封装 FEH 为 VSRReward callable

    实现 __call__(claim_text, evidence_text, **kwargs) -> float 接口，
    可直接传给 VSRReward(learned_verifier=verifier)。
    """

    def __init__(self, config: FEHVerifierConfig | None = None) -> None:
        self.config = config or FEHVerifierConfig()
        self._feh = None
        self._extractor = None
        self._visual_cache: dict[str, np.ndarray] = {}
        self._loaded = False

    @classmethod
    def create(
        cls,
        mode: str = "text_only",
        feh_checkpoint: str = "",
        visual_cache_dir: str = "",
        device: str = "cuda",
        **kwargs: Any,
    ) -> "FEHVerifier":
        """工厂方法，根据 mode 自动配置。

        Args:
            mode: "full" / "cached" / "text_only"
            feh_checkpoint: FEH checkpoint 路径
            visual_cache_dir: 视觉特征缓存目录
            device: 计算设备
            **kwargs: 额外配置参数

        Returns:
            配置好的 FEHVerifier 实例
        """
        config = FEHVerifierConfig(
            mode=mode,
            feh_checkpoint=feh_checkpoint,
            visual_cache_dir=visual_cache_dir,
            device=device,
            **{k: v for k, v in kwargs.items() if hasattr(FEHVerifierConfig, k)},
        )

        if mode in ("full", "cached") and feh_checkpoint and not Path(feh_checkpoint).exists():
            logger.warning(
                f"FEH checkpoint not found: {feh_checkpoint}, falling back to text_only mode"
            )
            config.mode = "text_only"

        return cls(config)

    def _lazy_load(self) -> None:
        """延迟加载模型 (首次调用时触发)。"""
        if self._loaded:
            return
        self._loaded = True

        if self.config.mode == "text_only":
            logger.info("FEHVerifier: text_only mode (no model loading)")
            return

        import torch
        from sciconsist_pilot.src.models.feh import FactualEntailmentHead, FEHConfig

        if self.config.feh_checkpoint and Path(self.config.feh_checkpoint).exists():
            logger.info(f"Loading FEH from {self.config.feh_checkpoint}")
            state = torch.load(self.config.feh_checkpoint, map_location=self.config.device)
            feh_config = state.get("config", FEHConfig())
            if isinstance(feh_config, dict):
                feh_config = FEHConfig(**feh_config)
            self._feh = FactualEntailmentHead(feh_config)
            self._feh.load_state_dict(state["model_state_dict"])
            self._feh.to(self.config.device)
            self._feh.eval()
        else:
            logger.warning("No FEH checkpoint, using untrained FEH (random weights)")
            self._feh = FactualEntailmentHead()
            self._feh.to(self.config.device)
            self._feh.eval()

        if self.config.mode == "cached" and self.config.visual_cache_dir:
            self._load_visual_cache()

        if self.config.mode == "full":
            from sciconsist_pilot.src.features.extract import ExtractionConfig, FeatureExtractor
            ext_config = ExtractionConfig(
                model_name=self.config.extractor_model,
                device=self.config.device,
            )
            self._extractor = FeatureExtractor(ext_config)

    def _load_visual_cache(self) -> None:
        """加载预提取的视觉特征。

        目录结构: visual_cache_dir/{paper_id}.npy -> (hidden_dim,) float32
        """
        cache_dir = Path(self.config.visual_cache_dir)
        if not cache_dir.exists():
            logger.warning(f"Visual cache dir not found: {cache_dir}")
            return
        count = 0
        for npy_file in cache_dir.glob("*.npy"):
            paper_id = npy_file.stem
            self._visual_cache[paper_id] = np.load(npy_file)
            count += 1
        logger.info(f"Loaded {count} visual feature vectors from {cache_dir}")

    def __call__(
        self,
        claim_text: str,
        evidence_text: str,
        paper_id: str = "",
        image_path: str = "",
    ) -> float:
        """计算 claim 的 learned verification reward。

        Args:
            claim_text: 待验证的 claim 文本
            evidence_text: 证据文本 (表格 caption / paper excerpt)
            paper_id: 论文 ID (cached 模式用于查找视觉特征)
            image_path: 图片路径 (full 模式用)

        Returns:
            reward 分数, 范围 [-1.0, 1.0]
        """
        self._lazy_load()

        if self.config.mode == "text_only":
            return self._text_only_verify(claim_text, evidence_text)

        if self.config.mode == "cached":
            return self._cached_verify(claim_text, evidence_text, paper_id)

        return self._full_verify(claim_text, evidence_text, image_path)

    # ── text_only 模式 ──────────────────────────────────────────

    def _text_only_verify(self, claim_text: str, evidence_text: str) -> float:
        """基于文本特征的轻量验证。

        融合 token overlap F1 + 关键实体覆盖 + 数值一致性检查。
        不需要任何模型，纯规则。
        """
        if not evidence_text.strip():
            return 0.0

        f1 = self._token_overlap_f1(claim_text, evidence_text)
        entity_score = self._entity_coverage(claim_text, evidence_text)
        numeric_score = self._numeric_consistency(claim_text, evidence_text)

        combined = (
            0.4 * f1
            + 0.3 * entity_score
            + 0.3 * numeric_score
        )

        # 映射到 reward 范围: [0, 1] -> [contradicts, entails]
        if combined >= 0.5:
            reward = self.config.reward_entails * min(combined * 1.5, 1.0)
        elif combined >= 0.2:
            reward = self.config.reward_neutral
        else:
            reward = self.config.reward_contradicts * (1.0 - combined * 3)

        return max(-1.0, min(1.0, reward))

    @staticmethod
    def _token_overlap_f1(text_a: str, text_b: str) -> float:
        """Token-level F1 overlap score。"""
        tokens_a = set(re.findall(r'\w+', text_a.lower()))
        tokens_b = set(re.findall(r'\w+', text_b.lower()))
        if not tokens_a or not tokens_b:
            return 0.0
        common = tokens_a & tokens_b
        if not common:
            return 0.0
        precision = len(common) / len(tokens_a)
        recall = len(common) / len(tokens_b)
        return 2 * precision * recall / (precision + recall)

    @staticmethod
    def _entity_coverage(claim_text: str, evidence_text: str) -> float:
        """检查 claim 中的关键实体是否出现在 evidence 中。

        识别大写开头的词组(方法名/模型名)和数据集名。
        """
        entity_pattern = re.compile(
            r'\b([A-Z][A-Za-z0-9]*(?:[-_][A-Za-z0-9]+)*)\b'
        )
        claim_entities = set(e.lower() for e in entity_pattern.findall(claim_text))
        if not claim_entities:
            return 0.5  # 没有实体默认中性

        evidence_lower = evidence_text.lower()
        covered = sum(1 for e in claim_entities if e in evidence_lower)
        return covered / len(claim_entities)

    @staticmethod
    def _numeric_consistency(claim_text: str, evidence_text: str) -> float:
        """检查 claim 中的数值是否出现在 evidence 中。"""
        num_pattern = re.compile(r'(\d+\.?\d*)\s*%?')
        claim_nums = set(num_pattern.findall(claim_text))
        if not claim_nums:
            return 0.5  # 没有数值默认中性

        evidence_nums = set(num_pattern.findall(evidence_text))
        if not evidence_nums:
            return 0.3  # evidence 无数值, 偏负面

        matched = sum(1 for n in claim_nums if n in evidence_nums)
        return matched / len(claim_nums)

    # ── cached 模式 ─────────────────────────────────────────────

    def _cached_verify(
        self, claim_text: str, evidence_text: str, paper_id: str
    ) -> float:
        """使用预提取视觉特征 + 实时文本特征 + FEH 分类。"""
        import torch

        if not self._feh:
            return self._text_only_verify(claim_text, evidence_text)

        h_visual = self._get_visual_feature(paper_id)
        if h_visual is None:
            return self._text_only_verify(claim_text, evidence_text)

        h_text = self._extract_text_feature(claim_text)
        if h_text is None:
            return self._text_only_verify(claim_text, evidence_text)

        h_v = torch.from_numpy(h_visual).unsqueeze(0).to(self.config.device)
        h_t = torch.from_numpy(h_text).unsqueeze(0).to(self.config.device)

        return self._feh_to_reward(h_v, h_t)

    def _get_visual_feature(self, paper_id: str) -> Optional[np.ndarray]:
        """从缓存获取视觉特征。"""
        if paper_id in self._visual_cache:
            return self._visual_cache[paper_id]
        paper_id_clean = paper_id.replace("/", "_").replace("\\", "_")
        return self._visual_cache.get(paper_id_clean)

    def _extract_text_feature(self, text: str) -> Optional[np.ndarray]:
        """实时提取文本特征。"""
        if self._extractor is None:
            from sciconsist_pilot.src.features.extract import ExtractionConfig, FeatureExtractor
            ext_config = ExtractionConfig(
                model_name=self.config.extractor_model,
                device=self.config.device,
            )
            self._extractor = FeatureExtractor(ext_config)

        try:
            features = self._extractor.extract_text_features([text])
            return features[0]
        except Exception as e:
            logger.warning(f"Text feature extraction failed: {e}")
            return None

    # ── full 模式 ───────────────────────────────────────────────

    def _full_verify(
        self, claim_text: str, evidence_text: str, image_path: str
    ) -> float:
        """完整 VLM 特征提取 + FEH 分类。"""
        import torch

        if not self._feh or not self._extractor:
            return self._text_only_verify(claim_text, evidence_text)

        if not image_path or not Path(image_path).exists():
            return self._text_only_verify(claim_text, evidence_text)

        try:
            text_feats = self._extractor.extract_text_features([claim_text])
            visual_feats = self._extractor.extract_visual_features([image_path])
            h_v = torch.from_numpy(visual_feats[0]).unsqueeze(0).to(self.config.device)
            h_t = torch.from_numpy(text_feats[0]).unsqueeze(0).to(self.config.device)
            return self._feh_to_reward(h_v, h_t)
        except Exception as e:
            logger.warning(f"Full FEH verify failed: {e}")
            return self._text_only_verify(claim_text, evidence_text)

    # ── 共用: FEH 输出 → reward ──────────────────────────────────

    def _feh_to_reward(self, h_visual: "torch.Tensor", h_text: "torch.Tensor") -> float:
        """FEH 前向推理并映射为 reward 值。

        Args:
            h_visual: 视觉特征 (1, hidden_dim)
            h_text: 文本特征 (1, hidden_dim)

        Returns:
            reward 分数
        """
        import torch

        with torch.no_grad():
            labels, probs = self._feh.predict(h_visual, h_text)

        label = labels[0].item()
        prob = probs[0]

        reward_map = {
            0: self.config.reward_entails,
            1: self.config.reward_neutral,
            2: self.config.reward_contradicts,
        }
        base = reward_map.get(label, 0.0)
        confidence = prob[label].item()

        return base * confidence
