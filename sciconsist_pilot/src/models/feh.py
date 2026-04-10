"""
Factual Entailment Head (FEH) — 核心模型。

在独立 VLM (InternVL2.5) 的隐空间上做事实蕴含三分类 (ENTAILS / NEUTRAL / CONTRADICTS)。
采用 multi-layer fusion + element-wise difference 特征拼接，
为 GRPO 提供 cross-model reward signal，杜绝 reward hacking。

典型用法:
    extractor = FeatureExtractor(model_name="OpenGVLab/InternVL2_5-8B", ...)
    feh = FactualEntailmentHead(hidden_dim=4096, latent_dim=512)
    h_visual, h_text = extractor.extract(figure, claim)
    logits, diff = feh(h_visual, h_text)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import torch
import torch.nn as nn
import torch.nn.functional as F


class EntailmentLabel(IntEnum):
    """FEH 三分类标签。"""

    ENTAILS = 0
    NEUTRAL = 1
    CONTRADICTS = 2


@dataclass
class FEHConfig:
    """FEH 模型配置。

    Attributes:
        hidden_dim: 上游 VLM 的 hidden state 维度 (InternVL2.5-8B = 4096)
        latent_dim: FEH 内部投影维度
        num_classes: 分类数 (固定为 3)
        dropout: Dropout 概率
        fuse_layer_indices: multi-layer fusion 使用的层号列表 (0-indexed)
    """

    hidden_dim: int = 4096
    latent_dim: int = 512
    num_classes: int = 3
    dropout: float = 0.1
    fuse_layer_indices: list[int] | None = None


class FactualEntailmentHead(nn.Module):
    """Cross-Model Factual Entailment Head。

    在独立 VLM (InternVL2.5) 隐空间上做事实蕴含三分类。
    特征提取与 policy model 完全解耦，杜绝 reward hacking。

    架构:
        visual_hidden → Visual Projector → h_v
        text_hidden   → Text Projector   → h_t
        [h_v; h_t; |h_v - h_t|] → Classifier → {ENTAILS, NEUTRAL, CONTRADICTS}

    Args:
        config: FEH 模型配置
    """

    def __init__(self, config: FEHConfig | None = None) -> None:
        super().__init__()
        if config is None:
            config = FEHConfig()
        self.config = config

        self.visual_proj = nn.Sequential(
            nn.Linear(config.hidden_dim, config.latent_dim),
            nn.GELU(),
            nn.Linear(config.latent_dim, config.latent_dim),
        )
        self.text_proj = nn.Sequential(
            nn.Linear(config.hidden_dim, config.latent_dim),
            nn.GELU(),
            nn.Linear(config.latent_dim, config.latent_dim),
        )
        # 输入维度 = latent_dim * 3: [h_v; h_t; |h_v - h_t|]
        self.classifier = nn.Sequential(
            nn.Linear(config.latent_dim * 3, config.latent_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.latent_dim, config.num_classes),
        )

    def forward(
        self,
        h_visual: torch.Tensor,
        h_text: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """前向传播。

        Args:
            h_visual: 视觉特征 (B, hidden_dim)，来自 InternVL 的 multi-layer fused representation
            h_text: 文本特征 (B, hidden_dim)，来自 InternVL 的 multi-layer fused representation

        Returns:
            logits: 三分类 logits (B, 3)
            diff: 差异向量 (B, latent_dim)，可用于 CONTRADICTS 时的 attention heatmap 可视化
        """
        v = self.visual_proj(h_visual)
        t = self.text_proj(h_text)
        diff = torch.abs(v - t)
        combined = torch.cat([v, t, diff], dim=-1)
        logits = self.classifier(combined)
        return logits, diff

    def predict(self, h_visual: torch.Tensor, h_text: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """推理时使用，返回预测标签和置信度。

        Args:
            h_visual: 视觉特征 (B, hidden_dim)
            h_text: 文本特征 (B, hidden_dim)

        Returns:
            labels: 预测标签 (B,)
            probs: 三分类概率 (B, 3)
        """
        logits, _ = self.forward(h_visual, h_text)
        probs = F.softmax(logits, dim=-1)
        labels = logits.argmax(dim=-1)
        return labels, probs

    @staticmethod
    def fuse_layers(
        all_hidden_states: tuple[torch.Tensor, ...],
        layer_indices: list[int],
    ) -> torch.Tensor:
        """Multi-layer fusion：取指定层 hidden states 的均值。

        Args:
            all_hidden_states: 模型所有层的 hidden states, 形状各为 (B, seq_len, hidden_dim)
            layer_indices: 要融合的层号列表 (0-indexed)

        Returns:
            fused: 融合后的表示 (B, seq_len, hidden_dim)
        """
        selected = torch.stack([all_hidden_states[i] for i in layer_indices])
        return selected.mean(dim=0)

    @staticmethod
    def pool_sequence(hidden_states: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        """对序列维度做 mean pooling，得到固定长度表示。

        Args:
            hidden_states: (B, seq_len, hidden_dim)
            attention_mask: (B, seq_len)，1 表示有效 token，0 表示 padding

        Returns:
            pooled: (B, hidden_dim)
        """
        if attention_mask is None:
            return hidden_states.mean(dim=1)
        mask = attention_mask.unsqueeze(-1).float()
        return (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)


class FEHReward:
    """基于 FEH 的 Reward 计算器，用于 GRPO 训练。

    封装了 tri-state reward + cross-modal bonus + conflict detection + informativeness gate。

    Args:
        feh: 训练好的 FEH 模型 (frozen)
        reward_entails: ENTAILS 的 reward 分数
        reward_neutral: NEUTRAL 的 reward 分数 (v3.1: -0.1)
        reward_contradicts: CONTRADICTS 的 reward 分数
        lambda_cross: 跨模态 bonus 权重
        lambda_conflict: 冲突检测 bonus 权重
        lambda_info: informativeness gate 权重
        min_entails_ratio: informativeness gate 阈值
    """

    REWARD_MAP = {
        EntailmentLabel.ENTAILS: 1.0,
        EntailmentLabel.NEUTRAL: -0.1,
        EntailmentLabel.CONTRADICTS: -0.5,
    }
    CONFLICT_BONUS = 1.5
    CONFLICT_PENALTY = -1.0
    INFO_GATE_PENALTY = -0.5

    def __init__(
        self,
        feh: FactualEntailmentHead,
        reward_entails: float = 1.0,
        reward_neutral: float = -0.1,
        reward_contradicts: float = -0.5,
        lambda_cross: float = 0.3,
        lambda_conflict: float = 0.2,
        lambda_info: float = 1.0,
        min_entails_ratio: float = 0.3,
    ) -> None:
        self.feh = feh
        self.feh.eval()
        self.REWARD_MAP = {
            EntailmentLabel.ENTAILS: reward_entails,
            EntailmentLabel.NEUTRAL: reward_neutral,
            EntailmentLabel.CONTRADICTS: reward_contradicts,
        }
        self.lambda_cross = lambda_cross
        self.lambda_conflict = lambda_conflict
        self.lambda_info = lambda_info
        self.min_entails_ratio = min_entails_ratio

    @torch.no_grad()
    def compute_claim_rewards(
        self,
        h_visuals: list[torch.Tensor],
        h_texts: list[torch.Tensor],
    ) -> list[tuple[float, EntailmentLabel]]:
        """对一组 (visual, text) claim 对计算 FEH reward。

        Args:
            h_visuals: N 个视觉特征 (hidden_dim,)
            h_texts: N 个文本特征 (hidden_dim,)

        Returns:
            列表，每个元素为 (reward_score, predicted_label)
        """
        if not h_visuals:
            return []
        h_v = torch.stack(h_visuals)
        h_t = torch.stack(h_texts)
        labels, _ = self.feh.predict(h_v, h_t)
        results = []
        for label in labels:
            lbl = EntailmentLabel(label.item())
            results.append((self.REWARD_MAP[lbl], lbl))
        return results

    def compute_trajectory_reward(
        self,
        claim_rewards: list[tuple[float, EntailmentLabel]],
        cross_modal_matches: list[bool] | None = None,
        model_reports_conflict: bool = False,
        has_real_conflict: bool = False,
    ) -> float:
        """计算整条 trajectory 的综合 reward (R_FER)。

        Args:
            claim_rewards: compute_claim_rewards 的输出
            cross_modal_matches: table-figure 数值匹配结果 (True=匹配)
            model_reports_conflict: 模型是否在输出中报告了冲突
            has_real_conflict: FEH 是否确认存在真实冲突 (任一 claim 为 CONTRADICTS)

        Returns:
            R_FER: 综合 reward 分数
        """
        if not claim_rewards:
            return 0.0

        # R_base
        r_base = sum(r for r, _ in claim_rewards) / len(claim_rewards)

        # Cross-modal bonus
        r_cross = 0.0
        if cross_modal_matches:
            r_cross = sum(1 for m in cross_modal_matches if m) / len(cross_modal_matches) * 0.5

        # Conflict detection bonus
        r_conflict = 0.0
        if model_reports_conflict:
            if has_real_conflict:
                r_conflict = self.CONFLICT_BONUS
            else:
                r_conflict = self.CONFLICT_PENALTY

        # Informativeness gate
        r_info = 0.0
        n_entails = sum(1 for _, lbl in claim_rewards if lbl == EntailmentLabel.ENTAILS)
        entails_ratio = n_entails / len(claim_rewards)
        if entails_ratio < self.min_entails_ratio:
            r_info = self.INFO_GATE_PENALTY

        return r_base + self.lambda_cross * r_cross + self.lambda_conflict * r_conflict + self.lambda_info * r_info
