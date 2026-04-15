"""
VSR-GRPO 训练脚本 — Stage 2 核心

用 Verification-Stratified Reward 做 GRPO (best-of-group weighted SFT)，
在 SciMDR 数据上训练 Qwen2.5-VL-7B-Instruct + LoRA。

Pipeline:
  1. 加载 SciMDR SFT 数据 (tqa/mqa/vqa, 已转为对话格式)
  2. 加载 paper_tables.jsonl 构建 TableIndex
  3. 对每个 sample: 生成 group_size 个候选回答
  4. 用 VSRReward 对每个候选计算 reward
  5. 选取最优候选, 计算 advantage-weighted SFT loss
  6. 监控 layer-wise reward trajectory

用法:
  cd /root/SciConsist/sciconsist_pilot
  python scripts/train_vsr_grpo.py \
    --data-path /root/shared-nvme/sciconsist_pilot/processed/scimdr_sft_train.jsonl \
    --table-index /root/shared-nvme/sciconsist_pilot/processed/table_structured/paper_tables.jsonl \
    --output-dir /root/shared-nvme/sciconsist_pilot/outputs/vsr_grpo \
    --max-samples 1000 --epochs 1 --group-size 4
"""

from __future__ import annotations

import argparse
import json
import random
import time
from collections import Counter
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, Callable

import numpy as np
import torch
from torch.optim import AdamW

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from sciconsist_pilot.src.vsr.table_index import TableIndex
from sciconsist_pilot.src.vsr.reward import VSRReward, VSRConfig, VSRRewardOutput
from sciconsist_pilot.src.vsr.symbolic import SymbolicVerifierConfig
from sciconsist_pilot.src.vsr.semi_symbolic import SemiSymbolicVerifierConfig
from sciconsist_pilot.src.vsr.types import StructuredTable


# ── 配置 ──────────────────────────────────────────────────────────


@dataclass
class VSRGRPOConfig:
    """VSR-GRPO 完整配置

    Attributes:
        data_path: SFT 格式训练数据路径
        val_data_path: 验证数据路径 (可选, 留空则从 data_path 划分)
        table_index_path: paper_tables.jsonl 路径
        output_dir: 输出目录
        policy_model: VLM 模型名/路径
        use_lora: 是否使用 LoRA 微调
        lora_r: LoRA rank
        lora_alpha: LoRA alpha
        lora_target_modules: LoRA 目标模块
        max_samples: 最大训练样本数 (0=全部)
        val_ratio: 验证集划分比例 (仅在无 val_data_path 时生效)
        val_max_samples: 验证集最大样本数
        epochs: 训练轮数
        group_size: GRPO group 大小
        max_new_tokens: 生成最大 token 数
        temperature: 采样温度
        lr: 学习率
        adv_scale: advantage 缩放系数
        max_seq_len: 最大序列长度
        ema_decay: EMA baseline 衰减系数
        seed: 随机种子
        log_interval: 日志间隔 (steps)
        eval_interval: 评估间隔 (steps, 0=每 epoch 末)
        save_interval: 保存间隔 (steps, 0=仅最优)
        splits_filter: 仅使用这些 split 的数据 ("tqa,mqa" 或 "all")
        source_filter: 仅使用该 source ("arxiv"/"nature"/"all")
        lambda_coverage: VSR coverage bonus 权重
        lambda_specificity: VSR specificity bonus 权重
        layer_weights: VSR 各层权重
        confidence_gate: VSR fallback 置信度门限
        entity_match_threshold: Layer 0 实体匹配阈值
        value_exact_tolerance: Layer 0 精确匹配容差
        value_approx_tolerance_pct: Layer 0 近似匹配容差 (%)
    """
    # 数据
    data_path: str = ""
    val_data_path: str = ""
    table_index_path: str = ""
    output_dir: str = ""
    # 模型
    policy_model: str = "Qwen/Qwen2.5-VL-7B-Instruct"
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_target_modules: str = "q_proj,k_proj,v_proj,o_proj"
    # 训练
    max_samples: int = 0
    val_ratio: float = 0.005
    val_max_samples: int = 200
    epochs: int = 1
    group_size: int = 4
    max_new_tokens: int = 256
    temperature: float = 0.8
    lr: float = 1e-5
    adv_scale: float = 0.5
    max_seq_len: int = 2048
    ema_decay: float = 0.95
    seed: int = 42
    log_interval: int = 10
    eval_interval: int = 0
    save_interval: int = 0
    # 数据过滤
    splits_filter: str = "tqa,mqa"
    source_filter: str = "arxiv"
    # VSR reward
    lambda_coverage: float = 0.1
    lambda_specificity: float = 0.05
    layer_weights: str = "0:0.5,1:0.3,2:0.2"
    confidence_gate: float = 0.7
    entity_match_threshold: float = 0.7
    value_exact_tolerance: float = 0.05
    value_approx_tolerance_pct: float = 1.0

    def parse_layer_weights(self) -> dict[int, float]:
        """解析 layer_weights 字符串为字典"""
        result = {}
        for pair in self.layer_weights.split(","):
            k, v = pair.split(":")
            result[int(k)] = float(v)
        return result

    def build_vsr_config(self) -> VSRConfig:
        """从 GRPO config 构建 VSRConfig"""
        return VSRConfig(
            lambda_coverage=self.lambda_coverage,
            lambda_specificity=self.lambda_specificity,
            layer_weights=self.parse_layer_weights(),
            fallback_confidence_gate=self.confidence_gate,
            symbolic_config=SymbolicVerifierConfig(
                entity_match_threshold=self.entity_match_threshold,
                value_exact_tolerance=self.value_exact_tolerance,
                value_approx_tolerance_pct=self.value_approx_tolerance_pct,
            ),
            semi_symbolic_config=SemiSymbolicVerifierConfig(),
        )


# ── 数据加载 ──────────────────────────────────────────────────────


@dataclass
class GRPOSample:
    """GRPO 训练样本 (已加载)

    Attributes:
        id: 样本 ID
        paper_id: 论文 ID
        split: 数据 split (tqa/mqa/vqa)
        source: 数据源 (arxiv/nature)
        question_type: 问题类型
        question_text: 用户问题文本 (纯文本, 去掉图片)
        answer_text: 参考答案文本
        image_paths: 图片路径列表
        evidence_text: caption + context (用于 Layer 2)
    """
    id: str
    paper_id: str
    split: str
    source: str
    question_type: str
    question_text: str
    answer_text: str
    image_paths: list[str] = field(default_factory=list)
    evidence_text: str = ""


def load_grpo_data(
    path: str | Path,
    max_samples: int = 0,
    splits_filter: str = "all",
    source_filter: str = "all",
) -> list[GRPOSample]:
    """从 SFT 格式的 JSONL 加载 GRPO 训练样本

    Args:
        path: scimdr_sft_train.jsonl 路径
        max_samples: 最大样本数 (0=全部)
        splits_filter: 仅保留的 split, 逗号分隔 ("tqa,mqa" 或 "all")
        source_filter: 仅保留的 source ("arxiv"/"nature"/"all")

    Returns:
        GRPOSample 列表
    """
    allowed_splits = (
        set(splits_filter.split(",")) if splits_filter != "all" else None
    )
    samples: list[GRPOSample] = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            meta = obj.get("metadata", {})
            split = meta.get("split", "")
            source = meta.get("source", "")

            if allowed_splits and split not in allowed_splits:
                continue
            if source_filter != "all" and source != source_filter:
                continue

            question_text, evidence_text = _extract_text_from_messages(
                obj.get("messages", [])
            )
            answer_text = _extract_answer_from_messages(obj.get("messages", []))

            samples.append(GRPOSample(
                id=obj.get("id", ""),
                paper_id=meta.get("paper_id", ""),
                split=split,
                source=source,
                question_type=meta.get("question_type", ""),
                question_text=question_text,
                answer_text=answer_text,
                image_paths=obj.get("image_paths", []),
                evidence_text=evidence_text,
            ))

            if max_samples > 0 and len(samples) >= max_samples:
                break

    return samples


def _extract_text_from_messages(messages: list[dict]) -> tuple[str, str]:
    """从 Qwen2.5-VL 对话格式提取纯文本问题和 evidence

    Returns:
        (question_text, evidence_text)
    """
    question = ""
    evidence = ""
    for msg in messages:
        if msg.get("role") != "user":
            continue
        content = msg.get("content", "")
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    text = item.get("text", "")
                    parts = text.split("\n\n")
                    for part in parts:
                        if part.startswith("Question:"):
                            question = part[len("Question:"):].strip()
                        elif part.startswith("Caption:"):
                            evidence += part[len("Caption:"):].strip() + " "
                        elif part.startswith("Context:"):
                            evidence += part[len("Context:"):].strip() + " "
                    if not question:
                        question = text
        elif isinstance(content, str):
            question = content
    return question.strip(), evidence.strip()


def _extract_answer_from_messages(messages: list[dict]) -> str:
    """从对话格式提取 assistant 回答"""
    for msg in messages:
        if msg.get("role") == "assistant":
            return str(msg.get("content", ""))
    return ""


# ── GRPO 核心 ─────────────────────────────────────────────────────


def normalize_rewards(rewards: list[float], eps: float = 1e-6) -> np.ndarray:
    """均值/标准差归一化 + clip

    Args:
        rewards: 原始 reward 列表
        eps: 防除零

    Returns:
        归一化后的 reward array, clip 到 [-2, 2]
    """
    arr = np.asarray(rewards, dtype=np.float32)
    if arr.size == 0:
        return arr
    mean, std = float(arr.mean()), float(arr.std())
    if std < eps:
        return arr - mean
    return np.clip((arr - mean) / (std + eps), -2.0, 2.0)


@dataclass
class StepLog:
    """单步训练日志"""
    step: int
    sample_id: str
    paper_id: str
    split: str
    question_type: str
    loss: float
    best_reward: float
    mean_reward: float
    advantage: float
    num_claims: int
    layer_dist: dict
    vsr_base: float
    vsr_coverage: float
    vsr_specificity: float


@dataclass
class TrainingState:
    """训练状态追踪"""
    ema_baseline: float = 0.0
    total_steps: int = 0
    epoch: int = 0
    step_logs: list[StepLog] = field(default_factory=list)
    # 按层统计的 reward 轨迹
    layer0_rewards: list[float] = field(default_factory=list)
    layer1_rewards: list[float] = field(default_factory=list)
    layer2_rewards: list[float] = field(default_factory=list)


class VSRGRPOTrainer:
    """VSR-GRPO Trainer

    封装完整的 GRPO 训练循环: 生成候选 → VSR reward → 选最优 → weighted SFT。
    支持 text-only 模式 (调试) 和 VLM 模式 (正式训练)。
    """

    def __init__(
        self,
        config: VSRGRPOConfig,
        model: torch.nn.Module,
        tokenizer,
        processor=None,
        table_index: Optional[TableIndex] = None,
        vsr_reward: Optional[VSRReward] = None,
    ) -> None:
        """
        Args:
            config: 训练配置
            model: policy 模型 (Qwen2.5-VL 或 text-only LM)
            tokenizer: tokenizer
            processor: VLM processor (Qwen2.5-VL 需要, text-only 可为 None)
            table_index: 表格索引
            vsr_reward: VSR reward 实例
        """
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.processor = processor
        self.table_index = table_index or TableIndex()
        self.vsr_reward = vsr_reward or VSRReward(config.build_vsr_config())
        self.device = next(model.parameters()).device
        self.state = TrainingState()

    def compute_reward(
        self, sample: GRPOSample, response: str
    ) -> tuple[float, VSRRewardOutput]:
        """对单条 (sample, response) 计算 VSR reward

        Args:
            sample: GRPO 训练样本
            response: 模型生成的回答

        Returns:
            (total_reward, VSRRewardOutput)
        """
        tables = self.table_index.get(sample.paper_id)
        output = self.vsr_reward.compute(
            response=response,
            tables=tables,
            evidence_text=sample.evidence_text,
        )
        return output.total_reward, output

    def generate_candidates(
        self, sample: GRPOSample
    ) -> list[str]:
        """为单个 sample 生成 group_size 个候选回答

        VLM 模式下通过 processor 处理图片 + 文本；
        text-only 模式仅用 tokenizer。

        Args:
            sample: 训练样本

        Returns:
            生成的候选文本列表
        """
        prompt = self._build_generation_prompt(sample)

        if self.processor is not None:
            return self._generate_vlm(prompt, sample)
        return self._generate_text_only(prompt)

    def _build_generation_prompt(self, sample: GRPOSample) -> str:
        """构建生成 prompt (纯文本部分)"""
        parts = []
        if sample.evidence_text:
            parts.append(f"Context: {sample.evidence_text[:1500]}")
        parts.append(f"Question: {sample.question_text}")
        parts.append(
            "Please provide a detailed, factual answer with specific numbers "
            "and comparisons when available. Show your reasoning step by step."
        )
        return "\n\n".join(parts)

    def _generate_text_only(self, prompt: str) -> list[str]:
        """text-only LM 生成"""
        inputs = self.tokenizer(
            prompt, return_tensors="pt", truncation=True,
            max_length=self.config.max_seq_len - self.config.max_new_tokens,
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                do_sample=True,
                temperature=self.config.temperature,
                top_p=0.9,
                num_return_sequences=self.config.group_size,
                max_new_tokens=self.config.max_new_tokens,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        prompt_len = inputs["input_ids"].shape[1]
        return [
            self.tokenizer.decode(seq[prompt_len:], skip_special_tokens=True).strip()
            for seq in outputs
        ]

    def _generate_vlm(
        self, prompt: str, sample: GRPOSample
    ) -> list[str]:
        """Qwen2.5-VL 多模态生成

        通过 processor 拼接图片 token 和文本 token。
        """
        messages = [
            {"role": "system", "content": (
                "You are a scientific document analysis assistant. "
                "Answer questions about scientific figures, tables, and charts "
                "based on the visual content. Provide step-by-step reasoning "
                "with specific numbers when possible."
            )},
            {"role": "user", "content": self._build_vlm_content(prompt, sample)},
        ]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        from qwen_vl_utils import process_vision_info
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.device)

        responses = []
        for _ in range(self.config.group_size):
            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs,
                    do_sample=True,
                    temperature=self.config.temperature,
                    top_p=0.9,
                    max_new_tokens=self.config.max_new_tokens,
                )
            gen_ids = output_ids[:, inputs["input_ids"].shape[1]:]
            text = self.processor.batch_decode(
                gen_ids, skip_special_tokens=True
            )[0].strip()
            responses.append(text)

        return responses

    def _build_vlm_content(
        self, prompt: str, sample: GRPOSample
    ) -> list[dict]:
        """构建 VLM user content (图片 + 文本)"""
        content: list[dict] = []
        for img_path in sample.image_paths:
            content.append({"type": "image", "image": img_path})
        content.append({"type": "text", "text": prompt})
        return content

    def train_step(
        self,
        sample: GRPOSample,
        optimizer: torch.optim.Optimizer,
    ) -> StepLog:
        """单步 GRPO 训练

        1. 生成 group_size 个候选
        2. 用 VSR 计算每个候选的 reward
        3. 归一化 reward, 选取最优候选
        4. 计算 advantage-weighted SFT loss

        Args:
            sample: 训练样本
            optimizer: 优化器

        Returns:
            StepLog 训练日志
        """
        self.model.eval()
        candidates = self.generate_candidates(sample)

        rewards = []
        vsr_outputs = []
        for cand in candidates:
            r, out = self.compute_reward(sample, cand)
            rewards.append(r)
            vsr_outputs.append(out)

        norm_rewards = normalize_rewards(rewards)
        best_idx = int(np.argmax(norm_rewards))
        best_resp = candidates[best_idx]
        best_output = vsr_outputs[best_idx]
        advantage = float(norm_rewards[best_idx])

        self.state.ema_baseline = (
            self.config.ema_decay * self.state.ema_baseline
            + (1 - self.config.ema_decay) * float(np.mean(norm_rewards))
        )
        stabilized_adv = advantage - self.state.ema_baseline

        # SFT loss on best response
        self.model.train()
        loss = self._compute_sft_loss(sample, best_resp)
        weighted_loss = loss * (1.0 + self.config.adv_scale * max(0.0, stabilized_adv))

        optimizer.zero_grad(set_to_none=True)
        weighted_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        optimizer.step()

        self.state.total_steps += 1

        # 记录 layer-wise reward
        self._track_layer_rewards(best_output)

        return StepLog(
            step=self.state.total_steps,
            sample_id=sample.id,
            paper_id=sample.paper_id,
            split=sample.split,
            question_type=sample.question_type,
            loss=float(weighted_loss.detach().cpu().item()),
            best_reward=float(rewards[best_idx]),
            mean_reward=float(np.mean(rewards)),
            advantage=stabilized_adv,
            num_claims=best_output.num_claims,
            layer_dist=best_output.layer_distribution,
            vsr_base=best_output.base_reward,
            vsr_coverage=best_output.coverage_bonus,
            vsr_specificity=best_output.specificity_bonus,
        )

    def _compute_sft_loss(
        self, sample: GRPOSample, response: str
    ) -> torch.Tensor:
        """计算 response 的 next-token CE loss"""
        prompt = self._build_generation_prompt(sample)
        full = prompt + "\n" + response
        enc = self.tokenizer(
            full, return_tensors="pt", truncation=True,
            max_length=self.config.max_seq_len,
        ).to(self.device)
        with torch.enable_grad():
            out = self.model(**enc, labels=enc["input_ids"])
            return out.loss

    def _track_layer_rewards(self, output: VSRRewardOutput) -> None:
        """追踪各层 reward (用于 anchor effect 可视化)"""
        for cr in output.claim_rewards:
            for layer_info in cr.get("layers", []):
                layer_name = layer_info.get("layer", "")
                r = layer_info.get("reward", 0.0)
                if layer_name == "SYMBOLIC":
                    self.state.layer0_rewards.append(r)
                elif layer_name == "SEMI_SYMBOLIC":
                    self.state.layer1_rewards.append(r)
                elif layer_name == "LEARNED":
                    self.state.layer2_rewards.append(r)

    def evaluate(
        self, val_samples: list[GRPOSample], max_eval: int = 100
    ) -> dict:
        """在验证集上评估

        Args:
            val_samples: 验证样本
            max_eval: 最大评估数

        Returns:
            评估指标字典
        """
        self.model.eval()
        rewards = []
        layer_dists = {"layer_0": [], "layer_1": [], "layer_2": []}

        for sample in val_samples[:max_eval]:
            candidates = self.generate_candidates(sample)
            best_r = -float("inf")
            best_out = None
            for cand in candidates[:1]:
                r, out = self.compute_reward(sample, cand)
                if r > best_r:
                    best_r = r
                    best_out = out
            rewards.append(best_r)
            if best_out:
                for k in layer_dists:
                    layer_dists[k].append(
                        best_out.layer_distribution.get(k, 0.0)
                    )

        return {
            "avg_reward": float(np.mean(rewards)) if rewards else 0.0,
            "std_reward": float(np.std(rewards)) if rewards else 0.0,
            "n_eval": len(rewards),
            "avg_layer_0_pct": float(np.mean(layer_dists["layer_0"])),
            "avg_layer_1_pct": float(np.mean(layer_dists["layer_1"])),
            "avg_layer_2_pct": float(np.mean(layer_dists["layer_2"])),
        }


# ── 训练循环 ──────────────────────────────────────────────────────


def set_seed(seed: int) -> None:
    """设置全局随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_model_and_tokenizer(
    config: VSRGRPOConfig,
) -> tuple:
    """加载 policy model + tokenizer + (可选) processor

    根据 model name 自动判断是 VLM 还是 text-only。
    VLM 时额外加载 processor 和 LoRA。

    Args:
        config: 训练配置

    Returns:
        (model, tokenizer, processor) — text-only 时 processor=None
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    is_vlm = "VL" in config.policy_model or "vl" in config.policy_model

    if is_vlm:
        from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
        print(f"Loading VLM: {config.policy_model}")
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            config.policy_model,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="auto",
        )
        processor = AutoProcessor.from_pretrained(config.policy_model)
        tokenizer = processor.tokenizer

        if config.use_lora:
            from peft import LoraConfig, get_peft_model
            target_modules = config.lora_target_modules.split(",")
            lora_config = LoraConfig(
                r=config.lora_r,
                lora_alpha=config.lora_alpha,
                target_modules=target_modules,
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM",
            )
            model = get_peft_model(model, lora_config)
            model.print_trainable_parameters()
    else:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        print(f"Loading text-only model: {config.policy_model}")
        model = AutoModelForCausalLM.from_pretrained(
            config.policy_model,
            torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
            trust_remote_code=True,
        ).to(device)
        tokenizer = AutoTokenizer.from_pretrained(
            config.policy_model, trust_remote_code=True
        )
        processor = None

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    if hasattr(model, "config") and hasattr(model.config, "use_cache"):
        model.config.use_cache = False

    return model, tokenizer, processor


def run_training(config: VSRGRPOConfig) -> dict:
    """完整训练流程

    Args:
        config: 训练配置

    Returns:
        训练结果字典
    """
    set_seed(config.seed)
    out_dir = Path(config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 保存 config
    (out_dir / "config.json").write_text(
        json.dumps(asdict(config), indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # 加载数据
    print("Loading training data...")
    all_samples = load_grpo_data(
        config.data_path,
        max_samples=config.max_samples,
        splits_filter=config.splits_filter,
        source_filter=config.source_filter,
    )
    print(f"  Total samples: {len(all_samples)}")
    print(f"  Split dist: {Counter(s.split for s in all_samples)}")
    print(f"  QType dist: {Counter(s.question_type for s in all_samples).most_common(10)}")

    # 划分 train/val
    rng = random.Random(config.seed)
    rng.shuffle(all_samples)
    if config.val_data_path:
        train_samples = all_samples
        val_samples = load_grpo_data(
            config.val_data_path, max_samples=config.val_max_samples
        )
    else:
        val_size = max(int(len(all_samples) * config.val_ratio), 1)
        val_samples = all_samples[:val_size]
        train_samples = all_samples[val_size:]

    # 过滤掉 table_index 中没有对应表格的样本 (可选, 保留但标记)
    print("\nLoading table index...")
    table_index = TableIndex.from_jsonl(config.table_index_path)
    n_with_tables = sum(1 for s in train_samples if table_index.has(s.paper_id))
    print(f"  Train samples with tables: {n_with_tables}/{len(train_samples)} "
          f"({n_with_tables/max(len(train_samples),1)*100:.1f}%)")

    # 加载模型
    print("\nLoading model...")
    model, tokenizer, processor = load_model_and_tokenizer(config)

    # 构建 trainer
    vsr_config = config.build_vsr_config()
    vsr_reward = VSRReward(vsr_config)

    trainer = VSRGRPOTrainer(
        config=config,
        model=model,
        tokenizer=tokenizer,
        processor=processor,
        table_index=table_index,
        vsr_reward=vsr_reward,
    )

    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=config.lr,
    )

    # 训练循环
    history = []
    best_val_reward = -float("inf")

    for epoch in range(config.epochs):
        trainer.state.epoch = epoch
        epoch_logs = []
        rng.shuffle(train_samples)

        for step_idx, sample in enumerate(train_samples):
            log = trainer.train_step(sample, optimizer)
            epoch_logs.append(log)

            if config.log_interval > 0 and (step_idx + 1) % config.log_interval == 0:
                recent = epoch_logs[-config.log_interval:]
                avg_loss = np.mean([l.loss for l in recent])
                avg_reward = np.mean([l.best_reward for l in recent])
                avg_claims = np.mean([l.num_claims for l in recent])
                print(
                    f"  [E{epoch+1} S{step_idx+1}/{len(train_samples)}] "
                    f"loss={avg_loss:.4f} reward={avg_reward:.3f} "
                    f"claims={avg_claims:.1f} adv={log.advantage:.3f}"
                )

            # GPU 内存清理
            if torch.cuda.is_available() and (step_idx + 1) % 8 == 0:
                torch.cuda.empty_cache()

        # Epoch 末评估
        print(f"\n  Evaluating epoch {epoch+1}...")
        val_result = trainer.evaluate(val_samples, max_eval=config.val_max_samples)
        print(f"  Val: reward={val_result['avg_reward']:.3f} "
              f"±{val_result['std_reward']:.3f} (n={val_result['n_eval']})")
        print(f"  Val layer dist: L0={val_result['avg_layer_0_pct']:.1f}% "
              f"L1={val_result['avg_layer_1_pct']:.1f}% "
              f"L2={val_result['avg_layer_2_pct']:.1f}%")

        epoch_summary = {
            "epoch": epoch + 1,
            "train_steps": len(epoch_logs),
            "train_avg_loss": float(np.mean([l.loss for l in epoch_logs])),
            "train_avg_reward": float(np.mean([l.best_reward for l in epoch_logs])),
            "train_avg_claims": float(np.mean([l.num_claims for l in epoch_logs])),
            "val": val_result,
        }
        history.append(epoch_summary)

        if val_result["avg_reward"] > best_val_reward:
            best_val_reward = val_result["avg_reward"]
            ckpt_dir = out_dir / "checkpoints"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            if config.use_lora and hasattr(model, "save_pretrained"):
                model.save_pretrained(ckpt_dir / "best_lora")
            else:
                torch.save(
                    {"model_state_dict": model.state_dict()},
                    ckpt_dir / "best_model.pt",
                )
            print(f"  Saved best model (reward={best_val_reward:.3f})")

    # 保存 layer-wise reward trajectory
    trajectory = {
        "layer0_rewards": trainer.state.layer0_rewards,
        "layer1_rewards": trainer.state.layer1_rewards,
        "layer2_rewards": trainer.state.layer2_rewards,
    }
    (out_dir / "layer_reward_trajectory.json").write_text(
        json.dumps(trajectory), encoding="utf-8"
    )

    # 保存完整训练结果
    result = {
        "config": asdict(config),
        "train_size": len(train_samples),
        "val_size": len(val_samples),
        "history": history,
        "best_val_reward": best_val_reward,
        "total_steps": trainer.state.total_steps,
    }
    (out_dir / "training_results.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # 保存详细 step logs
    (out_dir / "step_logs.jsonl").write_text(
        "\n".join(json.dumps(asdict(l)) for l in trainer.state.step_logs),
        encoding="utf-8",
    )

    print(f"\nTraining complete. Results saved to {out_dir}")
    return result


# ── CLI ───────────────────────────────────────────────────────────


def parse_args() -> VSRGRPOConfig:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="VSR-GRPO Training")
    # 遍历 VSRGRPOConfig 的字段自动添加参数
    config = VSRGRPOConfig()
    for f_name, f_val in asdict(config).items():
        flag = f"--{f_name.replace('_', '-')}"
        if isinstance(f_val, bool):
            parser.add_argument(flag, action="store_true", default=f_val)
        elif isinstance(f_val, int):
            parser.add_argument(flag, type=int, default=f_val)
        elif isinstance(f_val, float):
            parser.add_argument(flag, type=float, default=f_val)
        else:
            parser.add_argument(flag, type=str, default=f_val)

    args = parser.parse_args()
    return VSRGRPOConfig(**{
        k.replace("-", "_"): v
        for k, v in vars(args).items()
    })


def main() -> None:
    config = parse_args()
    print("=== VSR-GRPO Training ===")
    print(f"  Policy: {config.policy_model}")
    print(f"  Data: {config.data_path}")
    print(f"  Table index: {config.table_index_path}")
    print(f"  Output: {config.output_dir}")
    print(f"  VSR weights: {config.layer_weights}")
    print(f"  Group size: {config.group_size}, Epochs: {config.epochs}")
    print()
    run_training(config)


if __name__ == "__main__":
    main()
