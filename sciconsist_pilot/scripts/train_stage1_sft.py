"""
Stage 1: Qwen2.5-VL-7B SFT on SciMDR — 建立科研文档基础理解能力

在 SciMDR 364K 多模态 QA 数据上用 LoRA 微调 Qwen2.5-VL-7B-Instruct,
使模型学会读取科研表格/图表并给出包含具体数值的 CoT 回答。

架构:
  - Base: Qwen2.5-VL-7B-Instruct (bf16)
  - LoRA: r=16, alpha=32, target=qkv+o_proj
  - 4x RTX 4090, gradient_checkpointing, bf16 mixed precision
  - 数据: scimdr_sft_train.jsonl (364K, Qwen2.5-VL 对话格式)

用法 (单机 4 卡):
  cd /root/SciConsist
  torchrun --nproc_per_node=4 sciconsist_pilot/scripts/train_stage1_sft.py \
    --train-data /root/shared-nvme/sciconsist_pilot/processed/scimdr_sft_train.jsonl \
    --val-data /root/shared-nvme/sciconsist_pilot/processed/scimdr_sft_val.jsonl \
    --image-root /root/shared-nvme/sciconsist_pilot/raw/scimdr \
    --output-dir /root/shared-nvme/sciconsist_pilot/outputs/sft_stage1 \
    --num-epochs 1 --per-device-batch-size 2 --gradient-accumulation 8

或者用 accelerate:
  accelerate launch --num_processes=4 sciconsist_pilot/scripts/train_stage1_sft.py ...
"""

from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Any

import torch
from torch.utils.data import Dataset
from transformers import (
    Qwen2VLForConditionalGeneration,
    AutoProcessor,
    Trainer,
    TrainingArguments,
    TrainerCallback,
)
from peft import LoraConfig, get_peft_model, TaskType


# ── 配置 ──────────────────────────────────────────────────────────

IMAGE_ROOT_DEFAULT = "/root/shared-nvme/sciconsist_pilot/raw/scimdr"
MIN_PIXELS = 256 * 28 * 28
MAX_PIXELS = 1280 * 28 * 28


# ── Dataset ───────────────────────────────────────────────────────


class SciMDRSFTDataset(Dataset):
    """SciMDR SFT 数据集 — 惰性加载图片

    从 scimdr_sft_train.jsonl 读取 Qwen2.5-VL 对话格式数据,
    通过 processor 编码为模型输入。图片在 __getitem__ 时加载。

    Attributes:
        records: 原始 JSON 记录列表
        processor: Qwen2.5-VL processor
        image_root: 图片根目录
        max_length: 最大序列长度
    """

    def __init__(
        self,
        data_path: str | Path,
        processor: AutoProcessor,
        image_root: str = IMAGE_ROOT_DEFAULT,
        max_length: int = 2048,
        max_samples: int = 0,
    ) -> None:
        """
        Args:
            data_path: SFT JSONL 数据路径
            processor: Qwen2.5-VL AutoProcessor
            image_root: 图片文件根目录 (image_paths 相对于此)
            max_length: tokenizer 最大长度
            max_samples: 最大加载样本数 (0=全部)
        """
        self.processor = processor
        self.image_root = Path(image_root)
        self.max_length = max_length
        self.records: list[dict] = []

        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                self.records.append(obj)
                if max_samples > 0 and len(self.records) >= max_samples:
                    break

        # 过滤掉图片不存在的样本 (惰性: 只检查第一张)
        valid = []
        skipped = 0
        for rec in self.records:
            img_paths = rec.get("image_paths", [])
            if img_paths:
                first_img = self.image_root / img_paths[0]
                if not first_img.exists():
                    skipped += 1
                    continue
            valid.append(rec)
        self.records = valid
        if skipped > 0:
            print(f"  [Dataset] Skipped {skipped} samples (missing images)")
        print(f"  [Dataset] Loaded {len(self.records)} samples")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        rec = self.records[idx]
        messages = rec["messages"]

        # 把相对图片路径转为绝对路径
        messages_resolved = _resolve_image_paths(messages, self.image_root)

        # 用 processor 编码
        text = self.processor.apply_chat_template(
            messages_resolved,
            tokenize=False,
            add_generation_prompt=False,
        )

        from qwen_vl_utils import process_vision_info
        image_inputs, video_inputs = process_vision_info(messages_resolved)

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt",
        )

        # 把 batch dim squeeze 掉
        input_ids = inputs["input_ids"].squeeze(0)
        attention_mask = inputs["attention_mask"].squeeze(0)

        # labels = input_ids, 但把 prompt 部分 mask 为 -100
        labels = input_ids.clone()
        labels = _mask_prompt_labels(
            labels, input_ids, self.processor.tokenizer
        )

        result = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
        # pixel_values / image_grid_thw (Qwen2.5-VL 特有)
        if "pixel_values" in inputs:
            result["pixel_values"] = inputs["pixel_values"].squeeze(0)
        if "image_grid_thw" in inputs:
            result["image_grid_thw"] = inputs["image_grid_thw"].squeeze(0)

        return result


def _resolve_image_paths(
    messages: list[dict], image_root: Path
) -> list[dict]:
    """将消息中的相对图片路径转为绝对路径 (file:// URI)

    Args:
        messages: Qwen2.5-VL 对话消息列表
        image_root: 图片根目录

    Returns:
        路径已解析的消息列表 (深拷贝)
    """
    resolved = []
    for msg in messages:
        new_msg = {"role": msg["role"]}
        content = msg.get("content", "")
        if isinstance(content, list):
            new_content = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "image":
                    img_key = item.get("image", "")
                    abs_path = str(image_root / img_key)
                    new_content.append({
                        "type": "image",
                        "image": f"file://{abs_path}",
                    })
                else:
                    new_content.append(item)
            new_msg["content"] = new_content
        else:
            new_msg["content"] = content
        resolved.append(new_msg)
    return resolved


def _mask_prompt_labels(
    labels: torch.Tensor,
    input_ids: torch.Tensor,
    tokenizer,
) -> torch.Tensor:
    """将 prompt 部分的 labels 设为 -100 (只对 assistant 回答计算 loss)

    策略: 找到最后一个 assistant turn 的起始位置,
    之前的 token 全部 mask。

    Args:
        labels: 原始 labels (= input_ids)
        input_ids: token IDs
        tokenizer: tokenizer 实例

    Returns:
        masked labels
    """
    # Qwen2.5-VL 的 assistant header token
    # 在 chat template 中, assistant 回答前有 "<|im_start|>assistant\n"
    # 找 "assistant" token 的位置
    text = tokenizer.decode(input_ids, skip_special_tokens=False)

    # 简单策略: 找最后一个 "<|im_start|>assistant" 的 token 位置
    assistant_marker = "<|im_start|>assistant"
    marker_pos = text.rfind(assistant_marker)

    if marker_pos >= 0:
        prefix_text = text[:marker_pos + len(assistant_marker)]
        prefix_ids = tokenizer.encode(prefix_text, add_special_tokens=False)
        mask_len = min(len(prefix_ids), len(labels))
        labels[:mask_len] = -100
    else:
        # fallback: mask 前 80%
        mask_len = int(len(labels) * 0.8)
        labels[:mask_len] = -100

    # padding tokens 也 mask
    labels[labels == tokenizer.pad_token_id] = -100

    return labels


# ── Data Collator ─────────────────────────────────────────────────


class VLMDataCollator:
    """Qwen2.5-VL 多模态 data collator

    处理变长 pixel_values 和 image_grid_thw 的 batch 拼接。
    """

    def __call__(self, features: list[dict]) -> dict[str, torch.Tensor]:
        batch: dict[str, Any] = {}

        # 标准 text 字段: pad
        batch["input_ids"] = torch.stack([f["input_ids"] for f in features])
        batch["attention_mask"] = torch.stack([f["attention_mask"] for f in features])
        batch["labels"] = torch.stack([f["labels"] for f in features])

        # vision 字段: cat (Qwen2.5-VL 内部处理 batch 维度)
        if "pixel_values" in features[0]:
            batch["pixel_values"] = torch.cat(
                [f["pixel_values"] for f in features], dim=0
            )
        if "image_grid_thw" in features[0]:
            batch["image_grid_thw"] = torch.cat(
                [f["image_grid_thw"] for f in features], dim=0
            )

        return batch


# ── Callback ──────────────────────────────────────────────────────


class SFTProgressCallback(TrainerCallback):
    """训练进度回调: 每 N 步打印 layer-wise loss"""

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and state.global_step % 50 == 0:
            loss = logs.get("loss", 0)
            lr = logs.get("learning_rate", 0)
            print(f"  Step {state.global_step}: loss={loss:.4f}, lr={lr:.2e}")


# ── Main ──────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 1: Qwen2.5-VL SFT on SciMDR")

    # 数据
    parser.add_argument("--train-data", required=True,
                        help="scimdr_sft_train.jsonl path")
    parser.add_argument("--val-data", default="",
                        help="scimdr_sft_val.jsonl path")
    parser.add_argument("--image-root", default=IMAGE_ROOT_DEFAULT,
                        help="Image files root directory")
    parser.add_argument("--max-train-samples", type=int, default=0,
                        help="Max training samples (0=all)")
    parser.add_argument("--max-val-samples", type=int, default=500)

    # 模型
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--max-length", type=int, default=2048)

    # LoRA
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--lora-target-modules", default="q_proj,k_proj,v_proj,o_proj")

    # 训练
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-epochs", type=int, default=1)
    parser.add_argument("--per-device-batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--save-steps", type=int, default=500)
    parser.add_argument("--eval-steps", type=int, default=500)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)

    # 其他
    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--gradient-checkpointing", action="store_true", default=True)
    parser.add_argument("--dataloader-num-workers", type=int, default=4)
    parser.add_argument("--resume-from-checkpoint", default="")
    parser.add_argument("--local-rank", type=int, default=-1,
                        help="Set by torchrun")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("=== Stage 1: Qwen2.5-VL SFT on SciMDR ===")
    print(f"  Model: {args.model_name}")
    print(f"  Train data: {args.train_data}")
    print(f"  Image root: {args.image_root}")
    print(f"  Output: {args.output_dir}")
    print(f"  LoRA: r={args.lora_r}, alpha={args.lora_alpha}")
    print(f"  Batch: {args.per_device_batch_size} x {args.gradient_accumulation} x GPUs")
    print()

    # 加载 processor
    processor = AutoProcessor.from_pretrained(
        args.model_name,
        min_pixels=MIN_PIXELS,
        max_pixels=MAX_PIXELS,
    )

    # 加载模型
    print("Loading model...")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )

    # 启用 gradient checkpointing
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

    # LoRA
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=args.lora_target_modules.split(","),
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 数据集
    print("\nLoading training dataset...")
    train_dataset = SciMDRSFTDataset(
        data_path=args.train_data,
        processor=processor,
        image_root=args.image_root,
        max_length=args.max_length,
        max_samples=args.max_train_samples,
    )

    val_dataset = None
    if args.val_data:
        print("Loading validation dataset...")
        val_dataset = SciMDRSFTDataset(
            data_path=args.val_data,
            processor=processor,
            image_root=args.image_root,
            max_length=args.max_length,
            max_samples=args.max_val_samples,
        )

    # effective batch size: per_device * accumulation * n_gpus
    n_gpus = max(torch.cuda.device_count(), 1)
    effective_batch = args.per_device_batch_size * args.gradient_accumulation * n_gpus
    total_steps = len(train_dataset) * args.num_epochs // effective_batch
    print(f"\n  Effective batch size: {effective_batch}")
    print(f"  Total steps (approx): {total_steps}")
    print(f"  Train samples: {len(train_dataset)}")
    if val_dataset:
        print(f"  Val samples: {len(val_dataset)}")

    # TrainingArguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.per_device_batch_size,
        per_device_eval_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        bf16=args.bf16,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_strategy="steps" if val_dataset else "no",
        eval_steps=args.eval_steps if val_dataset else None,
        save_total_limit=3,
        load_best_model_at_end=bool(val_dataset),
        metric_for_best_model="eval_loss" if val_dataset else None,
        greater_is_better=False,
        dataloader_num_workers=args.dataloader_num_workers,
        remove_unused_columns=False,
        seed=args.seed,
        report_to="none",
        ddp_find_unused_parameters=False,
        gradient_checkpointing=args.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=VLMDataCollator(),
        callbacks=[SFTProgressCallback()],
    )

    # 训练
    print("\nStarting training...")
    resume = args.resume_from_checkpoint if args.resume_from_checkpoint else None
    train_result = trainer.train(resume_from_checkpoint=resume)

    # 保存
    print("\nSaving model...")
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)

    # 保存训练指标
    metrics = train_result.metrics
    metrics["train_samples"] = len(train_dataset)
    if val_dataset:
        eval_result = trainer.evaluate()
        metrics.update(eval_result)
        metrics["val_samples"] = len(val_dataset)

    trainer.save_metrics("all", metrics)
    trainer.save_state()

    # 保存完整配置
    config_out = {
        "model_name": args.model_name,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_target_modules": args.lora_target_modules,
        "num_epochs": args.num_epochs,
        "effective_batch_size": effective_batch,
        "lr": args.lr,
        "max_length": args.max_length,
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset) if val_dataset else 0,
        "metrics": metrics,
    }
    Path(args.output_dir).joinpath("sft_config.json").write_text(
        json.dumps(config_out, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"\n=== Training Complete ===")
    print(f"  Train loss: {metrics.get('train_loss', 'N/A')}")
    if "eval_loss" in metrics:
        print(f"  Val loss: {metrics['eval_loss']}")
    print(f"  Model saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
