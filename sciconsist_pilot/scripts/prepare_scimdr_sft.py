"""
SciMDR → Qwen2.5-VL SFT 训练数据转换

将 SciMDR 的 tqa/mqa/vqa 三个 split 转为 Qwen2.5-VL 多模态对话格式，
用于 Stage 1 SFT 训练。同时附加 verifiability 元数据，供 Stage 2 VSR-GRPO 使用。

输入: /root/shared-nvme/sciconsist_pilot/raw/scimdr/{tqa,mqa,vqa}.jsonl
输出:
  - scimdr_sft_train.jsonl  (训练集，~350K)
  - scimdr_sft_val.jsonl    (验证集，~2K)
  - scimdr_sft_stats.json   (数据统计)

对话格式 (Qwen2.5-VL chat template):
  [
    {"role": "user", "content": [
      {"type": "image", "image": "<image_path>"},  # 可多张
      {"type": "text", "text": "<question>"}
    ]},
    {"role": "assistant", "content": "<answer_text>"}
  ]
"""

import json
import random
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

# ── 配置 ──────────────────────────────────────────────────────────

SCIMDR_DIR = Path("/root/shared-nvme/sciconsist_pilot/raw/scimdr")
OUTPUT_DIR = Path("/root/shared-nvme/sciconsist_pilot/processed")
IMAGE_BASE = ""  # 图片路径前缀，留空则使用 SciMDR 原始相对路径
VAL_RATIO = 0.005  # 验证集占比
SEED = 42

# ── 数据类 ────────────────────────────────────────────────────────

@dataclass
class SFTSample:
    """统一的 SFT 训练样本"""
    id: str                     # "{split}_{paper_id}_{idx}"
    split: str                  # "tqa" / "mqa" / "vqa"
    paper_id: str
    source: str                 # "arxiv" / "nature"
    question_type: str
    messages: list[dict]        # Qwen2.5-VL 对话格式
    image_paths: list[str]
    answer_text: str            # 纯文本回答（用于 reward 计算）
    has_cot: bool               # 是否包含 chain-of-thought


# ── 回答提取 ──────────────────────────────────────────────────────

def _build_cot_answer(answer_dict: dict) -> str:
    """从 tqa/mqa 的 dict answer 构建带 CoT 的回答文本。

    格式: "Thinking: ... \n\nAnswer: <conclusion>"
    """
    parts = []
    cot_steps = answer_dict.get("chain_of_thought_answer", [])
    if cot_steps:
        reasoning_lines = []
        for step in cot_steps:
            if isinstance(step, dict):
                s = step.get("step", "")
                r = step.get("reasoning", "")
                reasoning_lines.append(f"Step {s}: {r}" if s else r)
        if reasoning_lines:
            parts.append("Thinking:\n" + "\n".join(reasoning_lines))

    conclusion = answer_dict.get("conclusion", "")
    if conclusion:
        parts.append(f"Answer: {conclusion}")

    return "\n\n".join(parts) if parts else str(answer_dict)


def _build_vqa_answer(record: dict) -> str:
    """从 vqa 记录构建回答文本。"""
    cot = record.get("chain_of_thought", "")
    answer = record.get("answer", "")
    if cot:
        return f"Thinking:\n{cot}\n\nAnswer: {answer}"
    return answer


# ── 消息构建 ──────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are a scientific document analysis assistant. "
    "Answer questions about scientific figures, tables, and charts "
    "based on the visual content and your knowledge. "
    "Provide step-by-step reasoning before your final answer."
)


def _build_user_content(
    question: str,
    image_paths: list[str],
    caption: str = "",
    refs: list[dict] | None = None,
) -> list[dict]:
    """构建 user message 的多模态 content。

    Args:
        question: 问题文本
        image_paths: 图片路径列表
        caption: figure/table 的 caption
        refs: references_in_text 列表

    Returns:
        Qwen2.5-VL content 格式的列表
    """
    content = []

    for img in image_paths:
        path = f"{IMAGE_BASE}{img}" if IMAGE_BASE else img
        content.append({"type": "image", "image": path})

    text_parts = []
    if caption:
        clean_cap = _clean_latex_noise(caption)
        text_parts.append(f"Caption: {clean_cap}")

    if refs:
        ref_text = " ".join(r.get("text", "") for r in refs if isinstance(r, dict))
        if ref_text.strip():
            clean_ref = _clean_latex_noise(ref_text)[:2000]
            text_parts.append(f"Context: {clean_ref}")

    text_parts.append(f"Question: {question}")
    content.append({"type": "text", "text": "\n\n".join(text_parts)})

    return content


def _clean_latex_noise(text: str) -> str:
    """清理 LaTeX 噪音（mathbb、subscript 等 HF 转换残留）。"""
    text = re.sub(r'(?:italic_|start_POSTSUBSCRIPT|end_POSTSUBSCRIPT|POSTSUPERSCRIPT)\s*', '', text)
    text = re.sub(r'\\[a-zA-Z]+\{([^}]*)\}', r'\1', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# ── 主转换 ────────────────────────────────────────────────────────

def convert_record(record: dict, split: str, idx: int) -> SFTSample:
    """将单条 SciMDR 记录转为 SFTSample。

    Args:
        record: SciMDR 原始记录
        split: "tqa" / "mqa" / "vqa"
        idx: 记录在该 split 中的序号

    Returns:
        SFTSample 实例
    """
    paper_id = record.get("paper_id", "unknown")
    source = record.get("source", "unknown")
    question = record.get("question", "")
    question_type = record.get("question_type", "")
    image_paths = record.get("image_paths", [])
    caption = record.get("combined_caption", "")
    refs = record.get("references_in_text", [])

    # 构建回答
    answer = record["answer"]
    if isinstance(answer, dict):
        answer_text = _build_cot_answer(answer)
        has_cot = bool(answer.get("chain_of_thought_answer"))
    else:
        answer_text = _build_vqa_answer(record)
        has_cot = bool(record.get("chain_of_thought"))

    user_content = _build_user_content(question, image_paths, caption, refs)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": answer_text},
    ]

    return SFTSample(
        id=f"{split}_{paper_id}_{idx}",
        split=split,
        paper_id=paper_id,
        source=source,
        question_type=question_type,
        messages=messages,
        image_paths=image_paths,
        answer_text=answer_text,
        has_cot=has_cot,
    )


def process_split(split_name: str) -> list[SFTSample]:
    """处理一个 split 的全部记录。"""
    fpath = SCIMDR_DIR / f"{split_name}.jsonl"
    if not fpath.exists():
        print(f"  [SKIP] {fpath} not found")
        return []

    samples = []
    skipped = 0
    with open(fpath) as f:
        for idx, line in enumerate(f):
            record = json.loads(line)
            if "answer" not in record or "question" not in record:
                skipped += 1
                continue
            sample = convert_record(record, split_name, idx)
            samples.append(sample)
    if skipped:
        print(f"  {split_name}: skipped {skipped} records (missing answer/question)")

    print(f"  {split_name}: {len(samples):,} samples converted")
    return samples


def write_output(
    samples: list[SFTSample],
    val_ratio: float = VAL_RATIO,
    seed: int = SEED,
) -> dict:
    """划分 train/val 并写入输出文件。

    Args:
        samples: 全部 SFTSample
        val_ratio: 验证集占比
        seed: 随机种子

    Returns:
        统计信息字典
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)
    rng.shuffle(samples)

    val_size = max(int(len(samples) * val_ratio), 1)
    val_samples = samples[:val_size]
    train_samples = samples[val_size:]

    train_path = OUTPUT_DIR / "scimdr_sft_train.jsonl"
    val_path = OUTPUT_DIR / "scimdr_sft_val.jsonl"

    for path, subset in [(train_path, train_samples), (val_path, val_samples)]:
        with open(path, "w") as f:
            for s in subset:
                record = {
                    "id": s.id,
                    "messages": s.messages,
                    "image_paths": s.image_paths,
                    "metadata": {
                        "split": s.split,
                        "paper_id": s.paper_id,
                        "source": s.source,
                        "question_type": s.question_type,
                        "has_cot": s.has_cot,
                    },
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # 统计
    split_counts = Counter(s.split for s in samples)
    qt_counts = Counter(s.question_type for s in samples)
    source_counts = Counter(s.source for s in samples)
    cot_count = sum(1 for s in samples if s.has_cot)
    avg_imgs = sum(len(s.image_paths) for s in samples) / max(len(samples), 1)
    avg_answer_len = sum(len(s.answer_text) for s in samples) / max(len(samples), 1)

    stats = {
        "total": len(samples),
        "train": len(train_samples),
        "val": len(val_samples),
        "by_split": dict(split_counts),
        "by_source": dict(source_counts),
        "top_question_types": dict(qt_counts.most_common(20)),
        "has_cot_pct": round(cot_count / len(samples) * 100, 1),
        "avg_images_per_sample": round(avg_imgs, 2),
        "avg_answer_length_chars": round(avg_answer_len, 1),
        "train_path": str(train_path),
        "val_path": str(val_path),
    }

    stats_path = OUTPUT_DIR / "scimdr_sft_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    return stats


# ── Main ──────────────────────────────────────────────────────────

def main() -> None:
    print("=== SciMDR → Qwen2.5-VL SFT Data Preparation ===\n")

    all_samples: list[SFTSample] = []
    for split in ["tqa", "mqa", "vqa"]:
        all_samples.extend(process_split(split))

    print(f"\n  Total: {len(all_samples):,} samples")

    stats = write_output(all_samples)

    print(f"\n=== Output ===")
    print(f"  Train: {stats['train']:,} → {stats['train_path']}")
    print(f"  Val:   {stats['val']:,} → {stats['val_path']}")
    print(f"  CoT:   {stats['has_cot_pct']}%")
    print(f"  Avg images/sample: {stats['avg_images_per_sample']}")
    print(f"  Avg answer length: {stats['avg_answer_length_chars']:.0f} chars")

    # 打印一个样本
    print(f"\n=== Sample (first record) ===")
    s = all_samples[0] if all_samples else None
    if s:
        print(json.dumps(s.messages, indent=2, ensure_ascii=False)[:1500])

    print("\nDone.")


if __name__ == "__main__":
    main()
