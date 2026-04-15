"""
Layer 2 对照实验脚本: text_only vs FEH cached

本脚本用于快速比较 VSR 在两种 Layer 2 配置下的 reward 差异:
  1) text_only: 纯规则轻量验证 (无 FEH/无视觉缓存)
  2) cached:    FEH + 预提取视觉特征缓存 (paper_id.npy)

设计目标:
  - 复用与 GRPO 一致的数据格式 (scimdr_sft_train.jsonl / grpo_train_30k.jsonl)
  - 以参考答案作为 response 输入，稳定比较 reward 函数行为
  - 提供可复现的 summary + 明细输出，便于后续画图和写 ablation

输出文件:
  - <output_dir>/comparison_summary.json
  - <output_dir>/comparison_details.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.features.extract import ExtractionConfig, FeatureExtractor
from src.vsr.learned import FEHVerifierConfig
from src.vsr.reward import VSRConfig, VSRReward
from src.vsr.table_index import TableIndex


@dataclass
class CompareSample:
    """对照实验单条样本。

    Attributes:
        sample_id: 样本唯一 ID
        paper_id: 论文 ID
        split: 数据 split (tqa/mqa/vqa)
        source: 数据来源 (arxiv/nature)
        question_type: 任务类型
        answer_text: 参考答案文本 (作为 response 输入)
        evidence_text: 证据文本 (caption/context)
        image_paths: 相对图片路径列表
    """

    sample_id: str
    paper_id: str
    split: str
    source: str
    question_type: str
    answer_text: str
    evidence_text: str
    image_paths: list[str]


@dataclass
class CompareResult:
    """单条样本对照结果。"""

    sample_id: str
    paper_id: str
    split: str
    source: str
    question_type: str
    has_tables: bool
    cache_hit: bool
    reward_text_only: float
    reward_feh_cached: float
    reward_delta: float
    num_claims_text_only: int
    num_claims_feh_cached: int
    layer2_claims_text_only: int
    layer2_claims_feh_cached: int


def _extract_text_from_messages(messages: list[dict[str, Any]]) -> tuple[str, str]:
    """从 Qwen 对话格式中提取 question_text 与 evidence_text。"""
    question = ""
    evidence = ""
    for msg in messages:
        if msg.get("role") != "user":
            continue
        content = msg.get("content", "")
        if isinstance(content, list):
            for item in content:
                if not isinstance(item, dict) or item.get("type") != "text":
                    continue
                text = item.get("text", "")
                parts = text.split("\n\n")
                for part in parts:
                    if part.startswith("Question:"):
                        question = part[len("Question:") :].strip()
                    elif part.startswith("Caption:"):
                        evidence += part[len("Caption:") :].strip() + " "
                    elif part.startswith("Context:"):
                        evidence += part[len("Context:") :].strip() + " "
                if not question:
                    question = text
        elif isinstance(content, str):
            question = content
    return question.strip(), evidence.strip()


def _extract_answer_from_messages(messages: list[dict[str, Any]]) -> str:
    """提取 assistant 回答文本。"""
    for msg in messages:
        if msg.get("role") == "assistant":
            return str(msg.get("content", ""))
    return ""


def load_samples(
    data_path: str | Path,
    max_samples: int,
    seed: int,
    splits_filter: str,
    source_filter: str,
) -> list[CompareSample]:
    """加载并筛选对照样本。"""
    allow_splits = (
        set(s.strip() for s in splits_filter.split(",")) if splits_filter != "all" else None
    )
    samples: list[CompareSample] = []
    with open(data_path, encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            meta = obj.get("metadata", {})
            split = str(meta.get("split", ""))
            source = str(meta.get("source", ""))
            if allow_splits and split not in allow_splits:
                continue
            if source_filter != "all" and source != source_filter:
                continue

            _, evidence_text = _extract_text_from_messages(obj.get("messages", []))
            answer_text = _extract_answer_from_messages(obj.get("messages", []))
            if not answer_text.strip():
                continue

            samples.append(
                CompareSample(
                    sample_id=str(obj.get("id", "")),
                    paper_id=str(meta.get("paper_id", "")),
                    split=split,
                    source=source,
                    question_type=str(meta.get("question_type", "")),
                    answer_text=answer_text,
                    evidence_text=evidence_text,
                    image_paths=[str(p) for p in obj.get("image_paths", [])],
                )
            )

    rng = random.Random(seed)
    rng.shuffle(samples)
    if max_samples > 0:
        samples = samples[:max_samples]
    return samples


def _resolve_image_path(rel_path: str, image_root: str | Path) -> Path:
    """将相对路径解析为绝对路径。"""
    p = Path(rel_path)
    if p.is_absolute():
        return p
    return Path(image_root) / rel_path


def build_visual_cache_for_papers(
    samples: list[CompareSample],
    visual_cache_dir: str | Path,
    image_root: str | Path,
    extractor_model: str,
    cache_build_max_papers: int,
) -> dict[str, int]:
    """根据样本中的图片构建 FEH cached 模式所需的 paper_id.npy 缓存。

    每个 paper 仅保存一个视觉向量（首张有效图片）。
    """
    cache_dir = Path(visual_cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    paper_to_image: dict[str, Path] = {}

    for s in samples:
        if not s.paper_id or s.paper_id in paper_to_image:
            continue
        for rel in s.image_paths:
            abs_path = _resolve_image_path(rel, image_root)
            if abs_path.exists():
                paper_to_image[s.paper_id] = abs_path
                break

    paper_items = list(paper_to_image.items())
    if cache_build_max_papers > 0:
        paper_items = paper_items[:cache_build_max_papers]

    to_build: list[tuple[str, Path]] = []
    existed = 0
    for paper_id, image_path in paper_items:
        cache_path = cache_dir / f"{paper_id}.npy"
        if cache_path.exists():
            existed += 1
        else:
            to_build.append((paper_id, image_path))

    if not to_build:
        return {
            "candidate_papers": len(paper_items),
            "already_cached": existed,
            "new_built": 0,
            "build_failures": 0,
        }

    extractor = FeatureExtractor(
        ExtractionConfig(
            model_name=extractor_model,
            batch_size=1,
            device="cuda",
        )
    )

    built = 0
    failed = 0
    for paper_id, image_path in to_build:
        try:
            feat = extractor.extract_visual_features([str(image_path)])[0]
            np.save(cache_dir / f"{paper_id}.npy", feat)
            built += 1
        except Exception:
            failed += 1

    return {
        "candidate_papers": len(paper_items),
        "already_cached": existed,
        "new_built": built,
        "build_failures": failed,
    }


def _count_layer2_claims(claim_rewards: list[dict[str, Any]]) -> int:
    """统计该 response 中触发 Layer 2 的 claim 数。"""
    count = 0
    for claim in claim_rewards:
        layers = claim.get("layers", [])
        if any(str(l.get("layer", "")) == "LEARNED" for l in layers):
            count += 1
    return count


def run_comparison(
    samples: list[CompareSample],
    table_index: TableIndex,
    feh_checkpoint: str,
    visual_cache_dir: str,
) -> tuple[list[CompareResult], dict[str, Any]]:
    """执行 text_only vs FEH cached 对照。"""
    base_layer_weights = {0: 0.5, 1: 0.3, 2: 0.2}

    text_config = VSRConfig(
        layer_weights=base_layer_weights,
        feh_config=FEHVerifierConfig(mode="text_only"),
    )
    cached_config = VSRConfig(
        layer_weights=base_layer_weights,
        feh_config=FEHVerifierConfig(
            mode="cached",
            feh_checkpoint=feh_checkpoint,
            visual_cache_dir=visual_cache_dir,
            device="cuda",
        ),
    )

    vsr_text = VSRReward(text_config)
    vsr_cached = VSRReward(cached_config)

    results: list[CompareResult] = []
    for s in samples:
        tables = table_index.get(s.paper_id)
        out_text = vsr_text.compute(
            response=s.answer_text,
            tables=tables,
            evidence_text=s.evidence_text,
            paper_id=s.paper_id,
            image_path=s.image_paths[0] if s.image_paths else "",
        )
        out_cached = vsr_cached.compute(
            response=s.answer_text,
            tables=tables,
            evidence_text=s.evidence_text,
            paper_id=s.paper_id,
            image_path=s.image_paths[0] if s.image_paths else "",
        )

        cache_file = Path(visual_cache_dir) / f"{s.paper_id}.npy"
        result = CompareResult(
            sample_id=s.sample_id,
            paper_id=s.paper_id,
            split=s.split,
            source=s.source,
            question_type=s.question_type,
            has_tables=table_index.has(s.paper_id),
            cache_hit=cache_file.exists(),
            reward_text_only=float(out_text.total_reward),
            reward_feh_cached=float(out_cached.total_reward),
            reward_delta=float(out_cached.total_reward - out_text.total_reward),
            num_claims_text_only=out_text.num_claims,
            num_claims_feh_cached=out_cached.num_claims,
            layer2_claims_text_only=_count_layer2_claims(out_text.claim_rewards),
            layer2_claims_feh_cached=_count_layer2_claims(out_cached.claim_rewards),
        )
        results.append(result)

    deltas = [r.reward_delta for r in results]
    abs_deltas = [abs(x) for x in deltas]
    changed = [r for r in results if abs(r.reward_delta) > 1e-8]
    cache_hits = sum(1 for r in results if r.cache_hit)
    with_tables = sum(1 for r in results if r.has_tables)

    summary = {
        "n_samples": len(results),
        "n_changed": len(changed),
        "changed_ratio": (len(changed) / len(results)) if results else 0.0,
        "cache_hit_samples": cache_hits,
        "cache_hit_ratio": (cache_hits / len(results)) if results else 0.0,
        "samples_with_tables": with_tables,
        "samples_with_tables_ratio": (with_tables / len(results)) if results else 0.0,
        "mean_reward_text_only": mean([r.reward_text_only for r in results]) if results else 0.0,
        "mean_reward_feh_cached": mean([r.reward_feh_cached for r in results]) if results else 0.0,
        "mean_delta": mean(deltas) if deltas else 0.0,
        "mean_abs_delta": mean(abs_deltas) if abs_deltas else 0.0,
        "max_abs_delta": max(abs_deltas) if abs_deltas else 0.0,
    }
    return results, summary


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="Compare VSR text_only vs FEH cached")
    parser.add_argument(
        "--data-path",
        default="/root/shared-nvme/sciconsist_pilot/processed/grpo_train_30k.jsonl",
        help="输入数据 (SFT/GRPO JSONL)",
    )
    parser.add_argument(
        "--table-index",
        default="/root/shared-nvme/sciconsist_pilot/processed/table_structured/paper_tables.jsonl",
        help="paper_tables.jsonl 路径",
    )
    parser.add_argument(
        "--feh-checkpoint",
        default="/root/shared-nvme/sciconsist_pilot/outputs/checkpoints_fixrun/feh_best.pt",
        help="FEH checkpoint 路径",
    )
    parser.add_argument(
        "--visual-cache-dir",
        default="/root/shared-nvme/sciconsist_pilot/processed/vsr_feh_cache",
        help="cached 模式视觉缓存目录 (paper_id.npy)",
    )
    parser.add_argument(
        "--image-root",
        default="/root/shared-nvme/sciconsist_pilot/raw/scimdr",
        help="图片根目录，用于解析相对 image_paths",
    )
    parser.add_argument(
        "--output-dir",
        default="/root/shared-nvme/sciconsist_pilot/outputs/layer2_compare",
        help="输出目录",
    )
    parser.add_argument("--max-samples", type=int, default=200, help="评估样本数")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument(
        "--splits-filter",
        type=str,
        default="tqa,mqa",
        help="保留的 split，逗号分隔或 all",
    )
    parser.add_argument(
        "--source-filter",
        type=str,
        default="arxiv",
        help="保留的数据源 arxiv/nature/all",
    )
    parser.add_argument(
        "--build-cache",
        action="store_true",
        help="是否先为样本构建 visual cache",
    )
    parser.add_argument(
        "--cache-build-max-papers",
        type=int,
        default=300,
        help="最多构建多少篇论文的视觉缓存 (0=不限)",
    )
    parser.add_argument(
        "--extractor-model",
        type=str,
        default="OpenGVLab/InternVL2_5-8B",
        help="构建缓存时使用的特征提取模型",
    )
    return parser.parse_args()


def main() -> None:
    """脚本入口。"""
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    samples = load_samples(
        data_path=args.data_path,
        max_samples=args.max_samples,
        seed=args.seed,
        splits_filter=args.splits_filter,
        source_filter=args.source_filter,
    )
    if not samples:
        raise RuntimeError("No samples loaded after filtering. Please check filters/data path.")

    print(f"Loaded samples: {len(samples)}")
    print(f"  splits: {args.splits_filter}, source: {args.source_filter}")

    cache_build_stats: dict[str, Any] | None = None
    if args.build_cache:
        print("Building visual cache for FEH cached mode...")
        cache_build_stats = build_visual_cache_for_papers(
            samples=samples,
            visual_cache_dir=args.visual_cache_dir,
            image_root=args.image_root,
            extractor_model=args.extractor_model,
            cache_build_max_papers=args.cache_build_max_papers,
        )
        print(f"  cache stats: {cache_build_stats}")

    print("Loading table index...")
    table_index = TableIndex.from_jsonl(args.table_index)

    print("Running comparison...")
    results, summary = run_comparison(
        samples=samples,
        table_index=table_index,
        feh_checkpoint=args.feh_checkpoint,
        visual_cache_dir=args.visual_cache_dir,
    )

    summary_payload = {
        "config": {
            "data_path": args.data_path,
            "table_index": args.table_index,
            "feh_checkpoint": args.feh_checkpoint,
            "visual_cache_dir": args.visual_cache_dir,
            "max_samples": args.max_samples,
            "splits_filter": args.splits_filter,
            "source_filter": args.source_filter,
            "build_cache": bool(args.build_cache),
            "cache_build_max_papers": args.cache_build_max_papers,
            "extractor_model": args.extractor_model,
        },
        "cache_build_stats": cache_build_stats,
        "summary": summary,
    }

    summary_path = output_dir / "comparison_summary.json"
    details_path = output_dir / "comparison_details.jsonl"

    summary_path.write_text(
        json.dumps(summary_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    with details_path.open("w", encoding="utf-8") as f:
        for item in results:
            f.write(json.dumps(asdict(item), ensure_ascii=False) + "\n")

    print("\n=== Done ===")
    print(f"Summary: {summary_path}")
    print(f"Details: {details_path}")
    print("Key metrics:")
    print(f"  changed_ratio: {summary['changed_ratio']:.3f}")
    print(f"  cache_hit_ratio: {summary['cache_hit_ratio']:.3f}")
    print(f"  mean_delta: {summary['mean_delta']:.4f}")
    print(f"  mean_abs_delta: {summary['mean_abs_delta']:.4f}")


if __name__ == "__main__":
    main()
