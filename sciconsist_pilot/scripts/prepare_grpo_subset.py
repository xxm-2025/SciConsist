"""
GRPO 训练数据子集筛选 — 从 SFT 数据中选取 VSR 高覆盖样本

筛选策略:
  1. 只保留 paper_id 在 table index 中的样本 (L0/L1 才有表格可验证)
  2. 优先 tqa (L0+L1=78.6%) > mqa (85.0%) > vqa (44.8%)
  3. tqa 内优先 Comparative Analysis (L0+L1=92.6%)
  4. 控制总量 (默认 30K)，按 split 配额分配
  5. 输出: 与 SFT 格式相同的 JSONL，可直接给 train_vsr_grpo.py 消费

用法:
    python sciconsist_pilot/scripts/prepare_grpo_subset.py \
        --sft-data /root/shared-nvme/.../scimdr_sft_train.jsonl \
        --table-index /root/shared-nvme/.../paper_tables.jsonl \
        --output /root/shared-nvme/.../grpo_train_30k.jsonl \
        --total 30000

输入:
    --sft-data: SFT 格式 JSONL (364K)
    --table-index: paper_tables.jsonl (7.5K papers)
    --output: 输出路径
    --total: 目标总样本数
    --split-ratio: tqa:mqa:vqa 比例 (默认 "0.5:0.35:0.15")

输出:
    GRPO 子集 JSONL + 采样统计 JSON
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


# tqa question_type 按 verifiability 覆盖率排序 (Week 1 分析结果)
TQA_TYPE_PRIORITY = [
    "Comparative Analysis",        # L0+1 = 92.6%
    "Descriptive/Quantitative",    # L0+1 高 (数值密集)
    "Descriptive/Quantitative Analysis",
    "Causal Reasoning",            # L0+1 ≈ 78%
    "Methodological Analysis",     # L0+1 ≈ 75%
    "Conceptual Understanding",    # L0+1 ≈ 65%
    "Critical Evaluation",         # L0+1 ≈ 60%
]


@dataclass
class SubsetConfig:
    """数据子集配置

    Attributes:
        total: 目标总样本数
        split_ratios: tqa/mqa/vqa 的比例
        seed: 随机种子
    """
    total: int = 30000
    split_ratios: dict[str, float] = None
    seed: int = 42

    def __post_init__(self) -> None:
        if self.split_ratios is None:
            self.split_ratios = {"tqa": 0.50, "mqa": 0.35, "vqa": 0.15}


def load_table_paper_ids(path: str | Path) -> set[str]:
    """加载 table index 中的 paper_id 集合。

    Args:
        path: paper_tables.jsonl 路径

    Returns:
        paper_id 集合
    """
    ids = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            pid = d.get("paper_id", "")
            if pid:
                ids.add(pid)
    return ids


def load_and_filter(
    sft_path: str | Path,
    table_paper_ids: set[str],
) -> dict[str, list[dict[str, Any]]]:
    """加载 SFT 数据并按 paper_id 过滤, 按 split 分组。

    Args:
        sft_path: SFT JSONL 路径
        table_paper_ids: 有表格数据的 paper_id 集合

    Returns:
        {split: [sample_dict, ...]}
    """
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    skipped = 0

    with open(sft_path, encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            meta = obj.get("metadata", {})
            pid = meta.get("paper_id", "")
            split = meta.get("split", "")

            if pid not in table_paper_ids:
                skipped += 1
                continue

            groups[split].append(obj)

    print(f"Loaded: {sum(len(v) for v in groups.values()):,} matched, {skipped:,} skipped")
    for s in sorted(groups):
        print(f"  {s}: {len(groups[s]):,}")

    return groups


def prioritized_sample(
    samples: list[dict[str, Any]],
    n: int,
    split: str,
    rng: random.Random,
) -> list[dict[str, Any]]:
    """按 question_type 优先级采样。

    tqa: 优先选 Comparative Analysis 等高覆盖类型
    mqa/vqa: 均匀采样

    Args:
        samples: 候选样本列表
        n: 目标采样数
        split: 数据 split (tqa/mqa/vqa)
        rng: 随机数生成器

    Returns:
        采样结果
    """
    if len(samples) <= n:
        return samples

    if split == "tqa":
        by_type: dict[str, list[dict]] = defaultdict(list)
        for s in samples:
            qt = s.get("metadata", {}).get("question_type", "other")
            by_type[qt].append(s)

        selected: list[dict] = []
        remaining = n

        for qt in TQA_TYPE_PRIORITY:
            if remaining <= 0:
                break
            pool = by_type.get(qt, [])
            if not pool:
                continue
            take = min(len(pool), max(remaining // 3, 500))
            rng.shuffle(pool)
            selected.extend(pool[:take])
            remaining -= take

        # 剩余配额从全部样本中均匀补齐
        if remaining > 0:
            used_ids = {s["id"] for s in selected}
            leftovers = [s for s in samples if s["id"] not in used_ids]
            rng.shuffle(leftovers)
            selected.extend(leftovers[:remaining])

        return selected[:n]

    rng.shuffle(samples)
    return samples[:n]


def compute_stats(
    selected: list[dict[str, Any]],
) -> dict[str, Any]:
    """统计采样结果。

    Args:
        selected: 最终选中的样本

    Returns:
        统计字典
    """
    split_count = Counter()
    qt_count = Counter()
    source_count = Counter()
    paper_ids = set()

    for s in selected:
        meta = s.get("metadata", {})
        sp = meta.get("split", "")
        split_count[sp] += 1
        qt_count[f"{sp}_{meta.get('question_type', '')}"] += 1
        source_count[meta.get("source", "")] += 1
        paper_ids.add(meta.get("paper_id", ""))

    return {
        "total": len(selected),
        "unique_papers": len(paper_ids),
        "split_distribution": dict(split_count.most_common()),
        "source_distribution": dict(source_count.most_common()),
        "question_type_top20": dict(qt_count.most_common(20)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="GRPO 训练数据子集筛选")
    parser.add_argument(
        "--sft-data",
        default="/root/shared-nvme/sciconsist_pilot/processed/scimdr_sft_train.jsonl",
        help="SFT 数据路径",
    )
    parser.add_argument(
        "--table-index",
        default="/root/shared-nvme/sciconsist_pilot/processed/table_structured/paper_tables.jsonl",
        help="paper_tables.jsonl 路径",
    )
    parser.add_argument(
        "--output",
        default="/root/shared-nvme/sciconsist_pilot/processed/grpo_train_30k.jsonl",
        help="输出路径",
    )
    parser.add_argument("--total", type=int, default=30000, help="目标样本数")
    parser.add_argument(
        "--split-ratio",
        default="0.50:0.35:0.15",
        help="tqa:mqa:vqa 比例 (冒号分隔)",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    ratios = [float(x) for x in args.split_ratio.split(":")]
    config = SubsetConfig(
        total=args.total,
        split_ratios={"tqa": ratios[0], "mqa": ratios[1], "vqa": ratios[2]},
        seed=args.seed,
    )
    rng = random.Random(config.seed)

    print("=" * 60)
    print("GRPO 训练数据子集筛选")
    print(f"目标: {config.total:,} 样本, 比例: {config.split_ratios}")
    print("=" * 60)

    # Step 1: 加载 table paper_ids
    print("\n[1/4] 加载 table index...")
    table_ids = load_table_paper_ids(args.table_index)
    print(f"Table index: {len(table_ids):,} papers")

    # Step 2: 过滤
    print("\n[2/4] 加载并过滤 SFT 数据...")
    groups = load_and_filter(args.sft_data, table_ids)

    # Step 3: 按配额采样
    print("\n[3/4] 分层采样...")
    all_selected: list[dict] = []
    for split in ["tqa", "mqa", "vqa"]:
        quota = int(config.total * config.split_ratios.get(split, 0))
        pool = groups.get(split, [])
        sampled = prioritized_sample(pool, quota, split, rng)
        print(f"  {split}: {len(sampled):,} / {quota:,} (pool={len(pool):,})")
        all_selected.extend(sampled)

    rng.shuffle(all_selected)
    print(f"\n总采样: {len(all_selected):,}")

    # Step 4: 写出
    print("\n[4/4] 写出...")
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for s in all_selected:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    stats = compute_stats(all_selected)
    stats_path = output_path.with_suffix(".stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print(f"\n输出: {output_path} ({output_path.stat().st_size / 1024 / 1024:.1f} MB)")
    print(f"统计: {stats_path}")
    print(f"\nSplit 分布: {stats['split_distribution']}")
    print(f"Unique papers: {stats['unique_papers']}")
    print(f"\nQuestion type top 10:")
    for qt, c in list(stats["question_type_top20"].items())[:10]:
        print(f"  {qt}: {c:,}")

    print("\nDone!")


if __name__ == "__main__":
    main()
