"""
数据准备脚本 — 下载原始数据 + 构造 FEH 训练数据 + 提取特征并缓存。

这是 pilot 实验的第一步：
  1. 下载 S1-MMAlign 子集 (或使用 placeholder)
  2. 构造三类训练数据 (ENTAILS/NEUTRAL/CONTRADICTS)
  3. 用 InternVL2.5-8B 提取特征并缓存到磁盘

用法:
    # 使用 placeholder 数据快速跑通 pipeline
    python scripts/prepare_data.py

    # 使用真实数据
    python scripts/prepare_data.py --real-data

    # 仅构造数据，不提取特征 (当没有 GPU 时)
    python scripts/prepare_data.py --skip-features
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rich.console import Console

from src.data.prepare import construct_feh_training_data, download_s1mmalign
from src.features.extract import ExtractionConfig, FeatureExtractor

logger = logging.getLogger(__name__)
console = Console()


def main() -> None:
    parser = argparse.ArgumentParser(description="SciConsist Pilot: Data Preparation")
    parser.add_argument("--raw-dir", default="data/raw", help="原始数据目录")
    parser.add_argument("--processed-dir", default="data/processed", help="处理后数据目录")
    parser.add_argument("--real-data", action="store_true", help="使用真实数据（需网络）")
    parser.add_argument("--skip-features", action="store_true", help="跳过特征提取")
    parser.add_argument("--num-samples", type=int, default=60000, help="S1-MMAlign 下载数量")
    parser.add_argument("--num-per-class", type=int, default=50000, help="每类训练样本数")
    parser.add_argument("--model", default="OpenGVLab/InternVL2_5-8B", help="特征提取模型")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    console.print("[bold blue]SciConsist Pilot: Data Preparation[/bold blue]")

    # Step 1: 下载/加载原始数据
    console.print("\n[bold]Step 1: Loading raw data...[/bold]")
    if args.real_data:
        samples = download_s1mmalign(args.raw_dir, max_samples=args.num_samples, seed=args.seed)
    else:
        console.print("[yellow]Using placeholder data (pass --real-data for real data)[/yellow]")
        from src.data.prepare import _create_placeholder_data
        samples = _create_placeholder_data(args.num_samples, args.seed)
    console.print(f"  Loaded {len(samples)} raw samples")

    # Step 2: 构造三分类训练数据
    console.print("\n[bold]Step 2: Constructing FEH training data...[/bold]")
    all_samples, all_labels, all_ratios = construct_feh_training_data(
        entails_samples=samples,
        num_entails=args.num_per_class,
        num_neutral=args.num_per_class,
        num_contradicts=args.num_per_class,
        seed=args.seed,
    )

    # Train/Val split
    rng = np.random.RandomState(args.seed)
    n_total = len(all_samples)
    n_train = int(n_total * 0.9)
    perm = rng.permutation(n_total)

    train_indices = perm[:n_train]
    val_indices = perm[n_train:]

    console.print(f"  Train: {len(train_indices)}, Val: {len(val_indices)}")

    # Step 3: 提取特征
    if args.skip_features:
        console.print("\n[yellow]Skipping feature extraction (--skip-features)[/yellow]")
        console.print("[yellow]Run scripts/train_feh.py which will create placeholder features automatically.[/yellow]")
        return

    console.print(f"\n[bold]Step 3: Extracting features with {args.model}...[/bold]")
    config = ExtractionConfig(model_name=args.model)
    extractor = FeatureExtractor(config)

    for split_name, indices in [("train", train_indices), ("val", val_indices)]:
        split_texts = [all_samples[i].text for i in indices]
        split_images = [all_samples[i].figure_path for i in indices]
        split_labels = [all_labels[i] for i in indices]
        split_ratios = [all_ratios[i] for i in indices]

        console.print(f"  Extracting {split_name} ({len(indices)} samples)...")
        extractor.extract_and_cache(
            texts=split_texts,
            images=split_images,
            labels=split_labels,
            output_dir=Path(args.processed_dir) / "features",
            split=split_name,
            perturbation_ratios=split_ratios,
        )

    console.print("\n[bold green]✅ Data preparation complete![/bold green]")
    console.print("Next: python scripts/train_feh.py")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    main()
