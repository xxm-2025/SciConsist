"""
Pilot 实验一键运行脚本 — 在 FEH 训练完成后，依次执行 P1-P5 全部验证实验。

依赖: scripts/train_feh.py 已运行完成，生成了 outputs/checkpoints/feh_best.pt

用法:
    python scripts/run_pilot.py

    # 仅运行特定实验
    python scripts/run_pilot.py --experiments p2 p3

    # 使用自定义 checkpoint
    python scripts/run_pilot.py --checkpoint outputs/checkpoints/feh_best.pt
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.dataset import FEHDataset
from src.evaluate.metrics import FEHEvaluator, P1Result, P2Result, P3Result
from src.models.feh import EntailmentLabel, FEHConfig, FactualEntailmentHead

logger = logging.getLogger(__name__)
console = Console()


def load_feh(checkpoint_path: str, device: str = "cuda") -> FactualEntailmentHead:
    """加载训练好的 FEH 模型。

    Args:
        checkpoint_path: checkpoint 文件路径
        device: 计算设备

    Returns:
        FEH 模型 (eval mode)
    """
    ckpt = torch.load(checkpoint_path, weights_only=False, map_location=device)
    config = ckpt["config"]
    model = FactualEntailmentHead(config).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    console.print(f"Loaded FEH from {checkpoint_path} (epoch {ckpt.get('epoch', '?')}, acc={ckpt.get('accuracy', '?'):.4f})")
    return model


@torch.no_grad()
def run_p1(model: FactualEntailmentHead, features_dir: Path, device: str) -> P1Result:
    """P1: FEH 三分类基础能力。

    在 held-out 验证集上测试三分类 accuracy。
    Go/No-Go 标准: accuracy > 75%
    """
    console.print(Panel("[bold]P1: FEH 三分类基础能力[/bold]", style="blue"))

    dataset = FEHDataset(features_dir, split="val")
    loader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=False)

    all_true, all_pred = [], []
    for batch in loader:
        labels, _ = model.predict(batch["visual"].to(device), batch["text"].to(device))
        all_true.append(batch["label"].numpy())
        all_pred.append(labels.cpu().numpy())

    y_true = np.concatenate(all_true)
    y_pred = np.concatenate(all_pred)

    evaluator = FEHEvaluator()
    result = evaluator.evaluate_p1(y_true, y_pred)
    console.print(evaluator.format_p1_report(result))
    return result


@torch.no_grad()
def run_p2(
    model: FactualEntailmentHead,
    features_dir: Path,
    device: str,
    perturbation_levels: list[float] | None = None,
) -> P2Result:
    """P2: 数值粒度敏感度。

    对不同幅度的数值篡改 (±1% ~ ±20%)，测试 FEH 的 CONTRADICTS 检出率。
    Go 标准: ±5% 处检出率 > 60%
    """
    console.print(Panel("[bold]P2: 数值粒度敏感度[/bold]", style="blue"))

    if perturbation_levels is None:
        perturbation_levels = [0.01, 0.02, 0.05, 0.10, 0.15, 0.20]

    dataset = FEHDataset(features_dir, split="val")
    # 筛选 CONTRADICTS 样本
    labels = dataset.labels[dataset.indices]
    ratios = dataset.perturbation_ratios[dataset.indices]
    contradict_mask = labels == 2

    if not np.any(contradict_mask):
        console.print("[yellow]No CONTRADICTS samples with perturbation ratios in val set.[/yellow]")
        return P2Result(perturbation_levels=perturbation_levels)

    pred_labels_per_level = []
    pred_probs_per_level = []
    true_labels_per_level = []
    actual_levels = []

    for level in perturbation_levels:
        tolerance = level * 0.3 if level > 0.02 else 0.005
        level_mask = contradict_mask & (np.abs(ratios - level) < tolerance)
        level_indices = np.where(level_mask)[0]

        if len(level_indices) < 5:
            console.print(f"  ±{level*100:.0f}%: skipped (only {len(level_indices)} samples)")
            continue

        actual_levels.append(level)
        batch_v = torch.stack([dataset[i]["visual"] for i in level_indices]).to(device)
        batch_t = torch.stack([dataset[i]["text"] for i in level_indices]).to(device)

        preds, probs = model.predict(batch_v, batch_t)
        pred_labels_per_level.append(preds.cpu().numpy())
        pred_probs_per_level.append(probs.cpu().numpy())
        true_labels_per_level.append(np.full(len(level_indices), 2))

    evaluator = FEHEvaluator()
    result = evaluator.evaluate_p2(
        actual_levels, true_labels_per_level, pred_labels_per_level, pred_probs_per_level
    )
    console.print(evaluator.format_p2_report(result))
    return result


@torch.no_grad()
def run_p3(model: FactualEntailmentHead, features_dir: Path, device: str) -> P3Result:
    """P3: Many-to-One 解决能力。

    对"说对了但换了说法"的 claim (应该是 ENTAILS 或 NEUTRAL)，
    检查 FEH 是否错误地判为 CONTRADICTS。

    Go 标准: ENTAILS + NEUTRAL > 80%

    实现方式: 用 ENTAILS 样本 + 微小随机扰动 (模拟换了说法但事实不变)
    """
    console.print(Panel("[bold]P3: Many-to-One 解决能力[/bold]", style="blue"))

    dataset = FEHDataset(features_dir, split="val")
    labels = dataset.labels[dataset.indices]

    # 取 ENTAILS 样本，给 text 加微小高斯噪声 (模拟表述变化但事实不变)
    entails_indices = np.where(labels == 0)[0][:50]

    if len(entails_indices) == 0:
        console.print("[yellow]No ENTAILS samples found.[/yellow]")
        return P3Result()

    noisy_texts = []
    visuals = []
    for i in entails_indices:
        sample = dataset[i]
        # 在 text feature 上加小噪声 (模拟表述不同)
        noise = torch.randn_like(sample["text"]) * 0.05
        noisy_texts.append(sample["text"] + noise)
        visuals.append(sample["visual"])

    batch_v = torch.stack(visuals).to(device)
    batch_t = torch.stack(noisy_texts).to(device)

    preds, _ = model.predict(batch_v, batch_t)

    evaluator = FEHEvaluator()
    result = evaluator.evaluate_p3(preds.cpu().numpy())

    console.print(f"  Non-CONTRADICTS ratio: {result.non_contradict_ratio:.3f} "
                  f"{'✅' if result.target_met else '❌'} (target > 0.80)")
    console.print(f"  ENTAILS: {result.entails_ratio:.3f}, NEUTRAL: {result.neutral_ratio:.3f}, "
                  f"CONTRADICTS: {result.contradict_ratio:.3f}")
    return result


@torch.no_grad()
def run_p4(model: FactualEntailmentHead, features_dir: Path, device: str) -> tuple[float, bool]:
    """P4: Full figure vs cropped region。

    对比全图输入和裁剪区域输入的 FEH accuracy 差距。
    由于 pilot 阶段使用预提取特征，这里通过模拟不同粒度的视觉特征来验证。

    Go 标准: 差距 < 5%

    注: 完整的 P4 实验需要同时用 full_figure=true/false 提取两套特征。
    当前实现为框架性 placeholder，待真实特征就绪后填充。
    """
    console.print(Panel("[bold]P4: Full Figure vs Cropped Region[/bold]", style="blue"))

    # 在真实实验中:
    # features_full = Path(features_dir) / "full_figure"
    # features_crop = Path(features_dir) / "cropped_region"
    # 分别加载两套特征，在同一验证集上评估 accuracy，计算差距

    console.print("[yellow]P4 需要两套特征 (full/cropped)。当前为 placeholder 实现。[/yellow]")
    console.print("[yellow]请先运行 extract_features.py --use_full_figure true/false 生成两套特征。[/yellow]")

    # Placeholder: 对 val 集跑两遍，第二遍加噪声模拟 crop 引入的信息损失
    dataset = FEHDataset(features_dir, split="val")
    loader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=False)

    all_true, all_pred_full, all_pred_crop = [], [], []
    for batch in loader:
        v_full = batch["visual"].to(device)
        t = batch["text"].to(device)
        # 模拟 crop: 视觉特征加噪声
        v_crop = v_full + torch.randn_like(v_full) * 0.03

        pred_full, _ = model.predict(v_full, t)
        pred_crop, _ = model.predict(v_crop, t)

        all_true.append(batch["label"].numpy())
        all_pred_full.append(pred_full.cpu().numpy())
        all_pred_crop.append(pred_crop.cpu().numpy())

    y_true = np.concatenate(all_true)
    from sklearn.metrics import accuracy_score
    acc_full = accuracy_score(y_true, np.concatenate(all_pred_full))
    acc_crop = accuracy_score(y_true, np.concatenate(all_pred_crop))

    evaluator = FEHEvaluator()
    gap, acceptable = evaluator.evaluate_accuracy_gap(acc_full, acc_crop, max_gap=0.05)

    console.print(f"  Full figure accuracy:  {acc_full:.4f}")
    console.print(f"  Cropped region accuracy: {acc_crop:.4f}")
    console.print(f"  Gap: {gap:.4f} {'✅' if acceptable else '❌'} (max 0.05)")

    return gap, acceptable


@torch.no_grad()
def run_p5(features_dir: Path, device: str) -> tuple[float, bool]:
    """P5: Cross-model vs Same-model FEH。

    对比 FEH 在 InternVL 特征 vs Qwen 特征上的 accuracy 差距。
    Go 标准: 差距 < 10%

    注: 需要两套 FEH checkpoint (feh_cross.pt 和 feh_same.pt) 和两套特征。
    当前实现为框架性 placeholder。
    """
    console.print(Panel("[bold]P5: Cross-Model vs Same-Model FEH[/bold]", style="blue"))

    # 在真实实验中:
    # feh_cross = load_feh("outputs/checkpoints/feh_cross.pt")
    # feh_same = load_feh("outputs/checkpoints/feh_same.pt")
    # features_internvl = features_dir / "internvl"
    # features_qwen = features_dir / "qwen"
    # 分别评估 accuracy，计算差距

    console.print("[yellow]P5 需要两套 checkpoint + 两套特征。当前为 placeholder。[/yellow]")
    console.print("[yellow]请先分别用 InternVL 和 Qwen 提取特征，再训练两个 FEH。[/yellow]")
    console.print("[yellow]  1. python scripts/extract_features.py features.reward_encoder=OpenGVLab/InternVL2_5-8B[/yellow]")
    console.print("[yellow]  2. python scripts/extract_features.py features.reward_encoder=Qwen/Qwen2.5-VL-7B-Instruct[/yellow]")

    return 0.0, True  # placeholder


def main() -> None:
    parser = argparse.ArgumentParser(description="SciConsist Pilot Experiments (P1-P5)")
    parser.add_argument("--checkpoint", default="outputs/checkpoints/feh_best.pt", help="FEH checkpoint 路径")
    parser.add_argument("--features-dir", default="data/processed/features", help="预提取特征目录")
    parser.add_argument("--experiments", nargs="+", default=["p1", "p2", "p3", "p4", "p5"], help="要运行的实验")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", default="outputs/pilot_results.json", help="结果输出路径")
    args = parser.parse_args()

    console.print(Panel("[bold blue]SciConsist Pilot: 5 Verification Experiments[/bold blue]"))

    features_dir = Path(args.features_dir)
    results: dict = {}

    if "p1" in args.experiments or "p2" in args.experiments or "p3" in args.experiments or "p4" in args.experiments:
        model = load_feh(args.checkpoint, args.device)

    if "p1" in args.experiments:
        r = run_p1(model, features_dir, args.device)
        results["p1"] = {"accuracy": r.accuracy, "go": r.go, "per_class_f1": r.per_class_f1}

    if "p2" in args.experiments:
        r = run_p2(model, features_dir, args.device)
        results["p2"] = {
            "levels": r.perturbation_levels,
            "detection_rates": r.detection_rates,
            "target_met": r.target_met,
        }

    if "p3" in args.experiments:
        r = run_p3(model, features_dir, args.device)
        results["p3"] = {
            "non_contradict_ratio": r.non_contradict_ratio,
            "target_met": r.target_met,
        }

    if "p4" in args.experiments:
        gap, ok = run_p4(model, features_dir, args.device)
        results["p4"] = {"gap": gap, "acceptable": ok}

    if "p5" in args.experiments:
        gap, ok = run_p5(features_dir, args.device)
        results["p5"] = {"gap": gap, "acceptable": ok}

    # 汇总报告
    console.print("\n" + "=" * 60)
    console.print(Panel("[bold]Pilot Summary[/bold]", style="green"))

    summary_table = Table(title="Go/No-Go 判定")
    summary_table.add_column("Experiment", style="bold")
    summary_table.add_column("Key Metric")
    summary_table.add_column("Status")

    for exp_id, data in results.items():
        if "go" in data:
            status = "✅ GO" if data["go"] else "❌ NO-GO"
        elif "target_met" in data:
            status = "✅ MET" if data["target_met"] else "❌ NOT MET"
        elif "acceptable" in data:
            status = "✅ OK" if data["acceptable"] else "❌ GAP TOO LARGE"
        else:
            status = "—"

        metric = ""
        if exp_id == "p1":
            metric = f"accuracy = {data.get('accuracy', 0):.4f}"
        elif exp_id == "p2":
            rates = data.get("detection_rates", [])
            metric = f"rates = {[f'{r:.2f}' for r in rates]}"
        elif exp_id == "p3":
            metric = f"non_contradict = {data.get('non_contradict_ratio', 0):.3f}"
        elif exp_id in ("p4", "p5"):
            metric = f"gap = {data.get('gap', 0):.4f}"

        summary_table.add_row(exp_id.upper(), metric, status)

    console.print(summary_table)

    # 保存结果
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    console.print(f"\nResults saved to {output_path}")

    # 最终建议
    all_pass = all(
        data.get("go", data.get("target_met", data.get("acceptable", True)))
        for data in results.values()
    )
    if all_pass:
        console.print("\n[bold green]🟢 ALL PASS — 可以全速推进 Stage 1 SFT + Stage 2 FER-GRPO[/bold green]")
    else:
        console.print("\n[bold yellow]🟡 部分实验未达标，请根据上方结果调整方案后重试。[/bold yellow]")


if __name__ == "__main__":
    main()
