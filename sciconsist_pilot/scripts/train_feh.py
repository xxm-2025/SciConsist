"""
FEH 训练脚本 — 在 InternVL2.5 预提取特征上训练三分类头。

训练完成后自动运行 P1 (基础准确率) 评估并输出 Go/No-Go 判定。
支持 Hydra 配置覆盖。

用法:
    # 使用默认配置
    python scripts/train_feh.py

    # 覆盖参数
    python scripts/train_feh.py training.epochs=10 training.lr=5e-4

    # 使用 placeholder 数据快速验证 pipeline
    python scripts/train_feh.py data.num_entails=1000 data.num_neutral=1000 data.num_contradicts=1000
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import hydra
import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig
from rich.console import Console
from rich.table import Table
from torch.utils.data import DataLoader

# 确保 src 在 import path 中
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.dataset import FEHDataset
from src.evaluate.metrics import FEHEvaluator
from src.models.feh import FEHConfig, FactualEntailmentHead

logger = logging.getLogger(__name__)
console = Console()


def train_one_epoch(
    model: FactualEntailmentHead,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: str,
) -> float:
    """训练一个 epoch。

    Args:
        model: FEH 模型
        dataloader: 训练数据加载器
        optimizer: 优化器
        criterion: 损失函数
        device: 计算设备

    Returns:
        平均 loss
    """
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch in dataloader:
        h_visual = batch["visual"].to(device)
        h_text = batch["text"].to(device)
        labels = batch["label"].to(device)

        logits, _ = model(h_visual, h_text)
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(
    model: FactualEntailmentHead,
    dataloader: DataLoader,
    device: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """在验证集上评估。

    Args:
        model: FEH 模型
        dataloader: 验证数据加载器
        device: 计算设备

    Returns:
        (y_true, y_pred, y_probs): 真实标签、预测标签、预测概率
    """
    model.eval()
    all_true, all_pred, all_probs = [], [], []

    for batch in dataloader:
        h_visual = batch["visual"].to(device)
        h_text = batch["text"].to(device)
        labels = batch["label"]

        pred_labels, probs = model.predict(h_visual, h_text)

        all_true.append(labels.numpy())
        all_pred.append(pred_labels.cpu().numpy())
        all_probs.append(probs.cpu().numpy())

    return np.concatenate(all_true), np.concatenate(all_pred), np.concatenate(all_probs)


def create_placeholder_features(cfg: DictConfig) -> None:
    """生成占位特征数据用于 pipeline 调试。

    当真实数据尚未准备好时，生成随机特征来验证训练 pipeline 能跑通。
    """
    rng = np.random.RandomState(cfg.data.seed)
    hidden_dim = cfg.model.hidden_dim

    total = cfg.data.num_entails + cfg.data.num_neutral + cfg.data.num_contradicts
    n_train = int(total * cfg.data.train_split)
    n_val = total - n_train

    for split, n in [("train", n_train), ("val", n_val)]:
        split_dir = Path(cfg.data.processed_dir) / "features" / split
        split_dir.mkdir(parents=True, exist_ok=True)

        # 生成有区分度的特征：
        # ENTAILS: visual 和 text 特征接近
        # NEUTRAL: visual 和 text 特征无关
        # CONTRADICTS: visual 和 text 特征方向相反
        n_per_class = n // 3
        features_v, features_t, labels, ratios = [], [], [], []

        for label in range(3):
            base = rng.randn(n_per_class, hidden_dim).astype(np.float32) * 0.1
            if label == 0:  # ENTAILS: 近似相同
                v = base + rng.randn(n_per_class, hidden_dim).astype(np.float32) * 0.02
                t = base + rng.randn(n_per_class, hidden_dim).astype(np.float32) * 0.02
            elif label == 1:  # NEUTRAL: 独立随机
                v = rng.randn(n_per_class, hidden_dim).astype(np.float32) * 0.1
                t = rng.randn(n_per_class, hidden_dim).astype(np.float32) * 0.1
            else:  # CONTRADICTS: 方向相反
                v = base + rng.randn(n_per_class, hidden_dim).astype(np.float32) * 0.02
                t = -base + rng.randn(n_per_class, hidden_dim).astype(np.float32) * 0.02

            features_v.append(v)
            features_t.append(t)
            labels.extend([label] * n_per_class)
            if label == 2:
                ratios.extend(rng.choice([0.01, 0.02, 0.05, 0.10, 0.20], n_per_class).tolist())
            else:
                ratios.extend([0.0] * n_per_class)

        perm = rng.permutation(len(labels))
        np.save(split_dir / "visual_features.npy", np.concatenate(features_v)[perm])
        np.save(split_dir / "text_features.npy", np.concatenate(features_t)[perm])
        np.save(split_dir / "labels.npy", np.array(labels)[perm])
        np.save(split_dir / "perturbation_ratios.npy", np.array(ratios)[perm])

    logger.info(f"Placeholder features created: train={n_train}, val={n_val}")


@hydra.main(config_path="../run/conf", config_name="pilot", version_base=None)
def main(cfg: DictConfig) -> None:
    """FEH 训练主入口。"""
    console.print("[bold blue]SciConsist Pilot: FEH Training[/bold blue]")

    device = cfg.training.device if torch.cuda.is_available() else "cpu"
    torch.manual_seed(cfg.training.seed)
    np.random.seed(cfg.training.seed)

    # 检查特征是否已提取
    features_dir = Path(cfg.data.processed_dir) / "features"
    if not (features_dir / "train" / "visual_features.npy").exists():
        console.print("[yellow]Features not found. Creating placeholder data for pipeline validation...[/yellow]")
        create_placeholder_features(cfg)

    # 加载数据
    train_ds = FEHDataset(features_dir, split="train")
    val_ds = FEHDataset(features_dir, split="val")

    console.print(f"Train: {len(train_ds)} samples, label distribution: {train_ds.label_counts}")
    console.print(f"Val:   {len(val_ds)} samples, label distribution: {val_ds.label_counts}")

    train_loader = DataLoader(train_ds, batch_size=cfg.training.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=cfg.training.batch_size, shuffle=False, num_workers=0)

    # 初始化模型
    feh_config = FEHConfig(
        hidden_dim=cfg.model.hidden_dim,
        latent_dim=cfg.model.latent_dim,
        num_classes=cfg.model.num_classes,
        dropout=cfg.model.dropout,
        fuse_layer_indices=cfg.model.fuse_layer_indices,
    )
    model = FactualEntailmentHead(feh_config).to(device)
    param_count = sum(p.numel() for p in model.parameters())
    console.print(f"FEH parameters: {param_count:,} ({param_count/1e6:.1f}M)")

    # 训练配置
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.training.lr, weight_decay=cfg.training.weight_decay)
    criterion = nn.CrossEntropyLoss()

    total_steps = len(train_loader) * cfg.training.epochs
    warmup_steps = int(total_steps * cfg.training.warmup_ratio)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps)

    # 训练循环
    best_acc = 0.0
    patience_counter = 0
    checkpoint_dir = Path(cfg.output.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    evaluator = FEHEvaluator()

    for epoch in range(cfg.training.epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        scheduler.step()

        y_true, y_pred, y_probs = evaluate(model, val_loader, device)
        p1_result = evaluator.evaluate_p1(y_true, y_pred, go_threshold=cfg.pilot.p1.go_threshold)

        console.print(
            f"Epoch {epoch+1}/{cfg.training.epochs} | "
            f"Loss: {train_loss:.4f} | "
            f"Val Acc: {p1_result.accuracy:.4f} | "
            f"{'✅' if p1_result.go else '❌'}"
        )

        if p1_result.accuracy > best_acc:
            best_acc = p1_result.accuracy
            patience_counter = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": feh_config,
                    "epoch": epoch,
                    "accuracy": best_acc,
                },
                checkpoint_dir / "feh_best.pt",
            )
        else:
            patience_counter += 1
            if patience_counter >= cfg.training.early_stopping_patience:
                console.print(f"[yellow]Early stopping at epoch {epoch+1}[/yellow]")
                break

    # 最终评估
    console.print("\n" + "=" * 60)
    console.print("[bold]Final Evaluation (Best Checkpoint)[/bold]")

    ckpt = torch.load(checkpoint_dir / "feh_best.pt", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])

    y_true, y_pred, y_probs = evaluate(model, val_loader, device)
    p1_result = evaluator.evaluate_p1(y_true, y_pred, go_threshold=cfg.pilot.p1.go_threshold)

    console.print(evaluator.format_p1_report(p1_result))

    # Go/No-Go 判定
    console.print("\n" + "=" * 60)
    if p1_result.accuracy >= cfg.pilot.p1.go_threshold:
        console.print(f"[bold green]🟢 GO: accuracy = {p1_result.accuracy:.4f} ≥ {cfg.pilot.p1.go_threshold}[/bold green]")
        console.print("[green]FEH 三分类能力达标，可以继续 P2-P5 pilot 实验。[/green]")
    elif p1_result.accuracy >= cfg.pilot.p1.nogo_threshold:
        console.print(
            f"[bold yellow]🟡 MARGINAL: accuracy = {p1_result.accuracy:.4f} "
            f"(between {cfg.pilot.p1.nogo_threshold} and {cfg.pilot.p1.go_threshold})[/bold yellow]"
        )
        console.print("[yellow]建议：增加 FEH 训练数据，或尝试更大的 latent_dim。[/yellow]")
    else:
        console.print(
            f"[bold red]🔴 NO-GO: accuracy = {p1_result.accuracy:.4f} < {cfg.pilot.p1.nogo_threshold}[/bold red]"
        )
        console.print("[red]FEH 基础能力不足，需要根本性调整（数据质量/模型架构）。[/red]")


if __name__ == "__main__":
    main()
