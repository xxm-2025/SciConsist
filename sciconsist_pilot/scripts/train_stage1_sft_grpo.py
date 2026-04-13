from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from rich.console import Console
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.dataset import FEHDataset
from src.evaluate.metrics import FEHEvaluator
from src.models.feh import FEHConfig, FactualEntailmentHead

console = Console()


@dataclass
class StageMetrics:
    p1_accuracy: float
    p1_go: bool
    p3_non_contradict: float
    p3_target_met: bool
    p2_target_met: bool


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_reward_feh(checkpoint_path: str, device: str) -> FactualEntailmentHead:
    ckpt = torch.load(checkpoint_path, weights_only=False, map_location=device)
    config = ckpt["config"]
    model = FactualEntailmentHead(config).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def evaluate_p1(model: FactualEntailmentHead, loader: DataLoader, device: str, threshold: float = 0.75):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for batch in loader:
            labels, _ = model.predict(batch["visual"].to(device), batch["text"].to(device))
            y_true.append(batch["label"].numpy())
            y_pred.append(labels.cpu().numpy())
    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    evaluator = FEHEvaluator()
    return evaluator.evaluate_p1(y_true, y_pred, go_threshold=threshold)


def evaluate_p2(model: FactualEntailmentHead, dataset: FEHDataset, device: str) -> bool:
    labels = dataset.labels[dataset.indices]
    ratios = dataset.perturbation_ratios[dataset.indices]
    contradict_mask = labels == 2
    if not np.any(contradict_mask):
        return False

    perturbation_levels = [0.01, 0.02, 0.05, 0.10, 0.15, 0.20]
    pred_labels_per_level = []
    pred_probs_per_level = []
    true_labels_per_level = []
    actual_levels = []

    with torch.no_grad():
        for level in perturbation_levels:
            tolerance = level * 0.3 if level > 0.02 else 0.005
            level_mask = contradict_mask & (np.abs(ratios - level) < tolerance)
            level_indices = np.where(level_mask)[0]
            if len(level_indices) < 5:
                continue

            actual_levels.append(level)
            batch_v = torch.stack([dataset[i]["visual"] for i in level_indices]).to(device)
            batch_t = torch.stack([dataset[i]["text"] for i in level_indices]).to(device)
            preds, probs = model.predict(batch_v, batch_t)
            pred_labels_per_level.append(preds.cpu().numpy())
            pred_probs_per_level.append(probs.cpu().numpy())
            true_labels_per_level.append(np.full(len(level_indices), 2))

    evaluator = FEHEvaluator()
    result = evaluator.evaluate_p2(actual_levels, true_labels_per_level, pred_labels_per_level, pred_probs_per_level)
    return bool(result.target_met)


def evaluate_p3(model: FactualEntailmentHead, dataset: FEHDataset, device: str, noise_scale: float = 0.05):
    labels = dataset.labels[dataset.indices]
    entails_indices = np.where(labels == 0)[0][:50]
    if len(entails_indices) == 0:
        return 0.0, False

    visuals = []
    noisy_texts = []
    for i in entails_indices:
        sample = dataset[i]
        visuals.append(sample["visual"])
        noisy_texts.append(sample["text"] + torch.randn_like(sample["text"]) * noise_scale)

    with torch.no_grad():
        batch_v = torch.stack(visuals).to(device)
        batch_t = torch.stack(noisy_texts).to(device)
        preds, _ = model.predict(batch_v, batch_t)

    evaluator = FEHEvaluator()
    result = evaluator.evaluate_p3(preds.cpu().numpy())
    return float(result.non_contradict_ratio), bool(result.target_met)


def train_stage1_sft(
    model: FactualEntailmentHead,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: str,
    epochs: int,
    lr: float,
    class_weights: torch.Tensor | None,
) -> tuple[FactualEntailmentHead, float]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    best_acc = -1.0
    best_state = None

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            v = batch["visual"].to(device)
            t = batch["text"].to(device)
            y = batch["label"].to(device)
            logits, _ = model(v, t)
            loss = criterion(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        p1 = evaluate_p1(model, val_loader, device)
        avg_loss = total_loss / max(len(train_loader), 1)
        console.print(f"[SFT] Epoch {epoch+1}/{epochs} | loss={avg_loss:.4f} | val_p1={p1.accuracy:.4f}")
        if p1.accuracy > best_acc:
            best_acc = p1.accuracy
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, float(best_acc)


def train_stage2_grpo(
    model: FactualEntailmentHead,
    reward_model: FactualEntailmentHead,
    train_loader: DataLoader,
    val_loader: DataLoader,
    val_dataset: FEHDataset,
    device: str,
    epochs: int,
    lr: float,
    kl_beta: float,
    entropy_coef: float,
    group_size: int,
    supervised_coef: float,
) -> tuple[FactualEntailmentHead, StageMetrics]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    ce_criterion = nn.CrossEntropyLoss()

    ref_model = FactualEntailmentHead(model.config).to(device)
    ref_model.load_state_dict(model.state_dict())
    ref_model.eval()

    reward_vec = torch.tensor([1.0, -0.1, -0.5], device=device)

    best_score = -1e9
    best_state = None
    best_metrics = StageMetrics(0.0, False, 0.0, False, False)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            v = batch["visual"].to(device)
            t = batch["text"].to(device)
            y = batch["label"].to(device)

            logits, _ = model(v, t)
            probs = torch.softmax(logits, dim=-1)
            log_probs = torch.log_softmax(logits, dim=-1)

            with torch.no_grad():
                reward_logits, _ = reward_model(v, t)
                reward_probs = torch.softmax(reward_logits, dim=-1)
                reward_top = reward_probs.argmax(dim=-1, keepdim=True)

                ref_logits, _ = ref_model(v, t)
                ref_log_probs = torch.log_softmax(ref_logits, dim=-1)
                ref_probs = torch.softmax(ref_logits, dim=-1)

            actions = torch.multinomial(probs, num_samples=group_size, replacement=True)
            chosen_logp = torch.gather(log_probs, 1, actions)

            action_base = reward_vec[actions]
            action_conf = torch.gather(reward_probs, 1, actions)
            alignment_bonus = (actions == reward_top).float() * 0.1
            rewards = action_base * (0.5 + action_conf) + alignment_bonus

            adv = (rewards - rewards.mean(dim=1, keepdim=True)) / (rewards.std(dim=1, keepdim=True) + 1e-6)
            pg_loss = -(adv.detach() * chosen_logp).mean()

            kl = (probs * (log_probs - ref_log_probs)).sum(dim=-1).mean()
            entropy = -(probs * log_probs).sum(dim=-1).mean()

            loss = pg_loss + kl_beta * kl - entropy_coef * entropy
            if supervised_coef > 0:
                ce_loss = ce_criterion(logits, y)
                loss = loss + supervised_coef * ce_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        p1 = evaluate_p1(model, val_loader, device)
        p3_ratio, p3_met = evaluate_p3(model, val_dataset, device)
        p2_met = evaluate_p2(model, val_dataset, device)
        score = p1.accuracy + (0.05 if p2_met else 0.0) + 0.05 * p3_ratio
        avg_loss = total_loss / max(len(train_loader), 1)

        console.print(
            f"[GRPO] Epoch {epoch+1}/{epochs} | loss={avg_loss:.4f} | val_p1={p1.accuracy:.4f} | "
            f"val_p3={p3_ratio:.3f}"
        )

        if score > best_score:
            best_score = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_metrics = StageMetrics(
                p1_accuracy=float(p1.accuracy),
                p1_go=bool(p1.go),
                p3_non_contradict=float(p3_ratio),
                p3_target_met=bool(p3_met),
                p2_target_met=bool(p2_met),
            )

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage1 SFT + FER-GRPO small-run framework")
    parser.add_argument("--features-dir", required=True, help="Feature root containing train/val npy files")
    parser.add_argument("--reward-checkpoint", required=True, help="Frozen FEH checkpoint used as FER reward model")
    parser.add_argument("--output-dir", default="outputs/stage1_grpo", help="Output directory")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--max-train-samples", type=int, default=1800)
    parser.add_argument("--max-val-samples", type=int, default=240)

    parser.add_argument("--latent-dim", type=int, default=768)
    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument("--batch-size", type=int, default=160)

    parser.add_argument("--sft-epochs", type=int, default=6)
    parser.add_argument("--sft-lr", type=float, default=8e-4)

    parser.add_argument("--grpo-epochs", type=int, default=8)
    parser.add_argument("--grpo-lr", type=float, default=5e-4)
    parser.add_argument("--kl-beta", type=float, default=0.04)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--grpo-supervised-coef", type=float, default=0.7)

    parser.add_argument("--class-weights", type=str, default="1.32,1.59,0.77")
    args = parser.parse_args()

    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    ckpt_dir = output_dir / "checkpoints"
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    train_ds = FEHDataset(args.features_dir, split="train", max_samples=args.max_train_samples, seed=args.seed)
    val_ds = FEHDataset(args.features_dir, split="val", max_samples=args.max_val_samples, seed=args.seed)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    sample = train_ds[0]["visual"]
    hidden_dim = int(sample.shape[0])
    config = FEHConfig(hidden_dim=hidden_dim, latent_dim=args.latent_dim, num_classes=3, dropout=args.dropout)
    policy = FactualEntailmentHead(config).to(args.device)

    weights = torch.tensor([float(x) for x in args.class_weights.split(",")], dtype=torch.float32, device=args.device)
    reward_model = load_reward_feh(args.reward_checkpoint, args.device)

    console.print("[bold blue]Stage 1: SFT[/bold blue]")
    policy, sft_best_p1 = train_stage1_sft(
        policy,
        train_loader,
        val_loader,
        args.device,
        epochs=args.sft_epochs,
        lr=args.sft_lr,
        class_weights=weights,
    )
    torch.save(
        {
            "model_state_dict": policy.state_dict(),
            "config": config,
            "stage": "sft",
            "best_p1": sft_best_p1,
        },
        ckpt_dir / "policy_stage1_best.pt",
    )

    console.print("[bold blue]Stage 2: FER-GRPO[/bold blue]")
    policy, grpo_metrics = train_stage2_grpo(
        policy,
        reward_model,
        train_loader,
        val_loader,
        val_ds,
        args.device,
        epochs=args.grpo_epochs,
        lr=args.grpo_lr,
        kl_beta=args.kl_beta,
        entropy_coef=args.entropy_coef,
        group_size=args.group_size,
        supervised_coef=args.grpo_supervised_coef,
    )

    torch.save(
        {
            "model_state_dict": policy.state_dict(),
            "config": config,
            "stage": "grpo",
            "metrics": asdict(grpo_metrics),
        },
        ckpt_dir / "policy_stage2_best.pt",
    )

    final_p1 = evaluate_p1(policy, val_loader, args.device)
    final_p3_ratio, final_p3_met = evaluate_p3(policy, val_ds, args.device)
    final_p2_met = evaluate_p2(policy, val_ds, args.device)

    result = {
        "sft_best_p1": sft_best_p1,
        "final": {
            "p1_accuracy": float(final_p1.accuracy),
            "p1_go": bool(final_p1.go),
            "p2_target_met": bool(final_p2_met),
            "p3_non_contradict": float(final_p3_ratio),
            "p3_target_met": bool(final_p3_met),
            "per_class_f1": final_p1.per_class_f1,
        },
        "args": vars(args),
        "paths": {
            "stage1_checkpoint": str(ckpt_dir / "policy_stage1_best.pt"),
            "stage2_checkpoint": str(ckpt_dir / "policy_stage2_best.pt"),
        },
    }

    out_json = output_dir / "stage1_grpo_results.json"
    out_json.write_text(json.dumps(result, indent=2), encoding="utf-8")

    console.print("\n[bold green]Stage1 SFT + FER-GRPO completed[/bold green]")
    console.print(f"Results: {out_json}")
    console.print(
        f"Final P1={final_p1.accuracy:.4f}, P2={'MET' if final_p2_met else 'NOT_MET'}, "
        f"P3={final_p3_ratio:.3f} ({'MET' if final_p3_met else 'NOT_MET'})"
    )


if __name__ == "__main__":
    main()
