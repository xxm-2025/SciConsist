from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.features.extract import ExtractionConfig, FeatureExtractor
from src.models.feh import FactualEntailmentHead


@dataclass
class Sample:
    sample_id: str
    claim_text: str
    label_id: int
    caption: str
    image_path: str


def normalize_rewards(rewards: list[float], eps: float = 1e-6) -> np.ndarray:
    arr = np.asarray(rewards, dtype=np.float32)
    if arr.size == 0:
        return arr
    mean = float(arr.mean())
    std = float(arr.std())
    if std < eps:
        return arr - mean
    # Clip normalized rewards to reduce high-variance updates.
    return np.clip((arr - mean) / (std + eps), -2.0, 2.0)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_manifest(path: Path, max_samples: int) -> list[Sample]:
    rows: list[Sample] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            if not obj.get("image_exists"):
                continue
            raw = obj.get("raw", {})
            rows.append(
                Sample(
                    sample_id=str(obj.get("sample_id", "")),
                    claim_text=str(obj.get("text", "")),
                    label_id=int(obj.get("label_id", 1)),
                    caption=str(raw.get("caption", "")),
                    image_path=str(obj.get("image_path", "")),
                )
            )
            if max_samples > 0 and len(rows) >= max_samples:
                break
    return rows


def split_data(rows: list[Sample], train_ratio: float, seed: int) -> tuple[list[Sample], list[Sample]]:
    rng = random.Random(seed)
    rows2 = rows.copy()
    rng.shuffle(rows2)
    n_train = max(1, int(len(rows2) * train_ratio))
    return rows2[:n_train], rows2[n_train:]


def split_data_with_fixed_val(
    rows: list[Sample],
    train_ratio: float,
    seed: int,
    fixed_val_ids: set[str] | None,
) -> tuple[list[Sample], list[Sample]]:
    if fixed_val_ids:
        val_rows = [r for r in rows if r.sample_id in fixed_val_ids]
        train_rows = [r for r in rows if r.sample_id not in fixed_val_ids]
        if not val_rows or not train_rows:
            raise ValueError("Fixed val ids produce empty train/val split")
        return train_rows, val_rows
    return split_data(rows, train_ratio, seed)


def load_fixed_val_ids(path: str | None) -> set[str] | None:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"fixed val id file not found: {p}")
    data = json.loads(p.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        ids = data.get("val_ids", [])
    else:
        ids = data
    return set(str(x) for x in ids)


def load_feh(checkpoint_path: str, device: str) -> FactualEntailmentHead:
    ckpt = torch.load(checkpoint_path, weights_only=False, map_location=device)
    config = ckpt["config"]
    model = FactualEntailmentHead(config).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def build_prompt(sample: Sample) -> str:
    return (
        "You are a scientific consistency auditor.\n"
        "Given a figure context and a candidate claim, write 2-4 factual statements and indicate if there is conflict.\n"
        "When you output CONFLICT, at least one claim must include concrete numeric evidence.\n\n"
        f"Figure caption:\n{sample.caption}\n\n"
        f"Candidate claim:\n{sample.claim_text}\n\n"
        "Output format:\n"
        "- Claims:\n"
        "  1) [number] vs [number], ...\n"
        "  2) [number] vs [number], ...\n"
        "- Conflict verdict: CONSISTENT or CONFLICT\n"
    )


_SENT_SPLIT = re.compile(r"[\n\.!?;]+")
_CLAIM_HINT = re.compile(r"\d|increase|decrease|higher|lower|more|less|significant|outperform|worse", re.IGNORECASE)


def extract_claims(text: str) -> list[str]:
    claims: list[str] = []
    for part in _SENT_SPLIT.split(text):
        s = part.strip(" -\t")
        if len(s) < 12:
            continue
        if _CLAIM_HINT.search(s):
            claims.append(s)
    if not claims:
        chunks = [x.strip() for x in text.split("\n") if len(x.strip()) > 12]
        claims = chunks[:3]
    return claims[:6]


def conflict_flag(text: str) -> bool:
    t = text.lower()
    return ("conflict" in t) or ("inconsisten" in t) or ("contradict" in t)


_NUM_PATTERN = re.compile(r"[-+]?\d+(?:\.\d+)?")


def has_numeric_evidence(text: str, min_numbers: int = 2) -> bool:
    return len(_NUM_PATTERN.findall(text)) >= min_numbers


def feh_claim_reward(
    feh: FactualEntailmentHead,
    extractor: FeatureExtractor,
    sample: Sample,
    response: str,
    device: str,
    c2_correct_bonus: float,
    c2_miss_penalty: float,
    c2_prob_bonus: float,
    false_contra_penalty: float,
    numeric_conflict_bonus: float,
) -> tuple[float, dict]:
    claims = extract_claims(response)
    if not claims:
        return -0.3, {
            "n_claims": 0,
            "non_contradict_ratio": 0.0,
            "has_conflict": False,
            "contradict_ratio": 0.0,
            "contradict_prob_mean": 0.0,
        }

    text_feats = extractor.extract_text_features(claims)
    visual_feats = extractor.extract_visual_features([sample.image_path] * len(claims))

    h_t = torch.from_numpy(text_feats).to(device)
    h_v = torch.from_numpy(visual_feats).to(device)

    with torch.no_grad():
        labels, probs = feh.predict(h_v, h_t)

    labels_np = labels.detach().cpu().numpy().tolist()
    probs_np = probs.detach().cpu().numpy()
    reward_map = {0: 1.0, 1: -0.1, 2: -0.5}
    base = float(np.mean([reward_map[int(x)] for x in labels_np]))

    has_real_conflict = any(int(x) == 2 for x in labels_np)
    contradict_ratio = float(sum(int(x) == 2 for x in labels_np) / len(labels_np))
    contradict_prob_mean = float(np.mean(probs_np[:, 2]))
    said_conflict = conflict_flag(response)
    numeric_evidence = has_numeric_evidence(response)
    conflict_bonus = 0.0
    if said_conflict:
        conflict_bonus = 1.5 if has_real_conflict else -1.0

    entails_ratio = float(sum(int(x) == 0 for x in labels_np) / len(labels_np))
    info_penalty = -0.5 if entails_ratio < 0.3 else 0.0

    # Soft constraint: reward/penalty scales continuously with contradiction ratio/probability.
    c2_bonus = 0.0
    if int(sample.label_id) == 2:
        c2_bonus += c2_correct_bonus * contradict_ratio
        c2_bonus += c2_prob_bonus * contradict_prob_mean
        c2_bonus -= c2_miss_penalty * (1.0 - contradict_ratio)
    else:
        c2_bonus -= false_contra_penalty * contradict_ratio

    evidence_bonus = 0.0
    if said_conflict and has_real_conflict and numeric_evidence:
        evidence_bonus += numeric_conflict_bonus
    if said_conflict and (not numeric_evidence):
        evidence_bonus -= 0.15

    total = base + 0.2 * conflict_bonus + info_penalty + c2_bonus + evidence_bonus
    non_contradict_ratio = float(sum(int(x) != 2 for x in labels_np) / len(labels_np))
    meta = {
        "n_claims": len(claims),
        "non_contradict_ratio": non_contradict_ratio,
        "has_conflict": has_real_conflict,
        "contradict_ratio": contradict_ratio,
        "contradict_prob_mean": contradict_prob_mean,
        "has_numeric_evidence": numeric_evidence,
        "labels": labels_np,
    }
    return total, meta


def build_balanced_epoch_rows(train_rows: list[Sample], seed: int, balance_train: bool, c2_oversample_factor: float) -> list[Sample]:
    rng = random.Random(seed)
    rows = train_rows.copy()
    if not balance_train:
        rng.shuffle(rows)
        return rows

    buckets: dict[int, list[Sample]] = defaultdict(list)
    for s in rows:
        buckets[int(s.label_id)].append(s)
    if not buckets:
        return rows

    max_count = max(len(v) for v in buckets.values() if len(v) > 0)
    sampled: list[Sample] = []
    for label, items in buckets.items():
        if not items:
            continue
        target = max_count
        if label == 2:
            target = int(max_count * max(1.0, c2_oversample_factor))
        for _ in range(target):
            sampled.append(items[rng.randrange(len(items))])

    rng.shuffle(sampled)
    return sampled


def generate_candidates(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    device: str,
    group_size: int,
    max_new_tokens: int,
    temperature: float,
) -> list[str]:
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(
        **inputs,
        do_sample=True,
        temperature=temperature,
        top_p=0.9,
        num_return_sequences=group_size,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.eos_token_id,
    )
    responses: list[str] = []
    prompt_len = inputs["input_ids"].shape[1]
    for seq in outputs:
        text = tokenizer.decode(seq[prompt_len:], skip_special_tokens=True)
        responses.append(text.strip())
    return responses


def sft_loss_on_response(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    response: str,
    device: str,
    max_seq_len: int,
) -> torch.Tensor:
    full = prompt + "\n" + response
    enc = tokenizer(full, return_tensors="pt", truncation=True, max_length=max_seq_len).to(device)
    with torch.enable_grad():
        out = model(**enc, labels=enc["input_ids"])
        return out.loss


def evaluate(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    feh: FactualEntailmentHead,
    extractor: FeatureExtractor,
    val_rows: list[Sample],
    device: str,
    max_new_tokens: int,
    c2_correct_bonus: float,
    c2_miss_penalty: float,
    c2_prob_bonus: float,
    false_contra_penalty: float,
    numeric_conflict_bonus: float,
) -> dict:
    if not val_rows:
        return {"avg_reward": 0.0, "avg_non_contradict_ratio": 0.0, "n": 0}

    rewards = []
    nc_ratios = []
    for s in val_rows:
        prompt = build_prompt(s)
        resp = generate_candidates(model, tokenizer, prompt, device, 1, max_new_tokens, temperature=0.7)[0]
        r, meta = feh_claim_reward(
            feh,
            extractor,
            s,
            resp,
            device,
            c2_correct_bonus=c2_correct_bonus,
            c2_miss_penalty=c2_miss_penalty,
            c2_prob_bonus=c2_prob_bonus,
            false_contra_penalty=false_contra_penalty,
            numeric_conflict_bonus=numeric_conflict_bonus,
        )
        rewards.append(r)
        nc_ratios.append(meta["non_contradict_ratio"])
    return {
        "avg_reward": float(np.mean(rewards)),
        "avg_non_contradict_ratio": float(np.mean(nc_ratios)),
        "n": len(val_rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage2: real policy generation + claim-level FEH reward")
    parser.add_argument("--manifest", default="/root/shared-nvme/sciconsist_pilot/raw/trainable/musciclaims_feh_manifest.jsonl")
    parser.add_argument("--reward-checkpoint", required=True)
    parser.add_argument("--policy-model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--extractor-model", default="OpenGVLab/InternVL2_5-8B")
    parser.add_argument("--output-dir", default="/root/shared-nvme/sciconsist_pilot/outputs/stage2_policy_grpo")
    parser.add_argument("--max-samples", type=int, default=60)
    parser.add_argument("--train-ratio", type=float, default=0.85)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--group-size", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--adv-scale", type=float, default=0.6)
    parser.add_argument("--max-seq-len", type=int, default=512)
    parser.add_argument("--balance-train", action="store_true", help="Enable class-balanced epoch sampling")
    parser.add_argument("--c2-oversample-factor", type=float, default=1.8)
    parser.add_argument("--c2-correct-bonus", type=float, default=0.8)
    parser.add_argument("--c2-miss-penalty", type=float, default=0.7)
    parser.add_argument("--c2-prob-bonus", type=float, default=0.5)
    parser.add_argument("--false-contra-penalty", type=float, default=0.25)
    parser.add_argument("--numeric-conflict-bonus", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--fixed-val-ids", default="", help="JSON file containing fixed validation sample_id list")
    args = parser.parse_args()

    set_seed(args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_manifest(Path(args.manifest), args.max_samples)
    fixed_val_ids = load_fixed_val_ids(args.fixed_val_ids)
    train_rows, val_rows = split_data_with_fixed_val(rows, args.train_ratio, args.seed, fixed_val_ids)

    # Persist validation IDs for reproducible comparisons across runs.
    val_ids_path = out_dir / "val_ids.json"
    val_ids_path.write_text(json.dumps([r.sample_id for r in val_rows], indent=2), encoding="utf-8")

    tokenizer = AutoTokenizer.from_pretrained(args.policy_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.policy_model,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if args.device == "cuda" else torch.float32,
    ).to(args.device)
    # Reduce activation memory for full-parameter updates.
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    if hasattr(model, "config") and hasattr(model.config, "use_cache"):
        model.config.use_cache = False

    optimizer = AdamW(model.parameters(), lr=args.lr)

    feh = load_feh(args.reward_checkpoint, args.device)
    extractor = FeatureExtractor(
        ExtractionConfig(
            model_name=args.extractor_model,
            batch_size=1,
            fail_on_placeholder=True,
        )
    )

    history = []
    best_val = -1e9
    best_state = None
    best_epoch = -1
    ema_baseline = 0.0
    train_label_counts = dict(Counter(int(s.label_id) for s in train_rows))

    for epoch in range(args.epochs):
        model.train()
        losses = []
        rewards = []

        epoch_rows = build_balanced_epoch_rows(
            train_rows,
            seed=args.seed + epoch,
            balance_train=args.balance_train,
            c2_oversample_factor=args.c2_oversample_factor,
        )
        epoch_label_counts = dict(Counter(int(s.label_id) for s in epoch_rows))
        for step_idx, sample in enumerate(epoch_rows):
            prompt = build_prompt(sample)
            cands = generate_candidates(
                model,
                tokenizer,
                prompt,
                args.device,
                args.group_size,
                args.max_new_tokens,
                args.temperature,
            )

            cand_rewards = []
            for c in cands:
                r, _ = feh_claim_reward(
                    feh,
                    extractor,
                    sample,
                    c,
                    args.device,
                    c2_correct_bonus=args.c2_correct_bonus,
                    c2_miss_penalty=args.c2_miss_penalty,
                    c2_prob_bonus=args.c2_prob_bonus,
                    false_contra_penalty=args.false_contra_penalty,
                    numeric_conflict_bonus=args.numeric_conflict_bonus,
                )
                cand_rewards.append(r)

            norm_rewards = normalize_rewards(cand_rewards)

            best_idx = int(np.argmax(norm_rewards))
            best_resp = cands[best_idx]
            advantage = float(norm_rewards[best_idx])
            ema_baseline = 0.95 * ema_baseline + 0.05 * float(np.mean(norm_rewards))
            stabilized_adv = advantage - ema_baseline

            loss = sft_loss_on_response(model, tokenizer, prompt, best_resp, args.device, args.max_seq_len)
            loss = loss * (1.0 + args.adv_scale * max(0.0, stabilized_adv))

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            if args.device == "cuda" and (step_idx + 1) % 8 == 0:
                torch.cuda.empty_cache()

            losses.append(float(loss.detach().cpu().item()))
            rewards.append(float(max(norm_rewards)))

        model.eval()
        val = evaluate(
            model,
            tokenizer,
            feh,
            extractor,
            val_rows,
            args.device,
            args.max_new_tokens,
            c2_correct_bonus=args.c2_correct_bonus,
            c2_miss_penalty=args.c2_miss_penalty,
            c2_prob_bonus=args.c2_prob_bonus,
            false_contra_penalty=args.false_contra_penalty,
            numeric_conflict_bonus=args.numeric_conflict_bonus,
        )
        avg_loss = float(np.mean(losses)) if losses else 0.0
        avg_reward = float(np.mean(rewards)) if rewards else 0.0
        epoch_log = {
            "epoch": epoch + 1,
            "epoch_size": len(epoch_rows),
            "epoch_label_counts": epoch_label_counts,
            "train_loss": avg_loss,
            "train_best_reward": avg_reward,
            "val_avg_reward": val["avg_reward"],
            "val_avg_non_contradict_ratio": val["avg_non_contradict_ratio"],
        }
        history.append(epoch_log)
        print(epoch_log)

        score = val["avg_reward"] + 0.3 * val["avg_non_contradict_ratio"]
        if score > best_val:
            best_val = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch + 1

    if best_state is not None:
        model.load_state_dict(best_state)

    final_eval = evaluate(
        model,
        tokenizer,
        feh,
        extractor,
        val_rows,
        args.device,
        args.max_new_tokens,
        c2_correct_bonus=args.c2_correct_bonus,
        c2_miss_penalty=args.c2_miss_penalty,
        c2_prob_bonus=args.c2_prob_bonus,
        false_contra_penalty=args.false_contra_penalty,
        numeric_conflict_bonus=args.numeric_conflict_bonus,
    )

    save_dir = out_dir / "checkpoints"
    save_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), "policy_model": args.policy_model}, save_dir / "policy_stage2_best.pt")

    result = {
        "policy_model": args.policy_model,
        "reward_checkpoint": args.reward_checkpoint,
        "extractor_model": args.extractor_model,
        "train_size": len(train_rows),
        "val_size": len(val_rows),
        "train_label_counts": train_label_counts,
        "balance_train": args.balance_train,
        "c2_oversample_factor": args.c2_oversample_factor,
        "c2_correct_bonus": args.c2_correct_bonus,
        "c2_miss_penalty": args.c2_miss_penalty,
        "c2_prob_bonus": args.c2_prob_bonus,
        "false_contra_penalty": args.false_contra_penalty,
        "numeric_conflict_bonus": args.numeric_conflict_bonus,
        "epochs": args.epochs,
        "best_epoch": best_epoch,
        "history": history,
        "final_eval": final_eval,
        "val_ids_file": str(val_ids_path),
    }

    out_json = out_dir / "stage2_policy_grpo_results.json"
    out_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"saved: {out_json}")


if __name__ == "__main__":
    main()
