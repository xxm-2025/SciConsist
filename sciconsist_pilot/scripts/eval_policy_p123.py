from __future__ import annotations

import argparse
import json
import random
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.features.extract import ExtractionConfig, FeatureExtractor
from src.data.perturbation import NumericalPerturber
from src.models.feh import FactualEntailmentHead


@dataclass
class Sample:
    sample_id: str
    text: str
    label_id: int
    perturbation_ratio: float
    caption: str
    image_path: str


_SENT_SPLIT = re.compile(r"[\n\.!?;]+")
_CLAIM_HINT = re.compile(r"\d|increase|decrease|higher|lower|more|less|significant|outperform|worse", re.IGNORECASE)


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
                    text=str(obj.get("text", "")),
                    label_id=int(obj.get("label_id", 1)),
                    perturbation_ratio=float(obj.get("perturbation_ratio", 0.0)),
                    caption=str(raw.get("caption", "")),
                    image_path=str(obj.get("image_path", "")),
                )
            )
            if max_samples > 0 and len(rows) >= max_samples:
                break
    return rows


def split_with_val_ids(rows: list[Sample], val_ids: set[str]) -> tuple[list[Sample], list[Sample]]:
    val_rows = [r for r in rows if r.sample_id in val_ids]
    train_rows = [r for r in rows if r.sample_id not in val_ids]
    return train_rows, val_rows


def load_val_meta(path: str) -> tuple[set[str], list[str], list[float]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, dict):
        val_ids = set(str(x) for x in data.get("val_ids", []))
        p2_seed_ids = [str(x) for x in data.get("p2_seed_ids", [])]
        ratios = [float(x) for x in data.get("ratios", [0.01, 0.02, 0.05, 0.10, 0.15, 0.20])]
    else:
        val_ids = set(str(x) for x in data)
        p2_seed_ids = []
        ratios = [0.01, 0.02, 0.05, 0.10, 0.15, 0.20]
    return val_ids, p2_seed_ids, ratios


def load_feh(checkpoint_path: str, device: str) -> FactualEntailmentHead:
    ckpt = torch.load(checkpoint_path, weights_only=False, map_location=device)
    model = FactualEntailmentHead(ckpt["config"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def load_policy(base_model: str, policy_ckpt: str, device: str):
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
    ).to(device)
    ckpt = torch.load(policy_ckpt, weights_only=False, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval()
    return model, tokenizer


def build_prompt(sample: Sample) -> str:
    return (
        "You are a scientific consistency auditor.\n"
        "Given figure context and claim, output factual statements and a conflict verdict.\n\n"
        f"Figure caption:\n{sample.caption}\n\n"
        f"Candidate claim:\n{sample.text}\n\n"
        "Output:\n"
        "- Claims:\n"
        "  1) ...\n"
        "  2) ...\n"
        "- Conflict verdict: CONSISTENT or CONFLICT\n"
    )


def generate_response(model, tokenizer, prompt: str, device: str, max_new_tokens: int, temperature: float) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(
        **inputs,
        do_sample=True,
        temperature=temperature,
        top_p=0.9,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.eos_token_id,
    )
    prompt_len = inputs["input_ids"].shape[1]
    return tokenizer.decode(outputs[0][prompt_len:], skip_special_tokens=True).strip()


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


def feh_labels_for_claims(feh, extractor, sample: Sample, claims: list[str], device: str) -> list[int]:
    if not claims:
        return []
    t = extractor.extract_text_features(claims)
    v = extractor.extract_visual_features([sample.image_path] * len(claims))
    h_t = torch.from_numpy(t).to(device)
    h_v = torch.from_numpy(v).to(device)
    with torch.no_grad():
        labels, _ = feh.predict(h_v, h_t)
    return labels.detach().cpu().numpy().astype(int).tolist()


def majority_label(labels: list[int]) -> int:
    if not labels:
        return 1
    cnt = defaultdict(int)
    for x in labels:
        cnt[int(x)] += 1
    return sorted(cnt.items(), key=lambda kv: (kv[1], -kv[0]), reverse=True)[0][0]


def safe_rate(num: int, den: int) -> float:
    return float(num / den) if den > 0 else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Policy-side P1/P2/P3 aligned evaluation")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--val-ids", required=True)
    parser.add_argument("--policy-base-model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--policy-checkpoint", required=True)
    parser.add_argument("--reward-checkpoint", required=True)
    parser.add_argument("--extractor-model", default="OpenGVLab/InternVL2_5-8B")
    parser.add_argument("--max-samples", type=int, default=120)
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", required=True)
    parser.add_argument("--p2-per-level", type=int, default=6)
    parser.add_argument("--trace-limit", type=int, default=0, help="0 means save full traces")
    args = parser.parse_args()

    set_seed(args.seed)

    val_ids, p2_seed_ids, p2_levels_cfg = load_val_meta(args.val_ids)
    rows = load_manifest(Path(args.manifest), args.max_samples)
    _, val_rows = split_with_val_ids(rows, val_ids)
    rows_by_id = {r.sample_id: r for r in rows}

    policy, tokenizer = load_policy(args.policy_base_model, args.policy_checkpoint, args.device)
    feh = load_feh(args.reward_checkpoint, args.device)
    extractor = FeatureExtractor(ExtractionConfig(model_name=args.extractor_model, batch_size=1, fail_on_placeholder=True))

    y_true = []
    y_pred = []

    p2_by_level = defaultdict(lambda: {"total": 0, "detected": 0})
    p3_total = 0
    p3_non_contradict = 0

    traces = []

    for s in val_rows:
        prompt = build_prompt(s)
        resp = generate_response(policy, tokenizer, prompt, args.device, args.max_new_tokens, args.temperature)
        claims = extract_claims(resp)
        labels = feh_labels_for_claims(feh, extractor, s, claims, args.device)
        pred = majority_label(labels)

        y_true.append(int(s.label_id))
        y_pred.append(int(pred))

        # P2: only contradicted samples with perturbation ratio > 0
        if int(s.label_id) == 2 and float(s.perturbation_ratio) > 0:
            lvl = round(float(s.perturbation_ratio), 2)
            p2_by_level[lvl]["total"] += 1
            p2_by_level[lvl]["detected"] += int(pred == 2)

        # P3: original entails should avoid contradict prediction under paraphrase-style generation
        if int(s.label_id) == 0:
            p3_total += 1
            p3_non_contradict += int(pred != 2)

        traces.append(
            {
                "sample_id": s.sample_id,
                "true_label": int(s.label_id),
                "pred_label": int(pred),
                "n_claims": len(claims),
                "claim_labels": labels,
            }
        )

    # Synthetic P2 fallback if manifest has no native perturbation ratios.
    if len(p2_by_level) == 0 and p2_seed_ids:
        perturber = NumericalPerturber(seed=args.seed)
        seed_rows = [rows_by_id[sid] for sid in p2_seed_ids if sid in rows_by_id]
        numeric_rows = [r for r in seed_rows if re.search(r"\d", r.text)]
        if numeric_rows:
            for lvl in p2_levels_cfg:
                used = 0
                for base in numeric_rows:
                    if used >= args.p2_per_level:
                        break
                    pr = perturber.perturb_text(base.text, ratio=float(lvl))
                    if pr.num_values_changed <= 0:
                        continue
                    claims = [pr.perturbed_text]
                    labels = feh_labels_for_claims(feh, extractor, base, claims, args.device)
                    if not labels:
                        continue
                    pred = int(labels[0])
                    p2_by_level[round(float(lvl), 2)]["total"] += 1
                    p2_by_level[round(float(lvl), 2)]["detected"] += int(pred == 2)
                    used += 1

    y_true_np = np.asarray(y_true, dtype=np.int64)
    y_pred_np = np.asarray(y_pred, dtype=np.int64)
    p1_acc = float((y_true_np == y_pred_np).mean()) if len(y_true_np) > 0 else 0.0

    confusion = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    for t, p in zip(y_true, y_pred):
        if 0 <= int(t) <= 2 and 0 <= int(p) <= 2:
            confusion[int(t)][int(p)] += 1

    levels = sorted(p2_by_level.keys())
    p2_rates = [safe_rate(p2_by_level[l]["detected"], p2_by_level[l]["total"]) for l in levels]
    p2_target_met = False
    p2_has_5pct = False
    for l, r in zip(levels, p2_rates):
        if abs(l - 0.05) < 1e-6:
            p2_has_5pct = True
            p2_target_met = r >= 0.60
            break

    p3_non_contradict_ratio = safe_rate(p3_non_contradict, p3_total)
    p3_target_met = p3_non_contradict_ratio >= 0.80

    traces_out = traces if args.trace_limit <= 0 else traces[: args.trace_limit]

    result = {
        "n_val": len(val_rows),
        "p1": {"accuracy": p1_acc, "go": p1_acc >= 0.75},
        "confusion_matrix": {
            "labels": [0, 1, 2],
            "rows_true_cols_pred": confusion,
        },
        "p2": {
            "levels": levels,
            "detection_rates": p2_rates,
            "has_5pct": p2_has_5pct,
            "target_met": p2_target_met,
        },
        "p3": {"non_contradict_ratio": p3_non_contradict_ratio, "target_met": p3_target_met},
        "p2_synthetic_used": len(levels) > 0,
        "traces": traces_out,
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
