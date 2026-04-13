from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import math
import re

import numpy as np
from PIL import Image

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.features.extract import ExtractionConfig, FeatureExtractor
from src.data.perturbation import NumericalPerturber


_NUM_RE = re.compile(r"\d")


def load_manifest(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if obj.get("image_exists") and obj.get("text"):
                rows.append(obj)
    return rows


def center_crop(img: Image.Image, ratio: float = 0.7) -> Image.Image:
    w, h = img.size
    nw, nh = max(1, int(w * ratio)), max(1, int(h * ratio))
    left = (w - nw) // 2
    top = (h - nh) // 2
    return img.crop((left, top, left + nw, top + nh))


def split_indices(n: int, train_ratio: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.RandomState(seed)
    idx = rng.permutation(n)
    n_train = int(n * train_ratio)
    return np.sort(idx[:n_train]), np.sort(idx[n_train:])


def stratified_split_indices(
    labels: np.ndarray,
    perturbation_ratios: np.ndarray,
    train_ratio: float,
    seed: int,
    min_val_per_ratio: int,
) -> tuple[np.ndarray, np.ndarray]:
    """按 label 分层；对于 CONTRADICTS 再按扰动幅度分层。"""
    rng = np.random.RandomState(seed)

    groups: dict[tuple[int, float], list[int]] = {}
    for i, (lbl, r) in enumerate(zip(labels.tolist(), perturbation_ratios.tolist())):
        key = (int(lbl), float(round(r, 2)) if int(lbl) == 2 else -1.0)
        groups.setdefault(key, []).append(i)

    train_idx: list[int] = []
    val_idx: list[int] = []
    for key, idxs in groups.items():
        idxs = idxs.copy()
        rng.shuffle(idxs)
        lbl, ratio = key
        n = len(idxs)
        n_val = max(1, int(round(n * (1.0 - train_ratio))))
        if lbl == 2 and ratio > 0:
            # Ensure P2 has enough validation samples per perturbation level.
            n_val = max(n_val, min_val_per_ratio)
        n_val = min(n_val, max(1, n - 1))
        val_idx.extend(idxs[:n_val])
        train_idx.extend(idxs[n_val:])

    return np.array(sorted(train_idx), dtype=np.int64), np.array(sorted(val_idx), dtype=np.int64)


def save_split(
    output_root: Path,
    split: str,
    visual_feats: np.ndarray,
    text_feats: np.ndarray,
    labels: np.ndarray,
    perturbation_ratios: np.ndarray,
) -> None:
    split_dir = output_root / split
    split_dir.mkdir(parents=True, exist_ok=True)
    np.save(split_dir / "visual_features.npy", visual_feats)
    np.save(split_dir / "text_features.npy", text_feats)
    np.save(split_dir / "labels.npy", labels)
    np.save(split_dir / "perturbation_ratios.npy", perturbation_ratios)


def build_augmented_rows_for_p2(
    rows: list[dict],
    ratios: list[float],
    seed: int,
    target_total: int,
    target_contradict_ratio: float,
    target_neutral_ratio: float,
    min_val_per_ratio: int,
    train_ratio: float,
    clean_entails_only: bool,
    clean_entails_min_text_len: int,
) -> list[dict]:
    """构造可训练且可评估(P2)的数据：控制类别比例并注入足量扰动样本。"""
    perturber = NumericalPerturber(seed=seed)
    rng = np.random.RandomState(seed)

    base_entails = [dict(r) for r in rows if int(r.get("label_id", 1)) == 0]
    base_neutral = [dict(r) for r in rows if int(r.get("label_id", 1)) == 1]
    base_contra = [dict(r) for r in rows if int(r.get("label_id", 1)) == 2]

    if clean_entails_only:
        base_entails = [
            r
            for r in base_entails
            if len(str(r.get("text", "")).strip()) >= clean_entails_min_text_len
            and float(r.get("perturbation_ratio", 0.0)) == 0.0
        ]
        if not base_entails:
            raise RuntimeError("No clean ENTAILS samples after filtering.")

    # Normalize metadata fields.
    for r in base_entails + base_neutral + base_contra:
        r["perturbation_ratio"] = float(r.get("perturbation_ratio", 0.0))

    numeric_entails = [r for r in base_entails if _NUM_RE.search(str(r.get("text", "")))]
    if not numeric_entails:
        raise RuntimeError("No numeric ENTAILS samples found for P2 augmentation.")

    n_total = int(target_total)
    n_contra = int(round(n_total * target_contradict_ratio))
    n_neutral = int(round(n_total * target_neutral_ratio))
    n_entails = max(1, n_total - n_contra - n_neutral)

    def sample_pool(pool: list[dict], n: int) -> list[dict]:
        if not pool:
            return []
        idx = rng.choice(len(pool), size=n, replace=n > len(pool))
        return [dict(pool[i]) for i in idx.tolist()]

    entails_sel = sample_pool(base_entails, n_entails)

    neutral_sel = sample_pool(base_neutral, min(n_neutral, len(base_neutral)))
    if len(neutral_sel) < n_neutral and base_entails:
        # Synthesize neutral by mismatching image/text among entails.
        needed = n_neutral - len(neutral_sel)
        for _ in range(needed):
            i, j = rng.choice(len(base_entails), size=2, replace=len(base_entails) < 2)
            a, b = base_entails[int(i)], base_entails[int(j)]
            nr = dict(a)
            nr["text"] = b.get("text", "")
            nr["label_id"] = 1
            nr["label_str"] = "NEUTRAL"
            nr["perturbation_ratio"] = 0.0
            neutral_sel.append(nr)

    # Reserve enough perturbed contradictions so each ratio has enough val samples.
    min_total_per_ratio = int(math.ceil(min_val_per_ratio / max(1e-6, (1.0 - train_ratio)))) + 4
    ratio_target_each = max(min_total_per_ratio, int(math.ceil(n_contra / max(1, len(ratios)))))

    perturbed_contra: list[dict] = []
    if numeric_entails:
        for ratio in ratios:
            made = 0
            attempts = 0
            max_attempts = ratio_target_each * 80
            while made < ratio_target_each and attempts < max_attempts:
                src = numeric_entails[int(rng.randint(0, len(numeric_entails)))]
                attempts += 1
                pr = perturber.perturb_text(src.get("text", ""), ratio=ratio)
                if pr.num_values_changed <= 0:
                    continue
                nr = dict(src)
                nr["text"] = pr.perturbed_text
                nr["label_id"] = 2
                nr["label_str"] = "CONTRADICTS"
                nr["perturbation_ratio"] = float(ratio)
                perturbed_contra.append(nr)
                made += 1

            if made < ratio_target_each:
                raise RuntimeError(
                    f"Insufficient perturbed contradictions for ratio={ratio:.2f}: "
                    f"made={made}, target={ratio_target_each}."
                )

    remain_contra = max(0, n_contra - len(perturbed_contra))
    # Only use base CONTRADICTS with non-zero perturbation ratio to avoid ratio=0 pollution in P2.
    base_contra_nonzero = [r for r in base_contra if float(r.get("perturbation_ratio", 0.0)) > 0]
    contra_sel = perturbed_contra + sample_pool(base_contra_nonzero, remain_contra)
    for r in contra_sel:
        r["label_id"] = 2
        r["label_str"] = "CONTRADICTS"
        r["perturbation_ratio"] = float(r.get("perturbation_ratio", 0.0))

    final_rows = entails_sel + neutral_sel[:n_neutral] + contra_sel[:n_contra]
    rng.shuffle(final_rows)
    return final_rows


def extract_one_setup(
    rows: list[dict],
    output_root: Path,
    model_name: str,
    use_crop: bool,
    train_ratio: float,
    seed: int,
    batch_size: int,
    max_samples: int,
    strict_visual_backend: bool,
    enable_p2_augmentation: bool,
    p2_ratios: list[float],
    target_total: int,
    target_contradict_ratio: float,
    target_neutral_ratio: float,
    min_val_per_ratio: int,
    clean_entails_only: bool,
    clean_entails_min_text_len: int,
) -> dict:
    if max_samples > 0:
        rows = rows[:max_samples]

    if enable_p2_augmentation:
        rows = build_augmented_rows_for_p2(
            rows,
            ratios=p2_ratios,
            seed=seed,
            target_total=target_total,
            target_contradict_ratio=target_contradict_ratio,
            target_neutral_ratio=target_neutral_ratio,
            min_val_per_ratio=min_val_per_ratio,
            train_ratio=train_ratio,
            clean_entails_only=clean_entails_only,
            clean_entails_min_text_len=clean_entails_min_text_len,
        )

    texts = [r["text"] for r in rows]
    labels = np.array([int(r.get("label_id", 1)) for r in rows], dtype=np.int64)
    perturbation_ratios = np.array([float(r.get("perturbation_ratio", 0.0)) for r in rows], dtype=np.float32)

    imgs_full: list[Image.Image] = []
    for r in rows:
        path = Path(r["image_path"])
        img = Image.open(path).convert("RGB")
        imgs_full.append(center_crop(img) if use_crop else img)

    extractor = FeatureExtractor(
        ExtractionConfig(
            model_name=model_name,
            batch_size=batch_size,
            fail_on_placeholder=True,
        )
    )

    text_feats = extractor.extract_text_features(texts)
    visual_feats = extractor.extract_visual_features(imgs_full)

    if strict_visual_backend and extractor.vision_backend == "image_stats_fallback":
        raise RuntimeError(
            f"Strict mode: visual backend fallback is not allowed for {model_name}. "
            "Please ensure InternVL/Qwen model weights are reachable and vision APIs are available."
        )

    train_idx, val_idx = stratified_split_indices(
        labels=labels,
        perturbation_ratios=perturbation_ratios,
        train_ratio=train_ratio,
        seed=seed,
        min_val_per_ratio=min_val_per_ratio,
    )
    save_split(
        output_root,
        "train",
        visual_feats[train_idx],
        text_feats[train_idx],
        labels[train_idx],
        perturbation_ratios[train_idx],
    )
    save_split(
        output_root,
        "val",
        visual_feats[val_idx],
        text_feats[val_idx],
        labels[val_idx],
        perturbation_ratios[val_idx],
    )

    meta = {
        "rows": len(rows),
        "train_rows": int(len(train_idx)),
        "val_rows": int(len(val_idx)),
        "model_name": model_name,
        "use_crop": use_crop,
        "text_dim": int(text_feats.shape[1]),
        "visual_dim": int(visual_feats.shape[1]),
        "vision_backend": extractor.vision_backend,
        "enable_p2_augmentation": enable_p2_augmentation,
        "p2_ratios": p2_ratios,
        "target_total": target_total,
        "target_contradict_ratio": target_contradict_ratio,
        "target_neutral_ratio": target_neutral_ratio,
        "min_val_per_ratio": min_val_per_ratio,
        "clean_entails_only": clean_entails_only,
        "clean_entails_min_text_len": clean_entails_min_text_len,
    }
    (output_root / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta


def main() -> None:
    parser = argparse.ArgumentParser(description="Cache real FEH features from MuSciClaims manifest")
    parser.add_argument(
        "--manifest",
        default="/root/shared-nvme/sciconsist_pilot/raw/trainable/musciclaims_feh_manifest.jsonl",
    )
    parser.add_argument("--output-root", default="/root/shared-nvme/sciconsist_pilot/processed/real_features")
    parser.add_argument("--train-ratio", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--internvl-model", default="OpenGVLab/InternVL2_5-8B")
    parser.add_argument("--qwen-model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--strict-visual-backend", action="store_true")
    parser.add_argument("--hf-endpoint", default="https://hf-mirror.com")
    parser.add_argument("--internvl-only", action="store_true")
    parser.add_argument("--enable-p2-augmentation", action="store_true")
    parser.add_argument("--p2-ratios", default="0.01,0.02,0.05,0.10,0.15,0.20")
    parser.add_argument("--target-total", type=int, default=3000)
    parser.add_argument("--target-contradict-ratio", type=float, default=0.35)
    parser.add_argument("--target-neutral-ratio", type=float, default=0.30)
    parser.add_argument("--min-val-per-ratio", type=int, default=20)
    parser.add_argument("--clean-entails-only", action="store_true")
    parser.add_argument("--clean-entails-min-text-len", type=int, default=40)
    args = parser.parse_args()

    os.environ["HF_ENDPOINT"] = args.hf_endpoint
    os.environ["HF_HUB_ENDPOINT"] = args.hf_endpoint
    os.environ.setdefault("HF_HOME", "/root/shared-nvme/sciconsist_pilot/cache/huggingface")
    os.environ.setdefault("HF_DATASETS_CACHE", "/root/shared-nvme/sciconsist_pilot/cache/huggingface/datasets")
    os.environ.setdefault("HF_HUB_CACHE", "/root/shared-nvme/sciconsist_pilot/cache/huggingface/hub")

    manifest = Path(args.manifest)
    out = Path(args.output_root)
    out.mkdir(parents=True, exist_ok=True)

    rows = load_manifest(manifest)
    if not rows:
        raise RuntimeError(f"No usable samples from manifest: {manifest}")

    p2_ratios = [float(x.strip()) for x in args.p2_ratios.split(",") if x.strip()]

    setups = [
        ("internvl_full", args.internvl_model, False),
        ("internvl_crop", args.internvl_model, True),
    ]
    if not args.internvl_only:
        setups.append(("qwen_full", args.qwen_model, False))

    summary: dict[str, dict] = {}
    for name, model_name, use_crop in setups:
        dst = out / name
        dst.mkdir(parents=True, exist_ok=True)
        meta = extract_one_setup(
            rows=rows,
            output_root=dst,
            model_name=model_name,
            use_crop=use_crop,
            train_ratio=args.train_ratio,
            seed=args.seed,
            batch_size=args.batch_size,
            max_samples=args.max_samples,
            strict_visual_backend=args.strict_visual_backend,
            enable_p2_augmentation=args.enable_p2_augmentation,
            p2_ratios=p2_ratios,
            target_total=args.target_total,
            target_contradict_ratio=args.target_contradict_ratio,
            target_neutral_ratio=args.target_neutral_ratio,
            min_val_per_ratio=args.min_val_per_ratio,
            clean_entails_only=args.clean_entails_only,
            clean_entails_min_text_len=args.clean_entails_min_text_len,
        )
        summary[name] = meta

    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
