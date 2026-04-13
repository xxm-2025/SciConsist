from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from huggingface_hub import snapshot_download


def ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log(msg: str) -> None:
    print(f"[{ts()}] {msg}", flush=True)


def _find_musciclaims_jsonl(repo_dir: Path) -> Path:
    candidates = sorted(repo_dir.rglob("*.jsonl"))
    for p in candidates:
        if "test_set" in p.name.lower():
            return p
    if not candidates:
        raise FileNotFoundError(f"No .jsonl file found under {repo_dir}")
    return candidates[0]


def _label_to_feh(label: str) -> tuple[str, int]:
    upper = (label or "").strip().upper()
    if upper in {"SUPPORT", "SUPPORTED", "ENTAILS", "ENTAILMENT"}:
        return "ENTAILS", 0
    if upper in {"CONTRADICT", "CONTRADICTS", "REFUTES"}:
        return "CONTRADICTS", 2
    return "NEUTRAL", 1


def build_trainable_manifest(repo_dir: Path, out_dir: Path) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    src_jsonl = _find_musciclaims_jsonl(repo_dir)

    manifest_path = out_dir / "musciclaims_feh_manifest.jsonl"
    missing_path = out_dir / "musciclaims_missing_images.jsonl"
    meta_path = out_dir / "musciclaims_manifest.meta.json"

    log(f"Building trainable manifest from {src_jsonl}")

    total = 0
    usable = 0
    missing = 0

    with src_jsonl.open("r", encoding="utf-8") as fin, \
        manifest_path.open("w", encoding="utf-8") as f_ok, \
        missing_path.open("w", encoding="utf-8") as f_missing:

        for line in fin:
            total += 1
            row = json.loads(line)

            claim = row.get("claim_text", "")
            caption = row.get("caption", "")
            text = claim if claim else caption

            rel_fig = row.get("associated_figure_filepath", "")
            fig_path = (repo_dir / rel_fig).resolve() if rel_fig else None

            feh_label_str, feh_label_id = _label_to_feh(row.get("label_3class", ""))

            sample = {
                "source": "MuSciClaims",
                "sample_id": row.get("claim_id", row.get("base_claim_id", total)),
                "text": text,
                "label_str": feh_label_str,
                "label_id": feh_label_id,
                "domain": row.get("domain", ""),
                "image_path": str(fig_path) if fig_path else "",
                "image_exists": bool(fig_path and fig_path.exists()),
                "raw": {
                    "claim_text": claim,
                    "caption": caption,
                    "label_3class": row.get("label_3class", ""),
                    "associated_figure_filepath": rel_fig,
                },
            }

            if sample["image_exists"] and sample["text"]:
                usable += 1
                f_ok.write(json.dumps(sample, ensure_ascii=False) + "\n")
            else:
                missing += 1
                f_missing.write(json.dumps(sample, ensure_ascii=False) + "\n")

            if total % 500 == 0:
                log(f"Manifest progress: total={total}, usable={usable}, missing={missing}")

    meta = {
        "created_at": ts(),
        "repo_dir": str(repo_dir),
        "src_jsonl": str(src_jsonl),
        "manifest_path": str(manifest_path),
        "missing_path": str(missing_path),
        "total": total,
        "usable": usable,
        "missing": missing,
        "usable_ratio": (usable / total) if total else 0.0,
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"Manifest done: usable={usable}/{total} ({meta['usable_ratio']:.2%})")
    return meta


def main() -> None:
    parser = argparse.ArgumentParser(description="Download trainable datasets to data disk in background")
    parser.add_argument("--root", default="/root/shared-nvme/sciconsist_pilot")
    args = parser.parse_args()

    root = Path(args.root)
    raw_dir = root / "raw"
    trainable_dir = raw_dir / "trainable"
    repo_dir = raw_dir / "musciclaims_repo"

    raw_dir.mkdir(parents=True, exist_ok=True)
    trainable_dir.mkdir(parents=True, exist_ok=True)

    os.environ["HF_ENDPOINT"] = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com")
    os.environ["HF_HUB_ENDPOINT"] = os.environ.get("HF_HUB_ENDPOINT", os.environ["HF_ENDPOINT"])
    os.environ["HF_HOME"] = os.environ.get("HF_HOME", str(root / "cache" / "huggingface"))
    os.environ["HF_DATASETS_CACHE"] = os.environ.get(
        "HF_DATASETS_CACHE", str(root / "cache" / "huggingface" / "datasets")
    )
    os.environ["HF_HUB_CACHE"] = os.environ.get("HF_HUB_CACHE", str(root / "cache" / "huggingface" / "hub"))

    log("Starting trainable data download job")
    log(f"HF_ENDPOINT={os.environ['HF_ENDPOINT']}")
    log(f"root={root}")

    log("Downloading MuSciClaims snapshot ...")
    local_path = snapshot_download(
        repo_id="StonyBrookNLP/MuSciClaims",
        repo_type="dataset",
        local_dir=str(repo_dir),
        local_dir_use_symlinks=False,
        resume_download=True,
    )
    log(f"MuSciClaims snapshot ready at {local_path}")

    meta = build_trainable_manifest(repo_dir=repo_dir, out_dir=trainable_dir)
    log(f"Completed. Trainable samples: {meta['usable']}")


if __name__ == "__main__":
    main()
