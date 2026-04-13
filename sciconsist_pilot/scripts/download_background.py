from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

from datasets import load_dataset


def _ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log(msg: str) -> None:
    print(f"[{_ts()}] {msg}", flush=True)


def save_musciclaims(raw_dir: Path) -> None:
    out_jsonl = raw_dir / "musciclaims_test.jsonl"
    out_meta = raw_dir / "musciclaims_test.meta.json"

    log("Loading StonyBrookNLP/MuSciClaims split=test ...")
    ds = load_dataset("StonyBrookNLP/MuSciClaims", split="test")
    log(f"MuSciClaims rows={len(ds)} columns={ds.column_names}")

    with out_jsonl.open("w", encoding="utf-8") as f:
        for row in ds:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    out_meta.write_text(
        json.dumps(
            {
                "dataset": "StonyBrookNLP/MuSciClaims",
                "split": "test",
                "rows": len(ds),
                "columns": ds.column_names,
                "saved_at": _ts(),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    log(f"Saved MuSciClaims to {out_jsonl}")


def save_s1_stream(raw_dir: Path, limit: int) -> None:
    out_jsonl = raw_dir / "s1_mmalign_stream_sample.jsonl"
    out_meta = raw_dir / "s1_mmalign_stream_sample.meta.json"

    log(f"Streaming ScienceOne-AI/S1-MMAlign up to {limit} rows ...")
    ds = load_dataset("ScienceOne-AI/S1-MMAlign", split="train", streaming=True)

    n = 0
    columns = None
    with out_jsonl.open("w", encoding="utf-8") as f:
        for row in ds:
            if columns is None:
                columns = list(row.keys())
                log(f"S1-MMAlign streaming columns={columns}")

            # Keep only JSON-serializable metadata fields from webdataset rows.
            row_serializable = {
                "__key__": row.get("__key__", ""),
                "__url__": row.get("__url__", ""),
                "has_png": "png" in row,
            }
            f.write(json.dumps(row_serializable, ensure_ascii=False) + "\n")
            n += 1
            if n % 1000 == 0:
                log(f"S1-MMAlign progress: {n}/{limit}")
            if n >= limit:
                break

    out_meta.write_text(
        json.dumps(
            {
                "dataset": "ScienceOne-AI/S1-MMAlign",
                "split": "train(streaming)",
                "rows": n,
                "columns": columns or [],
                "saved_at": _ts(),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    log(f"Saved S1-MMAlign stream sample to {out_jsonl}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Background dataset downloader for SciConsist pilot")
    parser.add_argument("--raw-dir", default="/root/shared-nvme/sciconsist_pilot/raw")
    parser.add_argument("--s1-limit", type=int, default=60000)
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)

    # Force mirror/cache env in detached mode. Do not rely on inherited shell vars.
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    os.environ["HF_HOME"] = "/root/shared-nvme/sciconsist_pilot/cache/huggingface"
    os.environ["HF_DATASETS_CACHE"] = "/root/shared-nvme/sciconsist_pilot/cache/huggingface/datasets"
    os.environ["HF_HUB_CACHE"] = "/root/shared-nvme/sciconsist_pilot/cache/huggingface/hub"
    os.environ["HF_HUB_ENDPOINT"] = "https://hf-mirror.com"

    log("Started background download job")
    log(f"raw_dir={raw_dir}")
    log(f"HF_ENDPOINT={os.environ.get('HF_ENDPOINT')}")

    save_musciclaims(raw_dir)
    save_s1_stream(raw_dir, args.s1_limit)

    log("All download tasks completed")


if __name__ == "__main__":
    main()
