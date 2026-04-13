from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
from datetime import datetime
from pathlib import Path


def ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log(msg: str) -> None:
    print(f"[{ts()}] {msg}", flush=True)


def sanitize_key(key: str) -> str:
    keep = []
    for ch in key:
        if ch.isalnum() or ch in {"-", "_", "."}:
            keep.append(ch)
        else:
            keep.append("_")
    return "".join(keep)[:160] or "unknown"


def main() -> None:
    parser = argparse.ArgumentParser(description="Stable S1-MMAlign streaming index downloader")
    parser.add_argument("--out-root", default="/root/shared-nvme/sciconsist_pilot/raw/s1_index")
    parser.add_argument("--limit", type=int, default=60000)
    parser.add_argument("--store-bytes", action="store_true", help="Persist raw bytes to disk")
    parser.add_argument("--bytes-dir", default="bytes")
    parser.add_argument("--save-base64", action="store_true", help="Embed base64 in index (large files)")
    parser.add_argument("--progress-every", type=int, default=1000)
    parser.add_argument("--hf-endpoint", default="https://hf-mirror.com")
    args = parser.parse_args()

    os.environ["HF_ENDPOINT"] = args.hf_endpoint
    os.environ["HF_HUB_ENDPOINT"] = args.hf_endpoint
    os.environ.setdefault("HF_HOME", "/root/shared-nvme/sciconsist_pilot/cache/huggingface")
    os.environ.setdefault("HF_DATASETS_CACHE", "/root/shared-nvme/sciconsist_pilot/cache/huggingface/datasets")
    os.environ.setdefault("HF_HUB_CACHE", "/root/shared-nvme/sciconsist_pilot/cache/huggingface/hub")

    # Import after env setup so endpoint/cache variables take effect.
    from datasets import Image, load_dataset

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    bytes_root = out_root / args.bytes_dir
    if args.store_bytes:
        bytes_root.mkdir(parents=True, exist_ok=True)

    index_jsonl = out_root / "s1_index.jsonl"
    meta_json = out_root / "s1_index.meta.json"

    log(f"Loading ScienceOne-AI/S1-MMAlign with streaming (limit={args.limit})")
    ds = load_dataset("ScienceOne-AI/S1-MMAlign", split="train", streaming=True)

    # Avoid PIL decode path that may trigger DecompressionBombError.
    if hasattr(ds, "features") and "png" in ds.features:
        ds = ds.cast_column("png", Image(decode=False))

    count = 0
    stored_bytes = 0
    errors = 0
    with index_jsonl.open("w", encoding="utf-8") as fw:
        for row in ds:
            try:
                key = str(row.get("__key__", ""))
                url = str(row.get("__url__", ""))

                png_field = row.get("png")
                img_bytes = b""
                if isinstance(png_field, dict):
                    raw = png_field.get("bytes")
                    if isinstance(raw, bytes):
                        img_bytes = raw
                elif isinstance(png_field, bytes):
                    img_bytes = png_field

                record = {
                    "idx": count,
                    "key": key,
                    "url": url,
                    "has_png": png_field is not None,
                    "byte_size": len(img_bytes),
                    "sha256": hashlib.sha256(img_bytes).hexdigest() if img_bytes else "",
                }

                if args.store_bytes and img_bytes:
                    shard = f"{count // 10000:05d}"
                    shard_dir = bytes_root / shard
                    shard_dir.mkdir(parents=True, exist_ok=True)
                    file_name = f"{count:08d}_{sanitize_key(key)}.bin"
                    byte_path = shard_dir / file_name
                    byte_path.write_bytes(img_bytes)
                    record["byte_path"] = str(byte_path.relative_to(out_root))
                    stored_bytes += 1

                if args.save_base64 and img_bytes:
                    record["bytes_b64"] = base64.b64encode(img_bytes).decode("ascii")

                fw.write(json.dumps(record, ensure_ascii=False) + "\n")
                count += 1

                if count % args.progress_every == 0:
                    log(f"progress: {count}/{args.limit}, stored_bytes={stored_bytes}, errors={errors}")

                if count >= args.limit:
                    break
            except Exception as exc:  # keep streaming even if single sample fails
                errors += 1
                if errors <= 20:
                    log(f"warn: sample failed at idx={count}, err={exc}")
                continue

    meta = {
        "dataset": "ScienceOne-AI/S1-MMAlign",
        "split": "train(streaming)",
        "rows": count,
        "stored_bytes": stored_bytes,
        "errors": errors,
        "index_file": str(index_jsonl),
        "store_bytes": args.store_bytes,
        "hf_endpoint": args.hf_endpoint,
        "generated_at": ts(),
    }
    meta_json.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"done: rows={count}, stored_bytes={stored_bytes}, errors={errors}")
    log(f"index={index_jsonl}")


if __name__ == "__main__":
    main()
