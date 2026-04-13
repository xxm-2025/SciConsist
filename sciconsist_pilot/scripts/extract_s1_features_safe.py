from __future__ import annotations

import argparse
import io
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image, ImageFile, UnidentifiedImageError

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.features.extract import ExtractionConfig, FeatureExtractor


def ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log(msg: str) -> None:
    print(f"[{ts()}] {msg}", flush=True)


def iter_index(index_file: Path):
    with index_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def safe_decode_image(
    img_bytes: bytes,
    max_pixels: int,
    max_side: int,
    retries: int,
    retry_sleep: float,
) -> tuple[Image.Image | None, str | None, int]:
    # Allow truncated image loading; common in webdataset shards.
    ImageFile.LOAD_TRUNCATED_IMAGES = True

    for attempt in range(1, retries + 1):
        try:
            with Image.open(io.BytesIO(img_bytes)) as probe:
                width, height = probe.size
                if width <= 0 or height <= 0:
                    return None, "invalid_size", attempt - 1
                if width * height > max_pixels:
                    return None, "too_many_pixels", attempt - 1

            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            width, height = img.size
            if max(width, height) > max_side:
                scale = max_side / max(width, height)
                new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
                img = img.resize(new_size, Image.BICUBIC)
            return img, None, attempt - 1
        except (UnidentifiedImageError, OSError):
            if attempt >= retries:
                return None, "decode_failed", attempt - 1
            time.sleep(retry_sleep)
        except Exception:
            if attempt >= retries:
                return None, "unexpected_error", attempt - 1
            time.sleep(retry_sleep)

    return None, "decode_failed", max(0, retries - 1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Safe decode + visual feature extraction for S1 index")
    parser.add_argument("--index-file", default="/root/shared-nvme/sciconsist_pilot/raw/s1_index/s1_index.jsonl")
    parser.add_argument("--index-root", default="/root/shared-nvme/sciconsist_pilot/raw/s1_index")
    parser.add_argument("--output-dir", default="/root/shared-nvme/sciconsist_pilot/processed/s1_safe_features")
    parser.add_argument("--decoded-dir", default="/root/shared-nvme/sciconsist_pilot/processed/s1_safe_decoded")
    parser.add_argument("--limit", type=int, default=20000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-bytes", type=int, default=64 * 1024 * 1024)
    parser.add_argument("--max-pixels", type=int, default=80_000_000)
    parser.add_argument("--max-side", type=int, default=2048)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--retry-sleep", type=float, default=0.2)
    parser.add_argument("--model", default="OpenGVLab/InternVL2_5-8B")
    args = parser.parse_args()

    index_file = Path(args.index_file)
    index_root = Path(args.index_root)
    out_dir = Path(args.output_dir)
    decoded_dir = Path(args.decoded_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    decoded_dir.mkdir(parents=True, exist_ok=True)

    extractor = FeatureExtractor(
        ExtractionConfig(
            model_name=args.model,
            batch_size=args.batch_size,
            cache_dir=str(out_dir),
        )
    )

    kept_records: list[dict] = []
    decoded_paths: list[str] = []
    keys: list[str] = []
    skipped = {
        "missing_bytes": 0,
        "oversize_bytes": 0,
        "too_many_pixels": 0,
        "decode_failed": 0,
        "invalid_size": 0,
        "unexpected_error": 0,
    }
    retry_stats = {
        "samples_with_retry": 0,
        "total_retry_attempts": 0,
        "max_retry_attempts_single_sample": 0,
    }

    total = 0
    for rec in iter_index(index_file):
        if total >= args.limit:
            break
        total += 1

        byte_rel = rec.get("byte_path", "")
        if not byte_rel:
            skipped["missing_bytes"] += 1
            continue

        byte_path = index_root / byte_rel
        if not byte_path.exists():
            skipped["missing_bytes"] += 1
            continue

        byte_size = int(rec.get("byte_size", byte_path.stat().st_size))
        if byte_size > args.max_bytes:
            skipped["oversize_bytes"] += 1
            continue

        img_bytes = byte_path.read_bytes()
        img, err, retries_used = safe_decode_image(
            img_bytes=img_bytes,
            max_pixels=args.max_pixels,
            max_side=args.max_side,
            retries=args.retries,
            retry_sleep=args.retry_sleep,
        )
        if retries_used > 0:
            retry_stats["samples_with_retry"] += 1
            retry_stats["total_retry_attempts"] += retries_used
            retry_stats["max_retry_attempts_single_sample"] = max(
                retry_stats["max_retry_attempts_single_sample"],
                retries_used,
            )
        if img is None:
            if err in skipped:
                skipped[err] += 1
            else:
                skipped["unexpected_error"] += 1
            continue

        out_img = decoded_dir / f"{int(rec.get('idx', total)):08d}.jpg"
        img.save(out_img, format="JPEG", quality=92)
        decoded_paths.append(str(out_img))
        keys.append(str(rec.get("key", "")))
        kept_records.append(rec)

        if len(decoded_paths) % 1000 == 0:
            log(f"decoded: {len(decoded_paths)} / scanned={total}")

    if not decoded_paths:
        log("no decodable images found; skip feature extraction")
        meta = {
            "scanned": total,
            "decoded": 0,
            "skipped": skipped,
            "retry_stats": retry_stats,
            "generated_at": ts(),
        }
        (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        return

    log(f"extract visual features for {len(decoded_paths)} images")
    visual_feats = extractor.extract_visual_features(decoded_paths)
    np.save(out_dir / "visual_features.npy", visual_feats)

    with (out_dir / "keys.jsonl").open("w", encoding="utf-8") as fw:
        for key, rec, img_path in zip(keys, kept_records, decoded_paths):
            fw.write(
                json.dumps(
                    {
                        "key": key,
                        "idx": rec.get("idx"),
                        "url": rec.get("url", ""),
                        "decoded_path": img_path,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    meta = {
        "scanned": total,
        "decoded": len(decoded_paths),
        "feature_shape": list(visual_feats.shape),
        "model": args.model,
        "skipped": skipped,
        "retry_stats": retry_stats,
        "generated_at": ts(),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"done: decoded={len(decoded_paths)}, scanned={total}, feature_shape={visual_feats.shape}")


if __name__ == "__main__":
    main()
