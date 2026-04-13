from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
import re


_NUM_RE = re.compile(r"\d")


def load_rows(manifest: Path, pool_size: int) -> list[dict]:
    rows: list[dict] = []
    with manifest.open("r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            if not obj.get("image_exists"):
                continue
            rows.append(obj)
            if pool_size > 0 and len(rows) >= pool_size:
                break
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Build fixed validation ids with class + P2 level coverage")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--pool-size", type=int, default=120)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-per-class", type=int, default=12)
    parser.add_argument("--min-per-ratio", type=int, default=4)
    parser.add_argument("--ratios", default="0.01,0.02,0.05,0.10,0.15,0.20")
    parser.add_argument("--extra-val", type=int, default=24)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    ratios = [round(float(x), 2) for x in args.ratios.split(",") if x.strip()]

    rows = load_rows(Path(args.manifest), args.pool_size)
    if len(rows) < args.min_per_class * 3:
        raise RuntimeError(f"Pool too small: {len(rows)} rows")

    by_class: dict[int, list[dict]] = defaultdict(list)
    contra_by_ratio: dict[float, list[dict]] = defaultdict(list)
    for r in rows:
        lbl = int(r.get("label_id", 1))
        by_class[lbl].append(r)
        if lbl == 2:
            pr = round(float(r.get("perturbation_ratio", 0.0)), 2)
            contra_by_ratio[pr].append(r)

    val_ids: set[str] = set()

    # 1) ensure each class has minimum coverage
    for cls in [0, 1, 2]:
        candidates = by_class.get(cls, [])
        rng.shuffle(candidates)
        take = min(args.min_per_class, len(candidates))
        for r in candidates[:take]:
            val_ids.add(str(r.get("sample_id", "")))

    # 2) ensure P2 perturbation-level coverage among CONTRADICTS
    for pr in ratios:
        candidates = contra_by_ratio.get(pr, [])
        if not candidates:
            continue
        rng.shuffle(candidates)
        take = min(args.min_per_ratio, len(candidates))
        for r in candidates[:take]:
            val_ids.add(str(r.get("sample_id", "")))

    # 3) add extra random val examples for medium scale
    pool_ids = [str(r.get("sample_id", "")) for r in rows]
    remaining = [x for x in pool_ids if x not in val_ids]
    rng.shuffle(remaining)
    for sid in remaining[: max(0, args.extra_val)]:
        val_ids.add(sid)

    val_ids_sorted = sorted(x for x in val_ids if x)

    # Synthetic P2 fallback seeds: numeric claims usable for perturbation-level evaluation.
    numeric_pool = [r for r in rows if _NUM_RE.search(str(r.get("text", "")))]
    rng.shuffle(numeric_pool)
    synthetic_seed_target = max(len(ratios) * args.min_per_ratio, 12)
    p2_seed_ids = [str(r.get("sample_id", "")) for r in numeric_pool[:synthetic_seed_target] if str(r.get("sample_id", ""))]

    # stats
    val_rows = [r for r in rows if str(r.get("sample_id", "")) in val_ids]
    cls_count: dict[int, int] = defaultdict(int)
    lvl_count: dict[float, int] = defaultdict(int)
    for r in val_rows:
        lbl = int(r.get("label_id", 1))
        cls_count[lbl] += 1
        if lbl == 2:
            lvl_count[round(float(r.get("perturbation_ratio", 0.0)), 2)] += 1

    out = {
        "pool_size": len(rows),
        "val_size": len(val_ids_sorted),
        "seed": args.seed,
        "ratios": ratios,
        "class_counts": {str(k): int(v) for k, v in sorted(cls_count.items())},
        "contra_ratio_counts": {f"{k:.2f}": int(v) for k, v in sorted(lvl_count.items())},
        "p2_seed_ids": p2_seed_ids,
        "val_ids": val_ids_sorted,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"saved: {out_path}")
    print(json.dumps({
        "val_size": out["val_size"],
        "class_counts": out["class_counts"],
        "contra_ratio_counts": out["contra_ratio_counts"],
    }, indent=2))


if __name__ == "__main__":
    main()
