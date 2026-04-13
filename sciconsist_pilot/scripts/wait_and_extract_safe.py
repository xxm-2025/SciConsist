from __future__ import annotations

import argparse
import json
import subprocess
import time
from datetime import datetime
from pathlib import Path


def ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log(msg: str) -> None:
    print(f"[{ts()}] {msg}", flush=True)


def count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    n = 0
    with path.open("r", encoding="utf-8") as f:
        for _ in f:
            n += 1
    return n


def main() -> None:
    parser = argparse.ArgumentParser(description="Wait index threshold then run safe decode+feature extraction")
    parser.add_argument("--index-file", default="/root/shared-nvme/sciconsist_pilot/raw/s1_index_full/s1_index.jsonl")
    parser.add_argument("--target-lines", type=int, default=60000)
    parser.add_argument("--check-interval", type=int, default=30)
    parser.add_argument("--extract-limit", type=int, default=60000)
    parser.add_argument("--output-dir", default="/root/shared-nvme/sciconsist_pilot/processed/s1_safe_features")
    parser.add_argument("--decoded-dir", default="/root/shared-nvme/sciconsist_pilot/processed/s1_safe_decoded")
    parser.add_argument("--extract-log", default="/root/shared-nvme/sciconsist_pilot/outputs/s1_extract_safe.log")
    args = parser.parse_args()

    index_file = Path(args.index_file)
    marker = Path(args.output_dir) / "auto_extract_trigger.json"
    marker.parent.mkdir(parents=True, exist_ok=True)

    log(f"watch start: index={index_file}, target_lines={args.target_lines}")
    while True:
        lines = count_lines(index_file)
        log(f"watch progress: lines={lines}/{args.target_lines}")
        if lines >= args.target_lines:
            break
        time.sleep(args.check_interval)

    cmd = [
        "python",
        "scripts/extract_s1_features_safe.py",
        "--index-file",
        str(index_file),
        "--index-root",
        str(index_file.parent),
        "--output-dir",
        args.output_dir,
        "--decoded-dir",
        args.decoded_dir,
        "--limit",
        str(args.extract_limit),
    ]

    log("threshold reached; launching safe extraction")
    with open(args.extract_log, "w", encoding="utf-8") as lf:
        proc = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT)

    marker.write_text(
        json.dumps(
            {
                "triggered_at": ts(),
                "index_file": str(index_file),
                "index_lines": count_lines(index_file),
                "target_lines": args.target_lines,
                "extract_pid": proc.pid,
                "extract_log": args.extract_log,
                "cmd": cmd,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    log(f"extract started: pid={proc.pid}, log={args.extract_log}")


if __name__ == "__main__":
    main()
