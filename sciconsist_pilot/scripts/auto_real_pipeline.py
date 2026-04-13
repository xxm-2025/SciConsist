from __future__ import annotations

import argparse
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
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for _ in f)


def run_cmd(cmd: list[str], log_path: Path) -> None:
    log(f"run: {' '.join(cmd)}")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as lf:
        proc = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT)
        code = proc.wait()
    if code != 0:
        raise RuntimeError(f"Command failed ({code}): {' '.join(cmd)}; log={log_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Auto pipeline: wait index -> cache features -> train FEH -> real P1-P5")
    parser.add_argument("--index-file", default="/root/shared-nvme/sciconsist_pilot/raw/s1_index_full/s1_index.jsonl")
    parser.add_argument("--target-lines", type=int, default=60000)
    parser.add_argument("--check-interval", type=int, default=30)
    parser.add_argument(
        "--manifest",
        default="/root/shared-nvme/sciconsist_pilot/raw/trainable/musciclaims_feh_manifest.jsonl",
    )
    parser.add_argument("--real-feature-root", default="/root/shared-nvme/sciconsist_pilot/processed/real_features")
    parser.add_argument("--run-root", default="/root/shared-nvme/sciconsist_pilot/outputs/auto_pipeline")
    parser.add_argument("--max-samples", type=int, default=0)
    args = parser.parse_args()

    index_file = Path(args.index_file)
    run_root = Path(args.run_root)
    run_root.mkdir(parents=True, exist_ok=True)

    log(f"watch index: {index_file}, target={args.target_lines}")
    while True:
        n = count_lines(index_file)
        log(f"index progress: {n}/{args.target_lines}")
        if n >= args.target_lines:
            break
        time.sleep(args.check_interval)

    run_cmd(
        [
            "python",
            "scripts/cache_real_feh_features.py",
            "--manifest",
            args.manifest,
            "--output-root",
            args.real_feature_root,
            "--max-samples",
            str(args.max_samples),
            "--strict-visual-backend",
            "--internvl-only",
            "--enable-p2-augmentation",
        ],
        run_root / "01_cache_real_features.log",
    )

    # Cross-model FEH training (InternVL features)
    run_cmd(
        [
            "python",
            "scripts/train_feh.py",
            f"data.processed_dir={args.real_feature_root}/internvl_full",
            "output.checkpoint_dir=/root/shared-nvme/sciconsist_pilot/outputs/checkpoints_cross",
            "training.require_real_features=true",
            "hydra.run.dir=/root/shared-nvme/sciconsist_pilot/outputs/hydra_train_cross",
        ],
        run_root / "02_train_cross.log",
    )

    # Same-model FEH training (Qwen features)
    run_cmd(
        [
            "python",
            "scripts/train_feh.py",
            f"data.processed_dir={args.real_feature_root}/qwen_full",
            "output.checkpoint_dir=/root/shared-nvme/sciconsist_pilot/outputs/checkpoints_same",
            "training.require_real_features=true",
            "hydra.run.dir=/root/shared-nvme/sciconsist_pilot/outputs/hydra_train_same",
        ],
        run_root / "03_train_same.log",
    )

    run_cmd(
        [
            "python",
            "scripts/run_pilot.py",
            "--checkpoint",
            "/root/shared-nvme/sciconsist_pilot/outputs/checkpoints_cross/feh_best.pt",
            "--features-dir",
            f"{args.real_feature_root}/internvl_full",
            "--p4-full-features-dir",
            f"{args.real_feature_root}/internvl_full",
            "--p4-crop-features-dir",
            f"{args.real_feature_root}/internvl_crop",
            "--p5-cross-checkpoint",
            "/root/shared-nvme/sciconsist_pilot/outputs/checkpoints_cross/feh_best.pt",
            "--p5-same-checkpoint",
            "/root/shared-nvme/sciconsist_pilot/outputs/checkpoints_same/feh_best.pt",
            "--p5-cross-features-dir",
            f"{args.real_feature_root}/internvl_full",
            "--p5-same-features-dir",
            f"{args.real_feature_root}/qwen_full",
            "--output",
            "/root/shared-nvme/sciconsist_pilot/outputs/pilot_results_real.json",
            "--experiments",
            "p1",
            "p2",
            "p3",
            "p4",
            "p5",
        ],
        run_root / "04_run_pilot_real.log",
    )

    log("auto real pipeline finished successfully")


if __name__ == "__main__":
    main()
