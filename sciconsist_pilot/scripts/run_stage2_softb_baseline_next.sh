#!/usr/bin/env bash
set -euo pipefail

# Baseline fixed from mid4 soft-grid winner: mid4_b_soft_b
VAL=/root/shared-nvme/sciconsist_pilot/outputs/stage2_policy_grpo_mid/val_ids_with_stats.json
MAN=/root/shared-nvme/sciconsist_pilot/raw/trainable/musciclaims_feh_manifest.jsonl
REW=/root/shared-nvme/sciconsist_pilot/outputs/checkpoints_week1/feh_week1_go.pt
OUT=/root/shared-nvme/sciconsist_pilot/outputs/stage2_policy_grpo_softb_baseline_next

cd /root/SciConsist/sciconsist_pilot

CUDA_VISIBLE_DEVICES=0 \
HF_ENDPOINT=https://hf-mirror.com \
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64 \
python scripts/train_stage2_policy_grpo.py \
  --manifest "$MAN" \
  --reward-checkpoint "$REW" \
  --policy-model Qwen/Qwen2.5-0.5B-Instruct \
  --extractor-model OpenGVLab/InternVL2_5-8B \
  --output-dir "$OUT" \
  --max-samples 180 \
  --epochs 2 \
  --group-size 1 \
  --max-new-tokens 48 \
  --max-seq-len 320 \
  --train-ratio 0.8 \
  --lr 1.6e-5 \
  --adv-scale 0.40 \
  --temperature 0.72 \
  --fixed-val-ids "$VAL" \
  --balance-train \
  --c2-oversample-factor 1.8 \
  --c2-correct-bonus 1.2 \
  --c2-miss-penalty 0.9 \
  --c2-prob-bonus 0.6 \
  --false-contra-penalty 0.25 \
  --numeric-conflict-bonus 0.35

CUDA_VISIBLE_DEVICES=0 \
HF_ENDPOINT=https://hf-mirror.com \
python scripts/eval_policy_p123.py \
  --manifest "$MAN" \
  --val-ids "$VAL" \
  --policy-checkpoint "$OUT/checkpoints/policy_stage2_best.pt" \
  --reward-checkpoint "$REW" \
  --output "$OUT/policy_p123.json" \
  --max-samples 1206 \
  --max-new-tokens 64 \
  --temperature 0.7 \
  --p2-per-level 8 \
  --trace-limit 0

echo "saved: $OUT"