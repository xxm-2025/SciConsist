#!/usr/bin/env bash
set -euo pipefail

VAL=/root/shared-nvme/sciconsist_pilot/outputs/stage2_policy_grpo_mid/val_ids_with_stats.json
MAN=/root/shared-nvme/sciconsist_pilot/raw/trainable/musciclaims_feh_manifest.jsonl
REW=/root/shared-nvme/sciconsist_pilot/outputs/checkpoints_week1/feh_week1_go.pt
BASE=/root/shared-nvme/sciconsist_pilot/outputs

run_one() {
  local tag="$1"
  local c2_correct="$2"
  local c2_miss="$3"
  local c2_prob="$4"
  local false_contra="$5"
  local over="$6"
  local out_dir="${BASE}/stage2_policy_grpo_${tag}"

  echo "[$(date '+%F %T')] start ${tag}"
  CUDA_VISIBLE_DEVICES=0 \
  HF_ENDPOINT=https://hf-mirror.com \
  PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64 \
  python scripts/train_stage2_policy_grpo.py \
    --manifest "$MAN" \
    --reward-checkpoint "$REW" \
    --policy-model Qwen/Qwen2.5-0.5B-Instruct \
    --extractor-model OpenGVLab/InternVL2_5-8B \
    --output-dir "$out_dir" \
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
    --c2-oversample-factor "$over" \
    --c2-correct-bonus "$c2_correct" \
    --c2-miss-penalty "$c2_miss" \
    --c2-prob-bonus "$c2_prob" \
    --false-contra-penalty "$false_contra" \
    --numeric-conflict-bonus 0.35

  CUDA_VISIBLE_DEVICES=0 \
  HF_ENDPOINT=https://hf-mirror.com \
  python scripts/eval_policy_p123.py \
    --manifest "$MAN" \
    --val-ids "$VAL" \
    --policy-checkpoint "${out_dir}/checkpoints/policy_stage2_best.pt" \
    --reward-checkpoint "$REW" \
    --output "${out_dir}/policy_p123.json" \
    --max-samples 1206 \
    --max-new-tokens 64 \
    --temperature 0.7 \
    --p2-per-level 8 \
    --trace-limit 0

  echo "[$(date '+%F %T')] done ${tag}"
}

cd /root/SciConsist/sciconsist_pilot

# Soft-constraint grid: mild / medium / strong
run_one "mid4_b_soft_a" 0.9 0.7 0.4 0.30 1.6
run_one "mid4_b_soft_b" 1.2 0.9 0.6 0.25 1.8
run_one "mid4_b_soft_c" 1.5 1.0 0.8 0.20 2.0

python - <<'PY'
import json
from pathlib import Path

base = Path('/root/shared-nvme/sciconsist_pilot/outputs')
rows = []
for tag in ['mid3_b_baseline', 'mid4_b_soft_a', 'mid4_b_soft_b', 'mid4_b_soft_c']:
    train_path = base / f'stage2_policy_grpo_{tag}' / 'stage2_policy_grpo_results.json'
    p123_path = base / f'stage2_policy_grpo_{tag}' / 'policy_p123.json'
    if not (train_path.exists() and p123_path.exists()):
        continue
    train = json.loads(train_path.read_text(encoding='utf-8'))
    p123 = json.loads(p123_path.read_text(encoding='utf-8'))
    rows.append(
        {
            'tag': tag,
            'best_epoch': train.get('best_epoch'),
            'val_avg_reward_final': train.get('final_eval', {}).get('avg_reward'),
            'val_avg_non_contradict_final': train.get('final_eval', {}).get('avg_non_contradict_ratio'),
            'p1_acc': p123['p1']['accuracy'],
            'p1_go': p123['p1']['go'],
            'p2_go': p123['p2']['target_met'],
            'p2_rates': p123['p2']['detection_rates'],
            'p3_ratio': p123['p3']['non_contradict_ratio'],
            'p3_go': p123['p3']['target_met'],
            'confusion': p123['confusion_matrix']['rows_true_cols_pred'],
            'n_val': p123['n_val'],
            'soft_params': {
                'c2_correct_bonus': train.get('c2_correct_bonus'),
                'c2_miss_penalty': train.get('c2_miss_penalty'),
                'c2_prob_bonus': train.get('c2_prob_bonus'),
                'false_contra_penalty': train.get('false_contra_penalty'),
                'c2_oversample_factor': train.get('c2_oversample_factor'),
            },
        }
    )
out = base / 'stage2_policy_grpo_mid4_softgrid_summary.json'
out.write_text(json.dumps(rows, indent=2), encoding='utf-8')
print(f'saved: {out}')
print(json.dumps(rows, indent=2))
PY
