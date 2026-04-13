#!/usr/bin/env bash
set -euo pipefail

VAL=/root/shared-nvme/sciconsist_pilot/outputs/stage2_policy_grpo_mid/val_ids_with_stats.json
MAN=/root/shared-nvme/sciconsist_pilot/raw/trainable/musciclaims_feh_manifest.jsonl
REW=/root/shared-nvme/sciconsist_pilot/outputs/checkpoints_week1/feh_week1_go.pt
BASE_OUT=/root/shared-nvme/sciconsist_pilot/outputs

run_train_eval() {
  local tag="$1"
  local lr adv temp g tok e

  case "$tag" in
    a) lr="1.2e-5"; adv="0.35"; temp="0.70"; g="2"; tok="56"; e="2" ;;
    b) lr="1.6e-5"; adv="0.45"; temp="0.75"; g="2"; tok="64"; e="2" ;;
    c) lr="1.4e-5"; adv="0.30"; temp="0.68"; g="2"; tok="56"; e="2" ;;
    *) echo "unknown tag: $tag"; exit 2 ;;
  esac

  local out_dir="${BASE_OUT}/stage2_policy_grpo_mid2_${tag}"
  local ckpt="${out_dir}/checkpoints/policy_stage2_best.pt"
  local p123="${out_dir}/policy_p123.json"

  echo "[$(date '+%F %T')] start train ${tag}"
  CUDA_VISIBLE_DEVICES=0 \
  HF_ENDPOINT=https://hf-mirror.com \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:64 \
  python scripts/train_stage2_policy_grpo.py \
    --manifest "$MAN" \
    --reward-checkpoint "$REW" \
    --policy-model Qwen/Qwen2.5-0.5B-Instruct \
    --extractor-model OpenGVLab/InternVL2_5-8B \
    --output-dir "$out_dir" \
    --max-samples 180 \
    --epochs "$e" \
    --group-size "$g" \
    --max-new-tokens "$tok" \
    --max-seq-len 384 \
    --train-ratio 0.8 \
    --lr "$lr" \
    --adv-scale "$adv" \
    --temperature "$temp" \
    --fixed-val-ids "$VAL"

  echo "[$(date '+%F %T')] start eval ${tag}"
  CUDA_VISIBLE_DEVICES=0 \
  HF_ENDPOINT=https://hf-mirror.com \
  python scripts/eval_policy_p123.py \
    --manifest "$MAN" \
    --val-ids "$VAL" \
    --policy-checkpoint "$ckpt" \
    --reward-checkpoint "$REW" \
    --output "$p123" \
    --max-samples 1206 \
    --max-new-tokens 72 \
    --temperature 0.7 \
    --p2-per-level 8

  echo "[$(date '+%F %T')] done ${tag}"
}

cd /root/SciConsist/sciconsist_pilot
run_train_eval a
run_train_eval b
run_train_eval c

python - <<'PY'
import json
from pathlib import Path

base = Path('/root/shared-nvme/sciconsist_pilot/outputs')
rows = []
for t in ['a', 'b', 'c']:
    p = base / f'stage2_policy_grpo_mid2_{t}' / 'policy_p123.json'
    d = json.loads(p.read_text(encoding='utf-8'))
    rows.append(
        {
            'tag': t,
            'n_val': d['n_val'],
            'p1_acc': d['p1']['accuracy'],
            'p1_go': d['p1']['go'],
            'p2_has_5pct': d['p2']['has_5pct'],
            'p2_go': d['p2']['target_met'],
            'p3_ratio': d['p3']['non_contradict_ratio'],
            'p3_go': d['p3']['target_met'],
        }
    )
out = base / 'stage2_policy_grpo_mid2_summary.json'
out.write_text(json.dumps(rows, indent=2), encoding='utf-8')
print(f'saved: {out}')
print(json.dumps(rows, indent=2))
PY
