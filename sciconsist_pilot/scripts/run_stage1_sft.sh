#!/usr/bin/env bash
set -euo pipefail

# Stage 1: Qwen2.5-VL-7B SFT on SciMDR
# 4x RTX 4090, LoRA, bf16, ~364K samples, ~1 epoch
# 预计耗时: ~8-12 小时 (364K * 1 epoch / effective_batch_64)

export HF_ENDPOINT=https://hf-mirror.com
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

DATA_ROOT=/root/shared-nvme/sciconsist_pilot
OUTPUT=${DATA_ROOT}/outputs/sft_stage1

cd /root/SciConsist

torchrun --nproc_per_node=4 \
  sciconsist_pilot/scripts/train_stage1_sft.py \
  --train-data ${DATA_ROOT}/processed/scimdr_sft_train.jsonl \
  --val-data ${DATA_ROOT}/processed/scimdr_sft_val.jsonl \
  --image-root ${DATA_ROOT}/raw/scimdr \
  --output-dir ${OUTPUT} \
  --model-name Qwen/Qwen2.5-VL-7B-Instruct \
  --max-length 2048 \
  --lora-r 16 \
  --lora-alpha 32 \
  --lora-target-modules q_proj,k_proj,v_proj,o_proj \
  --num-epochs 1 \
  --per-device-batch-size 2 \
  --gradient-accumulation 8 \
  --lr 1e-4 \
  --warmup-ratio 0.03 \
  --save-steps 500 \
  --eval-steps 500 \
  --logging-steps 10 \
  --seed 42

echo "=== Stage 1 SFT complete: ${OUTPUT} ==="
