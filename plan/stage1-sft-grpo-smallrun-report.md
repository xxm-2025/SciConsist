# Stage1 SFT + FER-GRPO Small-Run Report

Date: 2026-04-13

## Implemented

- Added executable pipeline script:
  - `scripts/train_stage1_sft_grpo.py`
- Pipeline includes:
  - Stage 1 SFT (supervised training on FEH feature pairs)
  - Stage 2 FER-GRPO (frozen FEH reward model + KL regularization + supervised anchor)
  - Automatic evaluation for P1/P2/P3
  - Checkpoint + JSON result export

## Reward Model Used

- Frozen Week1 GO FEH:
  - `/root/shared-nvme/sciconsist_pilot/outputs/checkpoints_week1/feh_week1_go.pt`

## Runs and Results

1. `stage1_grpo_smallrun1`
- Output: `/root/shared-nvme/sciconsist_pilot/outputs/stage1_grpo_smallrun1/stage1_grpo_results.json`
- Result: P1=0.3583, P2=NOT_MET, P3=1.000
- Note: overly conservative collapse under weak constraints.

2. `stage1_grpo_smallrun2`
- Output: `/root/shared-nvme/sciconsist_pilot/outputs/stage1_grpo_smallrun2/stage1_grpo_results.json`
- Result: P1=0.5458, P2=MET, P3=0.920
- Note: improved stability after adding supervised anchor.

3. `stage1_grpo_smallrun3`
- Output: `/root/shared-nvme/sciconsist_pilot/outputs/stage1_grpo_smallrun3/stage1_grpo_results.json`
- Result: P1=0.6458, P2=MET, P3=0.860
- Note: conservative GRPO settings further improved P1.

4. `stage1_grpo_smallrun4_fullval`
- Output: `/root/shared-nvme/sciconsist_pilot/outputs/stage1_grpo_smallrun4_fullval/stage1_grpo_results.json`
- Result: P1=0.6765, P2=MET, P3=0.820
- Note: medium-scale run gives stable baseline for Week2 integration.

## Current Best Config (for next iteration)

- `latent_dim=1024`
- `sft_epochs=8`, `sft_lr=6e-4`
- `grpo_epochs=5`, `grpo_lr=2e-4`
- `kl_beta=0.30`
- `grpo_supervised_coef=2.0`
- `group_size=2`
- `class_weights=1.32,1.59,0.77`

## Next Suggested Steps

- Integrate this loop with real policy generation (claim extraction + FEH claim-level scoring), replacing feature-space action proxy.
- Add KL scheduler and reward normalization per batch to reduce early-stage instability.
- Launch multi-seed runs (>=3) on the current best config before Week2 full training.
