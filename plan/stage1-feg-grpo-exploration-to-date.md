# Stage1 FEH-GRPO Exploration Summary (To Date)

Date: 2026-04-13
Scope: Stage1/early-Stage2 bridge exploration for FEH-guided GRPO effectiveness.

## 1. Objective and Evaluation Gate

- Objective: validate whether FEH-guided GRPO can improve policy-side factual consistency without sacrificing safety behavior.
- Unified gate (fixed val_ids):
  - P1: accuracy >= 0.75
  - P2: detection at 5% perturbation >= 0.60
  - P3: non-contradict ratio >= 0.80

## 2. What Was Implemented

- Stage1 SFT + GRPO proxy pipeline:
  - scripts/train_stage1_sft_grpo.py
- Real policy loop with claim-level FEH reward:
  - scripts/train_stage2_policy_grpo.py
  - scripts/eval_policy_p123.py
- Stability and comparability upgrades:
  - fixed validation IDs
  - best-epoch checkpoint selection
  - reward normalization + EMA baseline
  - full trace and confusion matrix export in evaluation

## 3. Stage1 Small-Run Progress (Historical)

- smallrun1: P1=0.3583, P2=NOT_MET, P3=1.000
- smallrun2: P1=0.5458, P2=MET, P3=0.920
- smallrun3: P1=0.6458, P2=MET, P3=0.860
- smallrun4_fullval: P1=0.6765, P2=MET, P3=0.820

Interpretation:
- Stage1 proxy pipeline successfully improved from collapse to stable learning behavior.
- It was sufficient to justify moving into real policy closed-loop training.

## 4. Real Policy Closed-Loop Findings (mid3 -> mid4)

### 4.1 mid3 baseline vs c2boost_r2

- baseline (mid3_b_baseline):
  - P1=0.3971, P2=0.3333, P3=0.9630
  - confusion row(true=2): [2, 19, 1]
- c2boost_r2:
  - P1=0.4706, P2=0.3333, P3=0.9630
  - confusion row(true=2): [11, 10, 1]

Interpretation:
- Softened class-2 emphasis improved P1 and shifted class-2 errors away from pure neutral collapse.
- True class-2 recall remained low (pred=2 count still small).

### 4.2 soft-grid r3 (soft_a / soft_b / soft_c)

Comparison baseline:
- mid3_b_baseline: P1=0.3971, P3=0.9630

Results:
- soft_a:
  - P1=0.3824, P2=0.3333, P3=1.0000
  - class-2 row: [7, 15, 0]
- soft_b:
  - P1=0.4853, P2=0.3333, P3=0.8889
  - class-2 row: [8, 12, 2]
- soft_c:
  - P1=0.3824, P2=0.3333, P3=0.9630
  - class-2 row: [9, 12, 1]

Interpretation:
- soft_b is the best balance so far (highest P1, P3 still above gate).
- soft_a is too conservative (nearly no class-2 output).
- soft_c is too aggressive/unstable for current setting.

## 5. Critical Method Note

- The current synthetic P2 fallback is weakly sensitive to policy differences in this setup.
- Therefore, P2=0.3333 plateau should not be over-interpreted as final evidence of no policy gain.
- For next round, policy-coupled P2 evaluation should be strengthened.

## 6. Decision (Fixed Baseline)

- Fixed next baseline: soft_b parameters from mid4 soft-grid.
- Baseline launcher created:
  - sciconsist_pilot/scripts/run_stage2_softb_baseline_next.sh

## 7. What Is Proven vs Not Yet Proven

Proven:
- FEH-guided GRPO can improve policy-side P1 over prior baseline.
- Soft constraints are safer than hard gates and preserve P3.

Not yet proven:
- Robust class-2 recall at target level.
- P2 target at 5% perturbation under policy-coupled evaluation.
- Multi-seed stability for promotion to next stage gate.

## 8. Next Actions

1. Keep soft_b as baseline and run 3-seed reproducibility.
2. Upgrade P2 evaluation to stronger policy-coupled protocol.
3. Run focused ablation around soft_b:
   - false_contra_penalty: 0.20 / 0.25 / 0.30
   - c2_prob_bonus: 0.5 / 0.6 / 0.7
4. Re-check gate with fixed val_ids + confusion matrix trend on class-2 row.
