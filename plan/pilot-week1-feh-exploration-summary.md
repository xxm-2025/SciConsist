# SciConsist Pilot Summary (Week1 FEH)

## 1. Objective

- Verify whether FEH can pass Week1 Go/No-Go and support the next stages.
- Practical targets during pilot iteration:
  - P1: push above threshold and stabilize
  - P2: keep numeric sensitivity consistently passing
  - P3: avoid many-to-one collapse and keep non-contradict ratio high

## 2. What P1/P2/P3 actually measure

- P1 (3-class base ability)
  - Evaluates ENTAILS/NEUTRAL/CONTRADICTS classification on validation split.
  - Most sensitive to: class weights, latent_dim, training length, LR, early stopping.
  - In this repo config, P1 is the gating metric for Week1 FEH Go/No-Go.
- P2 (numeric perturbation sensitivity)
  - Evaluates CONTRADICTS detection rates over perturbation levels 1%-20%.
  - Mainly checks whether the model catches numeric inconsistencies.
  - Across experiments, P2 was consistently MET and not the main bottleneck.
- P3 (many-to-one robustness)
  - Uses ENTAILS samples with small text noise to test whether model over-predicts CONTRADICTS.
  - High P3 means better tolerance to paraphrase-like variation and less collapse.

## 3. Exploration trajectory

- Early stage:
  - Pipeline and strict InternVL feature path were stabilized.
  - Real feature extraction + train/eval loops became reproducible.
- Mid stage:
  - Strong trade-off observed: pushing P1 often hurt P3, and vice versa.
  - Several runs achieved high P1 but low P3, or high P3 but low P1.
- Late stage (narrow sweeps + capacity tuning):
  - Narrow parameter sweeps around frontier points.
  - Increased latent_dim and longer training with controlled weights.
  - Finally reached all-pass solutions.

## 4. Key experimental outcomes

Representative points from result files:

- High P1 but weak P3:
  - `fixrun14_h`: P1=0.7937, P2=MET, P3=0.740
- Balanced all-pass frontier:
  - `fixrun19_u`: P1=0.7778, P2=MET, P3=0.800
  - `fixrun19_v`: P1=0.7778, P2=MET, P3=0.820
  - `fixrun19_w`: P1=0.7582, P2=MET, P3=0.820

Interpretation:

- Week1 FEH gating objective is achieved.
- A stable all-pass region exists near the `fixrun19` settings.

## 5. Week1 freeze decision

- Primary GO checkpoint:
  - Run: `fixrun19_v`
  - Frozen file: `/root/shared-nvme/sciconsist_pilot/outputs/checkpoints_week1/feh_week1_go.pt`
- Robust backup checkpoint:
  - Run: `fixrun19_w`
  - Frozen file: `/root/shared-nvme/sciconsist_pilot/outputs/checkpoints_week1/feh_week1_backup.pt`

Selection reason:

- Both pass P1/P2/P3 thresholds.
- `fixrun19_v` has stronger P3 while keeping P1 comfortably above GO.
- `fixrun19_w` is a nearby robust alternative with similar behavior.

## 6. Artifacts and evidence

- Freeze readme and checksums:
  - `/root/shared-nvme/sciconsist_pilot/outputs/checkpoints_week1/README_WEEK1.md`
- Main result JSONs:
  - `/root/shared-nvme/sciconsist_pilot/outputs/pilot_results_fixrun19_u_p123.json`
  - `/root/shared-nvme/sciconsist_pilot/outputs/pilot_results_fixrun19_v_p123.json`
  - `/root/shared-nvme/sciconsist_pilot/outputs/pilot_results_fixrun19_w_p123.json`
- Narrow-sweep logs:
  - `/root/shared-nvme/sciconsist_pilot/outputs/fixrun18_narrow.log`
  - `/root/shared-nvme/sciconsist_pilot/outputs/fixrun19_capacity_edge.log`

## 7. Recommended next step

- Use `feh_week1_go.pt` for Stage 1/2 integration path.
- Keep `feh_week1_backup.pt` as immediate fallback in case of integration drift.
- In Week2+, track P1/P3 together as a Pareto pair to avoid hidden regressions.
