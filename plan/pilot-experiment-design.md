# Pilot Experiment Design: Cross-Modal Consistency Verification

## Objective
Design and implement a pilot experiment to validate the Cross-Model Factual Entailment Head (FEH) and GRPO training pipeline as outlined in the research plan.

---

## Experiment Steps

### 1. Data Preparation
- **Datasets**:
  - MuSciClaims (IJCNLP 2025): ~2K figure-claim pairs with support/contradict labels.
  - SciClaimEval (2026.02): 1,664 modified figure/table pairs.
  - S1-MMAlign subsets:
    - Positive pairs (~50K): Automatically paired as ENTAILS.
    - Synthetic CONTRADICTS (~50K): Text values altered by ±20%.
    - Random NEUTRAL (~50K): Random figure-text pairs.
- **Preprocessing**:
  - Normalize text and figure data.
  - Split datasets into training, validation, and test sets.

### 2. FEH Pretraining
- **Model**: InternVL2.5-8B (frozen for feature extraction).
- **Training**:
  - Input: Multi-layer averaged features from InternVL.
  - Output: {ENTAILS, NEUTRAL, CONTRADICTS} classification.
  - Accuracy target: >75% on held-out MuSciClaims + SciClaimEval.

### 3. GRPO Training
- **Policy Model**: Qwen2.5-VL-7B.
- **Reward Signal**:
  - Use FEH to compute R_FER(τ_i).
  - Penalize trajectories with ENTAILS claim ratio <30%.
- **Training Pipeline**:
  - Standard SFT → GRPO with FEH reward.
  - Evaluate reward hacking resistance.

### 4. Evaluation
- **Metrics**:
  - Accuracy, precision, recall for FEH.
  - Reward stability and diversity in GRPO outputs.
- **Ablation Studies**:
  - Compare FEH-Same vs FEH-Cross.
  - Sensitivity analysis for conflict ratio (1%-50%).

### 5. Documentation
- Record all configurations and results in Obsidian project memory.
- Generate visualizations for FEH attention heatmaps.

---

## Next Steps
1. Implement data preprocessing scripts.
2. Set up FEH pretraining pipeline.
3. Configure GRPO training environment.
4. Validate the pilot experiment setup.