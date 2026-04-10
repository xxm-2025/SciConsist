# SciConsist v3.1: Factual Entailment Reward for Cross-Modal Scientific Document Verification

> v3.1 修正版 | 基于 v2 三条批评 + v3 四条批评 重构 | 生成日期: 2026-04-10

---

## 版本演进摘要

### v2 → v3 核心改动

| v2 的致命伤 | v3 的修正 |
|------------|----------|
| R_cycle 用 surface-level sim() → Many-to-One 退化，多样性崩塌 | **Factual Entailment Head**：在隐空间做 NLI 三分类（entails/neutral/contradicts），不再做文本表面匹配 |
| SFT → DPO → GRPO 三阶段 buff 堆砌，说不清谁是核心 | **砍到两阶段**：标准 SFT（不是贡献）+ Factual Entailment GRPO（唯一核心），DPO 降级为可选消融 |
| PRISMM-Bench 仅 384 样本，随机种子波动 5-10% | **6 个评估集** + 3 seeds 置信区间 + 跨领域泛化测试 + 长文档一致性测试 |
| 只学"一致性" → 当文图确实矛盾时模型无法处理 | **Conflict Detection Bonus**：模型检测到真实文图冲突时给额外奖励，从"刷一致性"升华为"多模态事实裁判" |

### v3 → v3.1 防御补丁（基于四条新批评）

| v3 的风险 | v3.1 的防御 |
|-----------|-----------|
| FEH 与 MLLM 共享特征空间 → Reward Hacking | **Cross-Model FEH**：FEH 在 InternVL2.5 特征上训练，reward 计算完全独立于 policy model |
| NEUTRAL=0 → 模型生成模糊废话规避风险 | **Informativeness Gate**：每条 trajectory 的 ENTAILS claim 占比 < 30% 则额外扣分 |
| 训练集冲突比例 30-50% vs 真实 ~1% → 过度报告冲突 | **冲突率校准**：训练时控制 5% 冲突比例 + Sensitivity Analysis (1%-50%) |
| FEH 判定不可解释 | **FEH Rationale**：CONTRADICTS 时输出 attention heatmap 标记冲突位置 |

---

## 一、核心 Insight（修正版）

> 科研文档中 text / table / figure 对同一事实的天然冗余 ≠ 保证一致。
> 冗余中既有"可验证的一致"，也有"作者手误导致的真实冲突"。
>
> 我们提出 **Factual Entailment Reward (FER)**——一种在隐空间做事实蕴含判定的 reward 信号，
> 它同时训练模型做两件事：(1) 生成与证据一致的回答 (2) 检测并标记真实的跨模态冲突。
> 这把故事从"刷 faithfulness"升级为"**多模态科研文档审计**"。

---

## 二、为什么 v2 的 sim() 会死——具体分析

### v2 的 Reward 设计（有缺陷）

```
R_cycle(τ) = sim(claim_from_figure, text_ground_truth)
```

这里 `sim()` = sentence embedding cosine similarity (SciBERT/BGE-M3)。

### 三类致命场景

| 场景 | 模型生成的 claim | 原文写法 | 事实正确？ | sim() 分数 | 后果 |
|------|----------------|---------|-----------|-----------|------|
| **语义等价、表述不同** | "Model B's BLEU is 3.2 points below Model A" | "Our method significantly outperforms all baselines" | ✅ 正确 | 🔴 低 (0.3) | 正确推理被惩罚 |
| **原文未提及** | "The error bar of Model C suggests high variance" | （原文没讨论这个） | ✅ 正确 | 🔴 极低 (0.1) | 正确观察被当幻觉 |
| **真实冲突** | "Fig.3 shows accuracy is 85%" | "We achieved 58% accuracy" | 🚩 作者手误 | 🔴 低 (0.2) | 应该检测冲突，却只给低分 |

**结论**：surface sim() 把"表述不同"、"未提及"、"真实冲突"三种完全不同的情况混为一谈，全部给低分。长期训练后模型只会复读原文 → diversity collapse。

---

## 三、方法设计（v3 重构）

### 核心创新：Cross-Model Factual Entailment Head (FEH)

**替代 surface sim()，在隐空间做事实蕴含判定。**
**v3.1 关键变更：FEH 的特征提取与 policy model 完全解耦，杜绝 reward hacking。**

#### 架构（v3.1 Cross-Model 设计）

```
   ┌─────────────────────────────────────────────────────────────────┐
   │                      REWARD 计算 (完全独立)                       │
   │                                                                  │
   │    ┌──────────────────────────┐                                  │
   │    │  InternVL2.5-8B (FROZEN) │  ← 独立于 policy model           │
   │    │  Vision + Language Encoder│                                  │
   │    └───────────┬──────────────┘                                  │
   │                │                                                  │
   │    ┌───────────┼──────────────────┐                              │
   │    ▼                              ▼                              │
   │  figure_region_hidden      text_claim_hidden                     │
   │  (multi-layer avg:         (multi-layer avg:                     │
   │   layers L/4,L/2,3L/4,L)   layers L/4,L/2,3L/4,L)              │
   │    │                              │                              │
   │    ▼                              ▼                              │
   │  Visual Proj (2-layer MLP)  Text Proj (2-layer MLP)             │
   │    │                              │                              │
   │    ▼                              ▼                              │
   │  h_v (d=512)                   h_t (d=512)                      │
   │    │                              │                              │
   │    └──────► [h_v ; h_t ; |h_v - h_t|] ◄──┘                     │
   │                     │                                            │
   │                     ▼                                            │
   │            Entailment Classifier                                 │
   │            (2-layer MLP → softmax)                               │
   │                     │                                            │
   │                     ▼                                            │
   │      {ENTAILS, NEUTRAL, CONTRADICTS}                             │
   │      + attention heatmap (if CONTRADICTS)                        │
   └─────────────────────────────────────────────────────────────────┘
                         │
                    R_FER(τ_i) ──→ GRPO gradient ──→ Qwen2.5-VL-7B (policy)
```

#### 为什么用 Cross-Model 而非 Same-Model（v3 → v3.1 关键变更）

**v3 的 Reward Hacking 风险**：FEH 用 Qwen 的 hidden states → Qwen 在 GRPO 中被训练 → Qwen 可以学会生成"语义上错误但 hidden states 恰好命中 FEH ENTAILS 边界"的输出 → 刷分不增智。

**v3.1 的解决方案**：
1. FEH 在 **InternVL2.5-8B** 的特征上预训练
2. GRPO 训练时，对每个 (figure, claim) 对，过一遍 **冻结的 InternVL2.5-8B** 取 hidden states → FEH 判定
3. Policy model (Qwen2.5-VL-7B) 的参数变化 **完全无法影响** InternVL 的特征空间
4. Reward hacking 在物理上被阻断

**额外实验价值**：如果 Cross-Model FEH 有效，证明 FEH 捕捉的是 **模型无关的事实结构**（model-agnostic factual structure），而非特定模型的 hidden state artifact。这本身就是一个强实验结论。

#### 关键设计选择

1. **Multi-Layer Fusion**：不取单一最后层，取 {L/4, L/2, 3L/4, L} 四层的平均。增加通过操纵特定层来欺骗的难度。（即使是 cross-model 设计，multi-layer 也有助于更稳定的特征表示）

2. **三分类而非二分类**。NEUTRAL 类专门处理"事实正确但原文没提"和"不确定"的情况，解决 Many-to-One 问题。

3. **特征拼接用 `[h_v; h_t; |h_v - h_t|]`**（借鉴 InferSent/ConflictAwareAH 的 element-wise difference），差异向量帮助捕捉冲突信号。

4. **CONTRADICTS 时输出 attention heatmap**（v3.1 新增）。利用 InternVL 的 cross-attention weights，标记 figure 中与 claim 冲突的具体区域。不参与 loss 计算，纯粹用于论文中的可解释性展示和 case study。

#### FEH 预训练

在用 FEH 做 reward 之前，先在 InternVL2.5-8B 特征上预训练：

| 数据来源 | 类型 | 数量 | 标注方式 |
|---------|------|------|---------|
| MuSciClaims (IJCNLP 2025) | figure-claim 对 + support/contradict 标签 | ~2K | 已有标注 |
| SciClaimEval (2026.02) | 修改过的 figure/table + 原始 claim | 1,664 | 已有标注 |
| S1-MMAlign 子集 | figure-text 配对（正例） | ~50K | 自动配对 = ENTAILS |
| S1-MMAlign 合成 | 把 text 数值篡改 ±20% | ~50K | 合成 = CONTRADICTS |
| 随机配对 | 随机 figure + 不相关 text | ~50K | 自动 = NEUTRAL |

预训练 FEH（参数量 ~10M）仅需 1-2h on single 4090。
InternVL2.5-8B 仅做 frozen inference 提特征，不需要训练。

**FEH 准确率要求**：在 held-out MuSciClaims + SciClaimEval 上三分类 accuracy > 75% 方可作为 reward。

#### Reward Hacking 防御的消融验证

| 配置 | 描述 | 预期 |
|------|------|------|
| FEH-Same (v3) | FEH 在 Qwen 特征上训练，reward 用 Qwen 特征 | 可能有 reward hacking |
| FEH-Cross (v3.1) | FEH 在 InternVL 特征上训练，reward 用 InternVL 特征 | 无 hacking |
| FEH-Cross-Check | 用 FEH-Same 训练完的模型，用 FEH-Cross 重新评估 | 如果 FEH-Same 分数高但 FEH-Cross 分数低 → 证实 hacking 发生 |

**A8 消融**：FEH-Same vs FEH-Cross。如果两者效果接近 → same-model 没 hacking（好事）；如果 FEH-Same 在自己的评分上高但 FEH-Cross 上低 → 证实 hacking → 必须用 cross-model。

---

### C1: Tri-State Factual Entailment Reward（核心贡献）

**替代 v2 的 R_cycle，解决退化问题。v3.1 新增 Informativeness Gate + 冲突率校准。**

```
给定 MLLM 生成的 trajectory τ（包含若干 factual claims）:

对每个 claim c_i:
    1. 将 (figure_region, c_i) 输入冻结的 InternVL2.5-8B 提取 hidden states
    2. FEH 判定: label_i = FEH(h_visual, h_text) ∈ {ENTAILS, NEUTRAL, CONTRADICTS}
    3. 赋分:
       R(c_i) = { +1.0   if ENTAILS
                 -0.1    if NEUTRAL       ← v3.1 修正: 轻微负分而非 0
                 -0.5    if CONTRADICTS }

整体 trajectory reward:
    R_FER(τ) = (1/N) Σ R(c_i)                    ... 基础分
             + λ_cross · cross_modal_bonus(τ)      ... 跨模态交叉验证 bonus
             + λ_conflict · conflict_detection(τ)   ... 冲突检测 bonus
             + λ_info · informativeness_gate(τ)     ... v3.1: 信息量门控
```

#### Informativeness Gate（v3.1 新增，防止 NEUTRAL 避难所效应）

**风险**：NEUTRAL 给 0 分（v3）→ 模型可能生成大量模糊废话来规避风险、稳拿 0 分。

**v3.1 双重防御**：

防御 1：NEUTRAL 分数从 0.0 改为 **-0.1**。轻微负分让模型有压力"做出判断"而非逃避。

防御 2：Informativeness Gate——基于 ENTAILS claim 占比的硬约束：

```
informativeness_gate(τ) = {
    0.0    if entails_ratio(τ) ≥ 0.3   ← 至少 30% claims 是 ENTAILS → 正常
   -0.5    if entails_ratio(τ) < 0.3    ← 太多模糊/无信息 → 扣分
}

其中 entails_ratio(τ) = count(ENTAILS claims) / count(all claims)
```

**为什么 0.3 阈值**：pilot 实验中统计正常高质量回答的 ENTAILS 比例，取 P25 作为阈值。

**GRPO 的内生防御**：即使不加 informativeness gate，GRPO 的 group 内归一化也提供了部分保护——如果所有 response 都是废话（全 NEUTRAL），advantage 全 ≈ 0，没有梯度信号。但这只能防止"全废话"，无法防止"少量 ENTAILS + 大量 NEUTRAL 填充"的策略。Gate 针对的正是后者。

#### Cross-Modal Bonus（跨模态交叉验证）

当 table 和 figure 对同一数据点给出一致的数值时，额外加分：
```
cross_modal_bonus = (1/M) Σ exact_match(value_from_figure_j, value_from_table_j) × 0.5
```
数值匹配用 relative tolerance < 5%。这部分用 exact match，不需要 FEH。

#### Conflict Detection Bonus（冲突检测奖励，v3.1 修正训练分布）

**v3 的独特贡献点，v3.1 修正了训练时的冲突比例问题。**

当模型生成的回答中 **主动报告了** 文图之间的真实冲突时，给予额外奖励：

```
conflict_detection(τ) = {
    +1.5   if τ 包含 "Detected conflict: [description]"
           AND FEH(figure_region, text_claim) = CONTRADICTS    ← 验证确实有冲突
    -1.0   if τ 报告了冲突 BUT FEH 判定为 ENTAILS            ← 虚假报告惩罚
     0.0   otherwise
}
```

**v3.1 冲突率校准**：

v3 的问题：如果训练数据中 30-50% 包含冲突，但真实论文只有 ~1% 冲突 → 模型会过度报告（"疑心病"）。

修正方案：
1. **训练时冲突比例控制在 5%**（而非 30-50%）：每 batch 中 95% 的样本是正常论文（无冲突），5% 是注入了冲突的样本
2. 模型在绝大多数样本上学到的是"正常一致性验证"，只在极少数样本上触发冲突检测
3. 冲突样本来源：PRISMM-Bench 真实不一致 + 人工数值篡改（±20%）

**v3.1 新增 Sensitivity Analysis 实验**（见实验设计部分）：
- 在冲突率 1%, 5%, 10%, 25%, 50% 的测试集上分别报告 Precision / Recall / F1
- 重点关注 **FPR@1% conflict rate**：在正常论文上的误报率
- 目标：FPR < 5%（即 100 篇正常论文中最多 5 篇被误报为有冲突）

### C2: 简化训练 Pipeline（解决 buff 堆砌）

**v2 的三阶段 → v3 的两阶段。DPO 从"核心组件"降级为"可选消融"。**

```
┌──────────────────────────────────────────────────────────┐
│                    v3 Training Pipeline                    │
│                                                           │
│  ┌──────────┐  ┌──────────┐                              │
│  │ S1-MMAlign│  │ SciMDR   │                              │
│  │ 15.5M     │  │ 300K QA  │                              │
│  └─────┬─────┘  └─────┬────┘                              │
│        │              │                                    │
│        ▼              ▼                                    │
│  ╔══════════════════════════════════════════╗              │
│  ║  Stage 1: Standard SFT (NOT our contrib)║              │
│  ║  Standard cross-entropy on SciMDR 300K  ║              │
│  ╚═══════════════════╤══════════════════════╝              │
│                      ▼                                     │
│  ╔══════════════════════════════════════════╗              │
│  ║  Stage 2: FER-GRPO (OUR CORE CONTRIB)   ║              │
│  ║                                          ║              │
│  ║  Reward = R_FER(τ)                       ║              │
│  ║        = Entailment Score                ║              │
│  ║        + Cross-Modal Bonus               ║              │
│  ║        + Conflict Detection Bonus        ║              │
│  ║                                          ║              │
│  ║  + VPPO-style claim-token reweighting    ║              │
│  ╚═══════════════════╤══════════════════════╝              │
│                      ▼                                     │
│              SciConsist-7B (Final)                          │
│                                                           │
│  ┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈               │
│  Optional ablation variant:                                │
│  Stage 1 → DPO (using FER as scoring) → Stage 2           │
│  (to show DPO adds little beyond FER-GRPO)                 │
│  ┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈               │
└──────────────────────────────────────────────────────────┘
```

**为什么砍掉 DPO 作为核心阶段**：
1. 如果 FER 足够好，GRPO 直接用 FER 做在线 reward 就够了，不需要先做离线偏好学习
2. DPO 的偏好对本身是用 FER 打分构造的，信息来源相同，多一轮中介反而引入噪声
3. **审稿人检验**：消融中对比 "SFT + FER-GRPO" vs "SFT + FER-DPO + FER-GRPO"，如果差距 < 2pp → 证明 GRPO 单独就够，pipeline 干净

**为什么不砍掉 SFT**：
- SciMDR 300K 的 SFT 建立科研文档的基础理解能力，没有它模型连 claim extraction 都做不好
- 但 SFT 不是我们的贡献，用现成的方法即可

### 核心贡献的纯粹性论证

当审稿人问"你的核心算法贡献到底是什么"时，回答极其清晰：

> **Factual Entailment Head (FEH)** + **Tri-State Reward with Conflict Detection**。
>
> FEH 是一个在 MLLM 隐空间上训练的轻量级事实蕴含判别器。
> 它解决了 cycle consistency 方法的 Many-to-One 退化问题（CycleReward/CycleCap 没解决）。
> Conflict Detection Bonus 让模型从"被动对齐"变为"主动审计"（没有前人做过）。
> 两者组合成 FER，直接作为 GRPO reward，不需要 DPO 中介。

---

## 四、与已有工作的区别（更新版）

### vs CycleReward/CycleCap

| 维度 | CycleReward/CycleCap | SciConsist v3 |
|------|---------------------|---------------|
| 匹配方式 | image → caption → regenerated image → pixel/CLIP 相似度 | **隐空间 NLI 三分类**（不做表面匹配） |
| 退化处理 | 无 (Many-to-One 退化) | **NEUTRAL 类** 吸收"正确但表述不同"的情况 |
| 冲突检测 | 无 | **Conflict Detection Bonus** |
| 应用域 | 通用 image captioning | 科研文档（利用结构化冗余） |
| Reward 粒度 | image-level | **claim-level**（每个 factual claim 独立评分） |

### vs GVF (Factual Anchor)

| 维度 | GVF | SciConsist v3 |
|------|-----|---------------|
| "Anchor"含义 | 数据增强中的事实锚点（prompt 层面） | **隐空间中的事实蕴含判别器**（representation 层面） |
| 训练方式 | SFT + Factual Consistency Loss | **GRPO + tri-state reward** |
| 冲突处理 | 用 counterfactual prompts 做增强 | **模型主动检测冲突，给奖励** |
| 领域 | 通用 VQA | 科研文档 |

### vs DeFacto

| 维度 | DeFacto | SciConsist v3 |
|------|---------|---------------|
| Counterfactual 来源 | 合成图像掩码 | **文档内天然冗余**（text↔figure↔table） |
| Reward 信号 | GRPO + counterfactual reasoning | **FEH 三分类** + conflict detection |
| 数据需求 | 需要合成 counterfactual images | **零合成**：文档本身就有"天然反事实"（不同模态的互相验证） |

### vs VISA (Visual Semantic Anchoring)

| 维度 | VISA | SciConsist v3 |
|------|------|---------------|
| 对齐目标 | MLLM 中间层 ↔ 冻结 VFM（表示层对齐） | **Claim ↔ Visual Evidence**（事实层对齐） |
| 任务 | 提升 fine-grained perception | 提升 **faithfulness** + **conflict detection** |
| 训练信号 | 表示相似度 loss | **事实蕴含 reward** |

---

## 五、实验设计（针对批评三 重构）

### 5.1 评估矩阵（6 评估集 × 3 seeds × 置信区间）

| 数据集 | 规模 | 评估维度 | 解决什么批评 |
|--------|------|---------|-------------|
| **PRISMM-Bench** | 384 样本 | 不一致检测 / 修复 / 配对 | 直接评估 (但不是唯一) |
| **MuSciClaims** (IJCNLP 2025) | ~2K samples | 图表 claim 验证 F1 | 补充规模 + 科研图表专用 |
| **SciClaimEval** (2026.02) | 1,664 samples (ML/NLP/Medicine) | 跨模态 claim 验证 | **跨领域泛化** |
| **SciMDR-Eval** (2026.03) | 专家标注子集 | 科研文档多模态 QA | 推理能力不 regression |
| **BRIDGE** (2026.03) | 长文档多跳推理 | 证据定位 + 推理链 | **长文档一致性** |
| **ChartQAPro** | 通用图表 QA | 图表理解 | 不 regression |

**报告标准**：所有结果报告 mean ± std across 3 random seeds。如果 std > 3pp，增加到 5 seeds。

### 5.2 新增评估维度

#### 长文档一致性测试（解决 "384 样本太小"）

从 S1-MMAlign 中筛选 >10 页的论文 200 篇：
- 在每篇中选 text claim（从 Introduction）和 figure evidence（在 Experiments section，距离 >5 页）
- 测模型能否正确跨越远距离验证事实
- 这是现有所有 benchmark 都没覆盖的维度

#### 冲突检测测试（v3 独有）

构造方式：
- 从 PRISMM-Bench 的 384 个真实不一致中抽取 100 个
- 从 SciClaimEval 的 figure-modified 样本中抽取 200 个
- 从 S1-MMAlign 中人工注入 100 个数值冲突
- 总计 400 样本的 Conflict Detection Test Set
- 指标：Conflict Detection F1（precision + recall）

#### 跨领域零样本测试

在 CS 论文上训练，在以下领域零样本测试：
- 医学（SciClaimEval Medicine subset）
- 生物学（从 PubMed Central 抽取 100 篇带图表的论文）
- 经济学（从 NBER 抽取 50 篇带图表的论文）

### 5.3 Baselines

| Baseline | 类型 | 说明 |
|----------|------|------|
| Qwen2.5-VL-7B (zero-shot) | 基座 | 最直接 baseline |
| + SciMDR SFT | SFT only | 消融 Stage 1 |
| + EMPO (EMNLP 2025) | DPO | hallucination-detection based 偏好对 |
| + PaLMR V-GRPO | RL | LLM judge reward (非 self-supervised) |
| + VPPO (ICLR 2026) | RL | Token-level visual dependency |
| + CycleReward-style | RL | surface sim() reward (v2 的方案，作为消融) |
| InternVL2.5-8B (zero-shot) | 跨 backbone | 泛化验证 |
| GPT-4o / Qwen2.5-VL-72B | API 上界 | upper bound |
| **SciConsist-7B (ours)** | FER-GRPO | Stage 1 SFT + Stage 2 FER-GRPO |

### 5.4 消融实验（v3.1：11 个配置，逐条回应已知批评）

**核心消融：证明每个设计选择的独立贡献**

| ID | 配置 | 去掉/替换什么 | 回应哪条批评 |
|----|------|-------------|-------------|
| A1 | SFT only | 所有 RL | baseline |
| A2 | SFT + surface-sim GRPO | FEH → surface sim() | v2 批评一 (退化) |
| A3 | SFT + FER-GRPO (binary) | NEUTRAL 类 → 只留 ENTAILS/CONTRADICTS | v2 批评一 |
| A4 | SFT + FER-GRPO (no conflict) | Conflict Detection Bonus | v3 卖点验证 |
| A5 | SFT + FER-DPO only | GRPO | v2 批评二 (buff 堆砌) |
| A6 | SFT + FER-DPO + FER-GRPO | 完整 3 阶段 | v2 批评二 |
| A7 | SFT + FER-GRPO (ours, v3.1) | — | **完整方案** |
| **A8** | **SFT + FEH-Same-Model GRPO** | **FEH 在 Qwen 特征上 (v3 设计)** | **v3 批评一 (reward hacking)** |
| **A9** | **SFT + GPT-4o-as-Judge GRPO** | **FEH → 冻结的 GPT-4o API judge** | **v3 批评四 (效率对比)** |
| **A10** | **SFT + FER-GRPO (NEUTRAL=0)** | **NEUTRAL 分数从 -0.1 改回 0** | **v3 批评二 (避难所效应)** |
| **A11** | **SFT + FER-GRPO (no info gate)** | **去掉 informativeness gate** | **v3 批评二** |

**每条批评的消融对照**：

**回应 v3 批评一 (Reward Hacking)**：
- A8 (FEH-Same) vs A7 (FEH-Cross) → 如果 A8 在 FEH-Same 评分上高但在 FEH-Cross 重评时低 → 证实 hacking → cross-model 设计必要
- **关键指标**：对 A8 训练完的模型，用 FEH-Cross 重新评分。如果 FEH-Same 分数 - FEH-Cross 分数 > 0.3 → hacking 严重

**回应 v3 批评二 (NEUTRAL 避难所)**：
- A10 (NEUTRAL=0) vs A7 (NEUTRAL=-0.1) → NEUTRAL 轻微负分是否改善信息量
- A11 (no info gate) vs A7 (with gate) → informativeness gate 是否有效防止废话
- **关键指标**：除了标准 accuracy/F1 外，还要报告 **distinct-n** (多样性)、**avg_claim_count** (信息量)、**entails_ratio** (有效claim占比)

**回应 v3 批评三 (冲突过度报告)**：
- Sensitivity Analysis（独立实验，见下方 5.5）

**回应 v3 批评四 (效率)**：
- A9 (GPT-4o judge) vs A7 (FEH) → 效果和效率对比
- **关键指标**：Accuracy 差距 + reward 评估速度（FEH: ~0.01s/sample vs GPT-4o: ~2-5s/sample，预期 200-500× 加速）
- 如果 FEH 效果 ≥ GPT-4o judge 且快 200× → 独立卖点

**审稿人最想看的 Top 5 对比**：
1. A2 vs A7 → FEH 隐空间判定 vs surface matching
2. A8 vs A7 → Cross-Model vs Same-Model (reward hacking 验证)
3. A9 vs A7 → FEH vs LLM-as-Judge (效率 + 效果)
4. A3 vs A7 → 三分类 vs 二分类 (NEUTRAL 的价值)
5. A5 vs A7 → DPO vs GRPO (pipeline 精简论证)

### 5.5 分析实验（v3.1 扩充版）

#### 核心分析（必做）

| 分析 | 内容 | 回应什么 |
|------|------|---------|
| **FEH vs Human Agreement** | FEH 的三分类判定和人类标注的 Cohen's κ | FEH 作为 reward 是否可靠 |
| **Diversity + Informativeness Analysis** | distinct-n, self-BLEU, **avg_claim_count**, **entails_ratio** | **v3 批评二 (避难所)** |
| **Reward Hacking Detection** | A8 模型用 FEH-Cross 重评 vs FEH-Same 自评的分数差 | **v3 批评一 (hacking)** |
| **Conflict Sensitivity Analysis** | 冲突率 1%/5%/10%/25%/50% 下的 P/R/F1 + FPR@1% | **v3 批评三 (过度报告)** |
| **FEH vs GPT-4o Judge** | 在相同样本上：三分类 accuracy + 每样本推理耗时 | **v3 批评四 (效率)** |

#### 辅助分析（如有余力）

| 分析 | 内容 | 回应什么 |
|------|------|---------|
| **Error Taxonomy** | PRISMM-Bench 错误类型分布 (数值/实体/趋势/条件) | 哪类最受益 |
| **Distance-Accuracy Curve** | 随 text-figure 距离增大，验证准确率如何变化 | 长文档能力 |
| **FEH Scaling** | FEH 训练数据从 10K → 50K → 150K 的效果变化 | FEH 的 data efficiency |
| **Cross-Model Transferability** | FEH 在 InternVL 上训练，但用 Llama-Vision 特征评估 | FEH 的模型无关性 |
| **FEH Rationale Visualization** | CONTRADICTS 时的 attention heatmap + diff vector 可视化 | 可解释性 (加分项) |

#### Conflict Sensitivity Analysis 详细设计

```
构造 5 个测试集（每个 1000 samples）：
  - test_1pct:   990 正常 + 10 冲突
  - test_5pct:   950 正常 + 50 冲突
  - test_10pct:  900 正常 + 100 冲突
  - test_25pct:  750 正常 + 250 冲突
  - test_50pct:  500 正常 + 500 冲突

正常样本：从 S1-MMAlign 中随机选取的 figure-text 一致对
冲突样本：50% 来自 PRISMM-Bench 真实不一致 + 50% 人工数值篡改

报告：
  1. 每个冲突率下的 Conflict Detection Precision / Recall / F1
  2. 每个冲突率下的 False Positive Rate（正常样本被误报为冲突的比率）
  3. 绘制 FPR vs Conflict Rate 曲线
  4. 目标：FPR@1% < 5%（在几乎没有冲突的正常论文上，误报率 < 5%）
```

#### Informativeness Analysis 详细设计

```
对比以下配置训练出的模型：
  - A7  (NEUTRAL=-0.1, with info gate)
  - A10 (NEUTRAL=0.0, with info gate)
  - A11 (NEUTRAL=-0.1, no info gate)
  - A2  (surface sim, 作为 baseline)

指标：
  1. avg_claim_count: 每个 response 中 factual claim 的平均数量
     → 越高说明模型越愿意做具体陈述
  2. entails_ratio: ENTAILS claims / total claims
     → 越高说明模型做出的陈述越有信心
  3. distinct-1/2/3: unigram/bigram/trigram 多样性
     → 防止 diversity collapse
  4. self-BLEU: 同一 prompt 下 G 个 response 之间的相似度
     → 越低说明生成越多样
  5. 标准 accuracy/F1 在各 benchmark 上的表现
     → 保证以上指标的改善不以性能为代价
```

---

## 六、Paper Narrative（v3 修正版）

### Title 候选

1. **SciConsist: Factual Entailment Reward for Cross-Modal Scientific Document Verification**
2. **Beyond Surface Matching: Latent Factual Entailment as Self-Supervised Reward for Scientific Document Understanding**
3. **Teach Models to Audit: Factual Entailment Reward with Conflict Detection for Multimodal Scientific Reasoning**

推荐 Title 3——"Teach Models to Audit" 直接传达故事升级。

### Abstract 核心论点（7 句）

1. [问题] 多模态大模型在科研文档推理中频繁产生跨模态不一致——文本 claim 与图表证据矛盾、数值偏差、趋势描述错误。
2. [现有方案的缺陷] 已有的 cycle consistency 方法（CycleReward, CycleCap）依赖表面文本匹配作为 reward，导致 Many-to-One 退化：当模型生成了事实正确但表述不同的回答时被错误惩罚，最终模型退化为复读机。
3. [我们的 Insight] 我们观察到科研文档中 text/table/figure 的冗余不仅包含"可验证的一致"，也包含"作者手误导致的真实冲突"——一个好的 reward 应该同时激励一致性和冲突检测。
4. [方法 - FEH] 我们提出 Factual Entailment Head (FEH)，一个在 MLLM 隐空间上训练的轻量级事实蕴含判别器，将 claim-evidence 对分类为 ENTAILS/NEUTRAL/CONTRADICTS，从根本上解决表面匹配的退化问题。
5. [方法 - FER] 基于 FEH，我们设计 Factual Entailment Reward (FER)，一个三态 reward 信号，并引入 Conflict Detection Bonus——当模型正确识别文图冲突时给予额外奖励。
6. [结果] SciConsist-7B 在 PRISMM-Bench (不一致检测)、MuSciClaims (claim 验证)、SciMDR-Eval (文档推理) 上均取得显著提升，同时在专用 Conflict Detection Test 上首次展示了 7B 模型的跨模态冲突检测能力。
7. [意义] 我们证明了在隐空间做事实蕴含判定比表面匹配更鲁棒，冲突检测 bonus 使模型从被动对齐器升级为主动审计器，为科研文档的自动化质量控制开辟了新路径。

### 预期 Contribution

| 贡献 | 类型 | 新颖度 |
|------|------|--------|
| **Factual Entailment Head (FEH)**：隐空间事实蕴含判别 | Algorithmic (trained component) | 🔴 核心新颖 |
| **Tri-State Reward + Conflict Detection Bonus** | Reward design | 🔴 核心新颖 |
| Conflict Detection Test Set (400 samples) | Evaluation resource | 🟡 附加 |
| 消融证明 FER > surface matching, NEUTRAL 类的关键作用 | Empirical insight | 🟡 附加 |
| 6 数据集跨领域评估 + diversity analysis | Thorough evaluation | 🟢 基础要求 |

---

## 七、关键技术细节

### FEH 训练细节（v3.1 Cross-Model 版本）

```python
class FactualEntailmentHead(nn.Module):
    """
    在独立 VLM (InternVL2.5) 隐空间上做事实蕴含三分类。

    v3.1: 特征来自 InternVL2.5-8B（frozen），与 policy model (Qwen) 完全解耦。
    Multi-layer fusion: 取 {L/4, L/2, 3L/4, L} 四层 hidden states 的平均。

    输入：InternVL2.5 对 figure_region 和 text_claim 编码后的 multi-layer averaged hidden states
    输出：{ENTAILS, NEUTRAL, CONTRADICTS} 的概率分布 + attention heatmap (optional)
    """
    def __init__(self, hidden_dim: int = 4096, latent_dim: int = 512, num_layers_to_fuse: int = 4):
        super().__init__()
        self.visual_proj = nn.Sequential(
            nn.Linear(hidden_dim, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, latent_dim),
        )
        self.text_proj = nn.Sequential(
            nn.Linear(hidden_dim, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, latent_dim),
        )
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim * 3, latent_dim),  # [h_v; h_t; |h_v - h_t|]
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(latent_dim, 3),  # ENTAILS=0, NEUTRAL=1, CONTRADICTS=2
        )
        self.layer_indices: list[int] = []  # 由 init 时根据模型总层数计算

    @staticmethod
    def fuse_layers(
        all_hidden_states: list[torch.Tensor],
        layer_indices: list[int],
    ) -> torch.Tensor:
        """取指定层的均值作为 multi-layer fused representation。"""
        selected = torch.stack([all_hidden_states[i] for i in layer_indices])
        return selected.mean(dim=0)  # (B, seq_len, hidden_dim)

    def forward(
        self,
        h_visual: torch.Tensor,
        h_text: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        v = self.visual_proj(h_visual)   # (B, d)
        t = self.text_proj(h_text)       # (B, d)
        diff = torch.abs(v - t)          # (B, d) — 冲突信号
        combined = torch.cat([v, t, diff], dim=-1)  # (B, 3d)
        logits = self.classifier(combined)  # (B, 3)
        return logits, diff  # diff 可用于 attention heatmap 可视化
```

### FER-GRPO 训练循环（v3.1）

```
# 初始化
policy_model = Qwen2.5-VL-7B (trainable)
reward_encoder = InternVL2.5-8B (FROZEN, 仅用于提取 FEH 输入特征)
feh = FactualEntailmentHead (FROZEN, 在 InternVL 特征上预训练)

for each prompt x (scientific document question):
    1. policy_model 生成 G 个 response {τ_1, ..., τ_G}  (G=8)
    
    2. 对每个 τ_i:
       a. 提取 factual claims: {c_1, ..., c_K}
       b. 对每个 claim c_j:
          - 将 (figure_region, c_j) 输入 reward_encoder (InternVL2.5)
          - 取 multi-layer fused hidden states: h_v, h_t
          - label_j, diff_j = FEH(h_v, h_t)
          - R(c_j) = {+1.0 if ENTAILS, -0.1 if NEUTRAL, -0.5 if CONTRADICTS}
       c. R_base(τ_i) = mean(R(c_j) for j in 1..K)
       d. cross_modal = mean(exact_match(figure_val_j, table_val_j)) × 0.5
       e. conflict = +1.5 if (τ_i reports conflict AND FEH confirms CONTRADICTS)
                   = -1.0 if (τ_i reports conflict BUT FEH says ENTAILS)
                   = 0.0 otherwise
       f. info_gate = 0.0 if (count_ENTAILS / K ≥ 0.3) else -0.5
       g. R_FER(τ_i) = R_base
                      + λ_cross × cross_modal
                      + λ_conflict × conflict
                      + λ_info × info_gate
    
    3. GRPO update:
       A_i = (R_FER(τ_i) - mean(R_FER)) / std(R_FER)  # 标准 GRPO advantage
       L = -Σ A_i × log π(τ_i | x) - β × KL(π || π_ref)
       + VPPO-style claim-token reweighting: 对 factual claim tokens 加 2× gradient weight
```

**v3 → v3.1 关键变化**：
1. reward_encoder 和 policy_model 是两个不同的模型（InternVL vs Qwen）
2. NEUTRAL 分数从 0.0 → -0.1（轻微推压）
3. 新增 info_gate 项防止废话填充

### λ 超参数

| 参数 | 默认值 | 范围 | 调优策略 |
|------|--------|------|---------|
| λ_cross | 0.3 | [0.1, 0.5] | grid search on pilot set |
| λ_conflict | 0.2 | [0.1, 0.4] | 设低一点防止模型过度报告冲突 |
| β (KL penalty) | 0.04 | [0.01, 0.1] | 跟随 PaLMR 的设置 |
| G (group size) | 8 | {4, 8, 16} | 8 是 GRPO 标准 |
| VPPO claim weight | 2.0 | [1.5, 3.0] | 跟随 VPPO 的设置 |

---

## 八、与 v2 逐篇对比表的增量（新增文献）

| # | 论文 | 借鉴什么 |
|---|------|---------|
| 21 | **GVF**: Grounded Visual Factualization (2025) | Factual anchor 的概念启发；但 GVF 是 prompt-level anchor，我们是 representation-level |
| 22 | **DeFacto**: Counterfactual Reasoning + GRPO (2025) | Counterfactual 训练范式；但 DeFacto 用合成掩码，我们用文档天然冗余 |
| 23 | **VISA**: Visual Semantic Anchoring (submitted ICLR 2026) | 表示层对齐的思路；但 VISA 是 VFM anchor，我们是 factual entailment |
| 24 | **ConflictAwareAH**: Conflict-Aware Multimodal Fusion (2026.03) | element-wise difference 捕捉冲突的特征设计 |
| 25 | **CROSSCHECK**: Vision-Language Conflict Detection (2025/2026) | 冲突检测 benchmark 设计参考 |
| 26 | **MJ1**: Multimodal Judgment via Grounded Verification (2026.03) | Counterfactual consistency reward 设计 |

---

## 九、计算资源估算（4× 4090，v3.1 修正版）

| 阶段 | 内容 | 时间 | 显存 |
|------|------|------|------|
| InternVL2.5 特征提取 | Frozen inference for FEH 训练集 (~150K) | 4-6h | 1×24GB |
| FEH 预训练 | 三分类头，~150K 训练样本 | 1-2h | 1×24GB |
| Stage 1 SFT | Qwen2.5-VL-7B LoRA on SciMDR 300K | 8-12h | 4×24GB (ZeRO-2) |
| FEH Scoring | InternVL2.5 inference + FEH (~50K) | 3-4h | 1×24GB |
| Stage 2 FER-GRPO | 7B policy + InternVL2.5 reward encoder | 15-20h | 4×24GB (Qwen 3卡 + InternVL 1卡) |
| 消融 A8 (FEH-Same) | 同上但 FEH 在 Qwen 特征上 | 12-18h | 4×24GB |
| 消融 A2-A6, A9-A11 | 各种消融配置 | 共 ~30-40h | 4×24GB |
| Evaluation | 6 datasets, 3 seeds, 11 配置 | 10-15h | 1×24GB |
| **Total** | | **~90-120h GPU** | |

**v3.1 比 v3 多出的开销**：
- InternVL2.5 特征提取 (+4-6h)
- GRPO 时需要同时跑 InternVL inference (+3-5h)
- A8 消融 (+12-18h)
- A9 GPT-4o judge：API 费用 ~$50-100（10K samples × 8 responses × ~$0.001/call）

**优化策略**：
- InternVL2.5 特征可以预计算缓存（对 FEH 训练集和 GRPO 训练集）→ 减少重复 inference
- A2-A6 消融共享 Stage 1 SFT checkpoint → 不重复 SFT
- 4× 4090 可以 2 个消融并行跑（每个用 2 卡）

---

## 十、执行 Timeline（6 周）

### Week 1: Cross-Model FEH Pilot（最关键的一周）

- [ ] 下载 MuSciClaims + SciClaimEval + S1-MMAlign 子集
- [ ] 部署 InternVL2.5-8B (frozen inference)，对训练集做特征提取并缓存
- [ ] 构造 FEH 预训练数据 (~150K: 50K ENTAILS + 50K NEUTRAL + 50K CONTRADICTS)
- [ ] 在 InternVL 特征上训练 FEH，在 held-out 上测三分类 accuracy
  - **Go/No-Go**: accuracy > 75% → 继续；< 65% → 检查数据质量或改架构
  - **同时训练 FEH-Same (在 Qwen 特征上)**，作为 A8 消融的 baseline
- [ ] Pilot: 对 50 篇论文跑 FEH scoring
  - 对比 FEH-Cross vs FEH-Same vs surface sim() → 三者分歧有多大？
  - 对比 FEH 判定 vs 人类判断 → Cohen's κ > 0.5?
  - 统计 NEUTRAL 类的占比 → 如果 < 10% 说明数据有偏
  - 检查 Many-to-One 案例：生成正确但换了说法的 claim，FEH 判定是否为 ENTAILS/NEUTRAL（而非 CONTRADICTS）
- [ ] 下载 SciMDR 300K

### Week 2: Stage 1 SFT + GRPO Warm-up

- [ ] Stage 1: 标准 SFT on SciMDR 300K (LoRA)
- [ ] 大规模 InternVL 特征预缓存 (GRPO 训练集 ~50K samples 的 figure 特征)
- [ ] 实现 FER-GRPO training loop (含 InternVL reward encoder 调用)
- [ ] 实现 Conflict Detection Bonus + Informativeness Gate 逻辑
- [ ] 小规模 GRPO 试跑 (~1K prompts) 检查：训练稳定性 / InternVL inference 瓶颈 / reward 分布

### Week 3: Stage 2 FER-GRPO Full Training

- [ ] Full FER-GRPO training (~10K prompts × 8 samples)
- [ ] 监控训练指标：reward 分布、KL divergence、claim diversity (distinct-n)
- [ ] 如果 diversity 下降 → 检查 NEUTRAL reward 是否生效
- [ ] 中间 checkpoint 评估 (PRISMM-Bench + SciMDR-Eval)
- [ ] （可选）跑 DPO 消融版本

### Week 4: 全面评估 + 消融

- [ ] 跑所有 baselines (zero-shot, SFT-only, EMPO, PaLMR, VPPO, surface-sim)
- [ ] 6 数据集 × 3 seeds 全面评估
- [ ] 11 个消融配置 (A1-A11) 全部跑完（可并行：2 个消融同时跑，每个用 2 卡）
- [ ] A8 reward hacking 检测：用 FEH-Cross 对 A8 (FEH-Same) 模型重评
- [ ] A9 GPT-4o judge：调用 API 做 reward 评估 + 记录耗时
- [ ] 构造 Conflict Detection Test Set (400 samples) + 5 个 sensitivity 测试集
- [ ] 跑 Conflict Detection F1 + Sensitivity Analysis

### Week 5: 分析 + 可视化

- [ ] FEH vs Human Agreement (Cohen's κ)
- [ ] Diversity + Informativeness Analysis (distinct-n, self-BLEU, avg_claim_count, entails_ratio)
- [ ] Reward Hacking Detection (A8 FEH-Cross 重评分析)
- [ ] Conflict Sensitivity Analysis (5 个冲突率 × P/R/F1 + FPR 曲线)
- [ ] FEH vs GPT-4o Judge (accuracy + latency 对比表)
- [ ] Error Taxonomy (PRISMM-Bench)
- [ ] Distance-Accuracy Curve (长文档)
- [ ] Cross-Domain Zero-Shot (Medicine, Biology)
- [ ] **FEH Rationale Visualization**: 选 3-5 个 CONTRADICTS 案例，展示 attention heatmap
- [ ] Case Study: 6 个典型案例
  - 2 个 "surface sim 会退化但 FEH 正确" (Many-to-One 解决)
  - 2 个 "成功检测冲突 + heatmap 标记冲突位置"
  - 1 个 "NEUTRAL 正确分类：事实正确但原文未提及"
  - 1 个 "失败案例" (诚实展示局限)

### Week 6: 写作

- [ ] Paper draft (推荐 Title 3: "Teach Models to Audit")
- [ ] 重点写好 Section 3.1 (FEH 设计) 和 Section 3.2 (Tri-State Reward)
- [ ] 消融表 + case study 图 + pipeline 总图
- [ ] Related Work: 明确区分 CycleReward family, GVF family, VISA/PRe family

---

## 十一、风险评估（v3.1 修正版）

| 风险 | 等级 | 应对 | 来源 |
|------|------|------|------|
| FEH 三分类 accuracy < 65% | 🔴 高 | Week 1 Go/No-Go；可尝试增加训练数据或换 contrastive 方式 | v3 |
| **Cross-Model FEH 在 GRPO 中信号太弱** | 🟡 中 | InternVL 和 Qwen 的语义空间不完全重叠 → FEH 可能对 Qwen 生成的 claim 判定不准。**备选**：如果 cross-model 效果差，退回 same-model + 增加 KL penalty | v3.1 新增 |
| **InternVL2.5 inference 瓶颈** | 🟡 中 | GRPO 每步需跑 InternVL 8B inference → 训练变慢。**解决**：预缓存 figure 的 InternVL 特征；只对 claim tokens 做增量推理 | v3.1 新增 |
| NEUTRAL 避难所效应 | 🟡 中 | NEUTRAL = -0.1 + informativeness gate (v3.1) | v3 批评 |
| 冲突过度报告 | 🟡 中 | 训练冲突比例 5% + sensitivity analysis (v3.1) | v3 批评 |
| Reward Hacking | 🟡 中 | Cross-Model FEH + A8 消融验证 (v3.1) | v3 批评 |
| GRPO 训练不稳定 | 🟡 中 | 先只做 SFT + DPO (已够发 Findings) | v2 |
| 计算预算不够 | 🟡 中 | v3.1 总 GPU ~90-120h（4×4090 可在 2-3 周内完成），比 v3 增加了消融和 InternVL inference | v3.1 |
| 和 CycleReward 区别度不够 | 🟢 低 | FEH + cross-model + conflict detection 和 CycleReward 差距极大 | v3 降级 |

---

## 十二、EMNLP 接收概率（v3.1 修正后）

| 条件 | 概率 |
|------|------|
| FEH accuracy > 75% + cross-model 有效 + A2 vs A7 > 5pp + FPR@1% < 5% + 6 数据集一致提升 | **75-85% (Main)** |
| 上述满足但 cross-model 效果略差，退回 same-model + 强 KL penalty | **65-75% (Main)** |
| FEH 有效但 conflict detection 一般 | **55-65% (Main/Findings)** |
| FEH 有效但 GRPO 增益有限，退回 DPO | **50-60% (Findings)** |
| FEH accuracy 60-65%，信号偏弱 | **需要调整** |

**v3 → v3.1 的概率提升原因：**
1. **封堵 reward hacking**：cross-model FEH 从根本上杜绝了"刷分不增智"的质疑
2. **封堵 NEUTRAL 避难所**：-0.1 轻微负分 + informativeness gate 双重防御
3. **封堵冲突过度报告**：5% 训练冲突率 + sensitivity analysis 提供完整数据支撑
4. **新增 FEH vs GPT-4o judge 对比**：如果 FEH 快 200× 且效果相当 → 独立卖点
5. **消融从 7 个增至 11 个**：完全透明的实验设计，审稿人找不到"你为什么不做 XX 对比"的攻击点

---

## 十三、全版本对照速查表

| 维度 | v2 | v3 | v3.1 |
|------|----|----|------|
| Reward | surface sim() | FEH 隐空间三分类 (same-model) | **FEH 隐空间三分类 (cross-model)** |
| 退化处理 | 无 | NEUTRAL = 0 | **NEUTRAL = -0.1 + informativeness gate** |
| 冲突处理 | 无 | Conflict Detection Bonus | **+ 冲突率校准 5% + sensitivity analysis** |
| Reward Hacking 防御 | N/A | 无 | **Cross-Model FEH (InternVL → Qwen)** |
| Pipeline | SFT → DPO → GRPO (3 阶段) | SFT → FER-GRPO (2 阶段) | SFT → FER-GRPO (2 阶段) |
| 核心贡献 | 模糊 (3 个都是) | FEH + Tri-State FER | **Cross-Model FEH + Tri-State FER + Conflict Detection** |
| 消融数量 | 8 | 7 | **11 (含 reward hacking / LLM judge / informativeness)** |
| 评估集 | 6 | 6 + 扩展 | **6 + Conflict Sensitivity + Informativeness Analysis** |
| 可解释性 | 无 | 无 | **FEH attention heatmap (CONTRADICTS 时)** |
| GPU time 估算 | ~50h | ~35-50h | **~90-120h (含 InternVL + 扩充消融)** |
| 新增文献 | — | +6 篇 | +6 篇 (同 v3) |
