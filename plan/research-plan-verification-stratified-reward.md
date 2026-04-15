# SciConsist v2: Verification-Stratified Reward for Faithful Scientific Document Reasoning

> Target: EMNLP 2026 Main | Updated: 2026-04-14

---

## 一、核心 Idea

### 1.1 一句话 Pitch

> 科研文档中的事实声明横跨一个 **可验证性谱系**——从符号可精确验证（数值、实体名）到只能语义判定（趋势概括、定性结论）。我们提出 Verification-Stratified Reward (VSR)，将每个 claim 维度路由到最优验证机制，利用科研文档的跨模态结构冗余提供免费、不可 hack 的符号验证信号，同时对不可验证部分使用 learned judge。

### 1.2 Problem Statement

训练 VLM 做科研文档问答时，需要 reward signal 来判定模型输出的事实正确性。现有方法全部使用 **one-size-fits-all reward**：

| 方法 | Reward 类型 | 致命缺陷 |
|------|-----------|---------|
| CycleReward (ICCV 2025) | 表面文本相似度 | Many-to-One 退化：语义等价但表述不同被惩罚 |
| LLM Judge (GPT-4o) | 黑箱概率判断 | 不可控、昂贵、自身会 hallucinate |
| Learned Classifier (FEH-style) | 在 VLM 特征上训的分类器 | 可被 reward hack、泛化受限于训练分布 |
| HERO (ICLR 2026) | Verifier + RM 融合 | 仅用于数学（binary exact-match）、非多模态 |
| RLVRR (ICLR 2026) | Content/Style 分层 | 文本域、content 验证仅是关键词匹配 |

**核心问题**：所有方法对所有类型的 claim 使用相同的验证策略。但事实上：

- "Method A achieves **85.3%** on BLEU" → 数值可以从 table/figure 精确验证
- "Method A **outperforms** Method B" → 可以从排序关系半符号验证
- "Results suggest that **scaling generally helps**" → 只能靠语义判断

用同一个 learned reward 处理这三种 claim，要么在数值精确性上不够严格（模型输出 85.1% 也拿高分），要么在定性概括上过于苛刻（合理的新观察被惩罚）。这就是为什么所有现有 reward 都在 accuracy-diversity tradeoff 上挣扎。

### 1.3 Key Insight

**科研文档提供了其他领域不存在的东西：跨模态结构冗余产生的免费符号验证信号。**

同一个实验数据点同时出现在：
- Table cell: `Method A | BLEU | 85.3`
- Figure bar: bar height ≈ 0.853
- Text: "Method A achieves 85.3% on BLEU"

这意味着对 **可符号化验证的 claim 维度**（数值、实体、单位），你可以构造 deterministic、unhackable 的 reward——不需要学一个 classifier，不需要 LLM judge。规则就是规则。

而对 **不可符号化验证的 claim 维度**（趋势概括、定性判断），你仍然需要 learned judgment。

**VSR 的创新在于：根据 claim 的 verifiability profile，自动路由到不同的验证层。**

### 1.4 与最相关工作的精确区分

| 方法 | 验证策略 | Verifiability 分层 | 多模态结构利用 | Reward Hacking 防护 |
|------|---------|-------------------|---------------|-------------------|
| HERO (ICLR 2026) | Binary verifier + dense RM | 二元（可验/不可验） | 无（数学域） | 部分（verifier 层） |
| RLVRR (ICLR 2026) | Keyword match + LLM style | 二元（content/style） | 无（文本域） | 部分（content 层） |
| DeFacto (ICLR 2026) | GRPO + counterfactual masking | 无 | 区域级（是否看对地方） | 无 |
| PaLMR (CVPR 2026) | Hierarchical reward fusion | 无 | Pseudo-ground-truth | 无 |
| VPPO (ICLR 2026 Spotlight) | Token-level visual dependency | 无 | 视觉依赖度加权 | 无 |
| FormalJudge (2026) | Neuro-symbolic decomposition | 有（atomic constraints） | 无（agent safety） | 完全（形式验证） |
| **VSR (Ours)** | **三层 verifiability hierarchy** | **三级谱系** | **跨模态结构提取** | **符号层完全防护** |

**核心差异**：
1. **vs HERO**: 我们有三级 verifiability hierarchy（不是 binary）；验证来自跨模态结构提取（不是 exact-match）；多模态设定
2. **vs RLVRR**: 我们的 content 验证是跨模态结构化提取+符号比较（不是关键词匹配）；多模态设定
3. **vs DeFacto**: 我们验证 factual CONTENT 是否正确（不只是模型是否看了对的 region）
4. **vs FormalJudge**: 我们将 neuro-symbolic 思路应用于 multimodal factual claims（不是 agent safety constraints）

---

## 二、方法设计

### 2.1 Overall Pipeline

```
Stage 1: Standard SFT（非贡献）
  Qwen2.5-VL-7B + LoRA on SciMDR 300K
  → 建立科研文档基础理解能力

Stage 2: VSR-GRPO（核心贡献）
  用 Verification-Stratified Reward 做 GRPO
  → 产出 SciConsist-7B
```

### 2.2 Verification-Stratified Reward (VSR) Architecture

```
Model Response τ
       │
       ▼
┌─────────────────┐
│  Claim Extractor │  (轻量规则 + LLM fallback)
│  提取 atomic     │
│  factual claims  │
└────────┬────────┘
         │  {c₁, c₂, ..., cₖ}
         ▼
┌─────────────────────────────────┐
│  Verifiability Router           │
│  对每个 claim 判定各维度的       │
│  verifiability level            │
│                                 │
│  维度:                          │
│  - numeric_value (有/无具体数值) │
│  - entity_identity (有/无实体名) │
│  - relation (有/无比较/排序)     │
│  - trend (有/无趋势描述)        │
│  - qualitative (纯定性描述)     │
└────────┬────────────────────────┘
         │
    ┌────┴────┬──────────┐
    ▼         ▼          ▼
┌────────┐ ┌────────┐ ┌────────┐
│ Layer 0│ │ Layer 1│ │ Layer 2│
│Symbolic│ │Semi-   │ │Learned │
│Verifier│ │Symbolic│ │Verifier│
└────┬───┘ └───┬────┘ └───┬────┘
     │         │          │
     ▼         ▼          ▼
  r_sym      r_semi    r_learned
     │         │          │
     └────┬────┴──────────┘
          ▼
   ┌──────────────┐
   │  Aggregator   │
   │  R(τ) = Σ wᵢrᵢ│
   └──────────────┘
```

### 2.3 三层验证机制

#### Layer 0: Symbolic Verification（精确事实）

**触发条件**：claim 包含具体数值、百分比、实体名+指标名

**验证流程**：
1. 从 response 中解析 claim 的结构化表示：`(entity, metric, value, unit)`
2. 从 table（OCR/HTML parse）中提取对应 cell
3. 从 figure（OCR + 坐标映射）中估算对应值
4. 符号比较：
   - Exact match (table cell): value == extracted_value → r = +1.0
   - Approximate match (figure): |value - estimated| < tolerance → r = +0.8
   - Mismatch → r = -1.0

**关键性质**：
- **Deterministic**: 不依赖任何 learned model
- **Unhackable**: policy model 无法通过参数更新影响符号比较规则
- **Cross-modal triangulation**: 当 table 和 figure 都有数据时，取更严格的验证结果

**Confidence-gated Fallback**: Layer 0 仅在高置信度时生效（entity match score > 0.9 且 value extraction confidence > 0.95），否则回退 Layer 2。这保证 pipeline 失败模式是 "退化为 learned reward"，而非 "给出错误 reward"。

**实现工具**：
- Table parsing: Qwen2.5-VL-72B（offline inference）提取 table → structured JSON；对有 LaTeX 源码的子集可直接从 `.tex` 提取（bypass OCR）
- Figure OCR: 提取 axis labels + data labels → value estimation
- 实体/指标匹配: 字符串匹配 + fuzzy matching (Levenshtein)

#### Layer 1: Semi-Symbolic Verification（关系事实）

**触发条件**：claim 包含比较关系（A > B）、排序（A ranks first）、趋势方向（increases/decreases）

**验证流程**：
1. 解析 claim 中的关系断言：`(entity_A, relation, entity_B, [metric])`
2. 从 table 中提取双方数值，验证关系是否成立
3. 从 figure 中通过视觉推理验证趋势方向
4. 逻辑判定：
   - 关系与提取数据一致 → r = +0.8
   - 关系正确但缺少一方数据（只能单源验证）→ r = +0.5
   - 关系与提取数据矛盾 → r = -0.8

**关键性质**：
- 基于符号逻辑，但允许 tolerance（bar chart 的视觉估算有误差）
- 比 Layer 0 宽容（不要求精确数值，只要求方向/排序正确）

#### Layer 2: Learned Verification（语义事实）

**触发条件**：claim 不包含可符号化验证的维度（纯定性描述、概括性结论）

**验证流程**：
1. 将 (figure/table evidence, claim) 送入 learned verifier
2. Verifier 输出 confidence score

**Verifier 选择**（通过消融确定最优）：
- Option A: FEH（在冻结 VLM hidden states 上的 MLP classifier）→ 快但可能弱
- Option B: LLM-as-Judge（Qwen2.5-72B / InternVL2.5-78B）→ 强但慢
- Option C: Self-consistency（policy model 对同一 evidence 生成多次，取共识）→ 无额外模型

**关键性质**：
- 这一层承认 learned reward 的局限性，但限制了其作用范围
- 只有约 30-50% 的 claim 落到这一层（其余被 Layer 0/1 处理）
- 即使 Layer 2 被 hack，Layer 0/1 的符号验证仍然锚定整体 reward

### 2.4 Claim Extractor & Verifiability Router

**Claim 提取**（轻量级，不是贡献重点）：
```python
# Rule-based extraction for structured responses
# LLM fallback for free-form responses
claims = extract_atomic_claims(response)
# Each claim: {"text": str, "entities": list, "values": list,
#              "relations": list, "trend": bool, "qualitative": bool}
```

**Verifiability Router**（规则系统，无需训练）：
```python
def route_claim(claim) -> list[VerificationTask]:
    tasks = []
    if claim.values:       # 有具体数值
        tasks.append(SymbolicVerification(claim))
    if claim.relations:    # 有比较/排序
        tasks.append(SemiSymbolicVerification(claim))
    if claim.qualitative or not tasks:  # 纯定性或无可验维度
        tasks.append(LearnedVerification(claim))
    return tasks
```

### 2.5 Reward Aggregation

对每个 response τ 中的每个 claim cᵢ：

```
R(cᵢ) = Σⱼ wⱼ · rⱼ(cᵢ)    # j ∈ {sym, semi, learned}
```

整体 reward：
```
R_VSR(τ) = (1/K) Σᵢ R(cᵢ)                    # 基础分
         + λ_coverage · coverage_bonus(τ)       # 鼓励覆盖更多数据点
         + λ_specific · specificity_bonus(τ)    # 鼓励输出可验证的具体 claim
```

其中：
- **Coverage bonus**: response 覆盖了 table/figure 中多少比例的数据点
- **Specificity bonus**: 可符号验证 claim 的占比越高，奖励越高（鼓励模型输出具体、可核查的声明而非模糊废话）

---

## 三、论文贡献

### 3.1 贡献列表

| # | 贡献 | 类型 | 级别 |
|---|------|------|------|
| 1 | **Verifiability Hierarchy for Factual Claims**: 形式化了 multimodal factual claim 的可验证性谱系，提出根据 claim 维度的 verifiability level 路由到不同验证机制 | Conceptual Framework | 核心 |
| 2 | **Verification-Stratified Reward (VSR)**: 三层验证架构——符号层（unhackable）+ 半符号层 + learned 层——利用科研文档跨模态结构冗余提供免费验证信号 | 方法 | 核心 |
| 3 | **Empirical evidence that stratified > holistic reward**: 证明 accuracy-diversity tradeoff 来自 one-size-fits-all reward 设计，分层验证可以打破这一 tradeoff | 实验发现 | 核心 |
| 4 | 在 7 个 benchmark 上的全面评估 + 详细消融 | 实验 | 支撑 |

### 3.2 推荐标题

**Verification-Stratified Reward: Exploiting Cross-Modal Structural Redundancy for Faithful Scientific Document Reasoning**

### 3.3 Paper Narrative

**Section 1 (Intro)**: 科研文档的事实声明横跨 verifiability spectrum。现有 RL reward 是 one-size-fits-all，导致 accuracy-diversity tradeoff。我们提出 VSR，根据 claim 的 verifiability profile 路由到最优验证机制。

**Section 2 (Related Work)**: 三条线索交汇——(1) VLM factual consistency training, (2) Hybrid/stratified reward design, (3) Scientific document understanding benchmarks。指出 gap：没有人在多模态 factual claim 上做 verifiability-aware reward design。

**Section 3 (Method)**: VSR 三层架构 + claim extraction + verifiability routing + reward aggregation。

**Section 4 (Experiments)**: Main results → Per-dimension analysis → Ablations → Reward hacking resistance → Verifiability coverage analysis。

**Section 5 (Analysis)**: 为什么 stratified 能打破 tradeoff？符号层锚定数值精确性，learned 层保护表达多样性，两者互不干扰。Visualize output distribution under different rewards。

---

## 四、Related Work Map

### 4.1 VLM Factual Consistency Training

| Paper | Venue | 方法 | 与本文关系 |
|-------|-------|------|-----------|
| CycleReward | ICCV 2025 | Cycle consistency surface sim 作为 reward | 直接 baseline；我们 argue surface sim 是 suboptimal |
| DeFacto | ICLR 2026 | Counterfactual masking + GRPO | 验证 visual grounding region，不验证 factual content |
| VPPO | ICLR 2026 Spotlight | Token-level visual dependency reward | 正交贡献：关注哪些 token 视觉相关，不关注内容正确性 |
| PaLMR | CVPR 2026 | Process-level multimodal alignment | 用 pseudo-GT 做 hierarchical reward；我们用真实 symbolic verification |
| Vision-SR1 | ICLR 2026 | Self-rewarding via reasoning decomposition | 自洽性作为 reward；不保证 factual correctness |
| GVF | 2025 | Factual anchor + counterfactual prompt | SFT 级方法，非 RL reward |
| VISA | ICLR 2026 sub | VFM 表示对齐抗幻觉 | 训练时对齐，非 reward 设计 |
| EMPO | EMNLP 2025 | Entity-centric multimodal DPO | 偏好对齐方向，非 reward 设计 |

### 4.2 Hybrid / Stratified Reward Design

| Paper | Venue | 方法 | 与本文关系 |
|-------|-------|------|-----------|
| HERO | ICLR 2026 | Sparse verifier + dense RM, stratified normalization | 最近亲：但仅 binary verifiability (math)，非多模态 |
| RLVRR | ICLR 2026 | Content/Style 分层 reward | 文本域，content 验证仅 keyword match |
| FormalJudge | arXiv 2026 | Neuro-symbolic decomposition + formal verification | Agent safety，非 factual claims |
| K2V | ICLR 2026 sub | 知识密集型推理的 verifiable sub-task decomposition | 文本域，非多模态 |
| TruthRL | arXiv 2025 | Ternary reward (correct/hallucination/abstention) | 三分类思路类似但无 verifiability stratification |

### 4.3 Claim Decomposition & Verification

| Paper | Venue | 方法 | 与本文关系 |
|-------|-------|------|-----------|
| Dynamic Decomposition | ACL 2025 | RL 优化 claim 分解粒度 | 文本 claim decomposition，可借鉴 atomicity 概念 |
| GRPO Claim Decomposition | EACL 2026 Findings | GRPO 联合优化分解+验证 | 文本域，非多模态 |
| Veri-R1 | arXiv 2025 | Online RL for claim verification | 搜索引擎检索验证，非结构化文档验证 |

### 4.4 Scientific Document Understanding

| Paper | Venue | 方法 | 与本文关系 |
|-------|-------|------|-----------|
| SciClaimEval | LREC 2026 | 1,664 标注 claim-evidence pairs | 评估 benchmark |
| PRISMM-Bench | ICLR 2026 | 384 reviewer-flagged inconsistencies | 评估 benchmark |
| MuSciClaims | IJCNLP 2025 | 图表 claim 验证数据集 | 评估 benchmark |
| SciMDR | arXiv 2026 | 300K QA + expert-annotated eval | 训练数据 + 评估 benchmark |
| BRIDGE | arXiv 2026 | 多跳推理 long document benchmark | 评估 benchmark；其 motivation 直接支持我们的 structural redundancy observation |
| ChartQAPro | ACL 2025 Findings | 图表 QA benchmark | 评估 benchmark |
| SciMMIR | ACL 2024 Findings | 530K 科研图文对 | 潜在训练数据来源 |

### 4.5 OCR & Structured Extraction

| Paper | Venue | 方法 | 与本文关系 |
|-------|-------|------|-----------|
| MOCR | arXiv 2026 | Multimodal OCR, figure→SVG | 可用于 Layer 0 table/figure parsing |
| Uni-Parser | arXiv 2025 | 工业级科研文档解析 | 可用于 structured extraction |
| CycleChart | arXiv 2025 | Chart parsing + generation | Chart 结构化提取 |

---

## 五、实验设计

### 5.1 评估集（7 个）

| 数据集 | 规模 | 评估什么 | 来源 |
|--------|------|---------|------|
| SciClaimEval | 1,664 | 跨模态 claim 验证 | LREC 2026 |
| PRISMM-Bench | 384 | 不一致检测/修复 | ICLR 2026 |
| MuSciClaims | ~2K | 图表 claim 验证 | IJCNLP 2025 |
| SciMDR-Eval | 专家标注 | 科研文档 QA | arXiv 2026 |
| BRIDGE | 长文档 | 多跳推理 | arXiv 2026 |
| ChartQAPro | 1,948 | 图表理解 | ACL 2025 |
| InfoGraphicsVQA | 通用 | 非科研泛化 | 已有 |

所有结果报告 mean +/- std (3 seeds)。

### 5.2 Baselines

| Baseline | 类型 | 来源 |
|----------|------|------|
| Qwen2.5-VL-7B (zero-shot) | 无训练参考 | - |
| Qwen2.5-VL-7B + SFT only | SFT 上界 | Stage 1 |
| + Surface Sim GRPO | CycleReward-style | ICCV 2025 |
| + LLM Judge GRPO | GPT-4o / Qwen2.5-72B 打分 | 常见 baseline |
| + **GPT-4o Multi-Aspect Judge GRPO** | 精细多维度 prompt（数值/趋势/实体分别打分） | Strong baseline |
| + FEH GRPO (holistic learned) | 原 SciConsist v1 设计 | 本项目已有 |
| + HERO-style hybrid | Binary verifier + learned RM | ICLR 2026 |
| + **VSR-GRPO (Ours)** | 三层分层验证 | 本文 |
| InternVL2.5-8B / GPT-4o / Qwen2.5-VL-72B | 上界参考 | - |

### 5.3 核心消融

| ID | 对比 | 验证什么 |
|----|------|---------|
| A1 | All-Symbolic (only Layer 0+1) vs VSR | 纯符号 reward 的覆盖率瓶颈 |
| A2 | All-Learned (only Layer 2) vs VSR | 纯 learned reward 的精确性缺陷 |
| A3 | 2-Layer (symbolic + learned) vs 3-Layer | 半符号层（Layer 1）的价值 |
| A4 | 无 cross-modal triangulation vs 有 | 三角验证是否比单源更稳健 |
| A5 | 无 specificity bonus vs 有 | 鼓励可验证 claim 是否改善输出质量 |
| A6 | 无 coverage bonus vs 有 | 覆盖度激励是否防止 cherry-picking |
| A7 | Random routing vs Verifiability routing | 路由策略是否 matters |
| A8 | Layer 2: FEH vs LLM Judge vs Self-consistency | Learned layer 的最优实现 |
| A9 | Layer 0 注入 5%/10%/20% OCR 噪声 | 符号层对 extraction 噪声的容忍度 |
| A10 | LaTeX 源码提取 vs OCR 提取（有源码的子集） | 验证 OCR 提取的可靠性上界 |

### 5.4 专项分析

1. **Verifiability Coverage Analysis**
   - 在 SciMDR 训练集上统计：多少比例的 claim 可被 Layer 0/1/2 分别覆盖
   - **按 task type 分层**：Table-grounded QA（预期 L0+1 60-80%）、Figure-grounded QA（预期 40-60%）、Free-form summarization（预期 15-30%）
   - **这是 Go/No-Go 决策点**：如果 Figure/Table QA 中 Layer 0+1 覆盖 < 30%，stratification 没有实际意义

2. **Reward Meta-Evaluation（主动防御 pipeline fragility）**
   - 人工标注 500 claims 的 ground-truth judgment（CORRECT / PARTIALLY_CORRECT / WRONG）
   - 计算 VSR 每一层 vs human 的 precision/recall/Cohen's kappa
   - 同时计算 GPT-4o judge / FEH 的相同指标，做 reward quality 的横向对比
   - **论文正文必须包含此分析**（Table: Reward Function Meta-Evaluation）

3. **Per-Dimension Error Analysis**
   - 对每个 baseline 分析：numeric precision, entity accuracy, trend correctness, diversity
   - 预期发现：surface sim GRPO 在 numeric precision 上差但 trend OK；VSR 两者都好
   - 同时报告 **Cost per reward evaluation**：VSR (Layer 0/1 ≈ 0, Layer 2 = model inference) vs GPT-4o judge ($0.01-0.02/claim)

4. **Reward Hacking Resistance**
   - 跑 long training（2x 正常 steps），监控各 baseline 的 reward vs actual accuracy divergence
   - 预期：holistic learned reward 出现 hacking（reward 涨但 accuracy 平/降），VSR 的符号层锚定不 hack

5. **Symbolic Anchor Effect Visualization**
   - 绘制 RL 训练过程中 Layer 0 reward 与 Layer 2 reward 的 trajectory：当 Layer 2 (learned) 出现 reward inflation 时，Layer 0 (symbolic) 是否将整体 reward 拽回真实水平
   - 这张图是论文的核心 figure 之一，直观展示分层锚定机制

6. **OCR Noise Robustness（A9 消融的可视化）**
   - 在 0%/5%/10%/20% OCR 噪声下报告 VSR-GRPO 的 downstream performance 曲线
   - 预期：5% 噪声下几乎无影响，20% 噪声下性能下降但仍优于 all-learned baseline

7. **Output Distribution Visualization**
   - t-SNE/UMAP on model outputs under different rewards
   - 预期：surface sim → 一个 cluster；binary → 两个 cluster；VSR → 分布均匀

---

## 六、与已有工作的兼容性

| 已有资产 | 在新方案中的角色 |
|---------|----------------|
| FEH (Week1 checkpoint) | Layer 2 的一个候选 learned verifier |
| GRPO pipeline | 不变，只换 reward function |
| SciMDR SFT | Stage 1 不变 |
| P1/P3 evaluation framework | 作为 motivation evidence：holistic reward 导致 tradeoff |
| soft_b GRPO exploration | 用作 "FEH GRPO (holistic)" baseline |

---

## 七、执行计划

### 资源：4x RTX 4090，预计总 GPU 时间 ~120h

| 周次 | 任务 | 关键节点 |
|------|------|---------|
| **Week 1** | Verifiability 分析 + Layer 0/1 实现 | |
| | 1. 在 SciMDR 上统计 claim verifiability 分布（按 task type 分层） | **Go/No-Go: Figure/Table QA 中 L0+1 覆盖率 > 30%** |
| | 2. 实现 claim extractor + verifiability router | |
| | 3. 实现 Layer 0 symbolic verifier (table parse + value compare + confidence gate) | |
| | 4. 实现 Layer 1 semi-symbolic verifier (relation extraction + logic check) | |
| | 5. 对有 LaTeX 源码的子集，实现 .tex 直接提取路径（作为 OCR 上界参照） | |
| **Week 2** | VSR 集成 + Meta-Evaluation + 小规模训练 | |
| | 1. 将 Layer 0/1/2 集成为统一 VSR reward function | |
| | 2. **标注 500 claims 做 Reward Meta-Evaluation**（VSR vs GPT-4o judge vs FEH） | |
| | 3. VSR-GRPO 小规模试跑（1K steps），验证 reward 信号质量 | |
| | 4. 校准 layer weights、confidence thresholds、aggregation 策略 | |
| **Week 3** | Full Training + Baselines | |
| | 1. VSR-GRPO 全量训练 (3 seeds) | |
| | 2. 跑所有 baselines：surface sim, LLM judge, **GPT-4o multi-aspect judge**, FEH holistic, HERO-style | |
| | 3. 监控 reward/diversity/KL curves + Layer 0 vs Layer 2 anchor trajectory | |
| **Week 4** | 评估 + 消融 | |
| | 1. 7 benchmarks x 3 seeds 全面评估 | |
| | 2. 10 组消融实验（含 OCR 噪声鲁棒性 A9 + LaTeX 对照 A10） | |
| | 3. Per-dimension error analysis + cost comparison | |
| **Week 5** | 分析 + 可视化 | |
| | 1. Verifiability coverage analysis（按 task type 分层图表） | |
| | 2. Symbolic anchor effect visualization（训练过程 L0 vs L2 trajectory） | |
| | 3. OCR noise robustness curve | |
| | 4. Reward hacking resistance + output distribution visualization | |
| | 5. Case studies | |
| **Week 6** | 写作 | |
| | 1. Paper draft，重点 Section 3 (VSR) + Section 5 (Analysis) | |
| | 2. 确保正文包含：meta-evaluation table、anchor effect figure、noise robustness curve | |

### Go/No-Go 决策点

| 时间 | 条件 | 如果不通过 |
|------|------|-----------|
| Week 1 Day 3 | Figure/Table QA 中 Layer 0+1 覆盖率 > 30% | 降级为 Route A (systematic study)，VSR 成为 study 中的一个 condition |
| Week 2 Day 3 | Meta-Evaluation: VSR Layer 0 precision > 90%, 整体 reward accuracy > GPT-4o judge | 检查 parsing 质量 / confidence threshold / 调整 tolerance |
| Week 3 Day 7 | VSR-GRPO 在至少 3 个 benchmark 上超过 best holistic baseline（含 GPT-4o multi-aspect） | 检查 aggregation 权重；如果 GPT-4o judge 全面碾压，pivot 叙事为 "cost-equivalent analysis" |

---

## 八、关键风险与预设防御

| 风险 | 等级 | 应对 | 论文中的防御 |
|------|------|------|-------------|
| Layer 0+1 覆盖率太低 | 中高 | Week 1 Go/No-Go；退回 Route A | 按 task type 分层报告覆盖率，明确 scope |
| Table/Figure parsing 不准确 | 中 | Qwen2.5-VL-72B offline parsing + confidence gate | A9 OCR 噪声鲁棒性实验 + A10 LaTeX 对照 |
| Claim extraction 不可靠 | 中 | LLM-based + regex fallback | Meta-evaluation 中报告 extractor accuracy |
| GPT-4o multi-aspect judge 全面碾压 VSR | 中 | 预期 VSR 在 numeric precision 上赢 + cost 低 100x | Per-dimension 对比 + cost table |
| "循环论证"质疑（OCR 与 policy 同源偏见） | 低 | OCR 读 cell 是近确定性任务 vs policy 生成 claim 是复杂推理 | A10 LaTeX 对照证明 OCR 可靠性 + A9 噪声鲁棒性 |
| VSR 在非科研 benchmark 上无优势 | 中 | 预期如此 | 论文诚实讨论 scope，InfoGraphicsVQA 作为边界测试 |

---

## 九、为什么这个 Idea 值得做

1. **Problem 是真的**：PRISMM-Bench 显示 SOTA 模型在科研文档一致性检测上只有 27-54%；BRIDGE 证明科研文档的跨模态依赖链是天然存在的。

2. **Insight 是新的**：没有人在多模态 factual claim 上做 verifiability-aware reward stratification。HERO 和 RLVRR 是同一精神的 text/math 版本，但多模态+结构冗余利用+三级 hierarchy 是新的组合。

3. **方法是合理的**：不是黑箱 trick，而是基于 "不同类型的事实需要不同的验证策略" 这一直觉明确的原则。Reviewer 读完会觉得 "this makes sense"。

4. **实验可以讲 clean story**：如果 VSR 在 numeric precision 上赢（靠 Layer 0）且在 diversity 上不输（靠 Layer 2），就可以说 "stratification 打破了 accuracy-diversity tradeoff"。这是一个 testable、falsifiable 的 claim。

5. **执行可行**：用已有的 FEH + GRPO 基础设施，只需要加 Layer 0/1 的 symbolic verification + routing logic。
