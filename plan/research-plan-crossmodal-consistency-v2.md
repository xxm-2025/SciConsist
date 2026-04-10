# SciConsist: Cross-Modal Cycle Consistency for Faithful Scientific Document Understanding

> 升级版研究计划 | 生成日期: 2026-04-09

---

## 一、核心 Insight（一句话）

> 科研文档中 text / table / figure 对同一事实的天然冗余表达 = 免费的 faithfulness 自监督信号。
> 我们把这个信号形式化为 **跨模态 cycle consistency**，用它驱动偏好数据构造 + RL 训练，
> 不需要人工标注就能显著提升模型在科研文档多模态推理中的 faithfulness。

---

## 二、为什么这个 insight 是新的——逐篇对比

| 已有工作 | 它做了什么 | 它没做什么（= 你的空间） |
|----------|-----------|------------------------|
| **CycleReward** (ICCV 2025) | image→caption→image 的 cycle consistency 作为 reward | 仅限通用 image captioning；不利用文档内部 text↔table↔figure 的结构化冗余 |
| **CycleCap** (2026.03) | VLM caption → T2I 重建 → GRPO reward | 同上：general captioning，不涉及文档内多模态 claim 验证 |
| **PaLMR** (2026.03) | V-GRPO：visual fidelity reward（用 LLM judge 判 faithfulness） | 需要外部 LLM judge；不是 self-supervised cycle；不利用文档冗余 |
| **VPPO** (ICLR 2026) | Token-level visual dependency 重加权 | 关注"哪些 token 依赖视觉"，不关注"跨模态事实是否一致" |
| **PGPO** (2026.04) | KL divergence 量化 token visual dependency | 同上：token 粒度的感知信号，不是 claim 粒度的一致性信号 |
| **EMPO** (EMNLP 2025 Main) | 自动构造多方面偏好数据做 DPO | 偏好数据基于 hallucination detection，不基于跨模态 consistency |
| **ReLoop** (EMNLP 2025 Findings) | 闭环训练：semantic reconstruction + visual consistency + attention supervision | "Visual consistency" 是 reconstruction-based，不是 cross-modal fact consistency；不针对文档 |
| **MoD-DPO** (2026.03) | 模态解耦偏好优化 + language-prior debiasing | 关注跨模态幻觉抑制，不利用文档内部冗余作为训练信号 |
| **SciMDR** (2026.03) | 300K QA 的 SFT 训练 | 纯 SFT，无 consistency objective，无 RL |
| **PRe** (2026.03) | 预测正则化防止视觉特征退化 | 关注 representation 层面，不关注 claim 层面的跨模态一致性 |

**结论：CycleReward/CycleCap 证明了 cycle consistency 作为 self-supervised reward 的有效性，但只用于通用 image captioning。PaLMR/VPPO/PGPO 证明了 faithfulness-aware RL 的有效性，但不利用文档冗余。EMPO/ReLoop 证明了自动偏好数据构造在 EMNLP 可发。没有人把这三条线交叉到"科研文档跨模态 cycle consistency"上。**

---

## 三、方法设计：三个深层技术贡献

### C1: Cross-Modal Cycle Consistency Reward（核心创新）

**灵感来源：** CycleReward (ICCV 2025) + CycleCap (2026)

CycleReward 的 cycle 是：image → caption → regenerated_image → 比较相似度。
我们的 cycle 是在科研文档内部：

```
Forward cycle (Figure → Claim → Text Verification):
    figure_image → MLLM 生成 factual claim → 和原文 text 对比 → consistency score

Backward cycle (Text → Figure Verification):
    text_claim (从论文文本中提取) → MLLM 定位 figure 中的证据 → 描述证据 → 和原始 claim 对比 → consistency score

Cross-modal cycle (Table ↔ Figure):
    table_cell → MLLM 在 figure 中找对应的数据点 → 抽取值 → 和 table 值对比 → consistency score
```

**Reward 形式化：**

```
R_cycle(τ) = λ_fwd · sim(claim_from_figure, text_ground_truth)
           + λ_bwd · sim(evidence_from_figure, original_claim)  
           + λ_cross · exact_match(value_from_figure, value_from_table)
```

- `sim()` = sentence embedding cosine similarity（用 SciBERT 或 BGE-M3）
- `exact_match()` = 数值匹配（允许小 tolerance）
- λ 权重通过消融确定

**为什么比 PaLMR 的 LLM judge 好：**
1. 不需要外部 judge model → 更高效
2. Signal 来自文档本身 → 更 domain-specific
3. 数值部分可做 exact match → 比 LLM judge 更精确
4. 完全 self-supervised → 可 scale 到 S1-MMAlign 的 15.5M pairs

### C2: Consistency-Aware Preference Data Construction（自动化偏好对）

**灵感来源：** EMPO (EMNLP 2025) + MoD-DPO (2026) + CHiP (2025)

不做人工标注，从 cycle consistency 分数自动构造 DPO 偏好对：

**Step 1: Multi-View Sampling**
- 对同一个 factual question，让 MLLM 生成 N 个 responses（N=8, standard BoN）
- 每个 response 包含若干 factual claims

**Step 2: Cross-Modal Consistency Scoring**
- 对每个 response 的每个 claim，计算 R_cycle
- Response-level score = mean(claim-level scores)
- 按分数排序

**Step 3: Preference Pair Construction**
- `chosen` = 最高 consistency score 的 response
- `rejected` = 最低 consistency score 的 response
- 额外的 hard negative（借鉴 MoD-DPO）：
  - **Modality-swap negative**: 把 text 中的数值替换成另一个 figure 的数值
  - **Trend-flip negative**: 把 "increases" 换成 "decreases"
  - **Entity-swap negative**: 把 "Method A" 换成 "Method B"
  - 这些 hard negatives 的 consistency score 自然很低

**与 EMPO 的区别：**
- EMPO 基于 hallucination detection（需要一个 detector）
- 我们基于 cross-modal consistency（不需要 detector，consistency score 本身就是信号）
- EMPO 是通用多模态；我们是文档特定的，可以利用结构化冗余

### C3: Progressive Training Pipeline（渐进式训练）

**灵感来源：** PaLMR (V-GRPO) + VPPO (token-level reweighting) + PRe (visual regularization)

三阶段训练，逐步加强 consistency 约束：

**Stage 1: Consistency-Aware SFT（1 epoch）**
- 在 SciMDR 300K QA 上做 SFT
- 额外加一个 **cross-modal consistency regularization loss**（借鉴 PRe）：
  - 对同一事实，model 从 text 输入和 figure 输入得到的 hidden representation 应该接近
  - L_consistency = KL(p(answer | text_evidence) || p(answer | figure_evidence))
  - 这迫使 model 对同一事实建立统一的内部表示，不管输入来自哪个模态
- 训练数据：SciMDR + S1-MMAlign 中可以配对的 text-figure 对

**Stage 2: Consistency DPO（1-2 epochs）**
- 用 C2 构造的偏好对做 DPO
- 加入 MoD-DPO 的 modality-aware regularization
- 确保模型对 cross-modal consistent 的 response 有更高偏好

**Stage 3: Cycle Consistency GRPO（1-2 epochs）**
- 用 C1 的 R_cycle 作为 reward 做 GRPO
- 借鉴 VPPO：在 token-level 加权，只对 factual claim tokens 加大 gradient（而非全部 tokens）
- 借鉴 PaLMR：任何 consistency score < threshold 的 trajectory 直接 reward = 0

**为什么要三阶段：**
- Stage 1 建立基础能力（从 SciMDR 学会科研文档推理）
- Stage 2 学会偏好（哪些 response 是 cross-modally consistent 的）
- Stage 3 自我强化（用 cycle consistency reward 做在线 RL）
- 逐步加强约束比一步到位更稳定（PaLMR 和 Chart-RL 都采用类似渐进策略）

---

## 四、完整 Pipeline 图

```
┌─────────────────────────────────────────────────────────────┐
│                    Training Data Sources                      │
│  ┌──────────┐  ┌──────────┐  ┌─────────────────────────┐    │
│  │ S1-MMAlign│  │ SciMDR   │  │ Self-Generated          │    │
│  │ 15.5M     │  │ 300K QA  │  │ Consistency Preference  │    │
│  │ fig-text  │  │ 20K papers│  │ Pairs (C2)             │    │
│  └─────┬─────┘  └─────┬────┘  └────────────┬────────────┘    │
│        │              │                     │                 │
│        ▼              ▼                     ▼                 │
│  ┌──────────────────────────────────────────────────────┐    │
│  │         Stage 1: Consistency-Aware SFT               │    │
│  │  L = L_SFT + α · L_cross_modal_KL                   │    │
│  └────────────────────────┬─────────────────────────────┘    │
│                           ▼                                   │
│  ┌──────────────────────────────────────────────────────┐    │
│  │         Stage 2: Consistency DPO                      │    │
│  │  chosen = high R_cycle response                       │    │
│  │  rejected = low R_cycle response + hard negatives     │    │
│  │  + MoD-DPO modality-aware regularization              │    │
│  └────────────────────────┬─────────────────────────────┘    │
│                           ▼                                   │
│  ┌──────────────────────────────────────────────────────┐    │
│  │         Stage 3: Cycle Consistency GRPO               │    │
│  │  R = R_cycle (self-supervised, from document)         │    │
│  │  + VPPO-style token-level claim weighting             │    │
│  │  + PaLMR-style zero-reward for low consistency        │    │
│  └────────────────────────┬─────────────────────────────┘    │
│                           ▼                                   │
│  ┌──────────────────────────────────────────────────────┐    │
│  │           SciConsist-7B (Final Model)                 │    │
│  └──────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘

Inference:
  User question about scientific doc
       → SciConsist-7B generates grounded answer
       → (Optional) Cycle verification: check each claim against figure/table
       → Final answer with evidence citations
```

---

## 五、实验设计

### 5.1 评估数据集

| 数据集 | 评估维度 | 为什么选 |
|--------|---------|---------|
| **PRISMM-Bench** | 多模态不一致检测/修复 (384 样本) | 审稿人期望的直接评估；当前 SOTA 仅 27-54% |
| **MuSciClaims** | 图表 claim 验证 F1 | 科研图表 claim 验证的标准评估 |
| **SciMDR-Eval** | 科研文档推理（专家标注） | 验证 consistency training 是否提升推理能力 |
| **BRIDGE** | 长文档多跳推理 + 证据定位 | 验证跨模态证据聚合能力 |
| **SciClaimEval** | 跨模态 claim 验证 (1664 样本) | 补充实验 |
| **MathVista / ChartQAPro** | 通用图表 QA | 验证不 regression |

### 5.2 Baselines

| Baseline | 类型 | 说明 |
|----------|------|------|
| Qwen2.5-VL-7B (zero-shot) | 基座模型 | 最直接的 baseline |
| Qwen2.5-VL-7B + SciMDR SFT | SFT only | 消融 Stage 1 的价值 |
| Qwen2.5-VL-7B + EMPO | EMNLP 2025 方法 | 通用多模态 DPO，无 consistency signal |
| Qwen2.5-VL-7B + PaLMR | 2026.03 方法 | V-GRPO with LLM judge reward |
| Qwen2.5-VL-7B + VPPO | ICLR 2026 方法 | Token-level visual dependency |
| InternVL2.5-8B (zero-shot) | 跨 backbone | 验证方法泛化性 |
| GPT-4o / Qwen2.5-VL-72B | 强 API | 大模型 upper bound |
| **SciConsist-7B (ours)** | 全 pipeline | Stage 1 + 2 + 3 |

### 5.3 消融实验

| 配置 | 去掉什么 | 验证什么 |
|------|---------|---------|
| w/o Cycle Reward (Stage 3) | 只做 SFT + DPO | GRPO 的额外增益 |
| w/o DPO (Stage 2) | 只做 SFT + GRPO | DPO 偏好学习的增益 |
| w/o KL Consistency (Stage 1) | SFT 不加 consistency loss | 跨模态 KL 正则的增益 |
| w/o Hard Negatives | DPO 只用 BoN，不加合成 negatives | Hard negative 的增益 |
| w/o Token Weighting | GRPO 不加 VPPO-style 权重 | Claim token 加权的增益 |
| w/o Cross-Modal Cycle | 只用 forward cycle，不做 backward + cross | 完整 cycle 的增益 |
| R_cycle → LLM Judge Reward | 用 PaLMR 的 judge 替换 cycle reward | Cycle consistency vs LLM judge 的对比 |
| S1-MMAlign → No Pretraining Data | 不用 15.5M figure-text pairs | 大规模数据的增益 |

### 5.4 分析实验（加分项）

| 分析 | 内容 |
|------|------|
| **Error Taxonomy** | 对 PRISMM-Bench 错误分类：数值不一致 / 实体不一致 / 趋势不一致 / 条件不一致。哪类提升最大？ |
| **Consistency Score Calibration** | R_cycle 和人类判断的相关性有多高？ |
| **Scaling Behavior** | S1-MMAlign 从 1M → 5M → 15M，consistency 提升曲线如何？ |
| **Cross-Domain Transfer** | 在 CS 论文上训练，在医学/经济学论文上测试，泛化如何？ |
| **Efficiency Analysis** | Cycle consistency scoring 的计算开销 vs PaLMR 的 LLM judge 开销 |

---

## 六、必读文献清单（按技术组件分组，20 篇）

### Cycle Consistency & Self-Supervised Reward

| # | 论文 | 时间/会议 | 借鉴什么 |
|---|------|----------|---------|
| 1 | **CycleReward**: Cycle Consistency as Reward | ICCV 2025 | Cycle consistency → preference pairs → reward model |
| 2 | **CycleCap**: Self-Supervised Cycle Consistency Fine-Tuning | 2026.03 | VLM → caption → T2I → compare → GRPO |
| 3 | **CyCLeGen**: Cycle-Consistent Learning via RL | 2026.03 | Image↔layout cycle + RL |

### Faithfulness-Aware RL Training

| # | 论文 | 时间/会议 | 借鉴什么 |
|---|------|----------|---------|
| 4 | **PaLMR**: Process Alignment via V-GRPO | 2026.03 | Visual fidelity reward; zero reward for hallucination path |
| 5 | **VPPO**: Visually-Perceptive Policy Optimization | ICLR 2026 | Token-level visual dependency reweighting |
| 6 | **PGPO**: Perception-Grounded Policy Optimization | 2026.04 | KL divergence for token visual dependency |
| 7 | **Chart-RL**: RL with Verifiable Rewards | 2026.03 | 数值 verifiable reward 设计 |

### Preference Data Construction & DPO

| # | 论文 | 时间/会议 | 借鉴什么 |
|---|------|----------|---------|
| 8 | **EMPO**: Entity-Centric Multimodal Preference Optimization | EMNLP 2025 Main | 自动偏好数据构造 (multi-aspect) |
| 9 | **ReLoop**: Closed-Loop Training with Consistency Feedback | EMNLP 2025 Findings | Semantic reconstruction consistency |
| 10 | **MoD-DPO**: Modality-Decoupled Preference Optimization | 2026.03 | Modality-aware regularization |
| 11 | **CHiP**: Cross-modal Hierarchical DPO | 2025 | Hierarchical preference optimization |

### Scientific Document Understanding

| # | 论文 | 时间/会议 | 借鉴什么 |
|---|------|----------|---------|
| 12 | **SciMDR**: Scientific Multimodal Document Reasoning | 2026.03 | 300K QA 训练数据 + synthesize-and-reground |
| 13 | **S1-MMAlign**: 15.5M Scientific Figure-Text Pairs | 2026.01 | 大规模预训练数据 |
| 14 | **BRIDGE**: Multi-hop Reasoning in Long Docs | 2026.03 | 评估集 + 多跳推理标注 |
| 15 | **PRISMM-Bench**: Peer-Review Multimodal Inconsistencies | 2025.10 | 主评估集（27-54% 的 gap） |

### Multimodal Representation & Alignment

| # | 论文 | 时间/会议 | 借鉴什么 |
|---|------|----------|---------|
| 16 | **PRe**: Predictive Regularization for Visual Representation | 2026.03 | Visual feature regularization |
| 17 | **AlignVLM**: Vision-Text Alignment for Document Understanding | 2025.02 | Document-specific alignment method |
| 18 | **MuSciClaims**: Multimodal Scientific Claim Verification | 2025 | 评估集 |
| 19 | **SciClaimEval**: Cross-Modal Claim Verification | 2026.02 | 评估集 + claim 数据格式 |
| 20 | **VisualPRM**: Process Reward Model for Multimodal | ICLR 2026 | PRM 设计参考（对比 baseline） |

---

## 七、计算资源估算（4× 4090）

| 阶段 | 模型 | 数据量 | 预估时间 | 显存 |
|------|------|--------|---------|------|
| Stage 1 SFT | Qwen2.5-VL-7B LoRA | 300K (SciMDR) | 8-12h | 4×24GB (DeepSpeed ZeRO-2) |
| Cycle Scoring | Qwen2.5-VL-7B inference | ~50K samples | 6-8h | 1×24GB |
| Stage 2 DPO | Qwen2.5-VL-7B LoRA | ~30K pairs | 4-6h | 4×24GB |
| Stage 3 GRPO | Qwen2.5-VL-7B | ~10K prompts × 8 samples | 12-18h | 4×24GB |
| Evaluation | inference | ~5K samples | 2-3h | 1×24GB |
| **Total** | | | **~40-50h GPU** | |

如果用 5090 (32GB)：可以不用 LoRA 做全参数微调，或用更大 batch size 加速。

---

## 八、执行 Timeline（6 周）

### Week 1: 数据准备 + Cycle Scoring Pilot

- [ ] 下载 S1-MMAlign (HuggingFace) + SciMDR
- [ ] 实现 R_cycle 的三个分量（forward/backward/cross-modal）
- [ ] 在 50 篇论文上 pilot：
  - 用 Qwen2.5-VL-7B 做 multi-view claim extraction
  - 计算 cycle consistency score
  - **关键验证：cycle consistency score 和 human judgment 的相关性**
  - 如果 Pearson > 0.6 → Go；< 0.4 → 需要调整 reward 设计
- [ ] 分析 consistency score 的分布：一致 / 不一致的比例

### Week 2: Stage 1 Training + 偏好对构造

- [ ] 实现 cross-modal KL consistency loss
- [ ] 在 SciMDR 300K 上做 Stage 1 SFT
- [ ] 开始大规模 cycle scoring（~50K samples）
- [ ] 实现偏好对构造 pipeline（BoN sampling + hard negative generation）
- [ ] 生成 ~30K 偏好对

### Week 3: Stage 2 DPO + Stage 3 GRPO

- [ ] 实现并运行 Stage 2 DPO
- [ ] 实现 GRPO with cycle reward
- [ ] 加入 VPPO-style token weighting for claim tokens
- [ ] 运行 Stage 3 GRPO
- [ ] 中间 checkpoint 评估（PRISMM-Bench + SciMDR-Eval）

### Week 4: 全面评估 + Baselines

- [ ] 运行所有 baselines（Qwen2.5-VL zero-shot, SFT-only, EMPO, PaLMR）
- [ ] 在全部 6 个评估集上跑 SciConsist-7B
- [ ] 运行完整消融实验（8 个配置）
- [ ] 如有余力：在 InternVL2.5-8B 上验证跨 backbone 泛化

### Week 5: 分析 + 可视化

- [ ] Error taxonomy 分析（PRISMM-Bench 上的错误类型分布）
- [ ] Consistency score calibration 分析
- [ ] Cross-domain transfer 测试
- [ ] Case study：选 5 个典型 case 可视化 cycle consistency 如何帮助
- [ ] Efficiency 对比（cycle reward vs LLM judge）

### Week 6: 写作

- [ ] Paper draft：Abstract, Introduction (动机 + contribution), Related Work
- [ ] Method section（三个 contribution 各一节）
- [ ] Experiments + Analysis
- [ ] 画 pipeline 总图 + 消融表 + case study 图

---

## 九、Paper Narrative

### Title 候选

1. **SciConsist: Cross-Modal Cycle Consistency for Faithful Scientific Document Understanding**
2. **Cycle Consistency as Free Supervision: Training Faithful Multimodal Reasoners from Document Redundancy**
3. **Learning to Be Consistent: Self-Supervised Cross-Modal Alignment for Scientific Document Reasoning**

### Abstract 核心论点（6 句）

1. [问题] 多模态大模型在科研文档推理中频繁产生跨模态不一致——文本 claim 与图表证据矛盾、数值偏差、趋势描述错误。
2. [观察] 我们观察到科研文档天然包含跨模态冗余：同一事实同时以文本、表格和图表形式存在。
3. [Insight] 这种冗余提供了免费的 faithfulness 自监督信号——不需要人工标注就能构造跨模态一致性 reward。
4. [方法] 我们提出 SciConsist，一个三阶段训练框架：(1) 跨模态 KL 一致性正则化的 SFT，(2) 基于 cycle consistency 评分自动构造偏好对的 DPO，(3) 以 cycle consistency 为 reward 的 GRPO。
5. [结果] 在 PRISMM-Bench 上，SciConsist-7B 从 baseline 的 X% 提升到 Y%，同时在 SciMDR-Eval 和 MuSciClaims 上也取得显著提升。
6. [意义] 我们证明了文档内部冗余可以作为训练多模态 faithfulness 的有效信号，为低成本提升科研文档理解质量提供了新路径。

### 预期 Contribution Summary

| 贡献 | 类型 |
|------|------|
| 跨模态 cycle consistency 作为 self-supervised reward | **Algorithmic novelty** |
| Consistency-based preference pair 自动构造 | **Data/Training methodology** |
| 三阶段 progressive training (SFT → DPO → GRPO) | **Training framework** |
| 在 PRISMM-Bench / MuSciClaims / SciMDR-Eval 上的 SOTA | **Empirical contribution** |
| Error taxonomy + cross-domain transfer 分析 | **Analysis contribution** |

---

## 十、风险评估（升级版）

| 风险 | 等级 | 应对 | 参照论文 |
|------|------|------|---------|
| Cycle consistency score 和人类判断相关性低 | 🔴 高 | Week 1 pilot 验证；如 < 0.4 需改 reward 设计 | CycleReward 报告 0.7+ 相关性 |
| Multi-view claim extraction 本身不准 | 🟡 中 | 用 Qwen2.5-VL-72B API 做 extraction（inference only），用 7B 做 training | MuSciClaims 报告 best F1=0.72 |
| GRPO 训练不稳定 | 🟡 中 | 先只做 SFT + DPO（已经够发 Findings）；GRPO 是 bonus | PaLMR, VPPO 都用了 GRPO 且稳定 |
| 和 CycleReward 区别度不够 | 🟡 中 | Narrative 强调"文档冗余"不是"image-caption cycle"；评估在文档理解任务上 | CycleReward 不涉及文档 |
| 审稿人觉得技术组合多但每个不够深 | 🟡 中 | 消融表证明每个组件有独立贡献 | EMPO (EMNLP 2025) 也是组合多组件 |
| 计算预算不够做全部 ablation | 🟢 低 | 4× 4090 完全够 7B LoRA 训练 | 估算 ~50h GPU time |

---

## 十一、EMNLP 接收概率的条件分析

| 条件 | 估计概率 |
|------|---------|
| 全部按计划执行 + PRISMM-Bench 提升 >15pp + 消融完整 | **65-75% (Main + Findings)** |
| Stage 1+2 有效 + Stage 3 GRPO 锦上添花 | **55-65%** |
| 只有 SFT + DPO 有效，GRPO 不稳定 | **45-55% (Findings likely)** |
| Cycle consistency score 和人类判断相关性 < 0.4 | **需要 pivot** |

**概率提升的关键：**
1. 在 PRISMM-Bench 上做到目前 21 个 LMM 的最高分（目前 53.9%）
2. 消融表清晰展示每个组件的独立贡献
3. Cross-domain transfer 证明方法不只对 CS 论文有效
4. R_cycle vs LLM judge 的效率 + 质量对比（我们更快且不需要额外 API）
