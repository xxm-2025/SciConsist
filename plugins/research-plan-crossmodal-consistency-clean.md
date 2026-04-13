# SciConsist: Factual Entailment Reward for Cross-Modal Scientific Document Verification

> 目标会议: EMNLP 2026 | 6 周执行周期

---

## 一、问题：MLLM 在科研文档上的跨模态不一致

多模态大模型处理科研文档时，生成的文本 claim 经常与图表证据矛盾——数值偏差、趋势描述错误、凭空编造。

科研文档有一个天然特性：**结构化冗余**。同一个实验结果会同时出现在正文、表格和图中。这种冗余理论上可以用来做自监督的一致性验证，但现有方法没用好。

---

## 二、现有方法为什么不行

以 CycleReward/CycleCap 为代表的 cycle consistency 方法，用**表面文本相似度**（cosine similarity）作为 reward：

```
R(τ) = sim(模型从 figure 生成的 claim, 原文对应文本)
```

这会导致三种致命场景：

| 场景 | 例子 | sim() 分数 | 后果 |
|------|------|-----------|------|
| 语义等价但表述不同 | 模型说 "Model B 低 3.2 BLEU" vs 原文 "our method significantly outperforms" | 低 | 正确推理被惩罚 |
| 原文未提及的正确观察 | 模型说 "Model C 方差很大" 但原文没讨论 | 极低 | 正确观察被当幻觉 |
| 真实冲突（作者手误） | Figure 写 85% 但正文写 58% | 低 | 应检测冲突，却只给低分 |

**本质**：表面相似度把三种完全不同的情况混为一谈，全部给低分。长期训练后模型退化为复读原文（Many-to-One collapse）。

---

## 三、核心 Idea

> 不在文本表面做匹配，而是在 MLLM 隐空间做**事实蕴含判定**（Factual Entailment）。
> 同时训练模型做两件事：(1) 生成与证据一致的回答 (2) 主动检测跨模态冲突。
> 从 "被动对齐器" 升级为 "多模态科研文档审计器"。

### 三个核心设计

**1. Factual Entailment Head (FEH)** — 替代 surface sim()

一个轻量级分类头（~10M 参数），接收冻结 VLM 的 hidden states，把 (figure_region, text_claim) 对分类为三类：
- **ENTAILS**: claim 被视觉证据支持
- **NEUTRAL**: claim 正确但证据不足以判定（解决"表述不同"和"原文未提及"的问题）
- **CONTRADICTS**: claim 与视觉证据矛盾

**2. Cross-Model 设计** — 防止 Reward Hacking

FEH 在独立模型（InternVL2.5-8B，冻结）的特征上训练和推理，与 policy model（Qwen2.5-VL-7B）完全解耦。Policy model 的参数更新无法影响 reward 的计算，从物理上阻断 reward hacking。

**3. Conflict Detection Bonus** — 从对齐到审计

当模型主动报告文图冲突，且 FEH 确认确实存在矛盾时，给额外奖励。虚假报告则扣分。这使模型不仅学会"说对的话"，还学会"找错的地方"。

---

## 四、方法概览

### 4.1 训练 Pipeline（两阶段）

```
Stage 1: Standard SFT（非贡献）
  Qwen2.5-VL-7B + LoRA on SciMDR 300K
  → 建立科研文档基础理解能力

Stage 2: FER-GRPO（核心贡献）
  用 Factual Entailment Reward 做 GRPO 强化学习
  → 产出 SciConsist-7B
```

DPO 被砍掉。理由：FER 直接做 GRPO 在线 reward 就够了，DPO 的偏好对本身也是用 FER 打分构造的，多一层中介只引入噪声。消融实验会验证这一点。

### 4.2 FEH 架构

```
InternVL2.5-8B (frozen)
       │
  ┌────┴────┐
  ▼         ▼
figure    claim
hidden    hidden
  │         │
  ▼         ▼
Visual    Text
Proj      Proj
  │         │
  ▼         ▼
 h_v       h_t    (d=512)
  │         │
  └──► [h_v ; h_t ; |h_v - h_t|] ◄──┘
                  │
                  ▼
         3-class Classifier
                  │
                  ▼
    {ENTAILS, NEUTRAL, CONTRADICTS}
```

关键设计选择：
- **Multi-layer fusion**: 取 {L/4, L/2, 3L/4, L} 四层 hidden states 均值，比单层更稳定
- **差异向量 |h_v - h_t|**: 借鉴 InferSent，帮助捕捉冲突信号
- **三分类而非二分类**: NEUTRAL 类是解决 Many-to-One 的关键

### 4.3 Reward 计算

对模型生成的每个 response τ，提取所有 factual claims {c₁, ..., cₖ}，逐个过 FEH：

```
每个 claim 的得分:
  R(cᵢ) = +1.0 (ENTAILS) / -0.1 (NEUTRAL) / -0.5 (CONTRADICTS)

整体 reward:
  R_FER(τ) = mean(R(cᵢ))                           基础分
           + λ_cross · cross_modal_bonus(τ)          跨模态数值精确匹配
           + λ_conflict · conflict_detection(τ)      冲突检测奖惩
           + λ_info · informativeness_gate(τ)        信息量门控
```

其中：
- **Cross-Modal Bonus**: table 和 figure 对同一数据点数值一致时加分（精确匹配，不需要 FEH）
- **Conflict Detection**: 模型报告冲突且 FEH 确认 → +1.5；虚假报告 → -1.0
- **Informativeness Gate**: ENTAILS claim 占比 < 30% 时扣分，防止模型生成模糊废话逃避风险

### 4.4 FEH 预训练数据

| 来源 | 标签 | 数量 |
|------|------|------|
| S1-MMAlign figure-text 配对（正例） | ENTAILS | ~50K |
| S1-MMAlign 数值篡改 ±20% | CONTRADICTS | ~50K |
| 随机 figure + 不相关 text | NEUTRAL | ~50K |
| MuSciClaims + SciClaimEval（人工标注） | 混合 | ~3.6K |

FEH 参数量 ~10M，单卡 4090 训练 1-2h。**Go/No-Go 门槛**：held-out 三分类 accuracy > 75%。

---

## 五、与相关工作的核心区别

| 方法 | 匹配方式 | 退化处理 | 冲突检测 |
|------|---------|---------|---------|
| CycleReward/CycleCap | 表面相似度 | 无 (Many-to-One) | 无 |
| GVF | Prompt-level factual anchor | 用 counterfactual prompts | 无 |
| DeFacto | 合成 counterfactual + GRPO | 无 | 无 |
| VISA | 表示层 VFM 对齐 | 无 | 无 |
| **SciConsist (Ours)** | **隐空间 NLI 三分类** | **NEUTRAL 类** | **Conflict Detection Bonus** |

本质差异：我们在 representation level 做 factual entailment，而非 surface level 做 similarity；我们不需要合成数据，利用科研文档的天然冗余。

---

## 六、实验设计

### 6.1 评估集（6 个）

| 数据集 | 规模 | 评估什么 |
|--------|------|---------|
| PRISMM-Bench | 384 | 不一致检测/修复 |
| MuSciClaims | ~2K | 图表 claim 验证 |
| SciClaimEval | 1,664 | 跨领域泛化（ML/NLP/Medicine） |
| SciMDR-Eval | 专家标注 | 科研文档 QA |
| BRIDGE | 长文档 | 长文档多跳推理 |
| ChartQAPro | 通用 | 图表理解 baseline |

所有结果报告 mean ± std (3 seeds)。

### 6.2 Baselines

- Qwen2.5-VL-7B (zero-shot / + SFT only)
- EMPO (DPO, EMNLP 2025)
- PaLMR V-GRPO (LLM judge reward)
- VPPO (token-level, ICLR 2026)
- CycleReward-style surface sim (即 v2 方案)
- InternVL2.5-8B / GPT-4o / Qwen2.5-VL-72B (上界参考)

### 6.3 关键消融

| 对比 | 验证什么 |
|------|---------|
| surface sim GRPO vs FEH GRPO (A2 vs A7) | FEH 隐空间判定 vs 表面匹配 |
| FEH-Same vs FEH-Cross (A8 vs A7) | Cross-Model 设计是否防住 reward hacking |
| 二分类 vs 三分类 (A3 vs A7) | NEUTRAL 类的价值 |
| 去掉 Conflict Detection (A4 vs A7) | 冲突检测 bonus 的贡献 |
| DPO vs GRPO (A5 vs A7) | Pipeline 精简的合理性 |
| GPT-4o Judge vs FEH (A9 vs A7) | FEH 效率优势（预期 200×加速） |

### 6.4 专项分析

- **冲突检测 Sensitivity Analysis**: 在 1%/5%/10%/25%/50% 冲突率测试集上报告 P/R/F1，目标 FPR@1% < 5%
- **FEH vs Human Agreement**: Cohen's κ
- **Diversity Analysis**: distinct-n, self-BLEU，验证没有 diversity collapse
- **跨领域零样本**: CS 训练 → Medicine/Biology 测试

---

## 七、Paper Narrative

### 推荐标题

**Teach Models to Audit: Factual Entailment Reward with Conflict Detection for Multimodal Scientific Reasoning**

### 贡献列表

| 贡献 | 类型 | 重要性 |
|------|------|--------|
| Cross-Model Factual Entailment Head (FEH) | 算法 | 核心 |
| Tri-State Reward + Conflict Detection Bonus | Reward 设计 | 核心 |
| 消融证明 FEH > surface matching / NEUTRAL 类的作用 | 实验 insight | 支撑 |
| Conflict Detection Test Set (400 samples) | 评估资源 | 附加 |

### 一句话 Pitch

> 用隐空间事实蕴含判定替代表面文本匹配做 reward，解决 cycle consistency 的 Many-to-One 退化；引入冲突检测 bonus 让模型从被动对齐器变为主动审计器。

---

## 八、执行计划

### 资源：4× RTX 4090，预计总 GPU 时间 ~100h

| 周次 | 任务 | 关键节点 |
|------|------|---------|
| Week 1 | FEH Pilot | 部署 InternVL2.5 → 特征提取 → 训练 FEH → **Go/No-Go: acc > 75%** |
| Week 2 | Stage 1 SFT + GRPO 框架 | SciMDR SFT + 实现 FER-GRPO 训练循环 + 小规模试跑 |
| Week 3 | Stage 2 Full Training | FER-GRPO 全量训练，监控 reward/diversity/KL |
| Week 4 | 评估 + 消融 | 6 数据集 × 3 seeds + 全部消融配置 |
| Week 5 | 分析 + 可视化 | Sensitivity analysis + case study + 所有图表 |
| Week 6 | 写作 | Paper draft，重点 FEH 设计 + Tri-State Reward 两节 |

### 关键风险

| 风险 | 等级 | 应对 |
|------|------|------|
| FEH 三分类 accuracy < 65% | 高 | Week 1 Go/No-Go；增加训练数据或改架构 |
| Cross-Model FEH 信号太弱 | 中 | InternVL 和 Qwen 语义空间不完全重叠 → 退回 Same-Model + 强 KL penalty |
| InternVL inference 瓶颈 | 中 | 预缓存 figure 特征，减少重复推理 |
| 冲突过度报告 | 中 | 训练冲突比例控制 5% + sensitivity analysis 验证 |
