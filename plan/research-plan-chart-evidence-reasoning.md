# ChartEvidence: Structured Evidence Extraction and Graph-Grounded Reasoning for Chart-Dense Scientific Documents

> 研究方向计划 | 生成日期: 2026-04-09

---

## 一、文献全景

### 1.1 图表理解（Chart Understanding）现状

#### 单图表 QA——接近饱和

| 数据集/方法 | 时间 | 特点 | 现状 |
|------------|------|------|------|
| **ChartQA** | 2022 | 经典单图表 QA | Claude Sonnet 3.5 已达 90.5%，饱和 |
| **ChartQAPro** | 2025.04 | 更多样的图表类型 + 更难的问题 | Claude 降到 55.8%，仍有空间但仍是单图 |
| **ChartMind** | 2025.05 | 7 类任务、多语言、开放域 | 提出 ChartLLM 框架，但仍是单图范式 |
| **ChartLlama** | 2023 | 指令微调的图表 LLM | chart-to-table、QA、描述等多任务 |
| **ChartAnchor** | 2026 | chart-to-code + chart-to-table 双向对齐 | 8000+ 图表-表格-代码三元组，重数值精度 |

**结论：** 单图表理解已经是红海，不值得再进。

#### 多图表推理——刚刚起步

| 数据集 | 时间 | 特点 | 发现 |
|--------|------|------|------|
| **MultiChartQA** | 2024.10 | 多图表 QA（比较、顺序、并行推理） | MLLM 与人类差距显著 |
| **ChartNexus** | ICLR 2026 | 1370 QA / 6793 图表 / 18 领域 / 文档上下文 | 顶级模型在单图 >90% 但在此降至 ~45% |
| **INTERCHART** | 2025 | 三级难度（事实→语义→跨图推理） | 跨图推理是最差的 |
| **ChartDiff** | 2026 | 8541 图表对的差异摘要 | 趋势/波动/异常差异描述 |

**结论：** 多图表推理是公认的 open challenge，ChartNexus 明确指出模型在 working memory、cross-modal reasoning、contextual integration 上系统性失败。**但这些都只是 benchmark，没有人提出有效的解法。**

### 1.2 科研文档多模态推理

| 工作 | 时间 | 做了什么 | 没做什么 |
|------|------|----------|----------|
| **BRIDGE** | 2026.03 | 长科研论文多跳推理 benchmark，显式标注推理链 | 只是 benchmark；发现模型在证据聚合上系统失败 |
| **SciMDR** | 2026.03 | 300K QA + 20K 论文，synthesize-and-reground 框架 | 重在数据合成方法，不是推理方法本身 |
| **HighlightBench** | 2026.03 | 表格中视觉标记（高亮/加粗）的推理 | 仅表格，不涉及图表 |
| **MLDocRAG** | 2026.02 | 多模态长文档 RAG，chunk-query graph | Graph 是检索级的，不是证据语义级的 |
| **DocR1** | 2025 | Evidence page-guided GRPO | 页面级证据定位，不做细粒度图表解析 |
| **PaperQA3** | 2026.02 | 读 150M+ 论文的图表 | 让 LLM "选择何时看图"，不做结构化抽取 |

**结论：** 这一块的 benchmark 很多（BRIDGE, SciMDR, HighlightBench），但**解法严重缺乏**。BRIDGE 明确指出"evidence aggregation and grounding 存在系统性缺陷"。

### 1.3 证据图 / 知识图谱上的推理

| 工作 | 时间 | 做了什么 | 局限 |
|------|------|----------|------|
| **PrunE** | 2025 | 从科研文档构建证据图，剪枝后做 claim verification | 只从**文本**构建，不处理图表 |
| **EvidenceNet** | 2026 | 从生物医学文献构建 evidence-based KG | 领域特定（HCC、CRC），不处理图表 |
| **KG-CRAFT** | 2026, EACL | 从 claim 构建 KG → 对比推理 → 事实核查 | 不涉及视觉输入 |
| **ClimateViz** | 2025 | 图表 → KG explanations → 事实验证 | **最接近**，但仅限气候领域的统计图表 |

**结论：** 证据图推理在文本领域已有成熟方法，但**从视觉图表构建证据图 → 多跳推理**这条路几乎没人走过。ClimateViz 做了一小步（单领域），但没有通用框架。

### 1.4 结构化抽取：从图表到结构化 claim

| 工作 | 时间 | 输入 | 输出 |
|------|------|------|------|
| Chart derendering (ChartReader, ChartAnchor) | 2023-2026 | 图表图像 | 数据表格（数值级） |
| Triplet extraction from tables | 2025 | 论文表格 | ⟨subject, measure, outcome⟩ triplets |
| **SciDaSynth** | 2025 | 科研论文（多模态） | 交互式结构化数据表 |
| **ExtracTable** | 2025 | 科研出版物 | 人在环路的结构化 KG 表示 |
| **Format Matters** | 2026 | 表格 vs 图表证据 | 发现 MLLM 处理图表比表格差很多 |

**结论：** 现有的 chart derendering 输出的是**低级数据表格**（x=3, y=7.2），而不是**高级语义 claim**（"方法 A 在指标 X 上优于方法 B 3.2%"）。这个从**数值级到语义级的抽象跳跃**没有人系统做过。

---

## 二、Gap Analysis：空白在哪里

### 2.1 Gap Map

```
现有能力                          空白                              你的机会
─────────────────────────────────────────────────────────────────────────────
图表 → 数据表格                   图表 → 语义级 claim               ← Gap ①
(ChartReader, ChartAnchor)        (没人做过通用框架)

多图表 benchmark 发现问题          多图表的解法                      ← Gap ②
(ChartNexus, INTERCHART)          (只有 benchmark，没有 method)

文本证据图 → 推理                 图表证据图 → 推理                 ← Gap ③
(PrunE, KG-CRAFT)                 (没人从图表构建证据图)

PaperQA3 "选择何时看图"           结构化图表证据接入 RAG            ← Gap ④
(不做结构化抽取)                  (没人测过 structured vs flat)
```

### 2.2 核心洞察

**整个 pipeline 中存在一个"语义断层"：**

- **上游**（Chart Understanding）的输出是**数值级**的：表格、数据点、坐标值
- **下游**（Document Reasoning / RAG）的输入期望是**语义级**的：claim、evidence、argument
- **中间没有桥梁**：没有人系统地把"图表中的数值比较"翻译成"可推理的语义 claim"

这个 gap 非常适合你做，因为：
1. 上下游都已经有成熟工具（chart derendering + evidence graph reasoning），你只需要建**中间这座桥**
2. 不需要训练新模型——用 MLLM 做抽取 + 简单图结构做推理
3. 有现成的 benchmark 可以评估（ChartNexus, BRIDGE, SciMDR）
4. 和 Deep Research 热点强关联，narrative 自然

---

## 三、研究定位

### 3.1 Research Question

> **在图表密集的科研文档中，能否通过将图表转化为结构化的比较证据（comparison claims），并在证据图上进行多跳推理，显著提升跨图表问答的质量？**

### 3.2 Proposed Method: ChartEvidence

一个三阶段框架：

```
Stage 1: Chart Evidence Extraction
    图表图像 → 结构化 claim triplets
    ⟨entity_A, relation, entity_B, metric, value, source_chart⟩
    例: ⟨GPT-4, outperforms, LLaMA-70B, MMLU, +5.3%, Table 2⟩

Stage 2: Evidence Graph Construction
    多个 claim triplets → 证据图
    节点 = 实体 (模型/方法/数据集/指标)
    边 = 比较关系 (outperforms / comparable / degrades)
    属性 = 具体数值 + 来源图表 + 置信度

Stage 3: Graph-Grounded Reasoning
    用户问题 → 在证据图上做子图检索 → 多跳推理 → 答案 + 证据溯源
    支持：跨图表比较、趋势分析、条件推理
```

### 3.3 三个技术贡献

| 贡献 | 内容 | 差异化 |
|------|------|--------|
| **C1: Chart → Claim 抽取器** | MLLM-based 的图表语义抽取，输出结构化 claim triplets | ChartReader 输出数据表格（数值级），我们输出语义 claims（推理级） |
| **C2: 跨图表证据图** | 自动构建连接多图表证据的轻量图结构 | PrunE 从文本构建证据图，我们从视觉图表构建 |
| **C3: Graph-Grounded Reasoning** | 在证据图上做子图检索 + 多跳推理 | MLDocRAG 用 chunk-query graph（检索级），我们用 evidence graph（语义级） |

### 3.4 目标会议

| 会议 | 侧重点 | 适合度 |
|------|--------|--------|
| **ACL / EMNLP 2026-2027** | 强调 claim extraction + NLP pipeline | ⭐⭐⭐⭐⭐ |
| **NeurIPS 2026** | 强调 graph reasoning + 多模态 | ⭐⭐⭐⭐ |
| **KDD 2027** | 强调科研文档应用 + 实际系统 | ⭐⭐⭐⭐ |
| **ICLR 2027** | 强调 reasoning 方法创新 | ⭐⭐⭐ |

---

## 四、详细技术方案

### 4.1 Stage 1: Chart Evidence Extraction

**输入：** 图表图像（柱状图、折线图、表格、散点图等）+ 图表标题/caption

**输出：** 结构化 claim triplets 列表

```
Schema:
{
  "entity_a": str,          # 比较主体（如 "GPT-4"）
  "entity_b": str,          # 比较对象（如 "LLaMA-70B"）
  "relation": enum,         # outperforms | comparable | underperforms | trend_up | trend_down
  "metric": str,            # 指标名（如 "MMLU accuracy"）
  "value_a": float | null,  # entity_a 的具体值
  "value_b": float | null,  # entity_b 的具体值
  "delta": str | null,      # 差异描述（如 "+5.3%"）
  "condition": str | null,  # 条件（如 "zero-shot setting"）
  "source": str,            # 来源（如 "Figure 3" / "Table 2"）
  "confidence": float       # 抽取置信度 [0, 1]
}
```

**方法：**
1. 用 MLLM（Qwen2.5-VL / InternVL2）对图表做 structured prompting
2. Prompt 设计要点：先让模型描述图表内容，再从描述中抽取 triplets
3. 对数值型 claim 做 chart derendering 交叉验证（和 ChartAnchor 的数据表格对比）
4. Confidence 基于：多次采样一致性 + 数值是否可从表格验证

**关键设计决策：**
- 不训练新模型，纯 prompt-based extraction（training-free）
- Triplet schema 设计要足够通用，覆盖比较、趋势、异常三类 claim
- 允许 null 值（图表中不一定所有数值都可读取）

### 4.2 Stage 2: Evidence Graph Construction

**输入：** 来自多个图表的 claim triplets

**输出：** 一个属性图（property graph）

```
节点类型：
  - Entity: 方法/模型/数据集/药物/...
  - Metric: 评估指标
  - Condition: 实验条件

边类型：
  - COMPARED_ON: (Entity_A) -[COMPARED_ON {relation, delta, source}]-> (Entity_B)
  - EVALUATED_WITH: (Entity) -[EVALUATED_WITH {value, source}]-> (Metric)
  - UNDER_CONDITION: (Comparison) -[UNDER_CONDITION]-> (Condition)
```

**构建流程：**
1. 实体对齐（Entity Resolution）：同一个方法在不同图表中可能写法不同（如 "GPT4" vs "GPT-4" vs "GPT-4o"），用 embedding similarity + 规则做对齐
2. 边合并：如果两个图表对同一对实体在同一指标上有比较，合并并标注是否一致
3. 冲突检测：如果两个图表的 claim 矛盾，保留两条边并标注 conflict

**实现：** 用 NetworkX 即可，不需要图数据库。图通常很小（一篇论文 < 100 节点）。

### 4.3 Stage 3: Graph-Grounded Reasoning

**输入：** 用户问题 + 证据图

**输出：** 答案 + 证据链（evidence trail）

**方法（三种递进方案，按复杂度）：**

**方案 A（最简，推荐先做）：Graph-as-Context**
- 将证据图序列化为结构化文本（如 markdown 表格或 JSON）
- 直接作为 context 输入 LLM
- 让 LLM 在结构化证据上推理
- 优点：实现简单；缺点：大图可能超 context

**方案 B：Subgraph Retrieval + Reasoning**
- 根据问题中的实体/指标，从证据图中检索相关子图
- 只将子图序列化后输入 LLM
- 优点：高效；适合长文档

**方案 C：Graph Traversal Reasoning**
- LLM 作为 agent，通过工具调用（get_neighbors, get_comparison, find_path）在图上导航
- 多步推理，每步选择下一个要探索的节点
- 优点：支持复杂多跳；缺点：实现复杂

**推荐策略：** 从方案 A 开始，验证证据图本身的价值。如果有效，再做方案 B。方案 C 作为 future work。

### 4.4 与 RAG Pipeline 的集成

```
传统 RAG:    PDF → 文本切片 → embedding → 检索 → LLM 回答
                     ↑
                  图表被忽略或扁平化为 caption 文本

ChartEvidence:  PDF → 文本切片 + 图表 claim 抽取 → evidence graph
                                                       ↓
                用户问题 → 子图检索 → 结构化证据 + 文本证据 → LLM 回答
```

---

## 五、实验设计

### 5.1 评估数据集

| 数据集 | 用途 | 为什么选它 |
|--------|------|-----------|
| **ChartNexus** | 主实验：跨图表推理 QA | 最权威的多图表推理 benchmark，文档上下文 |
| **BRIDGE** | 补充实验：长文档多跳推理 | 显式标注推理链，可评估中间步骤 |
| **SciMDR-Eval** | 补充实验：科研文档推理 | 专家标注，覆盖图表+文本 |
| **MultiChartQA** | 消融分析：不同推理类型对比 | 明确区分了比较/顺序/并行推理 |

### 5.2 Baselines

| Baseline | 类别 | 描述 |
|----------|------|------|
| **Direct MLLM** | 端到端 | Qwen2.5-VL / GPT-4o 直接看图回答 |
| **ChartLLM** | Chart Understanding | ChartMind 提出的上下文感知框架 |
| **MLDocRAG** | Document RAG | 多模态长文档 RAG，chunk-query graph |
| **Flat-text RAG** | Text RAG | 图表 caption + OCR 文本作为检索单元 |
| **Chart-to-Table + QA** | Pipeline | ChartAnchor 抽表格 → 表格 QA |
| **ChartEvidence (ours)** | 本方法 | Chart → Claims → Evidence Graph → Reasoning |

### 5.3 评估指标

| 指标 | 评估什么 |
|------|---------|
| **Answer Accuracy** | 最终答案正确率（exact match / F1） |
| **Evidence Precision** | 引用的证据是否正确（和 ground truth evidence 对比） |
| **Evidence Recall** | 需要的证据是否被找到 |
| **Claim Extraction F1** | Stage 1 抽取的 claim 质量（需要人工标注子集） |
| **Multi-hop Success Rate** | 需要 2+ 步推理的问题的成功率（BRIDGE 提供标注） |
| **Reasoning Faithfulness** | 推理过程是否忠于证据图（非幻觉） |

### 5.4 消融实验

| 配置 | 去掉什么 | 验证什么 |
|------|---------|---------|
| w/o Evidence Graph | 只做 claim extraction，不建图 | 图结构的增益 |
| w/o Claim Extraction | 直接用 chart-to-table 的数值 | 语义 claim 的增益 vs 原始数值 |
| w/o Entity Resolution | 不做跨图表实体对齐 | 实体对齐的重要性 |
| w/o Confidence Filtering | 保留所有 claim，不过滤低置信度的 | 置信度过滤的必要性 |
| Gold Evidence Graph | 用人工构建的完美证据图 | pipeline 的上界 |

---

## 六、关键开放问题

1. **Claim 抽取的准确率能到多少？** 这是整个 pipeline 的瓶颈——如果 Stage 1 抽错了，后面全崩。需要 pilot experiment 先验证 MLLM 在这个任务上的零样本能力。

2. **图表类型覆盖度？** 柱状图/折线图的比较 claim 相对容易抽取，但散点图、热力图、Venn 图呢？需要评估不同图表类型的抽取难度。

3. **实体对齐的精度？** 跨图表的实体名可能差异很大（缩写、大小写、版本号）。这是一个已知的难问题，但在小规模（单文档内）可能用简单方法就够。

4. **证据图的规模和稀疏性？** 一篇论文通常有 5-15 个图表，可能只产出 20-100 个 claims。图会很小、很稀疏。这是优点（高效）还是缺点（图结构带来的增益有限）？

5. **Claim schema 的通用性？** 设计的 triplet schema 是否能覆盖不同领域（CS、医学、经济学）的图表？可能需要领域适配。

6. **和 PaperQA3 的关系？** PaperQA3 已经能读图表了。你的方法相比"直接让强大的 MLLM 看图回答"，增益在哪里？核心论证点：**结构化证据在多跳推理中的优势是扁平化阅读无法替代的**。

---

## 七、具体推进步骤（5 周计划）

### Week 1：Pilot — Claim 抽取可行性验证

**目标：** 验证 MLLM 能否可靠地从科研图表中抽取结构化 claim

- [ ] 收集 20 篇 CS/医学论文的图表（混合柱状图、折线图、表格、散点图）
- [ ] 设计 claim extraction prompt（迭代 3-5 个版本）
- [ ] 用 Qwen2.5-VL-72B (API) 和 GPT-4o 分别抽取
- [ ] 人工标注 ground truth claims（约 200 个），计算 Precision / Recall / F1
- [ ] 分析不同图表类型的抽取难度差异
- [ ] 同时用 ChartAnchor / chart-to-table 做数值验证

**产出：** Claim 抽取的可行性报告 + 最佳 prompt 模板  
**判断标准：** 如果 F1 < 0.5，需要考虑加入 chart derendering 辅助；如果 < 0.3，方向需要调整

### Week 2：Evidence Graph 构建 + 初步集成

**目标：** 实现 claim → graph 的完整 pipeline

- [ ] 实现 claim triplet 解析器（JSON schema validation）
- [ ] 实现 entity resolution 模块（embedding similarity + 规则）
- [ ] 用 NetworkX 构建 evidence graph
- [ ] 实现 Graph-as-Context 序列化（方案 A）
- [ ] 在 5 篇论文上端到端跑通：PDF → claims → graph → 回答问题
- [ ] 定性分析：图结构是否确实捕获了有用的跨图表关系

**产出：** 可运行的 prototype

### Week 3：ChartNexus 实验

**目标：** 在主 benchmark 上全面评估

- [ ] 下载 ChartNexus 数据集，理解其标注格式
- [ ] 对 ChartNexus 的图表运行 claim extraction pipeline
- [ ] 构建 evidence graph
- [ ] 实现所有 baselines（Direct MLLM / Flat-text RAG / Chart-to-Table + QA）
- [ ] 运行 Answer Accuracy + Evidence Precision/Recall 评估
- [ ] 分析不同难度类别（ChartNexus 的 4 级 11 子类）的表现差异

**产出：** 主实验结果表格

### Week 4：消融 + BRIDGE 补充实验

**目标：** 验证每个模块的贡献 + 测试多跳推理

- [ ] 跑全部消融实验（5 个配置）
- [ ] 在 BRIDGE 上测试（重点看 multi-hop success rate）
- [ ] 如有余力，在 SciMDR-Eval 上也跑
- [ ] 实现 Subgraph Retrieval（方案 B），测试大文档场景的效率
- [ ] 可视化 case study：展示证据图如何帮助多跳推理

**产出：** 消融表格 + 补充实验 + case study 图

### Week 5：分析 + 写作

**目标：** 锁定 narrative + paper draft

- [ ] Error analysis：pipeline 在什么情况下失败？（抽取错误？对齐错误？推理错误？）
- [ ] 跨领域分析（CS vs 医学 vs 经济学的图表，抽取难度差异）
- [ ] Overhead 分析（claim extraction 耗时、graph 构建耗时）
- [ ] 和 PaperQA3 的定性对比（如果 API 可用）
- [ ] Paper draft：Abstract + Introduction + Related Work + Method
- [ ] 画 pipeline 总图 + 证据图可视化

**产出：** Paper 初稿（Method + Experiments 部分完成）

---

## 八、核心文献清单（必读 Top 15）

### 图表理解

| # | 论文 | 时间 | 与你的关系 |
|---|------|------|-----------|
| 1 | **ChartNexus**: Multi-Chart Reasoning (OpenReview) | ICLR 2026 | **主评测集**，多图表推理的权威 benchmark |
| 2 | **ChartQAPro** | 2025.04 | 单图表 QA 上界参考 |
| 3 | **ChartAnchor**: Chart Grounding with Data Recovery | 2026 | Chart derendering 最新方法，**Stage 1 辅助工具** |
| 4 | **ChartMind** + ChartLLM | 2025.05 | 上下文感知图表理解框架，**baseline** |
| 5 | **INTERCHART** | 2025 | 跨图表推理的三级难度分析 |
| 6 | **ChartDiff** | 2026 | 图表对差异摘要，related work |

### 科研文档推理

| # | 论文 | 时间 | 与你的关系 |
|---|------|------|-----------|
| 7 | **BRIDGE**: Multi-hop Reasoning in Long Documents (2603.07931) | 2026.03 | **补充评测集**，显式多跳推理标注 |
| 8 | **SciMDR**: Scientific Multimodal Document Reasoning (2603.12249) | 2026.03 | 大规模科研文档推理，数据合成方法参考 |
| 9 | **MLDocRAG** (2602.10271) | 2026.02 | 多模态长文档 RAG，**architecture 参考** |
| 10 | **PaperQA3** | 2026.02 | 最强 Deep Research agent，**need to differentiate** |
| 11 | **HighlightBench** (2603.26784) | 2026.03 | 表格推理 benchmark，related work |

### 证据图 / 结构化抽取

| # | 论文 | 时间 | 与你的关系 |
|---|------|------|-----------|
| 12 | **PrunE**: Evidence Graph for Claim Verification | 2025 | **图结构设计参考**（但从文本构建，非视觉） |
| 13 | **KG-CRAFT**: KG-based Fact-checking (EACL 2026) | 2026 | 从 claim 构建 KG → 推理，**方法参考** |
| 14 | **ClimateViz**: Chart Fact Verification with KG | 2025 | **最接近的工作**——图表 → KG → 验证 |
| 15 | **Format Matters**: Table vs Chart Evidence | 2026 | MLLM 处理图表比表格差，**动机支撑** |

---

## 九、风险评估

| 风险 | 等级 | 应对 |
|------|------|------|
| Claim 抽取精度不够 | 🟡 中 | Week 1 pilot 验证；备选方案：chart-to-table 后从表格抽取 |
| 证据图太小/稀疏，增益有限 | 🟡 中 | 消融中用 Gold Graph 测上界；如果图结构增益 < 2%，pivot 到 flat structured evidence |
| 和 PaperQA3 对比没有优势 | 🟡 中 | 聚焦 PaperQA3 弱点：需要多跳推理 + 精确数值比较的场景 |
| ChartNexus 数据格式适配困难 | 🟢 低 | 数据集已公开，格式清晰 |
| 实体对齐错误导致图结构损坏 | 🟡 中 | 用保守策略（高阈值），宁可漏合并也不错合并 |
| 审稿人认为"只是 pipeline 组合" | 🔴 高 | **narrative 要强调"语义断层"这个新问题**，而不是 pipeline 本身 |

---

## 十、Paper Narrative 草案

### Title 候选

1. **ChartEvidence: Bridging the Semantic Gap between Chart Understanding and Document Reasoning**
2. **From Charts to Claims: Structured Evidence Extraction for Multi-Hop Scientific Document Reasoning**
3. **Evidence Graphs from Visual Charts: Structured Reasoning for Chart-Dense Scientific Documents**

### Abstract 核心论点（5 句话）

1. [问题] 科研文档中的图表包含关键的比较证据，但现有系统要么忽略图表，要么将其退化为扁平文本，无法支撑跨图表的多跳推理。
2. [洞察] 我们发现，图表理解（输出数值）和文档推理（需要语义 claim）之间存在一个"语义断层"——chart derendering 产出的数据表格和推理器需要的结构化证据不在同一抽象层级。
3. [方法] 我们提出 ChartEvidence，一个三阶段框架：将图表转化为结构化比较 claims，构建跨图表证据图，并在图上进行多跳推理。
4. [结果] 在 ChartNexus 和 BRIDGE 上的实验表明，ChartEvidence 在多图表推理问答中显著优于直接 MLLM、Chart-to-Table pipeline 和 flat-text RAG。
5. [贡献] 据我们所知，这是第一个从视觉图表自动构建证据图并用于科研文档多跳推理的工作。

---

## 十一、和方向 2 的潜在融合

如果后续想融合方向 2（inference-time visual re-alignment）：

- **在 Stage 1 中引入 entropy-based 自适应抽取**：当模型对某个图表的 claim 抽取 confidence 低时（高 entropy），自动触发高分辨率重新解析
- **Evidence Memory**：图表的 claim 缓存就是天然的 evidence memory buffer
- 这样方向 2 的 selective re-examination 思路可以自然嵌入方向 5.3 的 pipeline 中

但建议**先把 5.3 单独做完做实**，再考虑融合。
