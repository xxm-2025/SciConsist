# Inference-time Visual Re-alignment with Memory-aware Reasoning

> 研究方向初步计划 | 生成日期: 2026-04-09

---

## 一、文献全景：这个赛道现在长什么样

### 1.1 核心问题已被广泛确认

你描述的"感知疲劳"/"视觉对齐丢失"已经是 2025-2026 年多模态推理领域的一个 **公认热点问题**。多篇论文给它起了不同的名字：

| 论文 | 术语 | 核心发现 |
|------|------|----------|
| **Deeper Thought, Weaker Aim** (2603.14184) | Perceptual Impairment / Attention Dispersion | 推理模式下视觉注意力分散，偏离关键区域 |
| **Learning When to Look** (2512.17227) | Visual Forgetting / "Think Longer, See Less" | 训练范式过早纠缠推理与感知，导致冷启动缺陷 |
| **SPARC** (2602.06566) | Perception-Reasoning Entanglement | 非结构化 CoT 中感知错误级联放大 |
| **VisRef** (2603.00207, CVPR 2026) | Visual Token Attention Decay | 长文本推理中视觉 token 注意力持续衰减 |

**结论：问题的存在性和重要性不需要你再论证，这是好事——说明 community 已经认可这个问题。**

### 1.2 现有解法的三大流派

#### 流派 A：Training-free / Plug-and-play（与你的想法最近）

| 方法 | 机制 | 优点 | 局限 |
|------|------|------|------|
| **VRGA** (Deeper Thought, Weaker Aim) | 基于 entropy-focus 准则选择视觉注意力头，重新加权注意力 | Training-free；有 entropy 信号 | 只做注意力重加权，不做区域级重新解析；无记忆机制 |
| **VisRef** (CVPR 2026) | 在推理过程中重新注入语义相关的视觉 token coreset | Training-free；不需要 RL | **总是**注入，非选择性触发；无 entropy 门控；无记忆 |
| **DeepScan** (2603.03857) | 分层扫描 + Refocusing + 证据增强推理 | 有 evidence memory；分层 | 完整 pipeline 而非轻量模块；非推理步级动态触发 |
| **See It, Say It, Sorted** (2602.21497) | 每步推理用视觉证据监督 | 迭代式 | 需要额外 evidence pool 构建 |

#### 流派 B：Latent Space / 架构修改

| 方法 | 机制 |
|------|------|
| **VaLR** (2602.04476) | 每步推理前动态生成 vision-aligned latent tokens |
| **HIVE** | 分层视觉线索注入到 latent space |
| **SPARC** (2602.06566) | 两阶段 perception-reasoning 解耦，200× token 压缩 |

#### 流派 C：Training / RL-based

| 方法 | 机制 |
|------|------|
| **PG-CoT** (Learning When to Look) | RL + Pivotal Perception Reward，学习"何时感知" |
| **VAPO** | Vision-anchored policy optimization |
| **PaLMR** (2603.06652) | 过程级对齐 + 感知对齐数据 |

### 1.3 记忆机制的相关工作

| 方法 | 记忆架构 | 与推理的关系 |
|------|----------|-------------|
| **MemOCR** (2601.21468) | 布局感知视觉记忆，自适应分配记忆空间 | 长程推理 |
| **MemVerse** (2512.03627) | 短期 + 层级长期记忆 + 知识图谱 + 周期蒸馏 | Lifelong learning |
| **MM-Mem** | 三层：Sensory Buffer → Episodic Stream → Symbolic Schema | Fuzzy-Trace Theory 启发 |
| **DeepScan** | Hybrid evidence memory（多粒度视图聚合） | 视觉推理中的证据缓存 |

### 1.4 Entropy-based 自适应计算

| 方法 | entropy 的用途 |
|------|---------------|
| **SAGE** (2602.00523) | 输出 entropy 控制投机解码树结构 |
| **EAD** | 滚动 entropy 动态切换大小模型 |
| **ThinkingViT** | 逐步 confidence 做 early exit |
| **VRGA** | Entropy-focus 准则选择注意力头 |

---

## 二、Gap Analysis：你的想法还能做吗？

### 2.1 坦率评估

**坏消息：**
- "推理中检测视觉对齐丢失 → 重新对齐"的 **大框架已被充分探索**
- VisRef（CVPR 2026 已中）做的就是推理时重新注入视觉 token——和你的"二次对齐"很像
- VRGA 用 entropy 做注意力头选择——和你的"token entropy 触发"很像
- DeepScan 已经有 evidence memory——和你的 memory buffer 很像

**好消息 / 仍然存在的空白：**

| Gap | 为什么还没人做 | 潜在贡献 |
|-----|---------------|---------|
| **① 没有人统一 entropy 触发 + memory buffer + 区域级重解析** | 现有方法各做了其中一部分，但没人把三件事闭环 | 唯一一个同时解决 when/what/how 三个问题的轻量框架 |
| **② "要不要重看"的决策粒度太粗** | VisRef 总是注入；VRGA 全局重加权；PG-CoT 需要 RL 训练 | 基于 entropy 的连续控制信号，按需、按区域、按粒度触发 |
| **③ 重复检视的冗余问题** | 没有方法记录"已经确认过的视觉证据"来避免重复计算 | Memory buffer 避免对同一区域反复 re-examine |
| **④ 语义名词 ↔ 视觉特征的显式匹配** | 现有方法多在 attention level 操作，不做 explicit semantic-visual alignment | 比 attention 重加权更精细、更可解释 |

### 2.2 结论

**方向可以做，但必须大幅差异化。** 如果只是"entropy 检测 + 重新看图"，会被审稿人判为 VisRef/VRGA 的 incremental combination。需要找到一个独特的叙事角度。

---

## 三、推荐的研究定位（3 个候选角度）

### 角度 A（推荐）：Selective Visual Re-examination with Evidence Memory

**核心 narrative**：现有方法要么"总是重看"（VisRef），要么"全局重加权"（VRGA），要么需要 RL 训练（PG-CoT）。我们提出第一个 **按需、按区域、有记忆** 的 training-free 推理时视觉重新检视框架。

**三个技术贡献：**

1. **Entropy-Gated Re-examination Trigger**
   - 监控推理过程中每步的 token entropy 和语义-视觉余弦相似度
   - 当两个信号同时异常时，触发区域级视觉重新解析
   - 区别于 VisRef（always inject）和 VRGA（global reweight）

2. **Evidence Memory Buffer**
   - 维护一个轻量的 key-value memory：key = 语义概念，value = 已确认的视觉证据（区域特征 + 置信度）
   - 重新检视前先查 memory，如果已有高置信度证据则跳过
   - 随推理链更新，支持跨步推理

3. **Adaptive Granularity Selection**
   - 根据 entropy 的严重程度决定重新解析的粒度：低 → 只查 memory；中 → 重新 attend 已有 token；高 → 裁剪高分辨率区域重新编码
   - 形成一个从轻到重的三级干预阶梯

**为什么新：** 第一个在 training-free 设定下同时解决 when（entropy gate）、what（semantic-visual matching）、whether（memory check）三个决策的框架。

**目标会议：** ICLR 2027 / ACL 2026 / NeurIPS 2026

### 角度 B：Reasoning-Aware Visual Token Budget Controller

把问题重新 frame 为"推理过程中的视觉 token 预算分配"：
- 每步推理有一个动态 token budget
- Entropy 高 → 分配更多视觉 token（重新裁剪高分辨率区域）
- Entropy 低 → 压缩或跳过视觉 token
- Memory 用来缓存已解析的视觉 token 避免重复计算
- 更偏效率方向，和 SPARC 的 200× 压缩形成对话

### 角度 C：Graph-Structured Visual Evidence Tracking

- 用轻量图结构跟踪推理链中每个语义实体和对应的视觉证据
- 节点 = 语义概念，边 = 空间/语义关系
- 推理每步检查图中未被视觉支撑的节点 → 触发重新检视
- 更像 MemVerse/MM-Mem 的方向，新颖性强但实现复杂度高
- **你提到的 "simple graph" 对应这个角度**

---

## 四、关键开放问题

1. **Entropy 阈值怎么定？** 是固定阈值还是自适应？现有工作（SAGE、EAD）都用不同策略，需要实验验证哪种最稳健。

2. **"语义名词"怎么提取？** 从推理中间 token 中提取关键语义名词是一个非平凡的问题。可以用 NER？attention pattern？还是直接用 LLM 自己的 hidden state？

3. **Memory buffer 的容量和淘汰策略？** 推理链越长 memory 越大，需要 eviction policy（LRU？confidence-based？）

4. **高分辨率区域重新编码的计算代价？** 这是最重的操作，需要确保不会让推理变得过慢。需要测量 overhead vs. 收益的 trade-off。

5. **评估指标？** 不能只看最终准确率。需要量化：(a) 视觉对齐恢复程度，(b) 不必要重检视的比例（precision），(c) 遗漏的需要重检视的比例（recall），(d) 推理延迟增加。

6. **跨模型泛化性？** 作为 plug-and-play 方法，必须在 Qwen2.5-VL、LLaVA、InternVL 等多个 backbone 上验证。

---

## 五、具体推进步骤（4 周计划）

### Week 1：Pilot Experiment（验证前提假设）

**目标：** 用最小实验验证"推理中确实存在可检测的视觉对齐退化"

- [ ] 选择一个开源 MLLM（推荐 Qwen2.5-VL-7B，社区活跃 + 性能强）
- [ ] 在 MathVista / V*Bench 上运行 CoT 推理，收集：
  - 每步 token 的 entropy 分布
  - 视觉 attention map 的变化
  - 最后一层 hidden state 中语义 token 与视觉 token 的余弦相似度
- [ ] 可视化并确认：entropy spike 是否与视觉对齐下降相关？
- [ ] 复现 VRGA 的 entropy-focus 分析作为 baseline

**产出：** 一组诊断图表，确认/否认核心假设

### Week 2：Prototype 实现

**目标：** 实现最小可行版本

- [ ] 实现 Entropy-Gated Trigger：
  - Hook 进模型的 forward pass
  - 在每个推理步后计算 token entropy
  - 当 entropy 超过阈值时标记需要重新检视
- [ ] 实现 Naive Re-examination：
  - 简单方案：裁剪注意力最高的区域，重新通过 vision encoder
  - 将重新编码的 token 注入后续推理
- [ ] 在 3-5 个 case 上定性验证效果

**产出：** 可运行的 prototype（不含 memory）

### Week 3：Memory Buffer + 消融实验

**目标：** 加入 memory，开始系统实验

- [ ] 实现 Evidence Memory Buffer：
  - Key: 语义概念 embedding
  - Value: 视觉区域特征 + 置信度 + 被引用次数
  - 查询：当触发重检视时，先用语义 key 查 memory
- [ ] 设计消融实验：
  - Baseline: 原始 CoT
  - +Entropy Gate only
  - +Entropy Gate + Re-examination
  - +Entropy Gate + Re-examination + Memory
- [ ] 在 MathVista, MMStar, V*Bench 上全面评估

**产出：** 消融表格 + 初步结果

### Week 4：分析 + 写作准备

**目标：** 完善实验 + 锁定 narrative

- [ ] 跨模型实验（Qwen2.5-VL-7B, LLaVA-OneVision-7B, InternVL2-8B）
- [ ] Overhead 分析（推理延迟、显存增加）
- [ ] 和 VisRef / VRGA / DeepScan 做对比实验
- [ ] 可视化 case study：展示 memory 避免冗余重检视的案例
- [ ] 开始 paper draft：Related Work + Method 初稿

---

## 六、核心文献清单（必读 Top 10）

| # | 论文 | 日期 | 与你的关系 |
|---|------|------|-----------|
| 1 | **VisRef**: Visual Refocusing while Thinking (2603.00207) | 2026.02, CVPR 2026 | **最直接竞争者**，推理时重注入视觉 token |
| 2 | **Deeper Thought, Weaker Aim** (2603.14184) | 2026.03 | VRGA 用 entropy 做注意力重加权，**技术最相关** |
| 3 | **Learning When to Look** / PG-CoT (2512.17227) | 2025.12 | RL-based "何时感知"，是 training-based 的上界参照 |
| 4 | **SPARC** (2602.06566) | 2026.02 | 感知-推理解耦，框架设计参考 |
| 5 | **VaLR** (2602.04476) | 2026.02 | Vision-aligned latent tokens，latent space 方案参考 |
| 6 | **DeepScan** (2603.03857) | 2026.03 | 分层扫描 + evidence memory，**memory 设计参考** |
| 7 | **UniT** (2602.12279) | 2026.02 | Meta 的统一多模态 CoT test-time scaling |
| 8 | **SAGE** (2602.00523) | 2026.02 | Entropy-guided 自适应解码，**entropy 用法参考** |
| 9 | **MemOCR** (2601.21468) | 2026.01 | 布局感知视觉记忆，memory 架构参考 |
| 10 | **See It, Say It, Sorted** (2602.21497) | 2026.02 | 迭代式视觉证据监督，training-free |

---

## 七、风险评估

| 风险 | 等级 | 应对 |
|------|------|------|
| VisRef/VRGA 已经覆盖了核心贡献 | 🔴 高 | 必须在 memory + 选择性触发上做出明确差异化 |
| 高分辨率重编码太慢 | 🟡 中 | 设计三级干预阶梯，只在必要时做重编码 |
| Entropy 信号不可靠 | 🟡 中 | Week 1 pilot 验证；备选方案用 attention entropy |
| 跨模型泛化差 | 🟡 中 | 设计为 plug-and-play，在 3+ backbone 上验证 |
| 赛道太拥挤，审稿人疲劳 | 🟡 中 | Narrative 要突出 memory + selective，避免和已有工作撞脸 |

---

## 八、如果这个方向不行，备选方向

### 备选 1：Multimodal Reasoning with Explicit Uncertainty Quantification
- 不做重对齐，而是在推理过程中为每个视觉 claim 标注不确定性
- 最终输出附带视觉 evidence 的 confidence map
- 偏 trustworthy AI 方向，competition 较少

### 备选 2：Efficient Visual Token Scheduling for Long-Context MLLM
- 关注超长上下文（视频/多图/文档）场景下视觉 token 的调度
- 和 SPARC 的 token 压缩方向对话，但聚焦于 scheduling 而非 pruning
- 更偏系统/效率方向

### 备选 3：Cross-Modal Consistency Verification during Reasoning
- 不做 re-alignment，而是做 verification
- 在推理链的每步验证文本 claim 和视觉 evidence 的一致性
- 类似 self-consistency 但跨模态
- 和 PaLMR 的 process-level alignment 方向对话
