# SciConsist FEH 讨论结论：采纳与消融并进执行稿

> 更新时间：2026-04-13  
> 目的：把当前讨论转为可执行实验路线与论文叙事资产

---

## 1. 核心判断

当前方向具备明确价值：从 surface-level 相似度转向 representation-level factual entailment，并增加冲突检测奖励，能直接回应 cycle consistency 的 Many-to-One 退化问题。

下一阶段的关键不是继续堆 RL 技巧，而是补齐 FEH 的可信证据链：
- 数据是否抗质疑（避免人工痕迹捷径）
- Claim 拆解是否可控可解释
- 奖励是否稳定且不压制有效新观察
- Cross-Model 是否能应对“共错”场景

---

## 2. 固定设计（建议采纳为主方法）

这些建议作为主方法固定，不作为可有可无选项。

### 2.1 Hard Negatives 升级为逻辑负样本族

在现有“数值篡改 ±20%”之外，至少加入以下负样本类型：
- Trend Flip：上升/下降趋势反转
- Entity Swap：模型或方法名对调（如 A 最优改写为 B 最优）
- Unit Mismatch：单位冲突（ms vs s、% vs fraction）
- Magnitude Distortion：数量级错误（10x、100x）
- Range Violation：超出图中可见范围

数据构造要求：
- 基于 OCR 提取图内文本，再由 LLM 生成“专业但错误”的 claim
- 每张图生成多种语义难负样本，避免模型仅靠图像篡改痕迹取巧

### 2.2 Claim Decomposition 显式化

在 policy 输出中要求显式 Observation 段，先给原子事实再给最终回答。推荐格式：
- 最多 3-5 条 atomic claims
- 每条含 entity、metric、value/trend、unit（如有）

说明：
- 不依赖隐藏思维链标签
- 便于 FEH 逐条打分
- 降低后处理解析复杂度，提升可解释性与复现性

### 2.3 OCR-Injected FEH（显式证据通道）

FEH 输入改为三路：
- 图像 patch/region 特征
- OCR token 序列
- claim token 序列

最小可行版本可先采用轻量融合（concat + gating），不强制第一版就上重 Cross-Attention。

### 2.4 Anti-Shortcut 验证（必须加入）

为防 FEH 学捷径，固定加入以下诊断：
- 仅 OCR（去图像）
- 仅图像（去 OCR）
- 坐标轴文本打乱
- 单位符号扰动

如果某一路单独也能异常高分，需要在方法中主动承认并修正。

---

## 3. 消融矩阵（建议由数据决定）

这些是论文中的核心 A/B 对比项。

### 3.1 模型解耦
- Cross-Model FEH vs Same-Model FEH
- 目标：验证物理隔离是否实质抑制 reward hacking

### 3.2 标签体系
- Tri-class（ENTAILS/NEUTRAL/CONTRADICTS）vs Bi-class
- 目标：量化 NEUTRAL 对误伤率与多样性退化的影响

### 3.3 FEH 架构
- MLP concat baseline vs Cross-Attention Anchor
- 目标：验证细粒度判定是否受益于 token-level anchoring

### 3.4 Reward 形式与权重
- 固定权重 vs Specificity-weighted
- CONTRADICTS 惩罚强度：-0.5 vs -1.0
- 冲突检测 bonus 系数 λ_conflict sweep

---

## 4. Reward 稳定性改造建议

原始固定权重存在“压制创新观察”风险，建议改为：

R(c) = w_label(c) * S(c) * C(c)

其中：
- S(c)：Evidence Specificity（实体、数值、单位、比较关系的可核对程度）
- C(c)：校准后置信度（temperature-scaled confidence）

执行要点：
- ENTAILS 但模糊描述，得分接近 0
- NEUTRAL 默认接近 0，而非固定负值
- 对“具体但无证据”才轻度惩罚

目标：鼓励模型输出可核查事实，而不是安全废话。

---

## 5. 关键质疑回应：Cross-Model 在共错场景会否失效

结论：会有失效风险。Cross-Model 可降低可操纵性，但不能保证真值独立性。

建议采用“异质审计三角”：
- Learning-based：FEH
- Rule-based：OCR + 数值/单位/实体一致性规则
- Symbolic optional：图表解析/程序化核验（高风险样本复核）

聚合策略建议：
- 常规样本：线性融合
- 高风险冲突样本：保守聚合（min 或 veto）

这将显著增强 epistemic reliability 叙事可信度。

---

## 6. 叙事升级：Epistemic Consistency + Self-Correction Loop

建议将论文主线从“视觉对齐”升级为“事实一致性闭环”：
1. 识别命题是否被证据支持（entailment）
2. 主动检测跨模态冲突（audit）
3. 根据冲突反馈修复回答（self-correction）

新增实验（建议必做）：
- Round 1：模型输出 + 冲突声明
- Round 2：给冲突反馈后重答

报告指标：
- Correction Rate
- Over-correction Rate
- Final Faithfulness Gain

---

## 7. 一周最小可执行实验包（MVP）

### 7.1 数据与模型
- 增加三类 hard negatives：trend/entity/unit（每类先 5k）
- 输出格式改为 structured observations（最多 3-5 条）
- FEH 加入 OCR 通道（先轻量融合）

### 7.2 训练与评估
- 采用 specificity-weighted reward（第一版不引入过多 RL 技巧）
- 建立共错诊断集（100-200 条冷门图表）
- 跑四组关键对比：
  - A：当前版本
  - B：A + hard negatives
  - C：B + structured claims
  - D：C + specificity reward

### 7.3 Go/No-Go 门槛建议
- FEH tri-class accuracy >= 75%
- 高冲突集 FPR@1% < 5%
- 相比 A 组，D 组在 faithfulness 与 diversity 至少一项显著提升且无明显退化

---
其他意见：
Conflict Detection 应该成为真正的 main contribution，而不是 FEH。FEH 可以降级为 "our reward design"，但 "主动审计、检测跨模态冲突" 这个故事更有吸引力、更独特。应该围绕这个讲故事，而不是围绕 "替代 surface matching"。
FEH 预训练数据中的 NEUTRAL 类需要重新构造。用"随机配对"来代表 NEUTRAL 太粗糙了。应该用：(a) 同一篇论文中不相邻段落的 figure-text 对（相关但不直接对应），(b) paraphrase 后的 text（内容相同但表述不同）。
泛化到非科研场景。至少在实验中加一个非科研 benchmark（比如 InfoGraphicsVQA、Chart-to-Text），否则 reviewer 会觉得这只是一个 domain-specific trick。
75-85% 的估计降一降预期。做好投 Findings 的心理准备。11 个消融和 6 个评估集的工作量是 Findings 级别的扎实度，但 novelty 撑不起 Main。
