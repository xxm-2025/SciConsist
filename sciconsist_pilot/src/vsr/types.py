"""
VSR 核心数据类型

定义 Claim、Table、Verification 相关的结构化类型，
所有 VSR 组件共享这些类型以保证接口一致性。
"""

from __future__ import annotations

import enum
import re
from dataclasses import dataclass, field
from typing import Optional


class VerificationLayer(enum.IntEnum):
    """验证层级"""
    SYMBOLIC = 0       # Layer 0: 精确数值验证
    SEMI_SYMBOLIC = 1  # Layer 1: 关系/趋势验证
    LEARNED = 2        # Layer 2: 语义判断


@dataclass
class NumericValue:
    """从文本中提取的数值

    Attributes:
        raw: 原始文本片段, e.g. "85.3%"
        value: 归一化数值, e.g. 85.3
        unit: 单位, e.g. "%"
        is_percentage: 是否百分比
    """
    raw: str
    value: float
    unit: str = ""
    is_percentage: bool = False


_ENTITY_ALIASES = {
    'our method': 'ours', 'our model': 'ours', 'our approach': 'ours',
    'our system': 'ours', 'proposed method': 'ours', 'proposed model': 'ours',
}


def normalize_entity(name: str) -> str:
    """实体名归一化: 别名替换 + 小写去空格

    Args:
        name: 原始实体名

    Returns:
        归一化后的名字
    """
    lower = name.lower().strip()
    alias = _ENTITY_ALIASES.get(lower)
    if alias:
        return alias
    return re.sub(r'[\s\-_]+', '', lower)


@dataclass
class EntityMention:
    """实体提及 (方法名/模型名/数据集)

    Attributes:
        name: 实体名称, e.g. "BERT-base"
        normalized: 归一化名 (小写去空格+别名), 用于模糊匹配
        entity_type: 类型标签 (method/dataset/metric), 可选
    """
    name: str
    normalized: str = ""
    entity_type: str = ""

    def __post_init__(self) -> None:
        if not self.normalized:
            self.normalized = normalize_entity(self.name)


@dataclass
class RelationAssertion:
    """比较关系断言

    Attributes:
        entity_a: 主体实体
        entity_b: 比较对象
        relation: 关系类型 (gt/lt/eq/gte/lte)
        metric: 关联指标 (可选)
        raw_text: 原始表述
    """
    entity_a: str
    entity_b: str
    relation: str  # "gt", "lt", "eq", "gte", "lte"
    metric: str = ""
    raw_text: str = ""


@dataclass
class TrendAssertion:
    """趋势断言

    Attributes:
        entity: 涉及的实体/变量
        direction: 趋势方向 ("increase", "decrease", "stable", "saturate")
        condition: 条件描述 (e.g. "as model size grows")
        raw_text: 原始表述
    """
    entity: str
    direction: str  # "increase", "decrease", "stable", "saturate"
    condition: str = ""
    raw_text: str = ""


@dataclass
class AtomicClaim:
    """一条原子事实声明 — ClaimExtractor 的输出单元

    一个 response 可能包含多条 AtomicClaim，每条可独立验证。

    Attributes:
        text: 原始 claim 文本
        numeric_values: 提取到的数值列表
        entities: 提取到的实体列表
        metrics: 提取到的指标名列表
        relations: 提取到的比较关系列表
        trends: 提取到的趋势断言列表
        is_qualitative: 纯定性描述标记
        source_span: 在 response 中的字符偏移 (start, end)
    """
    text: str
    numeric_values: list[NumericValue] = field(default_factory=list)
    entities: list[EntityMention] = field(default_factory=list)
    metrics: list[str] = field(default_factory=list)
    relations: list[RelationAssertion] = field(default_factory=list)
    trends: list[TrendAssertion] = field(default_factory=list)
    is_qualitative: bool = False
    source_span: tuple[int, int] = (0, 0)


@dataclass
class VerifiabilityProfile:
    """单条 claim 的可验证性特征 — Router 的判定结果

    Attributes:
        has_numeric: 含具体数值/百分比
        has_entity_metric: 含 (实体, 指标) 对
        has_comparison: 含 A > B 等比较关系
        has_trend: 含趋势描述
        is_qualitative_only: 纯定性描述
        assigned_layers: 应路由到的验证层列表 (一个 claim 可同时触发多层)
    """
    has_numeric: bool = False
    has_entity_metric: bool = False
    has_comparison: bool = False
    has_trend: bool = False
    is_qualitative_only: bool = False
    assigned_layers: list[VerificationLayer] = field(default_factory=list)

    @property
    def layer0_eligible(self) -> bool:
        return self.has_numeric and self.has_entity_metric

    @property
    def layer1_eligible(self) -> bool:
        return self.has_comparison or self.has_trend

    @property
    def best_layer(self) -> VerificationLayer:
        if self.layer0_eligible:
            return VerificationLayer.SYMBOLIC
        if self.layer1_eligible:
            return VerificationLayer.SEMI_SYMBOLIC
        return VerificationLayer.LEARNED


@dataclass
class TableRecord:
    """结构化表格中的一条记录 — 从 HTML table 解析得到

    Attributes:
        entity: 行实体 (方法名/条件名)
        metric: 列指标名
        value: 数值
        unit: 单位
        raw_text: 原始 cell 文本
        row_idx: 原始行索引
        col_idx: 原始列索引
    """
    entity: str
    metric: str
    value: Optional[float]
    unit: str = ""
    raw_text: str = ""
    row_idx: int = -1
    col_idx: int = -1

    @property
    def entity_normalized(self) -> str:
        return re.sub(r'[\s\-_]+', '', self.entity.lower())

    @property
    def metric_normalized(self) -> str:
        return re.sub(r'[\s\-_]+', '', self.metric.lower())


@dataclass
class StructuredTable:
    """一张完整的结构化表格

    Attributes:
        paper_id: 论文 ID
        table_id: 表格编号 (论文内)
        caption: 表格标题
        headers: 列头列表 (可能多级)
        records: 解析出的所有 (entity, metric, value) 记录
        raw_html: 原始 HTML (用于 debug)
    """
    paper_id: str
    table_id: str
    caption: str = ""
    headers: list[str] = field(default_factory=list)
    records: list[TableRecord] = field(default_factory=list)
    raw_html: str = ""

    def lookup(
        self, entity: str, metric: str, fuzzy_threshold: float = 0.6
    ) -> Optional[TableRecord]:
        """根据 entity + metric 查找最佳匹配记录

        对所有记录计算综合匹配分, 返回最高分记录。
        实体权重 0.7, 指标权重 0.3 (实体区分度更高)。

        Args:
            entity: 查找的实体名
            metric: 查找的指标名
            fuzzy_threshold: 最低匹配阈值

        Returns:
            匹配的 TableRecord, 无匹配时返回 None
        """
        entity_norm = normalize_entity(entity)
        metric_norm = re.sub(r'[\s\-_]+', '', metric.lower())

        best_rec = None
        best_score = 0.0

        for rec in self.records:
            e_sim = _levenshtein_similarity(entity_norm, rec.entity_normalized)
            m_sim = (
                _levenshtein_similarity(metric_norm, rec.metric_normalized)
                if metric_norm else 0.5
            )
            score = e_sim * 0.7 + m_sim * 0.3
            if score > best_score:
                best_score = score
                best_rec = rec

        if best_score >= fuzzy_threshold:
            return best_rec
        return None


@dataclass
class VerificationResult:
    """单次验证的结果

    Attributes:
        layer: 执行验证的层级
        reward: 验证得分 [-1.0, 1.0]
        confidence: 验证置信度 [0.0, 1.0]
        matched_evidence: 匹配到的证据描述 (可选)
        details: 额外信息 (debug 用)
    """
    layer: VerificationLayer
    reward: float
    confidence: float = 1.0
    matched_evidence: str = ""
    details: dict = field(default_factory=dict)

    @property
    def is_high_confidence(self) -> bool:
        """是否高置信度 (Layer 0 生效条件)"""
        return self.confidence >= 0.9


# ── 工具函数 ────────────────────────────────────────────────────

def _levenshtein_similarity(s1: str, s2: str) -> float:
    """Levenshtein 相似度 (0~1)"""
    if not s1 and not s2:
        return 1.0
    if not s1 or not s2:
        return 0.0
    max_len = max(len(s1), len(s2))
    dist = _levenshtein_distance(s1, s2)
    return 1.0 - dist / max_len


def _levenshtein_distance(s1: str, s2: str) -> int:
    """Levenshtein 编辑距离"""
    if len(s1) < len(s2):
        return _levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    prev_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = prev_row[j + 1] + 1
            deletions = curr_row[j] + 1
            substitutions = prev_row[j] + (c1 != c2)
            curr_row.append(min(insertions, deletions, substitutions))
        prev_row = curr_row
    return prev_row[-1]
