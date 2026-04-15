"""
Claim Extractor — 从 model response 中提取 atomic factual claims

采用轻量规则提取，不依赖外部 LLM。

核心策略:
  1. 句子级分割 (sentence splitting)
  2. 对每个句子提取: 数值、实体/指标、比较关系、趋势
  3. 过滤非事实句 (纯 meta 描述如 "Let me analyze...")

设计选择: 规则优先
  - 训练阶段每秒处理数千个 response，不能调 LLM
  - 科研文档回答结构化程度高，规则覆盖率足够
  - LLM fallback 仅在评估/标注时使用 (不在此模块)
"""

from __future__ import annotations

import re
from typing import Optional

from sciconsist_pilot.src.vsr.types import (
    AtomicClaim,
    NumericValue,
    EntityMention,
    RelationAssertion,
    TrendAssertion,
    normalize_entity,
)

# ── 正则模式 ────────────────────────────────────────────────────

# 数值: 85.3%, 0.853, 85.3 dB, p < 0.05, ±0.2
_RE_NUMERIC = re.compile(
    r'(?P<value>[+-]?\d+\.?\d*)\s*(?P<unit>%|‰|dB|ms|s|nm|mm|cm|Hz|kHz|MHz|GHz|eV|keV|MeV|K|M|B)?'
    r'(?:\s*[±]\s*\d+\.?\d*)?'
)

_RE_PERCENTAGE = re.compile(r'(\d+\.?\d*)\s*(%|percent)')

# 指标名
_METRIC_NAMES_LONG = {
    'accuracy', 'precision', 'recall', 'f1-score', 'f1 score',
    'bleu', 'rouge', 'rouge-1', 'rouge-2', 'rouge-l', 'meteor', 'cider',
    'perplexity', 'loss', 'auroc',
    'miou', 'psnr', 'ssim',
    'error rate', 'success rate', 'hit rate', 'win rate',
    'exact match', 'top-1', 'top-5',
    'mse', 'mae', 'rmse',
    'throughput', 'latency', 'fps', 'flops',
    'kappa', "cohen's kappa",
}

# 短名需要大写匹配 (避免 "is"、"em" 误匹配常见英文词)
_METRIC_NAMES_SHORT_UPPER = {
    'F1', 'AUC', 'MAP', 'AP', 'IoU', 'FID', 'IS',
    'PPL', 'EM', 'ACC', 'R2',
    'JSR', 'ASR', 'FPS', 'FLOPS',
}

_RE_METRIC_LONG = re.compile(
    r'\b(' + '|'.join(re.escape(m) for m in sorted(_METRIC_NAMES_LONG, key=len, reverse=True)) + r')\b',
    re.IGNORECASE,
)

_RE_METRIC_SHORT = re.compile(
    r'\b(' + '|'.join(re.escape(m) for m in sorted(_METRIC_NAMES_SHORT_UPPER, key=len, reverse=True)) + r')\b',
)

# 比较关系
_COMPARISON_PATTERNS = [
    # "A outperforms/surpasses B" — b 组限制为单个实体词 (不吃介词)
    (re.compile(
        r'(?P<a>[A-Z][\w\-\.]*)\s+'
        r'(?:outperform|surpass|exceed|beat)s?\s+'
        r'(?P<b>[A-Z][\w\-\.]*)',
        re.IGNORECASE,
    ), 'gt'),
    # "A achieves higher/lower X than B"
    (re.compile(
        r'(?P<a>[A-Z][\w\-\.]*)\s+'
        r'(?:achieve|obtain|get|reach|show)s?\s+'
        r'(?:(?:significantly|much|slightly|marginally)\s+)?'
        r'(?P<dir>higher|lower|better|worse)\s+'
        r'(?:\w+\s+)?'
        r'than\s+'
        r'(?P<b>[A-Z][\w\-\.]*)',
        re.IGNORECASE,
    ), 'dynamic'),
    # "A > B", "A < B"
    (re.compile(
        r'(?P<a>[A-Z][\w\-\.]*)\s*'
        r'(?P<op>[><≥≤]=?)\s*'
        r'(?P<b>[A-Z][\w\-\.]*)',
    ), 'operator'),
    # "compared to B, A is higher"
    (re.compile(
        r'compared\s+(?:to|with)\s+(?P<b>[A-Z][\w\-\.]*)\s*,?\s*'
        r'(?P<a>[A-Z][\w\-\.]*)\s+'
        r'(?:is|are|was|were|shows?)\s+'
        r'(?:(?:significantly|much|slightly)\s+)?'
        r'(?P<dir>higher|lower|better|worse|superior|inferior)',
        re.IGNORECASE,
    ), 'dynamic'),
]

# 趋势
_TREND_PATTERNS = [
    (re.compile(r'\b(?:increas|improv|grow|rise|climb|ascend)\w*', re.I), 'increase'),
    (re.compile(r'\b(?:decreas|declin|drop|fall|shrink|degrad|diminish)\w*', re.I), 'decrease'),
    (re.compile(r'\b(?:plateau|saturat|stabiliz|converg|level\s+off)\w*', re.I), 'saturate'),
    (re.compile(r'\b(?:remain|stay|maintain)\w*\s+(?:stable|constant|unchanged)', re.I), 'stable'),
]

_TREND_CONDITION = re.compile(
    r'(?:as|when|with)\s+(?:the\s+)?(\w+(?:\s+\w+)?)\s+'
    r'(?:increas|decreas|grow|is\s+scaled|goes\s+up|goes\s+down)\w*',
    re.IGNORECASE,
)

# 过滤: 非事实句
_RE_META_SENTENCE = re.compile(
    r'^(?:let me|i will|i\'ll|first,?\s+let|to answer|based on|'
    r'looking at|examining|analyzing|in summary|in conclusion|'
    r'overall,?\s|to summarize|the (?:question|figure|table) (?:asks?|shows?)|'
    r'we can (?:see|observe|note))\b',
    re.IGNORECASE,
)


# ── ClaimExtractor ──────────────────────────────────────────────

class ClaimExtractor:
    """从 model response 中提取 atomic factual claims

    用法:
        extractor = ClaimExtractor()
        claims = extractor.extract("Method A achieves 85.3% accuracy...")
    """

    def extract(self, response: str) -> list[AtomicClaim]:
        """提取 response 中的所有 atomic claims

        Args:
            response: 模型输出的完整文本

        Returns:
            AtomicClaim 列表, 每条对应一个可独立验证的事实声明
        """
        sentences = self._split_sentences(response)
        claims: list[AtomicClaim] = []

        offset = 0
        for sent in sentences:
            start = response.find(sent, offset)
            end = start + len(sent) if start >= 0 else offset + len(sent)

            if self._is_meta_sentence(sent):
                offset = end
                continue

            claim = self._extract_from_sentence(sent, (start, end))
            if claim:
                claims.append(claim)
            offset = end

        return claims

    def _split_sentences(self, text: str) -> list[str]:
        """句子分割

        处理: 标准句号分割 + CoT step 分割 + 列表项分割
        """
        sentences: list[str] = []

        # step 格式: "Step 1: ...", "1. ...", "- ..."
        step_split = re.split(r'(?:^|\n)\s*(?:Step\s+\d+[.:]\s*|\d+[.)]\s+|[-•]\s+)', text)

        for chunk in step_split:
            chunk = chunk.strip()
            if not chunk:
                continue
            # 句号分割 (注意不切割小数点和缩写)
            parts = re.split(r'(?<=[.!?])\s+(?=[A-Z])', chunk)
            for p in parts:
                p = p.strip()
                if len(p) > 15:
                    sentences.append(p)

        return sentences

    def _is_meta_sentence(self, sentence: str) -> bool:
        """判断是否是 meta 描述（非事实性）"""
        return bool(_RE_META_SENTENCE.match(sentence.strip()))

    def _extract_from_sentence(
        self, sentence: str, span: tuple[int, int]
    ) -> Optional[AtomicClaim]:
        """从单个句子中提取 claim 结构"""
        numerics = self._extract_numerics(sentence)
        metrics = self._extract_metrics(sentence)
        entities = self._extract_entities(sentence, exclude_metrics=metrics)
        relations = self._extract_relations(sentence)
        trends = self._extract_trends(sentence)

        has_factual_content = (
            numerics or relations or trends
            or (entities and metrics)
        )
        if not has_factual_content:
            return None

        is_qualitative = not numerics and not relations and not trends

        return AtomicClaim(
            text=sentence,
            numeric_values=numerics,
            entities=entities,
            metrics=metrics,
            relations=relations,
            trends=trends,
            is_qualitative=is_qualitative,
            source_span=span,
        )

    def _extract_numerics(self, text: str) -> list[NumericValue]:
        """提取数值"""
        results: list[NumericValue] = []
        seen_values: set[float] = set()

        for m in _RE_NUMERIC.finditer(text):
            try:
                val = float(m.group('value'))
            except ValueError:
                continue
            unit = m.group('unit') or ''
            if val in seen_values:
                continue
            seen_values.add(val)

            is_pct = unit == '%' or unit == '‰'
            results.append(NumericValue(
                raw=m.group(0).strip(),
                value=val,
                unit=unit,
                is_percentage=is_pct,
            ))

        return results

    def _extract_entities(
        self, text: str, exclude_metrics: list[str] | None = None,
    ) -> list[EntityMention]:
        """提取实体名 (方法名/模型名)

        Args:
            text: 输入文本
            exclude_metrics: 排除已识别为 metric 的词 (避免 BLEU 同时是实体和指标)

        启发式: CamelCase, ALL_CAPS, 带版本号的词, 引号内的名字
        """
        entities: list[EntityMention] = []
        seen: set[str] = set()

        metric_norms = set()
        if exclude_metrics:
            metric_norms = {re.sub(r'[\s\-_]+', '', m.lower()) for m in exclude_metrics}

        patterns = [
            # CamelCase: "RoBERTa", "InternVL"
            re.compile(r'\b([A-Z][a-z]+[A-Z]\w*(?:-\w+)?)\b'),
            # ALL_CAPS with optional version: "BERT", "GPT-4o"
            re.compile(r'\b([A-Z]{2,}(?:-?\d[\w.]*)?(?:-[A-Za-z]+)?)\b'),
            # name-version: "Qwen2.5-VL-7B", "Llama-3"
            re.compile(r'\b([A-Z][a-z]+\d[\w.]*(?:-\w+)*)\b'),
            # "Ours", "Our method/model"
            re.compile(r'\b((?:Our|Proposed)\s+(?:method|model|approach|system))\b', re.I),
        ]

        # 排除词: 非实体 + metric 名
        stopwords = {'the', 'and', 'for', 'with', 'from', 'that', 'this',
                     'table', 'figure', 'fig', 'tab', 'section', 'step',
                     'method', 'results', 'performance', 'overall'}

        for pat in patterns:
            for m in pat.finditer(text):
                name = m.group(1).strip()
                norm = normalize_entity(name)
                if norm in seen or len(name) < 2:
                    continue
                if norm in stopwords or norm in metric_norms:
                    continue
                seen.add(norm)
                entities.append(EntityMention(name=name, normalized=norm))

        return entities

    def _extract_metrics(self, text: str) -> list[str]:
        """提取指标名"""
        found: list[str] = []
        seen: set[str] = set()
        for m in _RE_METRIC_LONG.finditer(text):
            name = m.group(1).lower()
            if name not in seen:
                seen.add(name)
                found.append(m.group(1))
        for m in _RE_METRIC_SHORT.finditer(text):
            name = m.group(1).lower()
            if name not in seen:
                seen.add(name)
                found.append(m.group(1))
        return found

    def _extract_relations(self, text: str) -> list[RelationAssertion]:
        """提取比较关系"""
        results: list[RelationAssertion] = []

        for pattern, rel_type in _COMPARISON_PATTERNS:
            for m in pattern.finditer(text):
                a = m.group('a').strip()
                b = m.group('b').strip()

                if rel_type == 'dynamic':
                    direction = m.group('dir').lower()
                    if direction in ('higher', 'better', 'superior'):
                        relation = 'gt'
                    else:
                        relation = 'lt'
                elif rel_type == 'operator':
                    op = m.group('op')
                    if '>' in op:
                        relation = 'gte' if '=' in op else 'gt'
                    else:
                        relation = 'lte' if '=' in op else 'lt'
                else:
                    relation = rel_type

                # 尝试关联 metric
                metrics = self._extract_metrics(text)
                metric = metrics[0] if metrics else ""

                results.append(RelationAssertion(
                    entity_a=a,
                    entity_b=b,
                    relation=relation,
                    metric=metric,
                    raw_text=m.group(0),
                ))

        return results

    def _extract_trends(self, text: str) -> list[TrendAssertion]:
        """提取趋势断言"""
        results: list[TrendAssertion] = []

        for pattern, direction in _TREND_PATTERNS:
            if pattern.search(text):
                # 尝试提取条件
                cond_match = _TREND_CONDITION.search(text)
                condition = cond_match.group(1) if cond_match else ""

                entities = self._extract_entities(text)
                entity = entities[0].name if entities else ""

                results.append(TrendAssertion(
                    entity=entity,
                    direction=direction,
                    condition=condition,
                    raw_text=text[:100],
                ))
                break  # 一个句子只取一个主要趋势

        return results
