"""
HTML Table → StructuredTable 解析器

从 SciMDR paper JSON 的 `tables` 字段中解析 HTML 表格为结构化记录。
支持多行 header、合并单元格、特殊符号清洗。

核心流程:
  1. BeautifulSoup 解析 HTML <table> 元素
  2. 处理 <thead>/<tbody>，分离 header 和 data rows
  3. 对每个 data cell，匹配 (row_entity, col_metric, value) 三元组
  4. 数值提取 + 单位归一化

用法:
  parser = TableParser()
  tables = parser.parse_paper(paper_json, paper_id="2409.xxxxx")
  for t in tables:
      rec = t.lookup("BERT", "accuracy")
"""

from __future__ import annotations

import re
from typing import Optional

from bs4 import BeautifulSoup, Tag

from sciconsist_pilot.src.vsr.types import (
    StructuredTable,
    TableRecord,
)

# ── 数值提取 ────────────────────────────────────────────────────

_RE_NUMBER = re.compile(
    r'([+-]?\d+\.?\d*)\s*(%|‰)?'
)

_RE_BOLD_MARKER = re.compile(r'[*]|\\textbf|\\mathbf|\\bf\b')

_UNIT_ALIASES = {
    '%': '%', '\\%': '%', 'percent': '%',
    'dB': 'dB', 'ms': 'ms', 's': 's',
}


def extract_numeric(text: str) -> tuple[Optional[float], str]:
    """从 cell 文本中提取数值和单位

    Args:
        text: 表格单元格文本, e.g. "85.3%", "0.853", "85.3 ± 0.2"

    Returns:
        (数值, 单位) 元组，提取失败时返回 (None, "")
    """
    cleaned = text.strip()
    cleaned = _RE_BOLD_MARKER.sub('', cleaned).strip()
    # 去掉 ± 后的部分 (保留主值)
    cleaned = re.sub(r'[±]\s*\d+\.?\d*', '', cleaned).strip()
    # 去掉括号内容 (通常是 std)
    cleaned = re.sub(r'\([^)]*\)', '', cleaned).strip()

    m = _RE_NUMBER.search(cleaned)
    if m:
        val = float(m.group(1))
        unit = m.group(2) or ''
        return val, unit
    return None, ""


# ── HTML 表格解析 ──────────────────────────────────────────────

class TableParser:
    """HTML 表格解析器

    将 SciMDR paper JSON 中的 HTML table 转为 StructuredTable。
    """

    def parse_paper(
        self,
        paper_json: dict,
        paper_id: str,
    ) -> list[StructuredTable]:
        """解析一篇论文的全部表格

        兼容两种 JSON 格式:
          - arxiv: paper_json["tables"] = {table_id: {"table_html": ..., "caption": ...}}
          - nature: paper_json["table_set"] = {table_id: {"html": ..., "caption": ...}}
                    + paper_json["supplementary_table_set"] (同结构)

        Args:
            paper_json: paper JSON 数据
            paper_id: 论文 ID

        Returns:
            该论文的所有 StructuredTable 列表
        """
        results: list[StructuredTable] = []

        # arxiv 格式
        tables_dict = paper_json.get("tables", {})
        if isinstance(tables_dict, dict):
            results.extend(self._parse_table_dict(tables_dict, paper_id, html_key="table_html"))

        # nature 格式
        for field in ("table_set", "supplementary_table_set"):
            ts = paper_json.get(field, {})
            if isinstance(ts, dict) and ts:
                results.extend(self._parse_table_dict(ts, paper_id, html_key="html"))

        return results

    def _parse_table_dict(
        self,
        tables_dict: dict,
        paper_id: str,
        html_key: str = "table_html",
    ) -> list[StructuredTable]:
        """从 table dict 解析表格，兼容不同的 HTML 字段名。

        Args:
            tables_dict: {table_id: table_data_dict}
            paper_id: 论文 ID
            html_key: HTML 内容的字段名 ("table_html" for arxiv, "html" for nature)

        Returns:
            解析出的 StructuredTable 列表
        """
        results: list[StructuredTable] = []
        for table_id, table_data in tables_dict.items():
            if not isinstance(table_data, dict):
                continue
            html = table_data.get(html_key, "") or table_data.get("table_html", "") or table_data.get("html", "")
            caption = table_data.get("caption", "") or table_data.get("capture", "") or table_data.get("label", "")
            if not html:
                continue

            st = self._parse_single_table(html, paper_id, table_id, caption)
            if st and st.records:
                results.append(st)
        return results

    def _parse_single_table(
        self,
        html: str,
        paper_id: str,
        table_id: str,
        caption: str,
    ) -> Optional[StructuredTable]:
        """解析单张 HTML 表格

        Args:
            html: HTML 字符串
            paper_id: 论文 ID
            table_id: 表格编号
            caption: 表格标题

        Returns:
            StructuredTable 或 None (解析失败时)
        """
        soup = BeautifulSoup(html, "html.parser")

        table_tag = soup.find("table")
        if not table_tag:
            return None

        rows = self._extract_rows(table_tag)
        if len(rows) < 2:
            return None

        header_rows, data_rows = self._split_header_data(rows)
        if not header_rows or not data_rows:
            return None

        headers = self._merge_headers(header_rows)
        records = self._build_records(headers, data_rows)

        return StructuredTable(
            paper_id=paper_id,
            table_id=table_id,
            caption=caption,
            headers=headers,
            records=records,
            raw_html=html[:2000],
        )

    def _extract_rows(self, table_tag: Tag) -> list[list[str]]:
        """从 <table> 中提取所有行, 处理 colspan/rowspan"""
        all_rows: list[list[str]] = []
        rowspan_carry: dict[int, tuple[str, int]] = {}

        for tr in table_tag.find_all("tr"):
            cells: list[str] = []
            col_idx = 0

            for td in tr.find_all(["td", "th"]):
                # 填充 rowspan 遗留
                while col_idx in rowspan_carry:
                    text, remaining = rowspan_carry[col_idx]
                    cells.append(text)
                    if remaining > 1:
                        rowspan_carry[col_idx] = (text, remaining - 1)
                    else:
                        del rowspan_carry[col_idx]
                    col_idx += 1

                text = self._clean_cell_text(td.get_text())
                colspan = int(td.get("colspan", 1))
                rowspan = int(td.get("rowspan", 1))

                for _ in range(colspan):
                    cells.append(text)
                    if rowspan > 1:
                        rowspan_carry[col_idx] = (text, rowspan - 1)
                    col_idx += 1

            # 填充尾部 rowspan
            while col_idx in rowspan_carry:
                text, remaining = rowspan_carry[col_idx]
                cells.append(text)
                if remaining > 1:
                    rowspan_carry[col_idx] = (text, remaining - 1)
                else:
                    del rowspan_carry[col_idx]
                col_idx += 1

            if cells:
                all_rows.append(cells)

        return all_rows

    def _split_header_data(
        self, rows: list[list[str]]
    ) -> tuple[list[list[str]], list[list[str]]]:
        """将行列表分为 header 行和 data 行

        启发式: 如果前几行都不含数字，视为 header；
        或者使用 <thead>/<tbody> 标记（已在 extract_rows 中展平）。
        """
        # 找到第一个含数值的行作为 data 起点
        data_start = 0
        for i, row in enumerate(rows):
            has_number = any(
                extract_numeric(cell)[0] is not None
                for cell in row[1:]  # 跳过第一列（通常是 entity 名）
            )
            if has_number and i > 0:
                data_start = i
                break

        if data_start == 0:
            data_start = 1

        return rows[:data_start], rows[data_start:]

    def _merge_headers(self, header_rows: list[list[str]]) -> list[str]:
        """合并多行 header 为单行

        对多行 header，将同一列的文本用 " / " 连接。
        """
        if not header_rows:
            return []
        if len(header_rows) == 1:
            return header_rows[0]

        max_cols = max(len(r) for r in header_rows)
        merged: list[str] = []
        for col_idx in range(max_cols):
            parts = []
            for row in header_rows:
                if col_idx < len(row) and row[col_idx].strip():
                    parts.append(row[col_idx].strip())
            # 去重相邻
            unique_parts = []
            for p in parts:
                if not unique_parts or unique_parts[-1] != p:
                    unique_parts.append(p)
            merged.append(" / ".join(unique_parts))
        return merged

    def _build_records(
        self,
        headers: list[str],
        data_rows: list[list[str]],
    ) -> list[TableRecord]:
        """从 header + data 行构建 TableRecord 列表

        每行第一列作为 entity，后续列按 header 对应 metric。
        """
        records: list[TableRecord] = []
        for row_idx, row in enumerate(data_rows):
            if not row:
                continue

            entity = row[0].strip()
            if not entity:
                continue

            for col_idx in range(1, len(row)):
                cell_text = row[col_idx].strip()
                if not cell_text or cell_text == '-' or cell_text == '—':
                    continue

                metric = headers[col_idx] if col_idx < len(headers) else f"col_{col_idx}"
                value, unit = extract_numeric(cell_text)

                records.append(TableRecord(
                    entity=entity,
                    metric=metric,
                    value=value,
                    unit=unit,
                    raw_text=cell_text,
                    row_idx=row_idx,
                    col_idx=col_idx,
                ))

        return records

    @staticmethod
    def _clean_cell_text(text: str) -> str:
        """清洗 cell 文本"""
        text = re.sub(r'\s+', ' ', text).strip()
        # 去掉 LaTeX 残留
        text = text.replace('\\textbf{', '').replace('}', '')
        text = text.replace('\\bf ', '')
        text = re.sub(r'\\(?:text|math)\w*\{([^}]*)\}', r'\1', text)
        return text


# ── 批量加载 ────────────────────────────────────────────────────

def load_paper_tables(
    papers_dir: str,
    paper_ids: Optional[set[str]] = None,
) -> dict[str, list[StructuredTable]]:
    """批量加载论文表格

    Args:
        papers_dir: 解压后的 paper JSON 目录
        paper_ids: 只加载指定论文 (None 表示全部)

    Returns:
        {paper_id: [StructuredTable, ...]}
    """
    import json
    from pathlib import Path

    parser = TableParser()
    result: dict[str, list[StructuredTable]] = {}
    papers_path = Path(papers_dir)

    for json_file in papers_path.glob("*.json"):
        pid = json_file.stem
        if paper_ids and pid not in paper_ids:
            continue

        try:
            with open(json_file) as f:
                data = json.load(f)
            tables = parser.parse_paper(data, paper_id=pid)
            if tables:
                result[pid] = tables
        except Exception:
            continue

    return result
