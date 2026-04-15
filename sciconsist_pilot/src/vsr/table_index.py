"""
Table Index — paper_id → StructuredTable[] 快速检索

从预提取的 paper_tables.jsonl 加载结构化表格，建立内存索引。
供 VSR reward 在 GRPO 训练中实时查询表格数据。

用法:
    index = TableIndex.from_jsonl("/path/to/paper_tables.jsonl")
    tables = index.get("2505.21277v2")  # -> list[StructuredTable]

性能: ~7500 篇论文, ~36K 表格, 加载 < 10s, 内存 ~500MB
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

from sciconsist_pilot.src.vsr.types import (
    StructuredTable,
    TableRecord,
)


class TableIndex:
    """论文表格内存索引

    Attributes:
        _index: paper_id → list[StructuredTable] 映射
        num_papers: 索引包含的论文数
        num_tables: 索引包含的表格总数
        num_records: 索引包含的记录总数
    """

    def __init__(self) -> None:
        self._index: dict[str, list[StructuredTable]] = {}
        self.num_papers: int = 0
        self.num_tables: int = 0
        self.num_records: int = 0

    def get(self, paper_id: str) -> list[StructuredTable]:
        """根据 paper_id 获取所有结构化表格

        Args:
            paper_id: 论文 ID (如 "2505.21277v2")

        Returns:
            该论文的 StructuredTable 列表, 无匹配时返回空列表
        """
        return self._index.get(paper_id, [])

    def has(self, paper_id: str) -> bool:
        """检查是否包含指定论文"""
        return paper_id in self._index

    @property
    def paper_ids(self) -> list[str]:
        """所有已索引的 paper_id"""
        return list(self._index.keys())

    @classmethod
    def from_jsonl(
        cls,
        path: str | Path,
        max_papers: int = 0,
    ) -> "TableIndex":
        """从 paper_tables.jsonl 加载索引

        JSONL 每行格式:
          {"paper_id": str, "num_tables": int, "tables": [
            {"table_id": str, "caption": str, "headers": [...],
             "num_records": int, "records": [
               {"entity": str, "metric": str, "value": float|null,
                "unit": str, "raw_text": str}
             ]}
          ]}

        Args:
            path: paper_tables.jsonl 路径
            max_papers: 最大加载论文数 (0=全部)

        Returns:
            TableIndex 实例
        """
        index = cls()
        path = Path(path)
        t0 = time.time()

        with open(path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f):
                if max_papers > 0 and line_no >= max_papers:
                    break
                obj = json.loads(line)
                paper_id = obj["paper_id"]
                tables = _parse_tables(paper_id, obj.get("tables", []))
                if tables:
                    index._index[paper_id] = tables
                    index.num_tables += len(tables)
                    index.num_records += sum(len(t.records) for t in tables)

        index.num_papers = len(index._index)
        elapsed = time.time() - t0
        print(
            f"[TableIndex] Loaded {index.num_papers} papers, "
            f"{index.num_tables} tables, {index.num_records} records "
            f"in {elapsed:.1f}s"
        )
        return index


def _parse_tables(
    paper_id: str, raw_tables: list[dict]
) -> list[StructuredTable]:
    """将 JSONL 中的原始 table 字典转为 StructuredTable 对象

    Args:
        paper_id: 论文 ID
        raw_tables: JSONL 中的 tables 列表

    Returns:
        StructuredTable 列表
    """
    tables: list[StructuredTable] = []
    for raw in raw_tables:
        records = [
            TableRecord(
                entity=r.get("entity", ""),
                metric=r.get("metric", ""),
                value=r.get("value"),
                unit=r.get("unit", ""),
                raw_text=r.get("raw_text", ""),
            )
            for r in raw.get("records", [])
            if r.get("entity")
        ]
        table = StructuredTable(
            paper_id=paper_id,
            table_id=str(raw.get("table_id", "")),
            caption=raw.get("caption", ""),
            headers=raw.get("headers", []),
            records=records,
        )
        tables.append(table)
    return tables
