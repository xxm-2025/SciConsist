"""
批量解析 Paper JSON 中的 HTML 表格 → 结构化数据

从 papers.tar.gz 解压后的 JSON 文件中提取所有表格，
生成 paper_id → StructuredTable 的索引，供 Layer 0/1 符号验证使用。

输出:
  table_structured/paper_tables.jsonl  — 每行一个论文的所有表格
  table_structured/stats.json          — 统计摘要

用法:
  PYTHONPATH=/root/SciConsist python3 sciconsist_pilot/scripts/parse_latex_tables.py
"""

import json
import sys
import time
from pathlib import Path
from dataclasses import asdict
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sciconsist_pilot.src.vsr.table_parser import TableParser, load_paper_tables


def main() -> None:
    papers_dir = Path("/root/shared-nvme/sciconsist_pilot/raw/scimdr/arxiv/papers")
    output_dir = Path("/root/shared-nvme/sciconsist_pilot/processed/table_structured")
    output_dir.mkdir(parents=True, exist_ok=True)

    if not papers_dir.exists():
        print(f"[ERROR] papers 目录不存在: {papers_dir}")
        print("请先解压 arxiv/papers.tar.gz")
        sys.exit(1)

    json_files = sorted(papers_dir.glob("*.json"))
    total = len(json_files)
    print(f"[INFO] 共 {total} 个 paper JSON")

    parser = TableParser()

    # 统计
    papers_with_tables = 0
    total_tables = 0
    total_records = 0
    empty_tables = 0
    parse_errors = 0

    out_path = output_dir / "paper_tables.jsonl"
    t0 = time.time()

    with open(out_path, "w") as out_f:
        for idx, json_file in enumerate(json_files):
            pid = json_file.stem

            try:
                with open(json_file) as f:
                    data = json.load(f)
            except Exception as e:
                parse_errors += 1
                continue

            tables = parser.parse_paper(data, paper_id=pid)

            if tables:
                papers_with_tables += 1
                total_tables += len(tables)

                record = {
                    "paper_id": pid,
                    "num_tables": len(tables),
                    "tables": [],
                }

                for t in tables:
                    n_recs = len(t.records)
                    total_records += n_recs
                    if n_recs == 0:
                        empty_tables += 1

                    record["tables"].append({
                        "table_id": t.table_id,
                        "caption": t.caption[:300],
                        "headers": t.headers,
                        "num_records": n_recs,
                        "records": [
                            {
                                "entity": r.entity,
                                "metric": r.metric,
                                "value": r.value,
                                "unit": r.unit,
                                "raw_text": r.raw_text,
                            }
                            for r in t.records
                        ],
                    })

                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")

            if (idx + 1) % 1000 == 0 or idx == total - 1:
                elapsed = time.time() - t0
                rate = (idx + 1) / elapsed
                print(
                    f"  [{idx+1}/{total}] "
                    f"tables={total_tables} records={total_records} "
                    f"papers_with_tables={papers_with_tables} "
                    f"errors={parse_errors} "
                    f"({rate:.0f} papers/s)"
                )

    elapsed = time.time() - t0

    stats = {
        "total_papers": total,
        "papers_with_tables": papers_with_tables,
        "papers_without_tables": total - papers_with_tables - parse_errors,
        "total_tables": total_tables,
        "total_records": total_records,
        "empty_tables": empty_tables,
        "parse_errors": parse_errors,
        "avg_tables_per_paper": round(total_tables / max(papers_with_tables, 1), 2),
        "avg_records_per_table": round(total_records / max(total_tables, 1), 2),
        "elapsed_seconds": round(elapsed, 1),
    }

    stats_path = output_dir / "stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\n{'='*60}")
    print(f"  Table Extraction Complete")
    print(f"{'='*60}")
    print(f"  Papers processed:      {total}")
    print(f"  Papers with tables:    {papers_with_tables} ({papers_with_tables/total*100:.1f}%)")
    print(f"  Total tables:          {total_tables}")
    print(f"  Total records:         {total_records}")
    print(f"  Avg tables/paper:      {stats['avg_tables_per_paper']}")
    print(f"  Avg records/table:     {stats['avg_records_per_table']}")
    print(f"  Parse errors:          {parse_errors}")
    print(f"  Time:                  {elapsed:.1f}s")
    print(f"  Output:                {out_path}")
    print(f"  Stats:                 {stats_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
