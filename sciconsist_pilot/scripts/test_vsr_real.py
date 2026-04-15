"""
在真实 SciMDR tqa 数据上测试 VSR end-to-end 流程

流程: tqa record → 查找对应 paper 表格 → VSR reward
验证: 表格匹配率、Layer 路由分布、reward 分布

用法: PYTHONPATH=/root/SciConsist python3 sciconsist_pilot/scripts/test_vsr_real.py
"""

import json
import sys
import random
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sciconsist_pilot.src.vsr.types import StructuredTable, TableRecord
from sciconsist_pilot.src.vsr.reward import VSRReward


def load_table_index(
    table_jsonl: str,
) -> dict[str, list[StructuredTable]]:
    """加载预提取的表格索引

    Args:
        table_jsonl: paper_tables.jsonl 路径

    Returns:
        {paper_id: [StructuredTable, ...]}
    """
    index: dict[str, list[StructuredTable]] = {}

    with open(table_jsonl) as f:
        for line in f:
            rec = json.loads(line)
            pid = rec["paper_id"]
            tables: list[StructuredTable] = []

            for t in rec["tables"]:
                records = [
                    TableRecord(
                        entity=r["entity"],
                        metric=r["metric"],
                        value=r["value"],
                        unit=r.get("unit", ""),
                        raw_text=r.get("raw_text", ""),
                    )
                    for r in t["records"]
                    if r.get("value") is not None
                ]
                if records:
                    tables.append(StructuredTable(
                        paper_id=pid,
                        table_id=t["table_id"],
                        caption=t.get("caption", ""),
                        headers=t.get("headers", []),
                        records=records,
                    ))
            if tables:
                index[pid] = tables

    return index


def extract_answer_text(record: dict) -> str:
    """从 SciMDR record 提取答案文本"""
    ans = record.get("answer", "")
    if isinstance(ans, dict):
        parts = []
        conc = ans.get("conclusion", "")
        if conc:
            parts.append(conc)
        for step in ans.get("chain_of_thought_answer", []):
            r = step.get("reasoning", "") if isinstance(step, dict) else ""
            if r:
                parts.append(r)
        return " ".join(parts)
    cot = record.get("chain_of_thought", "")
    return f"{ans} {cot}".strip()


def main() -> None:
    table_jsonl = "/root/shared-nvme/sciconsist_pilot/processed/table_structured/paper_tables.jsonl"
    tqa_path = "/root/shared-nvme/sciconsist_pilot/raw/scimdr/tqa.jsonl"

    print("[1/4] 加载表格索引...")
    table_index = load_table_index(table_jsonl)
    print(f"  {len(table_index)} papers with tables loaded")

    print("[2/4] 采样 tqa 记录...")
    rng = random.Random(42)
    records = []
    with open(tqa_path) as f:
        for line in f:
            records.append(json.loads(line))
    sample = rng.sample(records, min(500, len(records)))
    print(f"  {len(sample)} records sampled from {len(records)} total")

    print("[3/4] 跑 VSR reward...")
    vsr = VSRReward()

    results = {
        "matched": 0,
        "unmatched": 0,
        "rewards": [],
        "layer_counts": Counter(),
        "per_qtype": {},
    }

    for i, rec in enumerate(sample):
        pid = rec["paper_id"]
        qtype = rec.get("question_type", "UNK")
        answer = extract_answer_text(rec)
        caption = rec.get("combined_caption", "")

        tables = table_index.get(pid, [])
        if not tables:
            results["unmatched"] += 1
            continue

        results["matched"] += 1
        out = vsr.compute(answer, tables, evidence_text=caption)

        results["rewards"].append(out.total_reward)
        for cr in out.claim_rewards:
            for l in cr["layers"]:
                results["layer_counts"][l["layer"]] += 1

        if qtype not in results["per_qtype"]:
            results["per_qtype"][qtype] = {"rewards": [], "count": 0}
        results["per_qtype"][qtype]["rewards"].append(out.total_reward)
        results["per_qtype"][qtype]["count"] += 1

        if (i + 1) % 100 == 0:
            print(f"  processed {i+1}/{len(sample)}")

    print(f"\n[4/4] Results")
    print(f"{'='*60}")
    print(f"  Matched papers:   {results['matched']}/{len(sample)} ({results['matched']/len(sample)*100:.1f}%)")
    print(f"  Unmatched:        {results['unmatched']}")

    if results["rewards"]:
        import statistics
        rews = results["rewards"]
        print(f"\n  Reward distribution:")
        print(f"    mean={statistics.mean(rews):.3f}  median={statistics.median(rews):.3f}")
        print(f"    min={min(rews):.3f}  max={max(rews):.3f}  std={statistics.stdev(rews):.3f}")

    print(f"\n  Layer distribution:")
    total_layers = sum(results["layer_counts"].values())
    for layer, cnt in sorted(results["layer_counts"].items()):
        print(f"    {layer:15s}  {cnt:5d}  ({cnt/total_layers*100:.1f}%)")

    print(f"\n  Per question_type (top 10):")
    sorted_qtypes = sorted(results["per_qtype"].items(), key=lambda x: -x[1]["count"])
    for qt, data in sorted_qtypes[:10]:
        avg_r = statistics.mean(data["rewards"]) if data["rewards"] else 0
        print(f"    {qt:30s}  N={data['count']:4d}  avg_r={avg_r:+.3f}")

    print(f"{'='*60}")


if __name__ == "__main__":
    main()
