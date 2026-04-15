"""
500 Claims 采样 + VSR 自动标注 — Meta-Evaluation 数据准备

从 SciMDR tqa/mqa GT 答案中采样 500 条 claim，运行 VSR 各层打分，
输出供人工 + GPT-4o judge 标注的 meta-evaluation 数据。

用途:
  1. 验证 VSR 各层的 reward 质量 (precision/recall vs human)
  2. 对比 VSR vs GPT-4o judge vs FEH 的 reward 准确度
  3. Week 2 Go/No-Go: VSR Layer 0 precision > 90%

输出:
  outputs/meta_evaluation/
    claims_500.jsonl          — 500 条 claim 详情 + VSR 打分
    claims_for_annotation.jsonl — 人工标注表 (精简格式)
    gpt4o_judge_prompt.txt    — GPT-4o multi-aspect judge prompt
    sampling_stats.json       — 采样统计

用法:
  cd /root/SciConsist/sciconsist_pilot
  python scripts/sample_claims_meta_eval.py \
    --table-index /root/shared-nvme/sciconsist_pilot/processed/table_structured/paper_tables.jsonl \
    --output-dir /root/shared-nvme/sciconsist_pilot/outputs/meta_evaluation \
    --n-claims 500
"""

from __future__ import annotations

import argparse
import json
import random
import time
from collections import Counter
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from sciconsist_pilot.src.vsr.table_index import TableIndex
from sciconsist_pilot.src.vsr.claim_extractor import ClaimExtractor
from sciconsist_pilot.src.vsr.router import VerifiabilityRouter, RoutingDecision
from sciconsist_pilot.src.vsr.symbolic import SymbolicVerifier, SymbolicVerifierConfig
from sciconsist_pilot.src.vsr.semi_symbolic import SemiSymbolicVerifier, SemiSymbolicVerifierConfig
from sciconsist_pilot.src.vsr.types import (
    AtomicClaim,
    VerificationLayer,
    VerificationResult,
    StructuredTable,
)

# ── 配置 ──────────────────────────────────────────────────────────

SCIMDR_DIR = Path("/root/shared-nvme/sciconsist_pilot/raw/scimdr")
DEFAULT_TABLE_INDEX = "/root/shared-nvme/sciconsist_pilot/processed/table_structured/paper_tables.jsonl"
DEFAULT_OUTPUT = "/root/shared-nvme/sciconsist_pilot/outputs/meta_evaluation"

# 采样配额: 按目标层分层采样, 确保 L0/L1/L2 均有足够覆盖
LAYER_QUOTAS = {
    "L0": 100,   # Layer 0 (Symbolic) — 需 precision > 90%
    "L1": 200,   # Layer 1 (Semi-Symbolic)
    "L2": 200,   # Layer 2 (Learned) — 兜底层
}


# ── 数据类 ────────────────────────────────────────────────────────


@dataclass
class ClaimAnnotation:
    """单条 claim 的 meta-evaluation 记录

    Attributes:
        claim_id: 唯一标识
        sample_id: 来源样本 ID
        paper_id: 论文 ID
        split: 数据 split (tqa/mqa)
        question_type: 问题类型
        question: 原始问题
        answer_text: GT 答案全文
        claim_text: 提取的 atomic claim 文本
        claim_entities: 提取的实体
        claim_metrics: 提取的指标
        claim_values: 提取的数值
        claim_relations: 提取的关系
        claim_trends: 提取的趋势
        primary_layer: 主路由层 (L0/L1/L2)
        vsr_reward: VSR 给出的 reward
        vsr_confidence: VSR 置信度
        vsr_details: VSR 验证详情
        matched_table_evidence: 匹配到的表格证据
        human_label: 人工标签 (CORRECT / PARTIALLY_CORRECT / WRONG / UNVERIFIABLE)
        human_notes: 人工备注
    """
    claim_id: str
    sample_id: str
    paper_id: str
    split: str
    question_type: str
    question: str
    answer_text: str
    claim_text: str
    claim_entities: list[str]
    claim_metrics: list[str]
    claim_values: list[dict]
    claim_relations: list[dict]
    claim_trends: list[dict]
    primary_layer: str
    vsr_reward: float
    vsr_confidence: float
    vsr_details: dict
    matched_table_evidence: str
    human_label: str = ""
    human_notes: str = ""


# ── 采样 ──────────────────────────────────────────────────────────


def load_raw_samples(
    split: str, max_per_split: int = 10000, seed: int = 42
) -> list[dict]:
    """从 SciMDR 原始 JSONL 加载样本

    Args:
        split: "tqa" 或 "mqa"
        max_per_split: 每个 split 最大加载数
        seed: 随机种子

    Returns:
        原始记录列表
    """
    fpath = SCIMDR_DIR / f"{split}.jsonl"
    if not fpath.exists():
        print(f"  [SKIP] {fpath} not found")
        return []

    records = []
    with open(fpath, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            if "answer" not in obj or "question" not in obj:
                continue
            records.append(obj)
            if len(records) >= max_per_split:
                break

    rng = random.Random(seed)
    rng.shuffle(records)
    return records


def extract_answer_text(record: dict) -> str:
    """从 SciMDR 记录提取答案纯文本"""
    answer = record.get("answer", {})
    if isinstance(answer, dict):
        parts = []
        cot = answer.get("chain_of_thought_answer", [])
        for step in cot:
            if isinstance(step, dict):
                parts.append(step.get("reasoning", ""))
        conclusion = answer.get("conclusion", "")
        if conclusion:
            parts.append(conclusion)
        return " ".join(parts)
    return str(answer)


def sample_and_annotate(
    table_index: TableIndex,
    n_claims: int = 500,
    seed: int = 42,
) -> tuple[list[ClaimAnnotation], dict]:
    """分层采样 + VSR 自动标注

    按 Layer 配额从 tqa/mqa 采样，确保 L0/L1/L2 覆盖均衡。
    对每条 claim 运行 VSR 各层验证，记录详细结果。

    Args:
        table_index: 表格索引
        n_claims: 目标 claim 数
        seed: 随机种子

    Returns:
        (annotations, stats) — 标注列表和统计信息
    """
    extractor = ClaimExtractor()
    router = VerifiabilityRouter()
    sym_verifier = SymbolicVerifier()
    semi_verifier = SemiSymbolicVerifier()

    # 按层收集 claim 候选
    buckets: dict[str, list[ClaimAnnotation]] = {"L0": [], "L1": [], "L2": []}
    quotas = {k: min(v, n_claims) for k, v in LAYER_QUOTAS.items()}
    total_quota = sum(quotas.values())

    stats = {
        "records_scanned": 0,
        "claims_extracted": 0,
        "papers_with_tables": 0,
        "papers_without_tables": 0,
    }

    for split in ["tqa", "mqa"]:
        records = load_raw_samples(split, max_per_split=15000, seed=seed)
        print(f"  {split}: loaded {len(records)} records")

        for rec_idx, record in enumerate(records):
            if all(len(buckets[k]) >= quotas[k] for k in quotas):
                break

            paper_id = record.get("paper_id", "")
            tables = table_index.get(paper_id)

            if tables:
                stats["papers_with_tables"] += 1
            else:
                stats["papers_without_tables"] += 1

            answer_text = extract_answer_text(record)
            if len(answer_text) < 20:
                continue

            claims = extractor.extract(answer_text)
            stats["claims_extracted"] += len(claims)
            stats["records_scanned"] += 1

            if not claims:
                continue

            decisions = router.route(claims)

            for dec_idx, decision in enumerate(decisions):
                claim = decision.claim
                layer_key = _primary_layer_key(decision)

                if len(buckets[layer_key]) >= quotas[layer_key]:
                    continue

                vsr_result, matched_evidence = _run_layer_verification(
                    claim, decision, tables, sym_verifier, semi_verifier
                )

                annotation = ClaimAnnotation(
                    claim_id=f"{split}_{rec_idx}_{dec_idx}",
                    sample_id=f"{split}_{paper_id}_{rec_idx}",
                    paper_id=paper_id,
                    split=split,
                    question_type=record.get("question_type", ""),
                    question=record.get("question", ""),
                    answer_text=answer_text[:500],
                    claim_text=claim.text,
                    claim_entities=[e.name for e in claim.entities],
                    claim_metrics=claim.metrics,
                    claim_values=[
                        {"raw": v.raw, "value": v.value, "unit": v.unit}
                        for v in claim.numeric_values
                    ],
                    claim_relations=[
                        {"entity_a": r.entity_a, "entity_b": r.entity_b,
                         "relation": r.relation, "metric": r.metric}
                        for r in claim.relations
                    ],
                    claim_trends=[
                        {"entity": t.entity, "direction": t.direction,
                         "condition": t.condition}
                        for t in claim.trends
                    ],
                    primary_layer=layer_key,
                    vsr_reward=vsr_result.reward,
                    vsr_confidence=vsr_result.confidence,
                    vsr_details={
                        "layer": vsr_result.layer.name,
                        **vsr_result.details,
                    },
                    matched_table_evidence=matched_evidence,
                )
                buckets[layer_key].append(annotation)

    all_annotations = []
    for key in ["L0", "L1", "L2"]:
        all_annotations.extend(buckets[key])

    stats["total_claims"] = len(all_annotations)
    stats["by_layer"] = {k: len(v) for k, v in buckets.items()}
    stats["by_split"] = dict(Counter(a.split for a in all_annotations))
    stats["by_question_type"] = dict(
        Counter(a.question_type for a in all_annotations).most_common(15)
    )

    return all_annotations, stats


def _primary_layer_key(decision: RoutingDecision) -> str:
    """获取 claim 的主路由层 key"""
    if VerificationLayer.SYMBOLIC in decision.layers:
        return "L0"
    if VerificationLayer.SEMI_SYMBOLIC in decision.layers:
        return "L1"
    return "L2"


def _run_layer_verification(
    claim: AtomicClaim,
    decision: RoutingDecision,
    tables: list[StructuredTable],
    sym_verifier: SymbolicVerifier,
    semi_verifier: SemiSymbolicVerifier,
) -> tuple[VerificationResult, str]:
    """对单条 claim 运行对应层的验证

    Returns:
        (VerificationResult, matched_evidence_description)
    """
    matched_evidence = ""

    if VerificationLayer.SYMBOLIC in decision.layers and tables:
        result = sym_verifier.verify(claim, tables)
        if result.confidence > 0:
            matched_evidence = result.matched_evidence
            return result, matched_evidence

    if VerificationLayer.SEMI_SYMBOLIC in decision.layers and tables:
        result = semi_verifier.verify(claim, tables)
        if result.confidence > 0:
            matched_evidence = result.matched_evidence
            return result, matched_evidence

    return VerificationResult(
        layer=VerificationLayer.LEARNED,
        reward=0.0,
        confidence=0.5,
        details={"note": "no symbolic/semi-symbolic verification available"},
    ), ""


# ── GPT-4o Multi-Aspect Judge Prompt ─────────────────────────────


GPT4O_JUDGE_PROMPT = '''You are an expert evaluator for scientific document analysis. Your task is to assess whether a factual claim extracted from a model's response is correct, given the evidence from a scientific paper.

## Input
- **Claim**: A factual statement from a model's response about a scientific paper
- **Evidence**: Table data, figure descriptions, or text from the paper
- **Question**: The original question that prompted the response

## Evaluation Dimensions

Rate the claim on EACH of the following dimensions independently:

### 1. Numeric Accuracy (if applicable)
- Does the claim cite specific numbers (percentages, scores, counts)?
- Are these numbers EXACTLY correct according to the evidence?
- Score: CORRECT (exact match), APPROX (within 1% relative error), WRONG (>1% error), N/A (no numbers)

### 2. Entity Accuracy
- Are method names, dataset names, and metric names correctly identified?
- Are there any entity confusions or misspellings that change meaning?
- Score: CORRECT, PARTIALLY_CORRECT (minor name variation), WRONG, N/A

### 3. Relation Accuracy (if applicable)
- Are comparative claims (A > B, A outperforms B) correct?
- Are ranking claims (A is best, A ranks first) correct?
- Score: CORRECT, WRONG (reversed relation), N/A

### 4. Trend Accuracy (if applicable)
- Are trend claims (increasing, decreasing, stable) correct?
- Is the direction of the trend accurate?
- Score: CORRECT, WRONG, N/A

### 5. Overall Factual Correctness
Based on the above dimensions, provide an OVERALL judgment:
- CORRECT: All applicable dimensions are correct
- PARTIALLY_CORRECT: Some dimensions correct, minor errors in others
- WRONG: At least one critical dimension is wrong (numeric error, reversed relation, etc.)
- UNVERIFIABLE: Insufficient evidence to verify the claim

## Output Format (JSON)
```json
{{
  "numeric_accuracy": {{"score": "CORRECT|APPROX|WRONG|N/A", "explanation": "..."}},
  "entity_accuracy": {{"score": "CORRECT|PARTIALLY_CORRECT|WRONG|N/A", "explanation": "..."}},
  "relation_accuracy": {{"score": "CORRECT|WRONG|N/A", "explanation": "..."}},
  "trend_accuracy": {{"score": "CORRECT|WRONG|N/A", "explanation": "..."}},
  "overall": {{"score": "CORRECT|PARTIALLY_CORRECT|WRONG|UNVERIFIABLE", "explanation": "..."}},
  "confidence": 0.0-1.0
}}
```

## Important Notes
- Be STRICT on numeric accuracy: 85.3% vs 85.1% is WRONG, not approximate
- Be LENIENT on entity naming: "BERT-base" vs "BERT-Base" is CORRECT
- If the evidence is insufficient to verify, say UNVERIFIABLE rather than guessing
- Provide brief but specific explanations for each dimension

---

## Now evaluate this claim:

**Question**: {question}

**Claim**: {claim}

**Evidence (Table Data)**:
{table_evidence}

**Evidence (Text Context)**:
{text_context}
'''


# ── 输出 ──────────────────────────────────────────────────────────


def write_outputs(
    annotations: list[ClaimAnnotation],
    stats: dict,
    output_dir: str | Path,
) -> None:
    """写入所有输出文件

    Args:
        annotations: 标注数据
        stats: 统计信息
        output_dir: 输出目录
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # 完整数据
    with open(out / "claims_500.jsonl", "w", encoding="utf-8") as f:
        for a in annotations:
            f.write(json.dumps(asdict(a), ensure_ascii=False) + "\n")

    # 人工标注表 (精简格式)
    with open(out / "claims_for_annotation.jsonl", "w", encoding="utf-8") as f:
        for a in annotations:
            record = {
                "claim_id": a.claim_id,
                "claim_text": a.claim_text,
                "paper_id": a.paper_id,
                "question": a.question[:200],
                "answer_excerpt": a.answer_text[:300],
                "primary_layer": a.primary_layer,
                "table_evidence": a.matched_table_evidence,
                "vsr_reward": round(a.vsr_reward, 3),
                "human_label": "",
                "human_notes": "",
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # GPT-4o judge prompt
    (out / "gpt4o_judge_prompt.txt").write_text(
        GPT4O_JUDGE_PROMPT, encoding="utf-8"
    )

    # GPT-4o judge 批量请求数据 (每条 claim 填入 prompt)
    with open(out / "gpt4o_judge_requests.jsonl", "w", encoding="utf-8") as f:
        for a in annotations:
            table_ev = a.matched_table_evidence or "(No table evidence available)"
            text_ctx = a.answer_text[:500]
            filled_prompt = GPT4O_JUDGE_PROMPT.format(
                question=a.question[:300],
                claim=a.claim_text,
                table_evidence=table_ev,
                text_context=text_ctx,
            )
            request = {
                "claim_id": a.claim_id,
                "model": "gpt-4o",
                "messages": [
                    {"role": "user", "content": filled_prompt},
                ],
            }
            f.write(json.dumps(request, ensure_ascii=False) + "\n")

    # 统计
    (out / "sampling_stats.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"\nOutput written to {out}/")
    print(f"  claims_500.jsonl: {len(annotations)} claims")
    print(f"  claims_for_annotation.jsonl: ready for human labeling")
    print(f"  gpt4o_judge_prompt.txt: multi-aspect prompt")
    print(f"  gpt4o_judge_requests.jsonl: {len(annotations)} API requests")
    print(f"  sampling_stats.json: statistics")


# ── Main ──────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sample 500 claims for VSR Meta-Evaluation"
    )
    parser.add_argument(
        "--table-index", default=DEFAULT_TABLE_INDEX,
        help="paper_tables.jsonl path"
    )
    parser.add_argument(
        "--output-dir", default=DEFAULT_OUTPUT,
        help="Output directory"
    )
    parser.add_argument(
        "--n-claims", type=int, default=500,
        help="Target number of claims"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed"
    )
    args = parser.parse_args()

    print("=== Meta-Evaluation: Claim Sampling + VSR Auto-Annotation ===\n")

    print("Loading table index...")
    table_index = TableIndex.from_jsonl(args.table_index)

    print("\nSampling and annotating claims...")
    t0 = time.time()
    annotations, stats = sample_and_annotate(
        table_index, n_claims=args.n_claims, seed=args.seed
    )
    elapsed = time.time() - t0
    stats["elapsed_seconds"] = round(elapsed, 1)

    print(f"\n  Sampling complete in {elapsed:.1f}s")
    print(f"  Records scanned: {stats['records_scanned']}")
    print(f"  Claims extracted: {stats['claims_extracted']}")
    print(f"  Claims sampled: {stats['total_claims']}")
    print(f"  By layer: {stats['by_layer']}")
    print(f"  By split: {stats['by_split']}")

    write_outputs(annotations, stats, args.output_dir)

    # VSR reward 分布统计
    layer_rewards = {"L0": [], "L1": [], "L2": []}
    for a in annotations:
        layer_rewards[a.primary_layer].append(a.vsr_reward)

    print("\n=== VSR Reward Distribution ===")
    for layer, rewards in layer_rewards.items():
        if rewards:
            import numpy as np
            arr = np.array(rewards)
            print(f"  {layer}: n={len(rewards)}, "
                  f"mean={arr.mean():.3f}, std={arr.std():.3f}, "
                  f"min={arr.min():.3f}, max={arr.max():.3f}")

    print("\nDone. Next steps:")
    print("  1. Human annotation: edit claims_for_annotation.jsonl")
    print("  2. GPT-4o evaluation: run gpt4o_judge_requests.jsonl")
    print("  3. Compare: VSR vs Human vs GPT-4o judge")


if __name__ == "__main__":
    main()
