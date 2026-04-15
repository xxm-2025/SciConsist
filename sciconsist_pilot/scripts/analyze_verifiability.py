"""
Claim Verifiability Coverage Analysis (Go/No-Go for VSR)

分析科研文档中 claims 的可验证性分布，判断 Layer 0 (symbolic) 和 Layer 1
(semi-symbolic) 的覆盖率是否达到 30% 阈值。

三阶段：
  Phase 1: 规则分析 MuSciClaims 现有 claims (bio/physics, 无需 API)
  Phase 2: LLM 生成 CS/ML figure descriptions + LLM 精细分类 (API)
  Phase 3: 规则分析 SciMDR 真实 QA 回答 (tqa/mqa/vqa, 无需 API)

输出: JSON 统计 + 终端摘要
"""

import json
import re
import os
import sys
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional
from collections import Counter

# ── 数据类 ──────────────────────────────────────────────────────

@dataclass
class VerifiabilityProfile:
    """单条 claim 的可验证性特征"""
    has_numeric: bool = False          # 含具体数值/百分比
    has_entity_metric: bool = False    # 含 (实体, 指标) 对
    has_comparison: bool = False       # 含 A > B / outperforms 等比较
    has_trend: bool = False            # 含 increase/decrease 趋势
    is_qualitative_only: bool = False  # 纯定性描述

    @property
    def layer0_eligible(self) -> bool:
        """可路由到 Layer 0 (Symbolic)"""
        return self.has_numeric and self.has_entity_metric

    @property
    def layer1_eligible(self) -> bool:
        """可路由到 Layer 1 (Semi-Symbolic)"""
        return self.has_comparison or self.has_trend

    @property
    def layer2_only(self) -> bool:
        """只能走 Layer 2 (Learned)"""
        return not self.layer0_eligible and not self.layer1_eligible

    @property
    def best_layer(self) -> int:
        if self.layer0_eligible:
            return 0
        if self.layer1_eligible:
            return 1
        return 2


# ── 规则分析器 ──────────────────────────────────────────────────

# 数值模式
RE_NUMERIC = re.compile(
    r'(?:'
    r'\d+\.\d+\s*%'            # 85.3%
    r'|\d+\s*%'                 # 85%
    r'|\d+\.\d+\s*(?:dB|ms|s|nm|mum|mm|cm|Hz|kHz|MHz|GHz|eV|keV|MeV)'  # 带单位
    r'|\b\d+\.\d{1,4}\b'       # 0.853, 85.32 (小数，1-4位)
    r'|(?:=|is|of|at|by|to)\s*\d+(?:\.\d+)?'  # = 85.3, is 42
    r'|\bp\s*[<>=]\s*\d'        # p < 0.05
    r')',
    re.IGNORECASE
)

# 实体+指标模式（CS/ML 风格）
RE_ENTITY_METRIC = re.compile(
    r'(?:'
    r'(?:accuracy|precision|recall|F1|BLEU|ROUGE|METEOR|CIDEr|perplexity|loss|AUC|mAP|IoU|PSNR|SSIM|FID)'
    r'|(?:error rate|success rate|hit rate|top-\d+|rank-\d+)'
    r')',
    re.IGNORECASE
)

# 实体+指标模式（Bio/Physics 风格）
RE_ENTITY_METRIC_BIO = re.compile(
    r'(?:'
    r'(?:concentration|intensity|fluorescence|expression|fold[ -]change|abundance|viability|survival)'
    r'|(?:IC50|EC50|Kd|Ki|half-life|melting point|wavelength|frequency|energy|voltage|current|resistance)'
    r'|(?:p-value|significance|correlation|coefficient|R\^?2|chi-square)'
    r'|(?:mean|median|SD|SEM|standard deviation|confidence interval|CI)'
    r')',
    re.IGNORECASE
)

# 比较模式
RE_COMPARISON = re.compile(
    r'(?:'
    r'\b(?:outperform|surpass|exceed|superior|inferior|better|worse|higher|lower|greater|less|more|fewer)\b'
    r'|\b(?:compared?\s+(?:to|with)|relative\s+to|versus|vs\.?)\b'
    r'|\b(?:significant(?:ly)?\s+(?:higher|lower|better|worse|more|less|greater|fewer|increased|decreased|improved|reduced))\b'
    r'|\b(?:rank(?:s|ed)?\s+(?:first|second|third|last|highest|lowest))\b'
    r')',
    re.IGNORECASE
)

# 趋势模式
RE_TREND = re.compile(
    r'(?:'
    r'\b(?:increas|decreas|grow|shrink|rise|fall|climb|drop|declin|improv|degrad|escalt|diminish)\w*\b'
    r'|\b(?:upward|downward|ascending|descending|monoton)\w*\b'
    r'|\b(?:converge|diverge|plateau|saturat|stabiliz)\w*\b'
    r'|\b(?:scales?\s+(?:with|linearly|logarithmically|exponentially))\b'
    r'|\b(?:positive|negative|inverse|direct)\s+(?:correlation|relationship|trend)\b'
    r')',
    re.IGNORECASE
)

# 定性标记词（无数值、无比较、无趋势时判定为纯定性）
RE_QUALITATIVE = re.compile(
    r'(?:'
    r'\b(?:suggest|indicate|demonstrate|reveal|show|imply|observe|note|appear|seem)\b'
    r'|\b(?:important|notable|interesting|striking|remarkable|consistent|similar|distinct)\b'
    r')',
    re.IGNORECASE
)


def classify_claim_rule(text: str) -> VerifiabilityProfile:
    """基于正则的 claim verifiability 分类"""
    p = VerifiabilityProfile()
    p.has_numeric = bool(RE_NUMERIC.search(text))
    p.has_entity_metric = bool(RE_ENTITY_METRIC.search(text) or RE_ENTITY_METRIC_BIO.search(text))
    p.has_comparison = bool(RE_COMPARISON.search(text))
    p.has_trend = bool(RE_TREND.search(text))
    p.is_qualitative_only = (
        not p.has_numeric
        and not p.has_comparison
        and not p.has_trend
        and bool(RE_QUALITATIVE.search(text))
    )
    return p


# ── Phase 1: MuSciClaims 规则分析 ──────────────────────────────

def phase1_musciclaims(data_path: str) -> list[dict]:
    """对 MuSciClaims 做规则 verifiability 分类"""
    results = []
    with open(data_path) as f:
        for line in f:
            item = json.loads(line)
            text = item.get("claim_text") or item.get("text", "")
            caption = item.get("caption", "")
            domain = item.get("domain", "unknown")
            label = item.get("label_3class") or item.get("label_str", "")

            profile = classify_claim_rule(text)

            # 也分析 caption（代表 figure 中可提取的信息）
            caption_profile = classify_claim_rule(caption)

            results.append({
                "text": text[:200],
                "domain": domain,
                "label": label,
                "claim_verifiability": asdict(profile),
                "best_layer": profile.best_layer,
                "caption_has_numeric": caption_profile.has_numeric,
                "caption_has_entity_metric": caption_profile.has_entity_metric,
            })
    return results


# ── Phase 2: LLM 精细分类 ──────────────────────────────────────

CLASSIFY_PROMPT = """You are analyzing scientific claims for their verifiability. For each claim, determine:

1. has_numeric: Does it contain specific numbers, percentages, or measurable values? (true/false)
2. has_entity_metric: Does it mention a specific entity (model/method/condition) paired with a metric? (true/false)
3. has_comparison: Does it compare two or more entities? (true/false)
4. has_trend: Does it describe a trend (increase/decrease/convergence)? (true/false)
5. is_qualitative_only: Is it purely qualitative with no verifiable numeric/relational content? (true/false)
6. best_layer: Which verification layer suits best?
   - 0 = Symbolic (has specific number + entity/metric, can be exactly checked)
   - 1 = Semi-symbolic (has comparison or trend that can be logically checked)
   - 2 = Learned only (qualitative, needs semantic judgment)

Respond as a JSON array. One object per claim."""

GENERATE_CSML_PROMPT = """You are a vision-language model answering questions about a scientific figure from an ML/NLP paper.

Given a figure caption, generate a realistic model response (3-5 sentences) describing the key findings shown in the figure. Include a mix of:
- Specific numerical values where appropriate
- Comparisons between methods/models
- Trend observations
- Qualitative summaries

Caption: {caption}

Generate a REALISTIC response as if you were describing the figure to a researcher. Be specific."""

# CS/ML 风格的 figure captions（手写，代表典型 ML 论文中的 figure）
CSML_CAPTIONS = [
    "Figure 3: Comparison of model performance on the GLUE benchmark. Results show accuracy (%) for BERT-base, RoBERTa, and our proposed method across 8 tasks. Error bars indicate standard deviation over 5 runs.",
    "Figure 2: Training loss curves for different learning rates (1e-5, 3e-5, 5e-5) on the SQuAD v2.0 dataset over 100K steps.",
    "Table 1: BLEU scores on WMT'14 En-De and En-Fr translation benchmarks. Our model achieves 29.3 BLEU on En-De and 43.2 on En-Fr.",
    "Figure 5: Ablation study on the effect of number of attention heads (1, 2, 4, 8, 16) on MMLU accuracy. Performance saturates around 8 heads.",
    "Figure 4: Scaling behavior of the model. Test perplexity as a function of model size (125M to 13B parameters) on the Pile validation set.",
    "Table 3: Results on SciClaimEval. Precision, Recall, and F1 for claim verification across ML, NLP, and Medicine domains.",
    "Figure 1: Distribution of claim types in the training set. 42% contain numerical values, 28% are comparative, and 30% are qualitative.",
    "Figure 6: Effect of LoRA rank (4, 8, 16, 32, 64) on downstream task performance. Higher rank improves accuracy but increases memory usage.",
    "Table 2: Zero-shot and few-shot (1, 5, 10 shot) performance on ChartQA. GPT-4o achieves 78.2% in zero-shot vs our model's 72.1%.",
    "Figure 7: Attention visualization showing that the model focuses on relevant table cells when answering numerical questions but attends to the full table for trend questions.",
    "Figure 8: Reward curves during GRPO training. Symbolic reward (Layer 0) remains stable while learned reward (Layer 2) shows signs of inflation after step 5000.",
    "Table 4: Comparison of inference latency (ms) and GPU memory (GB) across different model sizes. Our 7B model runs at 45ms/token on A100.",
    "Figure 9: Per-category accuracy on PRISMM-Bench. Text-figure inconsistency: 67.3%, Text-table: 78.9%, Figure-table: 45.2%.",
    "Figure 10: Cross-domain generalization. Model trained on CS papers tested on Medicine (F1=0.62) and Biology (F1=0.58) domains.",
    "Table 5: Hallucination rates on different question types. Numerical questions: 23.4%, Comparative: 15.7%, Descriptive: 8.2%.",
    "Figure 11: ROC curves for inconsistency detection at different confidence thresholds. AUC = 0.87 for our method vs 0.73 for the baseline.",
    "Figure 12: Training dynamics showing diversity (distinct-3) vs accuracy over epochs. Surface sim reward causes diversity collapse by epoch 3.",
    "Table 6: Multi-seed stability analysis (3 seeds). Mean accuracy 76.4% +/- 1.2% on SciClaimEval, 82.1% +/- 0.8% on MuSciClaims.",
    "Figure 13: Bar chart comparing human agreement (Cohen's kappa) for different reward types: symbolic (0.91), semi-symbolic (0.82), learned (0.67).",
    "Figure 14: The impact of OCR noise levels (0%, 5%, 10%, 20%) on downstream claim verification accuracy.",
]


def phase2_llm_classify(claims: list[str], api_key: str, base_url: str) -> list[dict]:
    """用 LLM 对 claims 做精细 verifiability 分类"""
    import openai
    client = openai.OpenAI(api_key=api_key, base_url=base_url)

    batch_size = 20
    all_results = []
    for i in range(0, len(claims), batch_size):
        batch = claims[i:i+batch_size]
        numbered = "\n".join(f"{j+1}. {c}" for j, c in enumerate(batch))
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": CLASSIFY_PROMPT},
                {"role": "user", "content": f"Classify these {len(batch)} claims:\n\n{numbered}"}
            ],
            temperature=0,
            max_tokens=4000,
        )
        content = resp.choices[0].message.content.strip()
        # 提取 JSON
        start = content.find("[")
        end = content.rfind("]") + 1
        if start >= 0 and end > start:
            try:
                parsed = json.loads(content[start:end])
                all_results.extend(parsed)
            except json.JSONDecodeError:
                print(f"  [WARN] JSON parse failed for batch {i//batch_size}, skipping")
        else:
            print(f"  [WARN] No JSON array found in batch {i//batch_size}")
    return all_results


def phase2_generate_csml_responses(api_key: str, base_url: str) -> list[str]:
    """用 LLM 生成 CS/ML figure descriptions"""
    import openai
    client = openai.OpenAI(api_key=api_key, base_url=base_url)

    responses = []
    for i, cap in enumerate(CSML_CAPTIONS):
        try:
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "user", "content": GENERATE_CSML_PROMPT.format(caption=cap)}
                ],
                temperature=0.7,
                max_tokens=300,
            )
            responses.append(resp.choices[0].message.content.strip())
        except Exception as e:
            print(f"  [WARN] Generation failed for caption {i}: {e}")
    return responses


# ── Phase 3: SciMDR 真实 QA 规则分析 ─────────────────────────────

def _extract_answer_text(record: dict) -> str:
    """从 SciMDR record 中提取回答文本用于 verifiability 分析。

    mqa/tqa: answer 是 dict，取 conclusion + chain_of_thought reasoning
    vqa: answer 是 str，另有 chain_of_thought 字段
    """
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
    # vqa: answer 是 string
    cot = record.get("chain_of_thought", "")
    return f"{ans} {cot}".strip()


def phase3_scimdr(
    scimdr_dir: str,
    sample_per_split: int = 5000,
    seed: int = 42,
) -> dict[str, list[dict]]:
    """对 SciMDR 各 split 的真实 QA 回答做规则 verifiability 分类。

    Args:
        scimdr_dir: SciMDR JSONL 所在目录
        sample_per_split: 每个 split 采样数 (<=0 表示全量)
        seed: 随机种子

    Returns:
        {split_name: [result_dict, ...]}
    """
    import random
    rng = random.Random(seed)
    results_by_split: dict[str, list[dict]] = {}

    for fname in ["tqa.jsonl", "mqa.jsonl", "vqa.jsonl"]:
        fpath = Path(scimdr_dir) / fname
        if not fpath.exists():
            print(f"  [SKIP] {fpath} not found")
            continue

        split_name = fname.replace(".jsonl", "")
        print(f"  Loading {fname}...")

        records = []
        with open(fpath) as f:
            for line in f:
                records.append(json.loads(line))

        if 0 < sample_per_split < len(records):
            records = rng.sample(records, sample_per_split)

        split_results = []
        for rec in records:
            answer_text = _extract_answer_text(rec)
            caption = rec.get("combined_caption", "")
            # 拼合 references_in_text 的文本（代表论文中的上下文描述）
            ref_texts = " ".join(
                r.get("text", "") for r in rec.get("references_in_text", [])
                if isinstance(r, dict)
            )

            # 分析 model answer 的 verifiability
            answer_profile = classify_claim_rule(answer_text)
            # 分析 source evidence 的 verifiability（caption + refs）
            evidence_profile = classify_claim_rule(f"{caption} {ref_texts}")

            split_results.append({
                "text": answer_text[:200],
                "question_type": rec.get("question_type", ""),
                "claim_verifiability": asdict(answer_profile),
                "best_layer": answer_profile.best_layer,
                "evidence_has_numeric": evidence_profile.has_numeric,
                "evidence_has_entity_metric": evidence_profile.has_entity_metric,
                "evidence_layer01": evidence_profile.layer0_eligible or evidence_profile.layer1_eligible,
            })

        results_by_split[split_name] = split_results
        print(f"  {split_name}: analyzed {len(split_results)} records")

    return results_by_split


# ── 统计与输出 ──────────────────────────────────────────────────

def compute_stats(results: list[dict], source_name: str) -> dict:
    """计算 verifiability 分布统计"""
    total = len(results)
    if total == 0:
        return {"source": source_name, "total": 0}

    layer_counts = Counter(r["best_layer"] for r in results)
    l0_count = layer_counts.get(0, 0)
    l1_count = layer_counts.get(1, 0)
    l2_count = layer_counts.get(2, 0)

    # 各维度覆盖
    dim_counts = {}
    for dim in ["has_numeric", "has_entity_metric", "has_comparison", "has_trend", "is_qualitative_only"]:
        if dim in results[0].get("claim_verifiability", {}):
            dim_counts[dim] = sum(1 for r in results if r.get("claim_verifiability", {}).get(dim, False))
        elif dim in results[0]:
            dim_counts[dim] = sum(1 for r in results if r.get(dim, False))

    stats = {
        "source": source_name,
        "total": total,
        "layer_0_symbolic": l0_count,
        "layer_1_semi_symbolic": l1_count,
        "layer_2_learned_only": l2_count,
        "layer_0_pct": round(l0_count / total * 100, 1),
        "layer_1_pct": round(l1_count / total * 100, 1),
        "layer_2_pct": round(l2_count / total * 100, 1),
        "layer_01_combined_pct": round((l0_count + l1_count) / total * 100, 1),
        "dimension_counts": dim_counts,
    }
    return stats


def print_report(all_stats: list[dict]):
    """打印最终报告"""
    print("\n" + "=" * 70)
    print("  VERIFIABILITY COVERAGE ANALYSIS — Go/No-Go Report")
    print("=" * 70)

    for s in all_stats:
        print(f"\n{'─' * 60}")
        print(f"  Source: {s['source']}  (N={s['total']})")
        print(f"{'─' * 60}")
        print(f"  Layer 0 (Symbolic):       {s['layer_0_symbolic']:4d}  ({s['layer_0_pct']:5.1f}%)")
        print(f"  Layer 1 (Semi-Symbolic):  {s['layer_1_semi_symbolic']:4d}  ({s['layer_1_pct']:5.1f}%)")
        print(f"  Layer 2 (Learned only):   {s['layer_2_learned_only']:4d}  ({s['layer_2_pct']:5.1f}%)")
        print(f"  ────────────────────────────────")
        print(f"  Layer 0+1 Combined:              ({s['layer_01_combined_pct']:5.1f}%)")
        if s.get("dimension_counts"):
            print(f"\n  Dimension breakdown:")
            for dim, cnt in s["dimension_counts"].items():
                print(f"    {dim:25s}  {cnt:4d}  ({cnt/s['total']*100:5.1f}%)")

    # Go/No-Go 判定
    print(f"\n{'=' * 70}")
    print("  GO / NO-GO DECISION")
    print(f"{'=' * 70}")
    csml_stats = [s for s in all_stats if "CS/ML" in s["source"]]
    bio_stats = [s for s in all_stats if "Bio" in s["source"] or "MuSci" in s["source"]]

    if csml_stats:
        csml_cov = csml_stats[0]["layer_01_combined_pct"]
        print(f"\n  CS/ML domain Layer 0+1 coverage: {csml_cov:.1f}%")
        if csml_cov >= 30:
            print(f"  >>> GO: Coverage exceeds 30% threshold <<<")
        else:
            print(f"  >>> WARNING: Coverage below 30% threshold <<<")

    if bio_stats:
        bio_cov = bio_stats[0]["layer_01_combined_pct"]
        print(f"  Bio/Physics domain Layer 0+1 coverage: {bio_cov:.1f}%")

    combined_cov = sum(s["layer_0_symbolic"] + s["layer_1_semi_symbolic"] for s in all_stats) / max(sum(s["total"] for s in all_stats), 1) * 100
    print(f"\n  Overall Layer 0+1 coverage: {combined_cov:.1f}%")
    if combined_cov >= 30:
        print(f"  >>> OVERALL GO <<<")
    elif combined_cov >= 20:
        print(f"  >>> MARGINAL — need CS/ML-specific validation <<<")
    else:
        print(f"  >>> NO-GO: Consider Route A (systematic study) <<<")
    print(f"{'=' * 70}\n")


# ── Main ────────────────────────────────────────────────────────

def main():
    output_dir = Path("/root/shared-nvme/sciconsist_pilot/outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    all_stats = []

    # ── Phase 1: MuSciClaims 规则分析 ──
    print("[Phase 1] Rule-based analysis on MuSciClaims...")
    musciclaims_path = "/root/shared-nvme/sciconsist_pilot/raw/musciclaims_test.jsonl"
    if Path(musciclaims_path).exists():
        musci_results = phase1_musciclaims(musciclaims_path)
        musci_stats = compute_stats(musci_results, "MuSciClaims (Bio/Physics) — Rule-based")
        all_stats.append(musci_stats)
        print(f"  Analyzed {len(musci_results)} claims")
        print(f"  Layer 0+1 coverage: {musci_stats['layer_01_combined_pct']:.1f}%")

        # 按 domain 细分
        for domain in set(r["domain"] for r in musci_results):
            domain_results = [r for r in musci_results if r["domain"] == domain]
            domain_stats = compute_stats(domain_results, f"MuSciClaims ({domain}) — Rule-based")
            all_stats.append(domain_stats)
            print(f"  [{domain}] Layer 0+1: {domain_stats['layer_01_combined_pct']:.1f}% (N={len(domain_results)})")
    else:
        print(f"  [SKIP] {musciclaims_path} not found")

    # ── Phase 1b: 分析 captions（figure 中可提取信息的 proxy）──
    print("\n[Phase 1b] Analyzing figure captions (proxy for extractable info)...")
    if Path(musciclaims_path).exists():
        with open(musciclaims_path) as f:
            captions = list({json.loads(l).get("caption", "") for l in f if json.loads(l).get("caption")})
        cap_results = []
        for cap in captions:
            p = classify_claim_rule(cap)
            cap_results.append({"best_layer": p.best_layer, "claim_verifiability": asdict(p)})
        cap_stats = compute_stats(cap_results, "Figure Captions (extractable evidence proxy)")
        all_stats.append(cap_stats)
        print(f"  {len(captions)} unique captions; {cap_stats['layer_01_combined_pct']:.1f}% have L0+1 content")

    # ── Phase 2: CS/ML 域（LLM 生成 + 分类）──
    api_key = os.environ.get("OPENAI_API_KEY", "")
    base_url = os.environ.get("OPENAI_BASE_URL", "")
    if api_key and base_url:
        print("\n[Phase 2] Generating CS/ML figure descriptions via LLM...")
        csml_responses = phase2_generate_csml_responses(api_key, base_url)
        print(f"  Generated {len(csml_responses)} responses")

        # 规则分析生成的 CS/ML responses
        csml_rule_results = []
        for resp in csml_responses:
            # 将 response 拆成句子级 claims
            sentences = [s.strip() for s in re.split(r'[.!?]\s+', resp) if len(s.strip()) > 15]
            for sent in sentences:
                p = classify_claim_rule(sent)
                csml_rule_results.append({"best_layer": p.best_layer, "claim_verifiability": asdict(p), "text": sent[:150]})

        csml_stats = compute_stats(csml_rule_results, "CS/ML Generated Responses — Rule-based")
        all_stats.append(csml_stats)
        print(f"  {len(csml_rule_results)} claims extracted; Layer 0+1: {csml_stats['layer_01_combined_pct']:.1f}%")

        # LLM 精细分类（验证规则分析的准确性）
        print("\n[Phase 2b] LLM-based fine-grained classification (validation)...")
        sample_claims = [r["text"] for r in csml_rule_results[:40]]
        if sample_claims:
            llm_results_raw = phase2_llm_classify(sample_claims, api_key, base_url)
            llm_results = []
            for r in llm_results_raw:
                bl = r.get("best_layer", 2)
                llm_results.append({"best_layer": bl, **{k: v for k, v in r.items() if k != "best_layer"}})
            if llm_results:
                llm_stats = compute_stats(llm_results, "CS/ML Claims — LLM-classified (validation)")
                all_stats.append(llm_stats)
                print(f"  LLM classified {len(llm_results)} claims; Layer 0+1: {llm_stats['layer_01_combined_pct']:.1f}%")

                # 规则 vs LLM 一致性
                agree = sum(1 for r, l in zip(csml_rule_results[:len(llm_results)], llm_results) if r["best_layer"] == l["best_layer"])
                print(f"  Rule vs LLM agreement: {agree}/{len(llm_results)} ({agree/len(llm_results)*100:.1f}%)")
    else:
        print("\n[Phase 2] SKIP — OPENAI_API_KEY / OPENAI_BASE_URL not set")

    # ── Phase 3: SciMDR 真实 QA 规则分析 ──
    print("\n[Phase 3] Rule-based analysis on SciMDR real QA answers...")
    scimdr_dir = "/root/shared-nvme/sciconsist_pilot/raw/scimdr"
    scimdr_results = phase3_scimdr(scimdr_dir, sample_per_split=5000)

    for split_name, split_results in scimdr_results.items():
        split_stats = compute_stats(split_results, f"SciMDR-{split_name} answers — Rule-based")
        all_stats.append(split_stats)
        print(f"  {split_name}: Layer 0+1 = {split_stats['layer_01_combined_pct']:.1f}%")

        # evidence 端覆盖率（caption + refs 中有可验证内容的比例）
        ev_l01 = sum(1 for r in split_results if r.get("evidence_layer01")) / max(len(split_results), 1) * 100
        print(f"  {split_name}: Evidence L0+1 coverage = {ev_l01:.1f}%")

        # 按 question_type 细分
        from collections import Counter as C2
        qt_layer = {}
        for r in split_results:
            qt = r.get("question_type", "UNK")
            qt_layer.setdefault(qt, []).append(r["best_layer"])
        print(f"  Per question_type:")
        for qt, layers in sorted(qt_layer.items(), key=lambda x: -len(x[1])):
            n = len(layers)
            l01 = sum(1 for l in layers if l <= 1) / n * 100
            print(f"    {qt:30s}  N={n:5d}  L0+1={l01:5.1f}%")

    # 合计 SciMDR
    all_scimdr = []
    for v in scimdr_results.values():
        all_scimdr.extend(v)
    if all_scimdr:
        combined_scimdr_stats = compute_stats(all_scimdr, "SciMDR-ALL answers — Rule-based")
        all_stats.append(combined_scimdr_stats)
        print(f"\n  SciMDR ALL: Layer 0+1 = {combined_scimdr_stats['layer_01_combined_pct']:.1f}%")

    # ── 输出 ──
    print_report(all_stats)

    # 保存 JSON
    out_path = output_dir / "verifiability_coverage_analysis.json"
    with open(out_path, "w") as f:
        json.dump(all_stats, f, indent=2, ensure_ascii=False)
    print(f"Full results saved to {out_path}")


if __name__ == "__main__":
    main()
