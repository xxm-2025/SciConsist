"""
Meta-Evaluation 对比分析: VSR vs GPT-4o Judge vs Human Annotation

对 500 条 claims 的三方标注结果进行对比:
- 标签分布
- 混淆矩阵
- Exact / Binary agreement rate
- 按 Layer 分层分析
- 错误检测率
- Cohen's Kappa

输入:
    - claims_annotated.csv: 人工标注 (human_label 列)
    - gpt4o_judge_results.jsonl: GPT-4o 多维度评分
    - claims_500.jsonl (optional): VSR 原始结果

输出:
    - meta_evaluation_report.json: 结构化报告
    - 终端打印详细对比
"""

import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

META_DIR = Path("/root/shared-nvme/sciconsist_pilot/outputs/meta_evaluation")
ANNOTATED_CSV = Path("/root/SciConsist/temp/claims_annotated.csv")
LABELS = ["CORRECT", "PARTIALLY_CORRECT", "WRONG", "UNVERIFIABLE"]


def vsr_reward_to_label(reward: float) -> str:
    """将 VSR reward 数值映射为离散标签。

    Args:
        reward: VSR reward 值 (-1.0 ~ +1.0)

    Returns:
        对应的标签字符串
    """
    if reward > 0.3:
        return "CORRECT"
    elif reward > 0:
        return "PARTIALLY_CORRECT"
    elif reward == 0:
        return "UNVERIFIABLE"
    elif reward > -0.3:
        return "PARTIALLY_CORRECT"
    else:
        return "WRONG"


def load_human_annotations(path: Path) -> dict[str, dict[str, Any]]:
    """加载人工标注 CSV。"""
    data = {}
    with open(path, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            data[r["claim_id"]] = {
                "label": r["human_label"].strip(),
                "notes": r.get("human_notes", "").strip(),
                "layer": r["primary_layer"],
                "vsr_reward": float(r["vsr_reward"]),
            }
    return data


def load_gpt4o_results(path: Path) -> dict[str, str]:
    """加载 GPT-4o judge 结果，提取 overall score。"""
    data = {}
    with open(path) as f:
        for line in f:
            obj = json.loads(line)
            score = obj["judge_result"].get("overall", {}).get("score", "UNKNOWN")
            data[obj["claim_id"]] = score
    return data


def compute_agreement(a: dict[str, str], b: dict[str, str], ids: list[str]) -> float:
    """计算两组标签的 exact agreement rate。"""
    if not ids:
        return 0.0
    return sum(1 for cid in ids if a.get(cid) == b.get(cid)) / len(ids)


def compute_cohens_kappa(a: dict[str, str], b: dict[str, str], ids: list[str], labels: list[str]) -> float:
    """计算 Cohen's Kappa 系数。

    Args:
        a, b: claim_id -> label 映射
        ids: 要计算的 claim_id 列表
        labels: 所有可能的标签

    Returns:
        Cohen's Kappa 值
    """
    n = len(ids)
    if n == 0:
        return 0.0
    po = sum(1 for cid in ids if a.get(cid) == b.get(cid)) / n
    pe = 0.0
    for label in labels:
        pa = sum(1 for cid in ids if a.get(cid) == label) / n
        pb = sum(1 for cid in ids if b.get(cid) == label) / n
        pe += pa * pb
    if pe == 1.0:
        return 1.0
    return (po - pe) / (1 - pe)


def confusion_matrix(a: dict[str, str], b: dict[str, str], ids: list[str], labels: list[str]) -> list[list[int]]:
    """生成混淆矩阵，行=a, 列=b。"""
    matrix = []
    for al in labels:
        row = []
        for bl in labels:
            row.append(sum(1 for cid in ids if a.get(cid) == al and b.get(cid) == bl))
        matrix.append(row)
    return matrix


def print_confusion(name_a: str, name_b: str, matrix: list[list[int]], labels: list[str]) -> None:
    """打印混淆矩阵。"""
    short = [l[:10] for l in labels]
    header = f"{name_a + ' / ' + name_b:<22}"
    for s in short:
        header += f"{s:>12}"
    print(header)
    print("-" * (22 + 12 * len(labels)))
    for i, label in enumerate(labels):
        row_str = f"{label:<22}"
        for j in range(len(labels)):
            row_str += f"{matrix[i][j]:>12}"
        print(row_str)


def to_binary(label: str) -> str:
    return "POS" if label in ("CORRECT", "PARTIALLY_CORRECT") else "NEG"


def main() -> None:
    human = load_human_annotations(ANNOTATED_CSV)
    gpt4o = load_gpt4o_results(META_DIR / "gpt4o_judge_results.jsonl")

    human_labels = {cid: v["label"] for cid, v in human.items()}
    vsr_labels = {cid: vsr_reward_to_label(v["vsr_reward"]) for cid, v in human.items()}
    all_ids = list(human.keys())

    print("=" * 70)
    print(f"META-EVALUATION: VSR vs GPT-4o vs Human (N={len(all_ids)})")
    print("=" * 70)

    # --- 标签分布 ---
    h_counts = Counter(human_labels.values())
    g_counts = Counter(v for cid, v in gpt4o.items() if cid in human)
    v_counts = Counter(vsr_labels.values())

    print(f"\n### 1. 标签分布")
    print(f"{'Label':<22} {'Human':>8} {'GPT-4o':>8} {'VSR':>8}")
    print("-" * 48)
    for label in LABELS:
        print(f"{label:<22} {h_counts.get(label, 0):>8} {g_counts.get(label, 0):>8} {v_counts.get(label, 0):>8}")

    # --- 混淆矩阵 ---
    print(f"\n### 2. Human vs GPT-4o 混淆矩阵")
    cm_hg = confusion_matrix(human_labels, gpt4o, all_ids, LABELS)
    print_confusion("Human", "GPT-4o", cm_hg, LABELS)

    print(f"\n### 3. Human vs VSR 混淆矩阵")
    cm_hv = confusion_matrix(human_labels, vsr_labels, all_ids, LABELS)
    print_confusion("Human", "VSR", cm_hv, LABELS)

    print(f"\n### 4. GPT-4o vs VSR 混淆矩阵")
    cm_gv = confusion_matrix(gpt4o, vsr_labels, all_ids, LABELS)
    print_confusion("GPT-4o", "VSR", cm_gv, LABELS)

    # --- Agreement ---
    exact_hg = compute_agreement(human_labels, gpt4o, all_ids)
    exact_hv = compute_agreement(human_labels, vsr_labels, all_ids)
    exact_gv = compute_agreement(gpt4o, vsr_labels, all_ids)

    kappa_hg = compute_cohens_kappa(human_labels, gpt4o, all_ids, LABELS)
    kappa_hv = compute_cohens_kappa(human_labels, vsr_labels, all_ids, LABELS)
    kappa_gv = compute_cohens_kappa(gpt4o, vsr_labels, all_ids, LABELS)

    print(f"\n### 5. Exact Agreement & Cohen's Kappa")
    print(f"  Human vs GPT-4o:  {exact_hg:.1%}  (kappa={kappa_hg:.3f})")
    print(f"  Human vs VSR:     {exact_hv:.1%}  (kappa={kappa_hv:.3f})")
    print(f"  GPT-4o vs VSR:    {exact_gv:.1%}  (kappa={kappa_gv:.3f})")

    # --- Binary ---
    bin_h = {cid: to_binary(v) for cid, v in human_labels.items()}
    bin_g = {cid: to_binary(v) for cid, v in gpt4o.items()}
    bin_v = {cid: to_binary(v) for cid, v in vsr_labels.items()}

    b_hg = compute_agreement(bin_h, bin_g, all_ids)
    b_hv = compute_agreement(bin_h, bin_v, all_ids)

    print(f"\n### 6. Binary Agreement (CORRECT+PARTIAL vs WRONG+UNVERIFIABLE)")
    print(f"  Human vs GPT-4o:  {b_hg:.1%}")
    print(f"  Human vs VSR:     {b_hv:.1%}")

    # --- 按 Layer ---
    print(f"\n### 7. 按 Layer 分析")
    for layer in ["L0", "L1", "L2"]:
        layer_ids = [cid for cid, v in human.items() if v["layer"] == layer]
        hg = compute_agreement(human_labels, gpt4o, layer_ids)
        hv = compute_agreement(human_labels, vsr_labels, layer_ids)
        kg = compute_cohens_kappa(human_labels, gpt4o, layer_ids, LABELS)
        kv = compute_cohens_kappa(human_labels, vsr_labels, layer_ids, LABELS)

        layer_h = Counter(human_labels[cid] for cid in layer_ids)
        layer_g = Counter(gpt4o.get(cid, "?") for cid in layer_ids)
        layer_v = Counter(vsr_labels[cid] for cid in layer_ids)

        print(f"\n  {layer} (n={len(layer_ids)})")
        print(f"    Agreement:  Human-GPT4o={hg:.1%} (k={kg:.3f})  Human-VSR={hv:.1%} (k={kv:.3f})")
        print(f"    Human:  {dict(layer_h)}")
        print(f"    GPT-4o: {dict(layer_g)}")
        print(f"    VSR:    {dict(layer_v)}")

    # --- 错误检测 ---
    wrong_ids = [cid for cid in all_ids if human_labels[cid] == "WRONG"]
    print(f"\n### 8. 错误检测率 (Human=WRONG, n={len(wrong_ids)})")
    if wrong_ids:
        gpt4o_catch_strict = sum(1 for cid in wrong_ids if gpt4o.get(cid) == "WRONG") / len(wrong_ids)
        gpt4o_catch_loose = sum(1 for cid in wrong_ids if gpt4o.get(cid) in ("WRONG", "PARTIALLY_CORRECT")) / len(wrong_ids)
        vsr_catch_strict = sum(1 for cid in wrong_ids if vsr_labels[cid] == "WRONG") / len(wrong_ids)
        vsr_catch_loose = sum(1 for cid in wrong_ids if vsr_labels[cid] in ("WRONG", "PARTIALLY_CORRECT")) / len(wrong_ids)
        print(f"  GPT-4o:  strict={gpt4o_catch_strict:.1%}  loose={gpt4o_catch_loose:.1%}")
        print(f"  VSR:     strict={vsr_catch_strict:.1%}  loose={vsr_catch_loose:.1%}")

        print(f"\n  Human=WRONG 但 GPT-4o=CORRECT 的 case:")
        for cid in wrong_ids:
            if gpt4o.get(cid) == "CORRECT":
                claim = human[cid]
                print(f"    {cid}: vsr={claim['vsr_reward']}, notes={claim['notes'][:80]}")
    else:
        print("  (无 WRONG 标注)")

    # --- 误判分析 ---
    print(f"\n### 9. 误判分析")
    correct_ids = [cid for cid in all_ids if human_labels[cid] == "CORRECT"]
    gpt4o_fp = sum(1 for cid in correct_ids if gpt4o.get(cid) == "WRONG") / len(correct_ids) if correct_ids else 0
    vsr_fp = sum(1 for cid in correct_ids if vsr_labels[cid] == "WRONG") / len(correct_ids) if correct_ids else 0
    print(f"  Human=CORRECT 但判为 WRONG (误杀率):")
    print(f"    GPT-4o: {gpt4o_fp:.1%} ({sum(1 for cid in correct_ids if gpt4o.get(cid) == 'WRONG')}/{len(correct_ids)})")
    print(f"    VSR:    {vsr_fp:.1%} ({sum(1 for cid in correct_ids if vsr_labels[cid] == 'WRONG')}/{len(correct_ids)})")

    # --- 保存报告 ---
    report = {
        "n_claims": len(all_ids),
        "distribution": {"human": dict(h_counts), "gpt4o": dict(g_counts), "vsr": dict(v_counts)},
        "exact_agreement": {"human_gpt4o": round(exact_hg, 4), "human_vsr": round(exact_hv, 4), "gpt4o_vsr": round(exact_gv, 4)},
        "cohens_kappa": {"human_gpt4o": round(kappa_hg, 4), "human_vsr": round(kappa_hv, 4), "gpt4o_vsr": round(kappa_gv, 4)},
        "binary_agreement": {"human_gpt4o": round(b_hg, 4), "human_vsr": round(b_hv, 4)},
        "confusion_human_gpt4o": cm_hg,
        "confusion_human_vsr": cm_hv,
    }
    out_path = META_DIR / "meta_evaluation_report.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n报告已保存: {out_path}")


if __name__ == "__main__":
    main()
