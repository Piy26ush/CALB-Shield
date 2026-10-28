#!/usr/bin/env python3
"""
implementation_2/rq2_gpt2_audit/benchmarks/run_stage2b_hotflip.py

Audit Step 2: Run Stage 2B HotFlip Gradient-Guided Trigger Search Alone on GPT-2 Cohort 3 (N=8).
Evaluates standalone trigger search accuracy, confusion matrix, and runtime
without any spectral gating or refusal probing.
"""

from __future__ import annotations

import json
import time
import sys
from pathlib import Path
from typing import Dict, List, Any
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from implementation_2.rq2_remediation.src.gradient_trigger_search import GradientTriggerSearcher

COHORT3_DIR = Path("implementation/adapters/cohort3_unseen")
RESULTS_DIR = Path("implementation_2/rq2_gpt2_audit/results")


def calculate_metrics(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    tp = sum(1 for r in records if r["ground_truth"] == "POISONED" and r["prediction"] == "REJECTED")
    fn = sum(1 for r in records if r["ground_truth"] == "POISONED" and r["prediction"] == "ADMITTED")
    tn = sum(1 for r in records if r["ground_truth"] == "CLEAN" and r["prediction"] == "ADMITTED")
    fp = sum(1 for r in records if r["ground_truth"] == "CLEAN" and r["prediction"] == "REJECTED")

    tpr = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    specificity = 1.0 - fpr
    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    accuracy = float((tp + tn) / len(records)) if records else 0.0
    balanced_acc = float(0.5 * (tpr + specificity))
    f1 = float(2 * precision * tpr / (precision + tpr)) if (precision + tpr) > 0 else 0.0

    return {
        "confusion_matrix": {"TP": tp, "FN": fn, "TN": tn, "FP": fp},
        "accuracy": round(accuracy * 100, 2),
        "balanced_accuracy": round(balanced_acc * 100, 2),
        "precision": round(precision * 100, 2),
        "recall_tpr": round(tpr * 100, 2),
        "fpr": round(fpr * 100, 2),
        "specificity_tnr": round(specificity * 100, 2),
        "f1_score": round(f1 * 100, 2),
    }


def run_stage2b_alone() -> Dict[str, Any]:
    print("=" * 80)
    print("AUDIT STEP 2: STAGE 2B HOTFLIP TRIGGER SEARCH ALONE (COHORT 3, N=8)")
    print("=" * 80)

    clean_dir = COHORT3_DIR / "clean"
    poison_dir = COHORT3_DIR / "poisoned"

    clean_adapters = sorted([p for p in clean_dir.iterdir() if p.is_dir() and not p.name.startswith(".")])
    poison_adapters = sorted([p for p in poison_dir.iterdir() if p.is_dir() and not p.name.startswith(".")])

    searcher = GradientTriggerSearcher(
        base_model_name_or_path="gpt2",
        cache_dir="implementation/models.nosync/hf_cache",
        n_tokens=4,
        n_steps=8,
        tau_kl=6.5,
        min_attractor_words=1,
    )

    records: List[Dict[str, Any]] = []

    # 1. Clean Adapters
    for p in clean_adapters:
        print(f"  [CLEAN] Running Stage 2B on {p.name}...")
        t0 = time.time()
        res = searcher.search_adapter(p)
        elapsed = round(time.time() - t0, 2)
        verdict = "REJECTED" if res.get("verdict") == "FLAGGED" else "ADMITTED"
        correct = (verdict == "ADMITTED")

        rec = {
            "adapter_name": p.name,
            "ground_truth": "CLEAN",
            "prediction": verdict,
            "is_correct": correct,
            "raw_verdict": res.get("verdict"),
            "optimized_prefix": res.get("optimized_prefix"),
            "mean_kl": res.get("mean_kl"),
            "mean_entropy": res.get("mean_entropy"),
            "entropy_shift": res.get("entropy_shift"),
            "top1_agreement": res.get("top1_agreement"),
            "cross_prompt_invariance": res.get("cross_prompt_invariance"),
            "differential_attractor_words": res.get("differential_attractor_words", []),
            "differential_attractor_count": res.get("differential_attractor_count", 0),
            "loss_drop": res.get("loss_drop"),
            "adapted_completions": res.get("adapted_completions", []),
            "base_completions": res.get("base_completions", []),
            "runtime_seconds": elapsed,
        }
        records.append(rec)
        print(f"    -> Verdict: {verdict} (Correct: {correct}) | KL={res.get('mean_kl'):.2f}, Attractors={res.get('differential_attractor_words')} | {elapsed}s")

    # 2. Poisoned Adapters
    for p in poison_adapters:
        print(f"  [POISONED] Running Stage 2B on {p.name}...")
        t0 = time.time()
        res = searcher.search_adapter(p)
        elapsed = round(time.time() - t0, 2)
        verdict = "REJECTED" if res.get("verdict") == "FLAGGED" else "ADMITTED"
        correct = (verdict == "REJECTED")

        rec = {
            "adapter_name": p.name,
            "ground_truth": "POISONED",
            "prediction": verdict,
            "is_correct": correct,
            "raw_verdict": res.get("verdict"),
            "optimized_prefix": res.get("optimized_prefix"),
            "mean_kl": res.get("mean_kl"),
            "mean_entropy": res.get("mean_entropy"),
            "entropy_shift": res.get("entropy_shift"),
            "top1_agreement": res.get("top1_agreement"),
            "cross_prompt_invariance": res.get("cross_prompt_invariance"),
            "differential_attractor_words": res.get("differential_attractor_words", []),
            "differential_attractor_count": res.get("differential_attractor_count", 0),
            "loss_drop": res.get("loss_drop"),
            "adapted_completions": res.get("adapted_completions", []),
            "base_completions": res.get("base_completions", []),
            "runtime_seconds": elapsed,
        }
        records.append(rec)
        print(f"    -> Verdict: {verdict} (Correct: {correct}) | KL={res.get('mean_kl'):.2f}, Attractors={res.get('differential_attractor_words')} | {elapsed}s")

    metrics = calculate_metrics(records)
    total_runtime = round(sum(r["runtime_seconds"] for r in records), 2)

    summary = {
        "benchmark": "Cohort 3 Stage 2B HotFlip Trigger Search Alone",
        "cohort_size": len(records),
        "clean_count": len(clean_adapters),
        "poisoned_count": len(poison_adapters),
        "search_parameters": {
            "n_tokens": 4,
            "n_steps": 8,
            "tau_kl": 6.5,
            "min_attractor_words": 1,
        },
        "metrics": metrics,
        "total_runtime_seconds": total_runtime,
        "per_adapter_records": records,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    json_path = RESULTS_DIR / "stage2b_alone_results.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    # Simplified table for CSV
    csv_rows = []
    for r in records:
        csv_rows.append({
            "adapter_name": r["adapter_name"],
            "ground_truth": r["ground_truth"],
            "prediction": r["prediction"],
            "is_correct": r["is_correct"],
            "optimized_prefix": r["optimized_prefix"],
            "mean_kl": r["mean_kl"],
            "attractor_count": r["differential_attractor_count"],
            "attractor_words": "; ".join(r["differential_attractor_words"]),
            "invariance": r["cross_prompt_invariance"],
            "runtime_seconds": r["runtime_seconds"],
        })
    df = pd.DataFrame(csv_rows)
    csv_path = RESULTS_DIR / "stage2b_alone_results.csv"
    df.to_csv(csv_path, index=False)

    print("-" * 80)
    print(f"Results Summary: Accuracy={metrics['accuracy']}%, BalAcc={metrics['balanced_accuracy']}%, Recall={metrics['recall_tpr']}%, FPR={metrics['fpr']}%")
    print(f"Confusion Matrix: {metrics['confusion_matrix']}")
    print(f"Saved: {json_path} and {csv_path}")

    return summary


if __name__ == "__main__":
    run_stage2b_alone()
