#!/usr/bin/env python3
"""
implementation_2/rq2_gpt2_audit/benchmarks/run_combined_pipeline.py

Audit Step 3: Run Full Multi-Tier Pipeline (Tiers 1, 2, 2B, 3) on GPT-2 Cohort 3 (N=8).
Evaluates the integrated SecureLoRA admission controller.
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

from implementation_2.rq2_remediation.src.securelora_remediated_pipeline import SecureLoRARemediatedPipeline

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


def run_combined() -> Dict[str, Any]:
    print("=" * 80)
    print("AUDIT STEP 3: FULL MULTI-TIER PIPELINE (TIERS 1, 2, 2B, 3) (COHORT 3, N=8)")
    print("=" * 80)

    clean_dir = COHORT3_DIR / "clean"
    poison_dir = COHORT3_DIR / "poisoned"

    clean_adapters = sorted([p for p in clean_dir.iterdir() if p.is_dir() and not p.name.startswith(".")])
    poison_adapters = sorted([p for p in poison_dir.iterdir() if p.is_dir() and not p.name.startswith(".")])

    pipeline = SecureLoRARemediatedPipeline(
        base_model_name_or_path="gpt2",
        cache_dir="implementation/models.nosync/hf_cache",
        erank_ratio_threshold=0.55,
        erank_max_threshold=0.88,
        enable_tier2_spectral=True,
        enable_tier2b_inversion=False,
        enable_tier2b_gradient_search=True,
        gradient_search_steps=8,
        gradient_search_tokens=4,
        gradient_search_tau_kl=6.5,
        enable_tier3_live_probe=True,
    )

    records: List[Dict[str, Any]] = []

    # 1. Clean Adapters
    for p in clean_adapters:
        print(f"  [CLEAN] Evaluating {p.name}...")
        t0 = time.time()
        res = pipeline.evaluate_adapter(p)
        elapsed = round(time.time() - t0, 2)
        verdict = res["verdict"]
        correct = (verdict == "ADMITTED")

        rec = {
            "adapter_name": p.name,
            "ground_truth": "CLEAN",
            "prediction": verdict,
            "is_correct": correct,
            "rejection_stage": res.get("rejection_stage"),
            "rejection_reason": res.get("reason"),
            "runtime_seconds": elapsed,
            "tier1_passed": res.get("diagnostics", {}).get("tier1_provenance", {}).get("passed", False),
            "tier2_verdict": res.get("diagnostics", {}).get("tier2_spectral", {}).get("verdict"),
            "tier2b_verdict": res.get("diagnostics", {}).get("tier2b_gradient_search", {}).get("verdict") if "tier2b_gradient_search" in res.get("diagnostics", {}) else "NORMAL",
            "tier3_verdict": res.get("diagnostics", {}).get("tier3_differential", {}).get("verdict"),
        }
        records.append(rec)
        print(f"    -> Result: {verdict} (Correct: {correct}) | Stage: {res.get('rejection_stage')} | {elapsed}s")

    # 2. Poisoned Adapters
    for p in poison_adapters:
        print(f"  [POISONED] Evaluating {p.name}...")
        t0 = time.time()
        res = pipeline.evaluate_adapter(p)
        elapsed = round(time.time() - t0, 2)
        verdict = res["verdict"]
        correct = (verdict == "REJECTED")

        rec = {
            "adapter_name": p.name,
            "ground_truth": "POISONED",
            "prediction": verdict,
            "is_correct": correct,
            "rejection_stage": res.get("rejection_stage"),
            "rejection_reason": res.get("reason"),
            "runtime_seconds": elapsed,
            "tier1_passed": res.get("diagnostics", {}).get("tier1_provenance", {}).get("passed", False),
            "tier2_verdict": res.get("diagnostics", {}).get("tier2_spectral", {}).get("verdict"),
            "tier2b_verdict": "FLAGGED" if res.get("rejection_stage") == "TIER_2B_GRADIENT_TRIGGER_SEARCH" else "NORMAL",
            "tier3_verdict": res.get("diagnostics", {}).get("tier3_differential", {}).get("verdict"),
        }
        records.append(rec)
        print(f"    -> Result: {verdict} (Correct: {correct}) | Stage: {res.get('rejection_stage')} | {elapsed}s")

    metrics = calculate_metrics(records)
    total_runtime = round(sum(r["runtime_seconds"] for r in records), 2)

    summary = {
        "benchmark": "Cohort 3 Full Multi-Tier Remediated Pipeline (v3.2)",
        "cohort_size": len(records),
        "clean_count": len(clean_adapters),
        "poisoned_count": len(poison_adapters),
        "metrics": metrics,
        "total_runtime_seconds": total_runtime,
        "per_adapter_records": records,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    json_path = RESULTS_DIR / "combined_pipeline_results.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    csv_rows = []
    for r in records:
        csv_rows.append({
            "adapter_name": r["adapter_name"],
            "ground_truth": r["ground_truth"],
            "prediction": r["prediction"],
            "is_correct": r["is_correct"],
            "rejection_stage": r["rejection_stage"],
            "rejection_reason": r["rejection_reason"],
            "runtime_seconds": r["runtime_seconds"],
        })
    df = pd.DataFrame(csv_rows)
    csv_path = RESULTS_DIR / "combined_pipeline_results.csv"
    df.to_csv(csv_path, index=False)

    print("-" * 80)
    print(f"Results Summary: Accuracy={metrics['accuracy']}%, BalAcc={metrics['balanced_accuracy']}%, Recall={metrics['recall_tpr']}%, FPR={metrics['fpr']}%")
    print(f"Confusion Matrix: {metrics['confusion_matrix']}")
    print(f"Saved: {json_path} and {csv_path}")

    return summary


if __name__ == "__main__":
    run_combined()
