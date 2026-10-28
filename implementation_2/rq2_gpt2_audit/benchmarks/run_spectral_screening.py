#!/usr/bin/env python3
"""
implementation_2/rq2_gpt2_audit/benchmarks/run_spectral_screening.py

Audit Step 1: Run Tier 2 Rank-Aware Spectral Screening Alone on GPT-2 Cohort 3 (N=8).
Evaluates standalone spectral screening accuracy, confusion matrix, and runtime
without any behavioral or trigger search assistance.
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

from implementation_2.rq2_remediation.src.rank_spectral_scanner import RankAwareSpectralScanner

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


def run_spectral_alone() -> Dict[str, Any]:
    print("=" * 80)
    print("AUDIT STEP 1: TIER 2 RANK-AWARE SPECTRAL SCREENING ALONE (COHORT 3, N=8)")
    print("=" * 80)

    clean_dir = COHORT3_DIR / "clean"
    poison_dir = COHORT3_DIR / "poisoned"

    clean_adapters = sorted([p for p in clean_dir.iterdir() if p.is_dir() and not p.name.startswith(".")])
    poison_adapters = sorted([p for p in poison_dir.iterdir() if p.is_dir() and not p.name.startswith(".")])

    scanner = RankAwareSpectralScanner(
        erank_ratio_threshold=0.55,
        erank_max_threshold=0.88,
    )

    records: List[Dict[str, Any]] = []

    # 1. Clean Adapters
    for p in clean_adapters:
        t0 = time.time()
        res = scanner.scan_adapter(p)
        elapsed = round(time.time() - t0, 3)
        verdict = "REJECTED" if res.get("verdict") == "FLAGGED" else "ADMITTED"
        correct = (verdict == "ADMITTED")

        mean_erank = float(np.mean([m.get("erank_ratio", 0.0) for m in res.get("per_layer_metrics", {}).values() if isinstance(m, dict)]))
        mean_top1 = float(np.mean([m.get("top1_ratio", 0.0) for m in res.get("per_layer_metrics", {}).values() if isinstance(m, dict)]))
        max_spectral_norm = float(max([m.get("spectral_norm", 0.0) for m in res.get("per_layer_metrics", {}).values() if isinstance(m, dict)], default=0.0))

        rec = {
            "adapter_name": p.name,
            "ground_truth": "CLEAN",
            "prediction": verdict,
            "is_correct": correct,
            "raw_verdict": res.get("verdict"),
            "anomaly_type": res.get("anomaly_type"),
            "anomaly_reason": res.get("anomaly_reason"),
            "mean_erank_ratio": round(mean_erank, 4),
            "mean_top1_ratio": round(mean_top1, 4),
            "max_spectral_norm": round(max_spectral_norm, 4),
            "runtime_seconds": elapsed,
        }
        records.append(rec)
        print(f"  [CLEAN] {p.name:<38} -> {verdict:<8} (Correct: {str(correct):<5}) | erank: {mean_erank:.3f}, top1: {mean_top1:.3f} | {elapsed:.3f}s")

    # 2. Poisoned Adapters
    for p in poison_adapters:
        t0 = time.time()
        res = scanner.scan_adapter(p)
        elapsed = round(time.time() - t0, 3)
        verdict = "REJECTED" if res.get("verdict") == "FLAGGED" else "ADMITTED"
        correct = (verdict == "REJECTED")

        mean_erank = float(np.mean([m.get("erank_ratio", 0.0) for m in res.get("per_layer_metrics", {}).values() if isinstance(m, dict)]))
        mean_top1 = float(np.mean([m.get("top1_ratio", 0.0) for m in res.get("per_layer_metrics", {}).values() if isinstance(m, dict)]))
        max_spectral_norm = float(max([m.get("spectral_norm", 0.0) for m in res.get("per_layer_metrics", {}).values() if isinstance(m, dict)], default=0.0))

        rec = {
            "adapter_name": p.name,
            "ground_truth": "POISONED",
            "prediction": verdict,
            "is_correct": correct,
            "raw_verdict": res.get("verdict"),
            "anomaly_type": res.get("anomaly_type"),
            "anomaly_reason": res.get("anomaly_reason"),
            "mean_erank_ratio": round(mean_erank, 4),
            "mean_top1_ratio": round(mean_top1, 4),
            "max_spectral_norm": round(max_spectral_norm, 4),
            "runtime_seconds": elapsed,
        }
        records.append(rec)
        print(f"  [POISONED] {p.name:<38} -> {verdict:<8} (Correct: {str(correct):<5}) | erank: {mean_erank:.3f}, top1: {mean_top1:.3f} | {elapsed:.3f}s")

    metrics = calculate_metrics(records)
    total_runtime = round(sum(r["runtime_seconds"] for r in records), 3)

    summary = {
        "benchmark": "Cohort 3 Tier 2 Spectral Screening Alone",
        "cohort_size": len(records),
        "clean_count": len(clean_adapters),
        "poisoned_count": len(poison_adapters),
        "metrics": metrics,
        "total_runtime_seconds": total_runtime,
        "per_adapter_records": records,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    json_path = RESULTS_DIR / "spectral_alone_results.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    df = pd.DataFrame(records)
    csv_path = RESULTS_DIR / "spectral_alone_results.csv"
    df.to_csv(csv_path, index=False)

    print("-" * 80)
    print(f"Results Summary: Accuracy={metrics['accuracy']}%, BalAcc={metrics['balanced_accuracy']}%, Recall={metrics['recall_tpr']}%, FPR={metrics['fpr']}%")
    print(f"Confusion Matrix: {metrics['confusion_matrix']}")
    print(f"Saved: {json_path} and {csv_path}")

    return summary


if __name__ == "__main__":
    run_spectral_alone()
