#!/usr/bin/env python3
"""
implementation_2/rq2_remediation/benchmarks/evaluate_cohort3_remediated.py

Rigorous Re-Evaluation and Comparative Benchmark of Frozen Cohort 3 (N=8 physical LoRA adapters):
SecureLoRA v3.1 (Remediated Spectral + Static Inversion)
vs.
SecureLoRA v3.2 (Remediated Spectral + EXP-RQ2-04 Gradient-Guided Trigger Search)

Scientific Constraints & Safeguards:
- Preserves all frozen adapter weights and labels under implementation/adapters/cohort3_unseen/.
- Real base-model vs. adapted model inference on Apple Silicon MPS/CPU (zero mocks).
- Development-calibrated thresholds only (tau_kl=6.5, erank=0.55/0.88).
- Evaluates per-adapter decisions, TP/FP/TN/FN, TPR, FPR, balanced accuracy, runtime, and memory.
- Writes comprehensive results to implementation_2/rq2_remediation/results/.
"""

from __future__ import annotations

import json
import time
import resource
import sys
from pathlib import Path
from typing import Dict, List, Any
import numpy as np
import pandas as pd
import torch

from implementation_2.rq2_remediation.src.securelora_remediated_pipeline import SecureLoRARemediatedPipeline


COHORT3_DIR = Path("implementation/adapters/cohort3_unseen")
RESULTS_DIR = Path("implementation_2/rq2_remediation/results")


def get_memory_mb() -> float:
    """Get current resident memory in MB."""
    if torch.backends.mps.is_available():
        mps_bytes = torch.mps.current_allocated_memory()
        if mps_bytes > 0:
            return round(mps_bytes / (1024 ** 2), 2)
    # Fallback to process RSS
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # On macOS ru_maxrss is in bytes, on Linux in KB
    factor = 1024 ** 2 if sys.platform == "darwin" else 1024
    return round(usage / factor, 2)


def evaluate_cohort(pipeline: SecureLoRARemediatedPipeline, version_name: str) -> Dict[str, Any]:
    """Evaluate all 8 Cohort 3 adapters under a specific pipeline configuration."""
    clean_dir = COHORT3_DIR / "clean"
    poison_dir = COHORT3_DIR / "poisoned"

    clean_adapters = sorted([p for p in clean_dir.iterdir() if p.is_dir() and not p.name.startswith(".")])
    poison_adapters = sorted([p for p in poison_dir.iterdir() if p.is_dir() and not p.name.startswith(".")])

    records: List[Dict[str, Any]] = []

    print(f"\n>>> Running Evaluation for {version_name} ({len(clean_adapters)} clean, {len(poison_adapters)} poisoned) <<<")

    # 1. Clean Controls
    for p in clean_adapters:
        print(f"  [{version_name} | CLEAN] {p.name}...")
        t0 = time.time()
        res = pipeline.evaluate_adapter(p)
        elapsed = round(time.time() - t0, 2)
        verdict = res["verdict"]
        correct = (verdict == "ADMITTED")
        rec = {
            "version": version_name,
            "adapter_name": p.name,
            "ground_truth": "CLEAN",
            "expected_verdict": "ADMITTED",
            "prediction": verdict,
            "is_correct": correct,
            "rejection_stage": res.get("rejection_stage"),
            "rejection_reason": res.get("reason"),
            "runtime_seconds": elapsed,
            "memory_mb": get_memory_mb(),
            "diagnostics": res.get("diagnostics", {}),
        }
        records.append(rec)
        print(f"    -> Result: {verdict} (Correct: {correct}) in {elapsed}s | Stage: {res.get('rejection_stage')}")

    # 2. Poisoned Trojans
    for p in poison_adapters:
        print(f"  [{version_name} | POISONED] {p.name}...")
        t0 = time.time()
        res = pipeline.evaluate_adapter(p)
        elapsed = round(time.time() - t0, 2)
        verdict = res["verdict"]
        correct = (verdict == "REJECTED")
        rec = {
            "version": version_name,
            "adapter_name": p.name,
            "ground_truth": "POISONED",
            "expected_verdict": "REJECTED",
            "prediction": verdict,
            "is_correct": correct,
            "rejection_stage": res.get("rejection_stage"),
            "rejection_reason": res.get("reason"),
            "runtime_seconds": elapsed,
            "memory_mb": get_memory_mb(),
            "diagnostics": res.get("diagnostics", {}),
        }
        records.append(rec)
        print(f"    -> Result: {verdict} (Correct: {correct}) in {elapsed}s | Stage: {res.get('rejection_stage')}")

    # Compute Metrics
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
    total_runtime = sum(r["runtime_seconds"] for r in records)
    mean_latency = float(np.mean([r["runtime_seconds"] for r in records]))
    max_mem = float(np.max([r["memory_mb"] for r in records]))

    return {
        "version": version_name,
        "total_adapters": len(records),
        "confusion_matrix": {
            "true_positives": tp,
            "false_negatives": fn,
            "true_negatives": tn,
            "false_positives": fp,
        },
        "metrics": {
            "tpr_recall": round(tpr, 4),
            "fpr_fallout": round(fpr, 4),
            "specificity": round(specificity, 4),
            "precision": round(precision, 4),
            "accuracy": round(accuracy, 4),
            "balanced_accuracy": round(balanced_acc, 4),
            "total_runtime_seconds": round(total_runtime, 2),
            "mean_latency_seconds": round(mean_latency, 2),
            "peak_memory_mb": round(max_mem, 2),
        },
        "per_adapter_results": records,
    }


def run_cohort3_full_comparison():
    print("=" * 80)
    print("SECURELORA COHORT 3 GENERALIZATION BENCHMARK: v3.1 vs. v3.2 COMPARISON")
    print("=" * 80)

    # 1. Pipeline v3.1: Remediated Spectral + Static Inversion (heuristic seeds)
    pipe_v31 = SecureLoRARemediatedPipeline(
        base_model_name_or_path="gpt2",
        cache_dir="implementation/models.nosync/hf_cache",
        erank_ratio_threshold=0.55,
        erank_max_threshold=0.88,
        tau_uas=3.5,
        tau_invariance=0.50,
        enable_tier2_spectral=True,
        enable_tier2b_inversion=True,
        enable_tier2b_gradient_search=False,
        enable_tier3_live_probe=True,
    )
    summary_v31 = evaluate_cohort(pipe_v31, "SecureLoRA v3.1 (Static Probing)")

    # 2. Pipeline v3.2: Remediated Spectral + EXP-RQ2-04 Gradient-Guided Trigger Search
    pipe_v32 = SecureLoRARemediatedPipeline(
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
    summary_v32 = evaluate_cohort(pipe_v32, "SecureLoRA v3.2 (Gradient-Guided Trigger Search)")

    # Historical Baseline
    historical_baseline = {
        "version": "SecureLoRA v3.0 (Historical Baseline)",
        "confusion_matrix": {"true_positives": 0, "false_negatives": 4, "true_negatives": 3, "false_positives": 1},
        "metrics": {
            "tpr_recall": 0.0,
            "fpr_fallout": 0.25,
            "specificity": 0.75,
            "precision": 0.0,
            "accuracy": 0.375,
            "balanced_accuracy": 0.375,
            "mean_latency_seconds": 15.2,
            "peak_memory_mb": 1150.0,
        },
    }

    comparison = {
        "benchmark": "Cohort 3 Frozen Generalization Stress Test (N=8)",
        "model_architecture": "GPT-2 (d=768, 12 layers, c_attn)",
        "timestamp": time.time(),
        "versions": {
            "v3.0_historical": historical_baseline,
            "v3.1_remediated_spectral": summary_v31,
            "v3.2_gradient_search": summary_v32,
        },
        "comparative_gains": {
            "v3.2_vs_historical_balanced_acc_gain": round(summary_v32["metrics"]["balanced_accuracy"] - historical_baseline["metrics"]["balanced_accuracy"], 4),
            "v3.2_vs_v3.1_balanced_acc_gain": round(summary_v32["metrics"]["balanced_accuracy"] - summary_v31["metrics"]["balanced_accuracy"], 4),
            "v3.2_vs_historical_tpr_gain": round(summary_v32["metrics"]["tpr_recall"] - historical_baseline["metrics"]["tpr_recall"], 4),
            "v3.2_vs_v3.1_tpr_gain": round(summary_v32["metrics"]["tpr_recall"] - summary_v31["metrics"]["tpr_recall"], 4),
            "fpr_maintained": summary_v32["metrics"]["fpr_fallout"] == 0.0,
        },
    }

    # Save to files
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    json_path = RESULTS_DIR / "cohort3_remediated_results.json"
    with open(json_path, "w") as f:
        json.dump(summary_v32, f, indent=2)

    comp_json_path = RESULTS_DIR / "cohort3_v31_vs_v32_comparison.json"
    with open(comp_json_path, "w") as f:
        json.dump(comparison, f, indent=2)

    # Save CSVs
    df_v32_rows = []
    for r in summary_v32["per_adapter_results"]:
        df_v32_rows.append({
            "adapter_name": r["adapter_name"],
            "ground_truth": r["ground_truth"],
            "prediction": r["prediction"],
            "is_correct": r["is_correct"],
            "rejection_stage": r["rejection_stage"],
            "rejection_reason": r["rejection_reason"],
            "runtime_seconds": r["runtime_seconds"],
            "memory_mb": r["memory_mb"],
        })
    df_v32 = pd.DataFrame(df_v32_rows)
    csv_path = RESULTS_DIR / "cohort3_remediated_results.csv"
    df_v32.to_csv(csv_path, index=False)

    df_comp_rows = []
    for r31, r32 in zip(summary_v31["per_adapter_results"], summary_v32["per_adapter_results"]):
        df_comp_rows.append({
            "adapter_name": r31["adapter_name"],
            "ground_truth": r31["ground_truth"],
            "v3.1_prediction": r31["prediction"],
            "v3.1_stage": r31["rejection_stage"],
            "v3.1_correct": r31["is_correct"],
            "v3.1_runtime": r31["runtime_seconds"],
            "v3.2_prediction": r32["prediction"],
            "v3.2_stage": r32["rejection_stage"],
            "v3.2_correct": r32["is_correct"],
            "v3.2_runtime": r32["runtime_seconds"],
        })
    df_comp = pd.DataFrame(df_comp_rows)
    comp_csv_path = RESULTS_DIR / "cohort3_v31_vs_v32_comparison.csv"
    df_comp.to_csv(comp_csv_path, index=False)

    # Print Formatted Report
    print("\n" + "=" * 80)
    print("COHORT 3 FROZEN BENCHMARK SUMMARY TABLE:")
    print("=" * 80)
    print(f"{'Metric':<25} | {'v3.0 (Historical)':<18} | {'v3.1 (Remediated)':<18} | {'v3.2 (EXP-RQ2-04)':<18}")
    print("-" * 80)
    m0 = historical_baseline["metrics"]
    m1 = summary_v31["metrics"]
    m2 = summary_v32["metrics"]
    print(f"{'True Positives (TP)':<25} | {historical_baseline['confusion_matrix']['true_positives']:<18} | {summary_v31['confusion_matrix']['true_positives']:<18} | {summary_v32['confusion_matrix']['true_positives']:<18}")
    print(f"{'False Negatives (FN)':<25} | {historical_baseline['confusion_matrix']['false_negatives']:<18} | {summary_v31['confusion_matrix']['false_negatives']:<18} | {summary_v32['confusion_matrix']['false_negatives']:<18}")
    print(f"{'True Negatives (TN)':<25} | {historical_baseline['confusion_matrix']['true_negatives']:<18} | {summary_v31['confusion_matrix']['true_negatives']:<18} | {summary_v32['confusion_matrix']['true_negatives']:<18}")
    print(f"{'False Positives (FP)':<25} | {historical_baseline['confusion_matrix']['false_positives']:<18} | {summary_v31['confusion_matrix']['false_positives']:<18} | {summary_v32['confusion_matrix']['false_positives']:<18}")
    print(f"{'TPR (Recall)':<25} | {m0['tpr_recall']:<18.1%} | {m1['tpr_recall']:<18.1%} | {m2['tpr_recall']:<18.1%}")
    print(f"{'FPR (Fallout)':<25} | {m0['fpr_fallout']:<18.1%} | {m1['fpr_fallout']:<18.1%} | {m2['fpr_fallout']:<18.1%}")
    print(f"{'Accuracy':<25} | {m0['accuracy']:<18.1%} | {m1['accuracy']:<18.1%} | {m2['accuracy']:<18.1%}")
    print(f"{'Balanced Accuracy':<25} | {m0['balanced_accuracy']:<18.1%} | {m1['balanced_accuracy']:<18.1%} | {m2['balanced_accuracy']:<18.1%}")
    print(f"{'Mean Latency':<25} | {m0['mean_latency_seconds']:<16.2f}s | {m1['mean_latency_seconds']:<16.2f}s | {m2['mean_latency_seconds']:<16.2f}s")
    print(f"{'Peak Memory':<25} | {m0['peak_memory_mb']:<15.1f}MB | {m1['peak_memory_mb']:<15.1f}MB | {m2['peak_memory_mb']:<15.1f}MB")
    print("=" * 80)
    print(f"Results saved to: {json_path}")
    print(f"Comparison saved to: {comp_json_path} and {comp_csv_path}")

    return comparison


if __name__ == "__main__":
    run_cohort3_full_comparison()
