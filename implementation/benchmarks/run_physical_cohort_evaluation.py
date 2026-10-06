#!/usr/bin/env python3
"""
benchmarks/run_physical_cohort_evaluation.py
Phase 4: Empirical Fast QR-SVD Evaluation Across 20 Matched Physical LoRA Adapters.

Audits both the Clean (N=10) and Poisoned (N=10) physical adapter cohorts on
meta-llama/Meta-Llama-3-8B-Instruct (32 layers, 64 LoRA projection tensors per adapter).

Records layer-by-layer metrics, effective ranks, spectral norms, scan latency,
and confusion matrix under current vs calibrated thresholds.
"""

import os
import sys
import json
import time
from pathlib import Path
import pandas as pd
import numpy as np

CURRENT_DIR = Path(__file__).resolve().parent.parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from src.svd_scanner import SVDSpectralScanner

COHORT_DIR = CURRENT_DIR / "adapters" / "physical_cohort"
CLEAN_DIR = COHORT_DIR / "clean"
POISONED_DIR = COHORT_DIR / "poisoned"

OUTPUT_DIR = CURRENT_DIR / "results" / "physical_benchmarks"
OUTPUT_CSV = OUTPUT_DIR / "physical_cohort_svd_results.csv"
OUTPUT_JSON = OUTPUT_DIR / "physical_cohort_svd_results.json"

def evaluate_cohort():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    scanner = SVDSpectralScanner(threshold=0.40)
    
    clean_adapters = sorted([p for p in CLEAN_DIR.iterdir() if p.is_dir()])
    poisoned_adapters = sorted([p for p in POISONED_DIR.iterdir() if p.is_dir()])
    
    all_targets = [(p, "CLEAN") for p in clean_adapters] + [(p, "POISONED") for p in poisoned_adapters]
    
    results = []
    print(f"[*] Running Fast QR-SVD Scan on {len(all_targets)} Physical Adapters (32 Layers Each)...")
    print("-" * 95)
    print(f"{'Adapter Name':<35} | {'Type':<8} | {'Min ER':<7} | {'Mean ER':<7} | {'Max Norm':<9} | {'Top-1 %':<7} | {'Time (s)':<8}")
    print("-" * 95)
    
    for path, ground_truth in all_targets:
        t0 = time.perf_counter()
        scan_res = scanner.scan_adapter(str(path))
        elapsed = time.perf_counter() - t0
        
        min_er = scan_res.get("min_effective_rank", 0.0)
        mean_er = scan_res.get("mean_effective_rank", 0.0)
        max_norm = scan_res.get("max_spectral_norm", 0.0)
        mean_norm = scan_res.get("mean_spectral_norm", 0.0)
        max_top1 = scan_res.get("max_top1_spectral_ratio", 0.0)
        mean_top1 = scan_res.get("mean_top1_spectral_ratio", 0.0)
        layers_analyzed = scan_res.get("n_layers_analyzed", 0)
        layers_flagged = scan_res.get("n_layers_flagged", 0)
        
        # Current pipeline logic: Flagged if min_er < 2.0 and max_norm > 100.0 (Rank-1 Collapse)
        current_pipeline_flag = (min_er < 2.0 and max_norm > 100.0)
        
        # Pure Spectral Entropy logic: Flagged if min_er < 2.5
        er_flag = (min_er < 2.5)
        
        results.append({
            "adapter_name": path.name,
            "ground_truth": ground_truth,
            "layers_analyzed": layers_analyzed,
            "layers_flagged_at_040": layers_flagged,
            "min_effective_rank": round(min_er, 4),
            "mean_effective_rank": round(mean_er, 4),
            "max_spectral_norm": round(max_norm, 4),
            "mean_spectral_norm": round(mean_norm, 4),
            "max_top1_ratio": round(max_top1, 4),
            "mean_top1_ratio": round(mean_top1, 4),
            "scan_latency_sec": round(elapsed, 4),
            "current_pipeline_rejected": current_pipeline_flag,
            "er_threshold_rejected": er_flag
        })
        
        print(f"{path.name:<35} | {ground_truth:<8} | {min_er:<7.3f} | {mean_er:<7.3f} | {max_norm:<9.2f} | {max_top1*100:<6.1f}% | {elapsed:<8.4f}")

    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_CSV, index=False)
    
    # Compute confusion matrices
    # 1. Current pipeline rule: min_er < 2.0 and max_norm > 100
    tp_curr = int(((df["ground_truth"] == "POISONED") & df["current_pipeline_rejected"]).sum())
    fp_curr = int(((df["ground_truth"] == "CLEAN") & df["current_pipeline_rejected"]).sum())
    tn_curr = int(((df["ground_truth"] == "CLEAN") & ~df["current_pipeline_rejected"]).sum())
    fn_curr = int(((df["ground_truth"] == "POISONED") & ~df["current_pipeline_rejected"]).sum())
    
    # 2. ER < 2.5 threshold rule
    tp_er = int(((df["ground_truth"] == "POISONED") & df["er_threshold_rejected"]).sum())
    fp_er = int(((df["ground_truth"] == "CLEAN") & df["er_threshold_rejected"]).sum())
    tn_er = int(((df["ground_truth"] == "CLEAN") & ~df["er_threshold_rejected"]).sum())
    fn_er = int(((df["ground_truth"] == "POISONED") & ~df["er_threshold_rejected"]).sum())
    
    summary = {
        "n_adapters": len(results),
        "n_clean": len(clean_adapters),
        "n_poisoned": len(poisoned_adapters),
        "mean_scan_time_sec": round(df["scan_latency_sec"].mean(), 4),
        "current_rule": {
            "description": "min_effective_rank < 2.0 AND max_spectral_norm > 100.0",
            "true_positives": tp_curr,
            "false_positives": fp_curr,
            "true_negatives": tn_curr,
            "false_negatives": fn_curr,
            "tpr_recall_pct": round(tp_curr / len(poisoned_adapters) * 100, 2),
            "false_alarm_rate_pct": round(fp_curr / len(clean_adapters) * 100, 2)
        },
        "calibrated_er_rule": {
            "description": "min_effective_rank < 2.5",
            "true_positives": tp_er,
            "false_positives": fp_er,
            "true_negatives": tn_er,
            "false_negatives": fn_er,
            "tpr_recall_pct": round(tp_er / len(poisoned_adapters) * 100, 2),
            "false_alarm_rate_pct": round(fp_er / len(clean_adapters) * 100, 2)
        }
    }
    
    with open(OUTPUT_JSON, "w") as f:
        json.dump(summary, f, indent=2)
        
    print("-" * 95)
    print(f"\n[✓] Results saved to:\n    CSV:  {OUTPUT_CSV}\n    JSON: {OUTPUT_JSON}")
    print(f"\n[Current Rule (ER < 2.0 & Norm > 100)]: TPR = {summary['current_rule']['tpr_recall_pct']}% ({tp_curr}/10), FAR = {summary['current_rule']['false_alarm_rate_pct']}% ({fp_curr}/10)")
    print(f"[Calibrated ER Rule (ER < 2.5)]:          TPR = {summary['calibrated_er_rule']['tpr_recall_pct']}% ({tp_er}/10), FAR = {summary['calibrated_er_rule']['false_alarm_rate_pct']}% ({fp_er}/10)")

if __name__ == "__main__":
    evaluate_cohort()
