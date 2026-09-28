#!/usr/bin/env python3
"""
run_probe_variance.py
Phase 5 Experiment 1: Probe Baseline Variance Verification across repeated inference runs.

Measures behavioral fingerprint consistency across K repeated runs (default K=5)
on the clean LLaMA-3 anchor (Meta-Llama-3-8B-Instruct.Q4_K_M.gguf) across all 30
diagnostic probes and 6 logit features (180 feature points total).

Success criterion: Coefficient of variation (CV% = sigma / |mu| * 100%) < 5.0%
across repeated runs on the same model checkpoint.
"""

import os
import sys
import json
import csv
import time
import argparse
from typing import List, Dict, Any
import numpy as np

# Add implementation root and src to path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from src.probe_runner import ProbeRunner
from src.probe_variance import (
    compute_variance_stats,
    format_variance_table,
    FEATURE_NAMES
)

def resolve_path(path_str: str) -> str:
    """Resolve relative path against either CWD or CURRENT_DIR."""
    if os.path.isabs(path_str):
        return path_str
    if os.path.exists(path_str):
        return os.path.abspath(path_str)
    candidate = os.path.join(CURRENT_DIR, path_str)
    if os.path.exists(candidate):
        return os.path.abspath(candidate)
    return os.path.abspath(path_str)

def main():
    parser = argparse.ArgumentParser(description="Run Probe Baseline Variance Verification (Phase 5 Exp 1)")
    parser.add_argument("--arch", type=str, default="llama3", help="Target architecture (default: llama3)")
    parser.add_argument("--model", type=str, default="models/llama3/Meta-Llama-3-8B-Instruct.Q4_K_M.gguf", help="Path to GGUF model")
    parser.add_argument("--probes", type=str, default="probes/probes_30.json", help="Path to probes JSON")
    parser.add_argument("--runs", type=int, default=5, help="Number of repeated inference passes (default: 5)")
    parser.add_argument("--n_gpu_layers", type=int, default=1, help="Number of GPU offload layers (default: 1)")
    parser.add_argument("--threshold", type=float, default=5.0, help="CV percent success threshold (default: 5.0 percent)")
    parser.add_argument("--output_json", type=str, default=None, help="Output JSON path")
    parser.add_argument("--output_csv", type=str, default=None, help="Output CSV path")
    args = parser.parse_args()

    model_path = resolve_path(args.model)
    probe_path = resolve_path(args.probes)
    num_runs = args.runs
    arch = args.arch
    threshold = args.threshold

    results_dir = os.path.join(CURRENT_DIR, "results")
    os.makedirs(results_dir, exist_ok=True)

    json_out = args.output_json or os.path.join(results_dir, f"probe_variance_{arch}.json")
    csv_out = args.output_csv or os.path.join(results_dir, f"probe_variance_{arch}.csv")

    print("=" * 88)
    print("CALB-SHIELD: PHASE 5 EXPERIMENT 1 — PROBE BASELINE VARIANCE VERIFICATION")
    print("=" * 88)
    print(f"Target Architecture:    {arch.upper()}")
    print(f"Model Path:             {model_path}")
    print(f"Probes Suite:           {probe_path}")
    print(f"Repeated Inference Runs:{num_runs}")
    print(f"Success Threshold:      CV < {threshold:.1f}%")
    print("-" * 88)

    if not os.path.exists(model_path):
        print(f"[ERROR] Model file not found at: {model_path}")
        sys.exit(1)

    if not os.path.exists(probe_path):
        print(f"[ERROR] Probes file not found at: {probe_path}")
        sys.exit(1)

    with open(probe_path, "r", encoding="utf-8") as f:
        probes = json.load(f)
    n_probes = len(probes)
    print(f"[INIT] Loaded {n_probes} probes.")

    print(f"[INIT] Loading GGUF model into ProbeRunner (n_gpu_layers={args.n_gpu_layers})...")
    init_start = time.time()
    runner = ProbeRunner(
        model_path=model_path,
        architecture=arch,
        n_gpu_layers=args.n_gpu_layers,
        verbose=False
    )
    init_duration = time.time() - init_start
    print(f"[INIT] Model loaded in {init_duration:.2f}s.")

    # Storage for all runs: shape (num_runs, n_probes, 6)
    runs_data = np.zeros((num_runs, n_probes, ProbeRunner.N_FEATURES), dtype=np.float64)
    run_durations = []

    print("-" * 88)
    print(f"[EXEC] Beginning {num_runs} independent passes over {n_probes} probes...")
    exp_start = time.time()

    for r in range(num_runs):
        run_idx = r + 1
        print(f"\n--- Run {run_idx}/{num_runs} ---")
        t_run_start = time.time()

        for p_idx, probe in enumerate(probes):
            probe_id = probe.get("probe_id", f"PRB-{p_idx+1:03d}")
            text = probe.get("probe_text", "")
            feat_vec = runner.run_single_probe(text)
            runs_data[r, p_idx, :] = feat_vec

        run_dur = time.time() - t_run_start
        run_durations.append(run_dur)
        print(f"Run {run_idx} complete: {run_dur:.2f}s ({run_dur / n_probes:.2f}s/probe)")

    total_exp_time = time.time() - exp_start
    print("\n" + "=" * 88)
    print(f"[STAT] All {num_runs} runs complete in {total_exp_time:.2f}s. Computing variance statistics...")

    stats = compute_variance_stats(runs_data, threshold_percent=threshold)

    # Print ASCII table
    print(format_variance_table(probes, stats))

    # Prepare structured JSON report
    per_probe_report = []
    mean_matrix = stats["mean"]      # (n_probes, 6)
    std_matrix = stats["std"]        # (n_probes, 6)
    cv_matrix = stats["cv_percent"]  # (n_probes, 6)

    for p_idx, probe in enumerate(probes):
        probe_id = probe.get("probe_id", f"PRB-{p_idx+1:03d}")
        domain = probe.get("domain", "unknown")
        probe_text = probe.get("probe_text", "")

        feature_stats = {}
        for f_idx, feat_name in enumerate(FEATURE_NAMES):
            feature_stats[feat_name] = {
                "mean": float(mean_matrix[p_idx, f_idx]),
                "std": float(std_matrix[p_idx, f_idx]),
                "cv_percent": float(cv_matrix[p_idx, f_idx]),
                "runs_values": [float(runs_data[r, p_idx, f_idx]) for r in range(num_runs)]
            }

        per_probe_report.append({
            "probe_id": probe_id,
            "domain": domain,
            "probe_text": probe_text,
            "probe_mean_cv": float(np.mean(cv_matrix[p_idx])),
            "probe_max_cv": float(np.max(cv_matrix[p_idx])),
            "features": feature_stats
        })

    report = {
        "experiment": "Phase 5 Experiment 1 — Probe Baseline Variance Verification",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model_info": {
            "model_path": model_path,
            "model_name": os.path.basename(model_path).replace(".gguf", ""),
            "architecture": arch,
            "n_gpu_layers": args.n_gpu_layers
        },
        "execution_summary": {
            "num_runs": num_runs,
            "num_probes": n_probes,
            "total_features": n_probes * ProbeRunner.N_FEATURES,
            "total_time_seconds": round(total_exp_time, 2),
            "avg_time_per_run_seconds": round(float(np.mean(run_durations)), 2),
            "run_durations_seconds": [round(d, 2) for d in run_durations]
        },
        "variance_metrics": {
            "mean_cv_percent": stats["mean_cv_percent"],
            "median_cv_percent": stats["median_cv_percent"],
            "max_cv_percent": stats["max_cv_percent"],
            "min_cv_percent": stats["min_cv_percent"],
            "threshold_percent": threshold,
            "passes_max_criterion": stats["passes_max_criterion"],
            "passes_mean_criterion": stats["passes_mean_criterion"],
            "status": "PASSED" if stats["passes_max_criterion"] else ("PASSED_MEAN" if stats["passes_mean_criterion"] else "FAILED")
        },
        "per_probe_results": per_probe_report
    }

    with open(json_out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\n[SAVE] Saved JSON report to: {json_out}")

    # Export CSV report
    with open(csv_out, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = ["probe_id", "domain", "feature_index", "feature_name"]
        for r in range(num_runs):
            header.append(f"run_{r+1}")
        header.extend(["mean", "std", "cv_percent"])
        writer.writerow(header)

        for p_idx, probe in enumerate(probes):
            probe_id = probe.get("probe_id", f"PRB-{p_idx+1:03d}")
            domain = probe.get("domain", "unknown")
            for f_idx, feat_name in enumerate(FEATURE_NAMES):
                row = [probe_id, domain, f_idx, feat_name]
                for r in range(num_runs):
                    row.append(f"{runs_data[r, p_idx, f_idx]:.6f}")
                row.append(f"{mean_matrix[p_idx, f_idx]:.6f}")
                row.append(f"{std_matrix[p_idx, f_idx]:.6f}")
                row.append(f"{cv_matrix[p_idx, f_idx]:.4f}")
                writer.writerow(row)

    print(f"[SAVE] Saved CSV report to:  {csv_out}")

    if stats["passes_max_criterion"]:
        print(f"\n[SUCCESS] SUCCESS CRITERION MET: Max CV ({stats['max_cv_percent']:.4f}%) is strictly below the {threshold:.1f}% threshold across all 180 features!")
        sys.exit(0)
    elif stats["passes_mean_criterion"]:
        print(f"\n[NOTICE] Mean CV ({stats['mean_cv_percent']:.4f}%) meets the {threshold:.1f}% threshold (Max CV: {stats['max_cv_percent']:.4f}%).")
        sys.exit(0)
    else:
        print(f"\n[FAIL] FAILED: CV exceeded threshold ({threshold:.1f}%). Mean CV: {stats['mean_cv_percent']:.4f}%, Max CV: {stats['max_cv_percent']:.4f}%.")
        sys.exit(1)

if __name__ == "__main__":
    main()
