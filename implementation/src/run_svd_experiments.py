#!/usr/bin/env python3
"""
src/run_svd_experiments.py
Batch runner for Phase 2 SVD spectral experiments on LoRA adapters.

Evaluates singular value distributions across a collection of clean and poisoned adapters.
Uses ExperimentTracker to hash inputs, record run provenance, and output reproducible CSV tables.
"""

import os
import sys
import time
import argparse
import pandas as pd
from pathlib import Path

from src.svd_scanner import SVDSpectralScanner
from src.experiment_tracker import ExperimentTracker

def run_batch_svd(adapters_dirs: list[str], output_csv: str, threshold: float = 0.40):
    tracker = ExperimentTracker(
        experiment_name="phase2_svd_spectral_benchmark",
        seed=42,
        notes="Comprehensive empirical spectral energy, norm, and effective rank scanning across clean and poisoned LoRA adapters"
    )
    
    scanner = SVDSpectralScanner(threshold=threshold)
    records = []

    all_adapter_paths = []
    for ad_dir in adapters_dirs:
        base_path = Path(ad_dir)
        if not base_path.exists():
            continue
        for p in base_path.iterdir():
            if p.is_dir() and not p.name.startswith("."):
                label = "POISONED" if "poisoned" in str(p) else "CLEAN"
                all_adapter_paths.append((p, label))

    print(f"\n[Experiment] Scanning {len(all_adapter_paths)} adapter directories across {adapters_dirs}...")

    for i, (ad_path, label) in enumerate(all_adapter_paths, 1):
        start_t = time.time()
        res = scanner.scan_adapter(str(ad_path))
        elapsed = time.time() - start_t

        if res.get("verdict") == "ERROR":
            print(f"[{i}/{len(all_adapter_paths)}] {ad_path.name}: ERROR ({res.get('reason')})")
            continue

        records.append({
            "adapter_name": ad_path.name,
            "ground_truth": label,
            "path": str(ad_path),
            "verdict": res["verdict"],
            "n_layers_analyzed": res["n_layers_analyzed"],
            "n_layers_flagged": res["n_layers_flagged"],
            "flagged_fraction": round(res["flagged_fraction"], 4),
            "max_top1_ratio": round(res["max_top1_spectral_ratio"], 4),
            "mean_top1_ratio": round(res["mean_top1_spectral_ratio"], 4),
            "std_top1_ratio": round(res["std_top1_spectral_ratio"], 4),
            "max_spectral_norm": round(res.get("max_spectral_norm", 0.0), 4),
            "mean_spectral_norm": round(res.get("mean_spectral_norm", 0.0), 4),
            "max_condition_number": round(res.get("max_condition_number", 0.0), 2),
            "mean_condition_number": round(res.get("mean_condition_number", 0.0), 2),
            "min_effective_rank": round(res.get("min_effective_rank", 0.0), 4),
            "mean_effective_rank": round(res.get("mean_effective_rank", 0.0), 4),
            "scan_time_sec": round(elapsed, 3)
        })

        print(f"[{i}/{len(all_adapter_paths)}] [{label}] {ad_path.name}: {res['verdict']} | "
              f"Max rho_1: {res['max_top1_spectral_ratio']:.4f} | "
              f"Max ||DeltaW||_2: {res.get('max_spectral_norm', 0.0):.2f} | "
              f"Max Cond#: {res.get('max_condition_number', 0.0):.1f} ({elapsed:.2f}s)")

    df = pd.DataFrame(records)
    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"\n[Output] Results saved to: {output_csv}")

    tracker.register_artifact("svd_results_csv", output_csv)
    tracker.log_metrics({
        "total_adapters_scanned": len(records),
        "clean_adapters": int(sum(1 for r in records if r["ground_truth"] == "CLEAN")),
        "poisoned_adapters": int(sum(1 for r in records if r["ground_truth"] == "POISONED")),
        "flagged_adapters": int(sum(1 for r in records if r["verdict"] == "FLAGGED")),
        "normal_adapters": int(sum(1 for r in records if r["verdict"] == "NORMAL"))
    })
    manifest_path = tracker.finalize(status="completed")
    print(f"[Provenance] Run manifest saved to: {manifest_path}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run batch SVD spectral scans on adapters.")
    parser.add_argument("--adapters-dirs", nargs="+", default=["adapters/clean", "adapters/poisoned"], help="Folders containing adapter subdirectories")
    parser.add_argument("--output", default="results/svd_benchmark_results.csv", help="Output CSV path")
    parser.add_argument("--threshold", type=float, default=0.40, help="Energy concentration cutoff")
    args = parser.parse_args()

    run_batch_svd(args.adapters_dirs, args.output, args.threshold)
