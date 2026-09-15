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

def run_batch_svd(adapters_dir: str, output_csv: str, threshold: float = 0.40):
    tracker = ExperimentTracker(
        experiment_name="phase2_svd_spectral_benchmark",
        seed=42,
        notes="Empirical spectral energy ratio scanning on LoRA adapters"
    )
    
    scanner = SVDSpectralScanner(threshold=threshold)
    records = []

    base_path = Path(adapters_dir)
    adapter_paths = [p for p in base_path.iterdir() if p.is_dir() and not p.name.startswith(".")]

    print(f"\n[Experiment] Scanning {len(adapter_paths)} adapter directories under {adapters_dir}...")

    for i, ad_path in enumerate(adapter_paths, 1):
        start_t = time.time()
        res = scanner.scan_adapter(str(ad_path))
        elapsed = time.time() - start_t

        if res.get("verdict") == "ERROR":
            print(f"[{i}/{len(adapter_paths)}] {ad_path.name}: ERROR ({res.get('reason')})")
            continue

        records.append({
            "adapter_name": ad_path.name,
            "path": str(ad_path),
            "verdict": res["verdict"],
            "n_layers_analyzed": res["n_layers_analyzed"],
            "n_layers_flagged": res["n_layers_flagged"],
            "flagged_fraction": res["flagged_fraction"],
            "max_top1_ratio": res["max_top1_spectral_ratio"],
            "mean_top1_ratio": res["mean_top1_spectral_ratio"],
            "std_top1_ratio": res["std_top1_spectral_ratio"],
            "scan_time_sec": round(elapsed, 3)
        })

        print(f"[{i}/{len(adapter_paths)}] {ad_path.name}: {res['verdict']} | "
              f"Max rho_1: {res['max_top1_spectral_ratio']:.4f} | "
              f"Mean rho_1: {res['mean_top1_spectral_ratio']:.4f} ({elapsed:.1f}s)")

    df = pd.DataFrame(records)
    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"\n[Output] Results saved to: {output_csv}")

    tracker.register_artifact("svd_results_csv", output_csv)
    tracker.log_metrics({
        "total_adapters_scanned": len(records),
        "flagged_adapters": int(sum(1 for r in records if r["verdict"] == "FLAGGED")),
        "normal_adapters": int(sum(1 for r in records if r["verdict"] == "NORMAL"))
    })
    manifest_path = tracker.finalize(status="completed")
    print(f"[Provenance] Run manifest saved to: {manifest_path}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run batch SVD spectral scans on adapters.")
    parser.add_argument("--adapters-dir", default="adapters/clean", help="Folder containing adapter subdirectories")
    parser.add_argument("--output", default="results/svd_benchmark_results.csv", help="Output CSV path")
    parser.add_argument("--threshold", type=float, default=0.40, help="Energy concentration cutoff")
    args = parser.parse_args()

    run_batch_svd(args.adapters_dir, args.output, args.threshold)
