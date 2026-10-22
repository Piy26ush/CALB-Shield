#!/usr/bin/env python3
"""
implementation_2/rq2_remediation/src/calibrate_spectral.py

Calibrates rank-aware spectral thresholds strictly on development cohorts:
- implementation/adapters/functional_cohort/ (5 clean, 5 poison)
- implementation/adapters/physical_cohort/ (10 clean, 10 poison)

DOES NOT TOUCH OR READ FROZEN EVALUATION COHORTS (cohort3_unseen, phase6_unseen_cohort).
Outputs calibration results to implementation_2/rq2_remediation/results/calibration_spectral.json.
"""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np

from implementation_2.rq2_remediation.src.rank_spectral_scanner import RankAwareSpectralScanner


def find_adapter_paths(parent_dir: Path) -> list[Path]:
    adapters = []
    for d in sorted(parent_dir.iterdir()):
        if d.is_dir() and not d.name.startswith("."):
            st = d / "adapter_model.safetensors"
            bin_f = d / "adapter_model.bin"
            if st.exists() or bin_f.exists():
                adapters.append(d)
    return adapters


def run_calibration() -> dict:
    dev_cohorts = [
        Path("implementation/adapters/functional_cohort"),
        Path("implementation/adapters/physical_cohort"),
    ]

    clean_paths: list[Path] = []
    poison_paths: list[Path] = []

    for base in dev_cohorts:
        c_dir = base / "clean"
        p_dir = base / "poisoned"
        if c_dir.exists():
            clean_paths.extend(find_adapter_paths(c_dir))
        if p_dir.exists():
            poison_paths.extend(find_adapter_paths(p_dir))

    print(f"Loaded {len(clean_paths)} clean dev adapters, {len(poison_paths)} poison dev adapters.")

    scanner = RankAwareSpectralScanner()
    clean_records = []
    for p in clean_paths:
        res = scanner.scan_adapter(p)
        if res.get("verdict") != "ERROR":
            clean_records.append({
                "path": str(p),
                "name": p.name,
                "label": "CLEAN",
                "metrics": res["summary_metrics"],
                "nominal_rank": res["nominal_rank"],
            })

    poison_records = []
    for p in poison_paths:
        res = scanner.scan_adapter(p)
        if res.get("verdict") != "ERROR":
            poison_records.append({
                "path": str(p),
                "name": p.name,
                "label": "POISONED",
                "metrics": res["summary_metrics"],
                "nominal_rank": res["nominal_rank"],
            })

    # Calibrate candidate features:
    # 1. min_erank_ratio (flag if < threshold)
    # 2. min_norm_entropy (flag if < threshold)
    # 3. max_top1_ratio (old metric for historical comparison)
    # 4. max_nsed (flag if > threshold)
    
    results = {
        "dataset_summary": {
            "n_clean_dev": len(clean_records),
            "n_poison_dev": len(poison_records),
            "dev_sources": [str(c) for c in dev_cohorts],
        },
        "features": {},
    }

    # Evaluate min_erank_ratio (lower is suspicious)
    c_erank = [r["metrics"]["min_erank_ratio"] for r in clean_records]
    p_erank = [r["metrics"]["min_erank_ratio"] for r in poison_records]
    all_erank = sorted(c_erank + p_erank)
    best_erank = {"threshold": 0.5, "j": -1, "tpr": 0, "fpr": 1, "bal_acc": 0}
    for i in range(len(all_erank) - 1):
        t = (all_erank[i] + all_erank[i+1]) / 2.0
        tpr = sum(1 for v in p_erank if v < t) / len(p_erank)
        fpr = sum(1 for v in c_erank if v < t) / len(c_erank)
        j = tpr - fpr
        if j > best_erank["j"]:
            best_erank = {"threshold": float(t), "j": float(j), "tpr": float(tpr), "fpr": float(fpr), "bal_acc": float(0.5 * (tpr + 1.0 - fpr))}
    results["features"]["min_erank_ratio"] = best_erank

    # Evaluate min_norm_entropy (lower is suspicious)
    c_ent = [r["metrics"]["min_norm_entropy"] for r in clean_records]
    p_ent = [r["metrics"]["min_norm_entropy"] for r in poison_records]
    all_ent = sorted(c_ent + p_ent)
    best_ent = {"threshold": 0.7, "j": -1, "tpr": 0, "fpr": 1, "bal_acc": 0}
    for i in range(len(all_ent) - 1):
        t = (all_ent[i] + all_ent[i+1]) / 2.0
        tpr = sum(1 for v in p_ent if v < t) / len(p_ent)
        fpr = sum(1 for v in c_ent if v < t) / len(c_ent)
        j = tpr - fpr
        if j > best_ent["j"]:
            best_ent = {"threshold": float(t), "j": float(j), "tpr": float(tpr), "fpr": float(fpr), "bal_acc": float(0.5 * (tpr + 1.0 - fpr))}
    results["features"]["min_norm_entropy"] = best_ent

    # Evaluate old max_top1_ratio (higher is suspicious)
    c_top1 = [r["metrics"]["max_top1_ratio"] for r in clean_records]
    p_top1 = [r["metrics"]["max_top1_ratio"] for r in poison_records]
    all_top1 = sorted(c_top1 + p_top1)
    best_top1 = {"threshold": 0.4, "j": -1, "tpr": 0, "fpr": 1, "bal_acc": 0}
    for i in range(len(all_top1) - 1):
        t = (all_top1[i] + all_top1[i+1]) / 2.0
        tpr = sum(1 for v in p_top1 if v > t) / len(p_top1)
        fpr = sum(1 for v in c_top1 if v > t) / len(c_top1)
        j = tpr - fpr
        if j > best_top1["j"]:
            best_top1 = {"threshold": float(t), "j": float(j), "tpr": float(tpr), "fpr": float(fpr), "bal_acc": float(0.5 * (tpr + 1.0 - fpr))}
    results["features"]["max_top1_ratio"] = best_top1

    # Record fixed old baseline performance (top-1 > 0.40)
    fixed_040_tpr = sum(1 for v in p_top1 if v > 0.40) / len(p_top1)
    fixed_040_fpr = sum(1 for v in c_top1 if v > 0.40) / len(c_top1)
    results["historical_baseline_top1_040"] = {
        "threshold": 0.40,
        "tpr": float(fixed_040_tpr),
        "fpr": float(fixed_040_fpr),
        "bal_acc": float(0.5 * (fixed_040_tpr + 1.0 - fixed_040_fpr)),
    }

    results["records"] = {
        "clean": clean_records,
        "poisoned": poison_records,
    }

    out_file = Path("implementation_2/rq2_remediation/results/calibration_spectral.json")
    out_file.write_text(json.dumps(results, indent=2))
    print(f"Calibration saved to {out_file}")
    print("\n=== CALIBRATION RESULTS ON DEVELOPMENT DATA ===")
    print(f"Historical Top-1 > 0.40: TPR={fixed_040_tpr:.1%}, FPR={fixed_040_fpr:.1%}, BalAcc={results['historical_baseline_top1_040']['bal_acc']:.1%}")
    print(f"Remediated min_erank_ratio (t={best_erank['threshold']:.3f}): TPR={best_erank['tpr']:.1%}, FPR={best_erank['fpr']:.1%}, BalAcc={best_erank['bal_acc']:.1%}")
    print(f"Remediated min_norm_entropy (t={best_ent['threshold']:.3f}): TPR={best_ent['tpr']:.1%}, FPR={best_ent['fpr']:.1%}, BalAcc={best_ent['bal_acc']:.1%}")

    return results


if __name__ == "__main__":
    run_calibration()
