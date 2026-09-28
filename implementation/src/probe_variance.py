#!/usr/bin/env python3
"""
src/probe_variance.py
Phase 5 Experiment 1: Probe Baseline Variance Verification.

Provides mathematical utilities to evaluate behavioral fingerprint stability
across repeated inference passes. Calculates mean, standard deviation, and
Coefficient of Variation (CV = (sigma / |mu|) * 100%) across runs.
"""

from typing import List, Dict, Any, Optional, Tuple
import numpy as np

FEATURE_NAMES = [
    "output_entropy",
    "logit_gap",
    "top5_prob_mass",
    "top1_prob",
    "distribution_spread",
    "logprob_mean",
]

def compute_variance_stats(
    runs_matrix: np.ndarray,
    threshold_percent: float = 5.0,
    eps: float = 1e-12
) -> Dict[str, Any]:
    """
    Compute variance and Coefficient of Variation (CV) across K runs.

    Args:
        runs_matrix: np.ndarray of shape (K, N) or (K, n_probes, n_features)
                     where K is the number of repeated runs (K >= 2).
        threshold_percent: Success criterion threshold for CV% (default: 5.0%).
        eps: Small constant to prevent divide-by-zero.

    Returns:
        Dictionary containing:
            - runs_count: K
            - mean: array of means
            - std: array of sample standard deviations (ddof=1)
            - cv_percent: array of CV percentages
            - mean_cv_percent: float
            - median_cv_percent: float
            - max_cv_percent: float
            - min_cv_percent: float
            - passes_max_criterion: bool (max_cv < threshold)
            - passes_mean_criterion: bool (mean_cv < threshold)
            - threshold_percent: float
    """
    if not isinstance(runs_matrix, np.ndarray):
        runs_matrix = np.array(runs_matrix, dtype=np.float64)

    K = runs_matrix.shape[0]
    if K < 2:
        raise ValueError(f"At least 2 runs are required to compute sample variance, got K={K}")

    # Sample mean along run axis
    mean = np.mean(runs_matrix, axis=0)

    # Sample standard deviation (ddof=1)
    std = np.std(runs_matrix, axis=0, ddof=1)

    # Safe Coefficient of Variation computation:
    # If std is effectively 0 (< eps), CV is exactly 0.0%
    # If mean is non-zero, CV = (std / |mean|) * 100.0
    # If mean is zero and std is zero, CV = 0.0%
    # If mean is zero and std > 0, CV = (std / eps) * 100.0
    abs_mean = np.abs(mean)
    is_std_zero = std < eps
    is_mean_zero = abs_mean < eps

    cv_percent = np.zeros_like(mean, dtype=np.float64)

    # Normal case: std > 0 and mean != 0
    normal_mask = (~is_std_zero) & (~is_mean_zero)
    cv_percent[normal_mask] = (std[normal_mask] / abs_mean[normal_mask]) * 100.0

    # Edge case: std > 0 and mean == 0
    zero_mean_var_mask = (~is_std_zero) & is_mean_zero
    cv_percent[zero_mean_var_mask] = (std[zero_mean_var_mask] / eps) * 100.0

    # Ensure exactly 0.0 where std is effectively zero
    cv_percent[is_std_zero] = 0.0

    mean_cv = float(np.mean(cv_percent))
    median_cv = float(np.median(cv_percent))
    max_cv = float(np.max(cv_percent))
    min_cv = float(np.min(cv_percent))

    return {
        "runs_count": K,
        "mean": mean,
        "std": std,
        "cv_percent": cv_percent,
        "mean_cv_percent": mean_cv,
        "median_cv_percent": median_cv,
        "max_cv_percent": max_cv,
        "min_cv_percent": min_cv,
        "passes_max_criterion": bool(max_cv < threshold_percent),
        "passes_mean_criterion": bool(mean_cv < threshold_percent),
        "threshold_percent": float(threshold_percent)
    }

def format_variance_table(
    probes: List[Dict[str, Any]],
    stats: Dict[str, Any]
) -> str:
    """
    Format variance metrics into a clean human-readable ASCII summary table.
    """
    lines = []
    lines.append("=" * 88)
    lines.append(f"{'PROBE ID':<10} | {'DOMAIN':<22} | {'MEAN CV%':<10} | {'MAX CV%':<10} | {'STATUS'}")
    lines.append("-" * 88)

    cv_matrix = stats["cv_percent"]  # shape (n_probes, 6)
    threshold = stats["threshold_percent"]

    for i, probe in enumerate(probes):
        probe_id = probe.get("probe_id", f"PRB-{i+1:03d}")
        domain = probe.get("domain", "unknown")
        probe_cvs = cv_matrix[i]
        probe_mean_cv = float(np.mean(probe_cvs))
        probe_max_cv = float(np.max(probe_cvs))
        status = "PASS (< 5%)" if probe_max_cv < threshold else "WARN (>= 5%)"

        lines.append(
            f"{probe_id:<10} | {domain[:22]:<22} | {probe_mean_cv:>9.4f}% | {probe_max_cv:>9.4f}% | {status}"
        )

    lines.append("=" * 88)
    lines.append(f"OVERALL SUMMARY ({stats['runs_count']} runs, {len(probes)} probes, {len(probes)*6} total features):")
    lines.append(f"  Mean CV%:   {stats['mean_cv_percent']:.4f}%")
    lines.append(f"  Median CV%: {stats['median_cv_percent']:.4f}%")
    lines.append(f"  Max CV%:    {stats['max_cv_percent']:.4f}%")
    lines.append(f"  Min CV%:    {stats['min_cv_percent']:.4f}%")
    criterion_status = "PASSED" if stats["passes_max_criterion"] else ("PASSED (Mean < 5%)" if stats["passes_mean_criterion"] else "FAILED")
    lines.append(f"  Criterion (CV < {threshold:.1f}%): {criterion_status}")
    lines.append("=" * 88)

    return "\n".join(lines)
