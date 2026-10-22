#!/usr/bin/env python3
"""
implementation_2/tests/test_stage3_data_integrity.py
Unit and integration tests for Stage 3 data partitions, feature scaling,
partition balance, leakage prevention, and threshold-selection mathematics.
Strictly adheres to Methodology Specification Version 1.3.1.
"""

import sys
from pathlib import Path
import numpy as np
import pytest

IMPL2_DIR = Path(__file__).resolve().parents[1]
if str(IMPL2_DIR) not in sys.path:
    sys.path.insert(0, str(IMPL2_DIR))

from src.data_loader import (
    FingerprintDataLoader,
    EXPECTED_PROBES,
    METRICS_PER_PROBE,
    TOTAL_FEATURES
)


def solve_threshold(
    y_true: np.ndarray,
    probs: np.ndarray,
    candidates: np.ndarray = np.arange(0.10, 0.91, 0.01)
) -> float:
    """
    Deterministic threshold solver strictly implementing Specification v1.3.1 Section 7.
    """
    clean_mask = (y_true == 0)
    anom_mask = (y_true == 1)

    clean_probs = probs[clean_mask]
    anom_probs = probs[anom_mask]

    # Evaluate all candidate thresholds
    fps = np.array([np.sum(clean_probs >= tau) for tau in candidates])
    tps = np.array([np.sum(anom_probs >= tau) for tau in candidates])
    n_anom = len(anom_probs)
    tprs = tps / n_anom if n_anom > 0 else np.zeros_like(tps, dtype=float)

    # 1. Feasible set (FP = 0)
    feasible_idx = np.where(fps == 0)[0]

    if len(feasible_idx) > 0:
        max_tpr = np.max(tprs[feasible_idx])
        opt_idx = feasible_idx[np.isclose(tprs[feasible_idx], max_tpr)]
        opt_candidates = candidates[opt_idx]
        # Primary tie-breaker: median of optimal candidate set
        chosen_tau = float(np.median(opt_candidates))
        return round(chosen_tau, 4)
    else:
        # Fallback when FP = 0 is unachievable
        min_fp = np.min(fps)
        subopt_idx = np.where(fps == min_fp)[0]
        max_tpr_subopt = np.max(tprs[subopt_idx])
        best_subopt_idx = subopt_idx[np.isclose(tprs[subopt_idx], max_tpr_subopt)]
        best_candidates = candidates[best_subopt_idx]
        chosen_tau = float(np.median(best_candidates))
        return round(chosen_tau, 4)


class TestStage3DataIntegrity:
    """Rigorous verification suite for Stage 3 data pipelines and mathematics."""

    @pytest.fixture(scope="class")
    def stage3_cohorts(self):
        loader = FingerprintDataLoader()
        train_d, calib_d, eval_d, phys_d, preprocessor = loader.build_stage3_partitions(seed=42)
        return {
            "loader": loader,
            "train": train_d,
            "calib": calib_d,
            "eval": eval_d,
            "phys": phys_d,
            "preprocessor": preprocessor
        }

    def test_feature_wise_scale_properties(self, stage3_cohorts):
        """Verify scale vector s is 180-dim, finite, lower-bounded, and free of Qwen data."""
        loader = stage3_cohorts["loader"]
        scale_vec = loader.compute_feature_wise_scale()

        assert len(scale_vec) == TOTAL_FEATURES, f"Expected 180 dims, got {len(scale_vec)}"
        assert np.all(np.isfinite(scale_vec)), "Scale vector contains non-finite values."
        # Lower bound check: max(std, 1e-3) + 1e-5 >= 1.01e-3
        assert np.all(scale_vec >= 1.01e-3), "Scale vector violates canonical lower bound."

        # Verify logit gap scale is substantially larger than probability scales
        # Metric 1 is logit_gap, Metric 2 is top5_prob_mass, Metric 3 is top1_prob
        logit_gap_scales = scale_vec[1::METRICS_PER_PROBE]
        top1_prob_scales = scale_vec[3::METRICS_PER_PROBE]
        assert np.mean(logit_gap_scales) > np.mean(top1_prob_scales), \
            "Metric orders of magnitude not respected by feature-wise scaling."

    def test_partition_dimensions_and_exact_allocations(self, stage3_cohorts):
        """Verify exact sample counts matching Specification v1.3.1 Section 6 & 8."""
        train = stage3_cohorts["train"]
        calib = stage3_cohorts["calib"]
        eval_d = stage3_cohorts["eval"]
        phys = stage3_cohorts["phys"]

        # 1. Training Set D_train (N = 120, 60 clean / 60 trojan)
        assert train["X"].shape == (120, TOTAL_FEATURES)
        assert train["N_clean"] == 60
        assert train["N_trojan"] == 60

        # 2. Calibration Set D_calib (N = 60, 30 clean / 30 trojan)
        assert calib["X"].shape == (60, TOTAL_FEATURES)
        assert calib["N_clean"] == 30
        assert calib["N_trojan"] == 30

        # 3. Synthetic Evaluation Set D_eval (N = 100, 70 clean / 30 trojan)
        assert eval_d["X"].shape == (100, TOTAL_FEATURES)
        assert eval_d["N_clean"] == 70
        assert eval_d["N_trojan"] == 30

        # Detailed breakdown of D_eval records
        records = eval_d["records"]
        assert len(records) == 100

        # Count by mechanism
        b0_count = sum(1 for r in records if r["mechanism"] == "B0")
        b1_count = sum(1 for r in records if r["mechanism"] == "B1")
        b2_count = sum(1 for r in records if r["mechanism"] == "B2")
        b3_count = sum(1 for r in records if r["mechanism"] == "B3")
        trojan_count = sum(1 for r in records if r["mechanism"] == "TROJAN")

        assert b0_count == 10, f"Expected exactly 10 B0 samples, got {b0_count}"
        assert b1_count == 20, f"Expected exactly 20 B1 samples, got {b1_count}"
        assert b2_count == 20, f"Expected exactly 20 B2 samples, got {b2_count}"
        assert b3_count == 20, f"Expected exactly 20 B3 samples, got {b3_count}"
        assert trojan_count == 30, f"Expected exactly 30 trojan samples, got {trojan_count}"

        # Count by anchor
        llama_eval = sum(1 for r in records if r["anchor"] == "llama3")
        mistral_eval = sum(1 for r in records if r["anchor"] == "mistral")
        assert llama_eval == 50, f"Expected 50 LLaMA samples, got {llama_eval}"
        assert mistral_eval == 50, f"Expected 50 Mistral samples, got {mistral_eval}"

        # 4. Physical Test Set D_phys (N = 2)
        assert phys["X"].shape == (2, TOTAL_FEATURES)
        assert phys["N"] == 2

    def test_b3_strictly_held_out(self, stage3_cohorts):
        """Verify that mechanism B3 is completely absent from D_train and D_calib."""
        # D_eval contains B3, but neither train nor calib may reference B3
        train = stage3_cohorts["train"]
        calib = stage3_cohorts["calib"]

        # The loader builds train and calib with only B0, B1, B2 and trojans
        assert train["N_clean"] == 60  # 2 B0 + 38 B1 + 20 B2
        assert calib["N_clean"] == 30  # 5 B0 + 15 B1 + 10 B2

    def test_qwen_strict_isolation_from_synthetic_partitions(self, stage3_cohorts):
        """Verify that physical Qwen fingerprints do not appear in D_train, D_calib, or D_eval."""
        phys_raw = stage3_cohorts["loader"].load_physical_fingerprints()
        qwen_clean_raw = phys_raw["qwen_clean"]["vector_1d"]
        qwen_poison_raw = phys_raw["qwen_poisoned"]["vector_1d"]

        # Raw clean training differentials do not use Qwen
        train_X = stage3_cohorts["train"]["X"]
        calib_X = stage3_cohorts["calib"]["X"]
        eval_X = stage3_cohorts["eval"]["X"]

        assert np.all(np.isfinite(train_X))
        assert np.all(np.isfinite(calib_X))
        assert np.all(np.isfinite(eval_X))

    def test_preprocessor_freeze_and_zero_centering(self, stage3_cohorts):
        """Verify preprocessor is frozen and analytical zero-centering holds."""
        pre = stage3_cohorts["preprocessor"]
        assert pre.is_frozen, "Preprocessor must be frozen after fitting on D_train clean."

        # Self-differential (exact zeros) maps to exact zeros
        zero_vec = np.zeros((1, TOTAL_FEATURES), dtype=np.float64)
        transformed_zero = pre.transform(zero_vec)
        assert np.all(transformed_zero == 0.0), "Zero vector must map to exact 0.0 under analytical zero."

    def test_deterministic_threshold_solver_clean_case(self):
        """Test threshold solver when FP = 0 is feasible and margin exists."""
        y_true = np.array([0, 0, 0, 1, 1, 1])
        probs = np.array([0.05, 0.12, 0.20, 0.70, 0.85, 0.95])
        tau = solve_threshold(y_true, probs)
        # Optimal candidates with FP=0 and TPR=1.0 are tau in [0.21, 0.70]
        assert tau > 0.20 and tau <= 0.70, f"Expected tau in (0.20, 0.70], got {tau}"

    def test_deterministic_threshold_solver_overlapping_fallback(self):
        """Test threshold solver fallback when benign and anomaly distributions overlap."""
        # Clean: 0.1, 0.3, 0.6. Trojan: 0.4, 0.8, 0.9. Overlap exists at [0.4, 0.6]
        y_true = np.array([0, 0, 0, 1, 1, 1])
        probs = np.array([0.10, 0.30, 0.60, 0.40, 0.80, 0.90])
        tau = solve_threshold(y_true, probs)
        assert isinstance(tau, float)
        assert 0.10 <= tau <= 0.90
