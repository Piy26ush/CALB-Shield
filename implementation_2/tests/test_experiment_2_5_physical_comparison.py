#!/usr/bin/env python3
"""
implementation_2/tests/test_experiment_2_5_physical_comparison.py
Regression and integrity tests for Experiment 2.5: RQ1 Physical Validation & Five-Classifier Comparison.
"""

import os
import sys
import json
import numpy as np
import pytest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

IMPL2_DIR = PROJECT_ROOT / "implementation_2"
if str(IMPL2_DIR) not in sys.path:
    sys.path.insert(0, str(IMPL2_DIR))

from implementation_2.src.data_loader import FingerprintDataLoader
from implementation_2.src.baseline_classifiers import BaselineSuite
from implementation_2.experiments.run_experiment_2_5_physical_comparison import (
    wilson_score_interval,
    count_model_parameters,
    run_physical_classifier_comparison
)


def test_wilson_score_interval_properties():
    """Verify mathematical properties of Wilson score interval."""
    # 0 successes out of 2
    low0, high0 = wilson_score_interval(0, 2)
    assert 0.0 <= low0 < high0 <= 1.0
    assert np.isclose(low0, 0.0, atol=1e-3)
    assert high0 > 0.5  # With N=2, high bound is wide

    # 2 successes out of 2
    low2, high2 = wilson_score_interval(2, 2)
    assert 0.0 <= low2 < high2 <= 1.0
    assert np.isclose(high2, 1.0, atol=1e-3)
    assert low2 < 0.5  # With N=2, low bound is wide

    # Symmetry around 1/2
    low1, high1 = wilson_score_interval(1, 2)
    assert np.isclose(0.5 - low1, high1 - 0.5, atol=1e-4)

    # Empty sample
    low_e, high_e = wilson_score_interval(0, 0)
    assert low_e == 0.0 and high_e == 1.0


def test_classifier_parameter_counts():
    """Verify parameter count helper for all 5 classifiers."""
    models = ["logistic_regression", "random_forest", "gradient_boosting", "pytorch_mlp", "mbconv"]
    for m in models:
        clf = BaselineSuite.create_classifier(m)
        params = count_model_parameters(clf, m)
        assert params > 0, f"Model {m} parameter count must be positive, got {params}"

    # PyTorch MLP has 13,889 trainable parameters
    mlp_clf = BaselineSuite.create_classifier("pytorch_mlp")
    assert count_model_parameters(mlp_clf, "pytorch_mlp") == 13889

    # Structured MBConv has 4,321 trainable parameters
    mbconv_clf = BaselineSuite.create_classifier("mbconv")
    assert count_model_parameters(mbconv_clf, "mbconv") == 4321


def test_zero_qwen_leakage_in_training_cohort():
    """Verify that training cohorts strictly exclude Qwen data."""
    loader = FingerprintDataLoader()

    # Regime A
    X_tr_A, y_tr_A, X_te_A, y_te_A, meta_A = loader.build_supervised_training_cohort(
        n_samples_per_arch=25, seed=42
    )
    assert "qwen" in meta_A["test_architecture"]
    assert "qwen" not in meta_A["train_architectures"]
    assert len(X_te_A) == 2  # exactly clean + poisoned

    # Regime B
    X_tr_B, y_tr_B, X_te_B, y_te_B, meta_B = loader.build_regime_b_cohort(
        n_samples_per_arch=25, seed=42
    )
    assert len(X_te_B) == 2
    # Verify clean Qwen differential is exactly zero vector
    assert np.all(X_te_B[0] == 0.0)


def test_experiment_2_5_execution_and_schema():
    """Execute Experiment 2.5 and assert schema and output constraints."""
    result = run_physical_classifier_comparison(seed=42)

    assert "results" in result
    records = result["results"]
    assert len(records) == 10  # 5 classifiers x 2 regimes

    regimes = set(r["regime"] for r in records)
    assert regimes == {"Regime A (Zero-Reference)", "Regime B (Reference-Anchored)"}

    classifiers = set(r["classifier_name"] for r in records)
    assert len(classifiers) == 5

    for r in records:
        assert r["physical_sample_count"] == 2
        assert r["synth_val_sample_count"] == 60
        assert 0.0 <= r["physical_clean_prob"] <= 1.0
        assert 0.0 <= r["physical_poison_prob"] <= 1.0
        assert r["physical_clean_prediction"] in ("CLEAN", "POISONED")
        assert r["physical_poison_prediction"] in ("CLEAN", "POISONED")
        assert 0.0 <= r["physical_accuracy"] <= 1.0
        assert 0.0 <= r["synth_val_accuracy"] <= 1.0
        assert 0.0 <= r["synth_val_f1_score"] <= 1.0

        # Physical confusion matrix arithmetic
        assert r["physical_tp"] + r["physical_fn"] == 1
        assert r["physical_tn"] + r["physical_fp"] == 1
        assert r["physical_tp"] + r["physical_tn"] + r["physical_fp"] + r["physical_fn"] == 2
