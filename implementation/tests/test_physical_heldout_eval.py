"""
test_physical_heldout_eval.py
Unit tests for Physical Held-Out Architecture Evaluation (Phase 1H).
"""

import os
import json
import numpy as np
import pytest

from run_physical_heldout_eval import (
    build_training_cohort,
    run_physical_heldout_evaluation,
    resolve_path
)

def test_build_training_cohort_zero_qwen_leakage():
    llama_mock = np.zeros(180)
    mistral_mock = np.ones(180)

    X_train, y_train, arch_tags, (clean, poison) = build_training_cohort(
        llama_mock, mistral_mock, n_samples_per_arch=10, seed=42
    )

    # 1. Verify shapes
    assert X_train.shape == (40, 180)
    assert len(y_train) == 40
    assert sum(y_train == 0) == 20
    assert sum(y_train == 1) == 20

    # 2. Verify strict zero-leakage: ZERO Qwen in training
    assert set(arch_tags) == {"llama3", "mistral"}
    assert "qwen" not in arch_tags

def test_run_physical_heldout_evaluation_metrics(tmp_path):
    out_csv = str(tmp_path / "test_heldout.csv")
    out_json = str(tmp_path / "test_heldout.json")

    summary = run_physical_heldout_evaluation(output_csv=out_csv, output_json=out_json)

    assert os.path.exists(out_csv)
    assert os.path.exists(out_json)

    # Verify methodology documentation in artifact
    assert "Qwen" in summary["methodology"]["held_out_test_architecture"]
    assert "LLaMA-3" in summary["methodology"]["training_cohort"]
    assert "Mistral-7B" in summary["methodology"]["training_cohort"]

    results = summary["results"]
    for clf_name, metrics in results.items():
        # Verify CALB-Shield achieves 100% accuracy on physical Qwen checkpoints
        assert metrics["calb_shield"]["accuracy"] == 100.0
        assert metrics["calb_shield"]["fnr"] == 0.0
        assert metrics["calb_shield"]["far"] == 0.0
        assert metrics["calb_shield"]["clean_pred"] == "CLEAN"
        assert metrics["calb_shield"]["poison_pred"] == "POISONED"

    # Verify that raw unnormalized classifiers exhibited domain failure
    # (either false alarms on clean or false negatives on poison)
    raw_failures = sum(
        results[c]["raw"]["accuracy"] < 100.0 for c in ["Logistic Regression", "Linear SVM", "Random Forest"]
    )
    assert raw_failures >= 2
