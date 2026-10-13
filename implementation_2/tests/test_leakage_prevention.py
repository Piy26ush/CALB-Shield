#!/usr/bin/env python3
"""
implementation_2/tests/test_leakage_prevention.py
Unit tests verifying strict prevention of data leakage across model splits.
"""

import numpy as np
import pytest
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from implementation_2.src.data_loader import FingerprintDataLoader


def test_zero_leakage_in_training_cohort():
    """Verify that the held-out evaluation architecture is strictly absent from training."""
    loader = FingerprintDataLoader()
    X_train, y_train, X_test, y_test, metadata = loader.build_supervised_training_cohort(
        n_samples_per_arch=15,
        seed=42,
        normalized=True
    )

    # 1. Assert metadata guarantees
    assert "qwen" not in metadata["train_architectures"], "Data leakage! Qwen found in training architectures!"
    assert metadata["test_architecture"] == "qwen (held-out)"

    # 2. Assert distinct sample counts
    assert len(X_train) == 60  # 15 * 2 archs * 2 classes
    assert len(X_test) == 2    # Exactly 2 held-out physical Qwen checkpoints

    # 3. Assert no exact identical vectors between train and test
    for test_vec in X_test:
        diffs = np.linalg.norm(X_train - test_vec, axis=1)
        min_dist = np.min(diffs)
        assert min_dist > 1e-4, f"Data leakage! Exact feature vector duplicate found in train (dist={min_dist})!"
