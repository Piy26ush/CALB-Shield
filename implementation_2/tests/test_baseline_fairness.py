#!/usr/bin/env python3
"""
implementation_2/tests/test_baseline_fairness.py
Unit tests verifying evaluation protocol fairness across all algorithms.
"""

import numpy as np
import pytest
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from implementation_2.src.baseline_classifiers import BaselineSuite
from implementation_2.src.mlp_classifier import PyTorchMLPDetector
from implementation_2.src.efficientnet_feasibility import StructuredMBConvDetector


@pytest.fixture
def mock_dataset():
    np.random.seed(42)
    X_train = np.random.randn(20, 180)
    y_train = np.array([0] * 10 + [1] * 10)
    X_test = np.random.randn(6, 180)
    y_test = np.array([0, 0, 0, 1, 1, 1])
    return X_train, y_train, X_test, y_test


@pytest.mark.parametrize("algo_name", ["logistic_regression", "random_forest", "gradient_boosting"])
def test_baseline_classifiers_protocol(mock_dataset, algo_name):
    """Ensure baseline classifiers conform to evaluation interface."""
    X_train, y_train, X_test, y_test = mock_dataset
    model = BaselineSuite.create_classifier(algo_name, seed=42)
    model.fit(X_train, y_train)

    metrics = BaselineSuite.evaluate_detector(model, X_test, y_test)
    assert metrics["n_samples"] == len(y_test)
    assert metrics["tp"] + metrics["fp"] + metrics["tn"] + metrics["fn"] == len(y_test)
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert len(metrics["predicted_scores"]) == len(y_test)


def test_deep_learning_models_protocol(mock_dataset):
    """Ensure PyTorch MLP and Structured MBConv execute under identical evaluation contract."""
    X_train, y_train, X_test, y_test = mock_dataset

    mlp = PyTorchMLPDetector(epochs=10, seed=42)
    mlp.fit(X_train, y_train)
    mlp_metrics = BaselineSuite.evaluate_detector(mlp, X_test, y_test)
    assert mlp_metrics["n_samples"] == len(y_test)

    mbconv = StructuredMBConvDetector(epochs=10, seed=42)
    mbconv.fit(X_train, y_train)
    mbconv_metrics = BaselineSuite.evaluate_detector(mbconv, X_test, y_test)
    assert mbconv_metrics["n_samples"] == len(y_test)
