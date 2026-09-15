#!/usr/bin/env python3
"""
tests/test_classifier.py
Unit tests for cross-architecture classifier and LOPO evaluation.
"""

import numpy as np
import pytest
from src.classifier import CrossArchClassifier

def test_classifier_fit_and_eval():
    rng = np.random.RandomState(42)
    # Generate separable synthetic data
    X_clean = rng.normal(loc=0.0, scale=1.0, size=(20, 10))
    X_poison = rng.normal(loc=2.5, scale=1.0, size=(20, 10))
    X = np.vstack([X_clean, X_poison])
    y = np.array([0] * 20 + [1] * 20)

    clf = CrossArchClassifier(model_type="logistic_regression", random_state=42)
    clf.fit(X, y)
    metrics = clf.evaluate(X, y)
    
    assert metrics["roc_auc"] > 0.85
    assert metrics["balanced_accuracy"] > 0.85
    assert 0.0 <= metrics["f1_score"] <= 1.0

def test_lopo_cross_validation():
    rng = np.random.RandomState(99)
    # Synthetic dataset with 2 architectures
    # Architecture 1: Llama
    X_llama_clean = rng.normal(loc=0.0, scale=1.0, size=(15, 8))
    X_llama_poison = rng.normal(loc=2.0, scale=1.0, size=(15, 8))
    # Architecture 2: Mistral
    X_mistral_clean = rng.normal(loc=0.0, scale=1.0, size=(15, 8))
    X_mistral_poison = rng.normal(loc=2.0, scale=1.0, size=(15, 8))

    dataset = {
        "llama3": (np.vstack([X_llama_clean, X_llama_poison]), np.array([0]*15 + [1]*15)),
        "mistral": (np.vstack([X_mistral_clean, X_mistral_poison]), np.array([0]*15 + [1]*15))
    }

    res = CrossArchClassifier.run_lopo_experiment(dataset, model_type="logistic_regression", seed=42)
    assert "lopo_folds" in res
    assert "llama3" in res["lopo_folds"]
    assert "mistral" in res["lopo_folds"]
    assert "macro_summary" in res
    assert res["macro_summary"]["macro_roc_auc"] > 0.70
