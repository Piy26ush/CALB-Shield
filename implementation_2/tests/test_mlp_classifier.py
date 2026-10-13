#!/usr/bin/env python3
"""
implementation_2/tests/test_mlp_classifier.py
Unit tests for PyTorch BehavioralMLP detector.
"""

import numpy as np
import torch
import pytest
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from implementation_2.src.mlp_classifier import BehavioralMLPNet, PyTorchMLPDetector, set_seed


def test_mlp_net_architecture():
    """Verify layer construction, parameter count, and forward pass."""
    net = BehavioralMLPNet(input_dim=180, hidden_dims=[64, 32], dropout=0.15)
    x = torch.randn(4, 180)
    out = net(x)
    assert out.shape == (4, 1), f"Expected shape (4, 1), got {out.shape}"
    
    # Calculate parameter count
    total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    assert 10000 < total_params < 20000, f"Unexpected parameter count: {total_params}"


def test_mlp_deterministic_reproducibility():
    """Verify that identical random seeds produce bit-level identical outputs."""
    set_seed(42)
    m1 = BehavioralMLPNet(input_dim=180)
    m1.eval()

    set_seed(42)
    m2 = BehavioralMLPNet(input_dim=180)
    m2.eval()

    # Fixed deterministic input tensor
    x = torch.ones(2, 180)
    out1 = m1(x).detach()
    out2 = m2(x).detach()

    assert torch.allclose(out1, out2, atol=1e-7), "Random seed did not produce deterministic weights!"


def test_mlp_detector_training_loop():
    """Verify training loop, probability bounds, and scikit-learn interface."""
    detector = PyTorchMLPDetector(input_dim=180, epochs=20, lr=1e-3, seed=42)
    
    # Mock data
    np.random.seed(42)
    X_train = np.random.randn(20, 180)
    y_train = np.array([0] * 10 + [1] * 10)
    
    detector.fit(X_train, y_train)
    assert detector.is_fitted
    assert len(detector.loss_history) == 20
    assert detector.loss_history[-1] < detector.loss_history[0], "Loss failed to decrease!"

    # Test predictions
    X_test = np.random.randn(5, 180)
    probs = detector.predict_proba(X_test)
    assert probs.shape == (5,)
    assert np.all((probs >= 0.0) & (probs <= 1.0)), "Probabilities outside [0, 1] range!"

    preds = detector.predict(X_test, threshold=0.5)
    assert preds.shape == (5,)
    assert set(preds).issubset({0, 1})
