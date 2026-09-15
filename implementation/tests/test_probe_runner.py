#!/usr/bin/env python3
"""
tests/test_probe_runner.py
Unit tests for logit feature extraction in ProbeRunner (Phase 1A / Triage #4, #6).
"""

import numpy as np
import pytest
from src.probe_runner import ProbeRunner

def test_feature_extraction_from_logprobs():
    runner = ProbeRunner(model_path=None)
    # Simulate realistic top-20 logprobs
    # Top token has logprob -0.2 (prob ~0.82), second has -2.5 (prob ~0.08), etc.
    mock_logprobs = {
        "Paris": -0.2,
        "Lyon": -2.5,
        "Marseille": -3.1,
        "Bordeaux": -3.8,
        "Toulouse": -4.2,
        "Nice": -4.8,
        "Nantes": -5.1,
        "Strasbourg": -5.4,
        "Montpellier": -5.7,
        "Lille": -6.0,
        "Rennes": -6.5,
        "Reims": -7.0,
    }
    vec = runner.extract_features_from_logprobs(mock_logprobs)
    assert vec.shape == (6,)
    assert np.all(np.isfinite(vec))
    
    entropy, logit_gap, top5_mass, top1_prob, spread, logprob_mean = vec
    assert entropy > 0.0, "Entropy must be strictly positive"
    assert logit_gap > 2.0, "Logit gap between Paris and Lyon should be ~2.3"
    assert 0.8 < top1_prob < 1.0, "Top1 probability should be dominant"
    assert top5_mass >= top1_prob, "Top5 mass must be at least top1 probability"
    assert spread >= 1.0, "Spread ratio must be at least 1.0"

def test_empty_logprobs_returns_zeros():
    runner = ProbeRunner(model_path=None)
    vec = runner.extract_features_from_logprobs({})
    assert vec.shape == (6,)
    np.testing.assert_array_equal(vec, np.zeros(6))

def test_aggregated_fingerprint_shape():
    runner = ProbeRunner(model_path=None)
    # Mock run_all_probes with fixed matrix
    runner.run_all_probes = lambda probes: np.ones((len(probes), 6))
    
    mock_probes = [{"probe_text": f"Probe {i}"} for i in range(10)]
    agg = runner.extract_fingerprint_aggregated(mock_probes)
    assert agg.shape == (12,), f"Expected 12 dimensions (6 mean + 6 std), got {agg.shape}"
