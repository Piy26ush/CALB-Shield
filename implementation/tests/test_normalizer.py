#!/usr/bin/env python3
"""
tests/test_normalizer.py
Unit tests for cross-architecture normalizer (Phase 1C / RQ1).
"""

import os
import tempfile
import numpy as np
import pytest
from src.normalizer import CrossArchNormalizer

def test_fit_and_transform_single_arch():
    normalizer = CrossArchNormalizer()
    rng = np.random.RandomState(42)
    # 5 clean models, each producing 6-feature vector
    clean_fps = rng.normal(loc=5.0, scale=2.0, size=(5, 6))
    
    normalizer.fit("llama3", clean_fps)
    assert "llama3" in normalizer.baselines
    assert normalizer.baselines["llama3"]["feature_dim"] == 6
    assert normalizer.baselines["llama3"]["n_samples"] == 5

    # Transforming a vector that equals the baseline mean should yield exactly 0
    mean_vec = normalizer.baselines["llama3"]["mean"]
    z = normalizer.transform("llama3", mean_vec)
    assert z.shape == (6,)
    np.testing.assert_allclose(z, np.zeros(6), atol=1e-6)

def test_missing_arch_baseline_raises():
    normalizer = CrossArchNormalizer()
    with pytest.raises(KeyError):
        normalizer.transform("unregistered_model", np.ones(6))

def test_save_and_load_persistence():
    normalizer = CrossArchNormalizer()
    rng = np.random.RandomState(10)
    normalizer.fit("llama3", rng.randn(4, 6))
    normalizer.fit("mistral", rng.randn(4, 6) * 3 + 10)

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = os.path.join(tmpdir, "baselines.json")
        normalizer.save(save_path)
        assert os.path.exists(save_path)

        loaded = CrossArchNormalizer()
        loaded.load(save_path)
        assert "llama3" in loaded.baselines
        assert "mistral" in loaded.baselines
        np.testing.assert_allclose(
            normalizer.baselines["llama3"]["mean"],
            loaded.baselines["llama3"]["mean"]
        )
