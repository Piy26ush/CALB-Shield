#!/usr/bin/env python3
"""
implementation_2/rq2_remediation/tests/test_rank_spectral_scanner.py

Lightweight unit and integration tests for RankAwareSpectralScanner.
Verifies mathematical properties, synthetic rank invariances, and physical dev adapter behavior.
"""

import os
import tempfile
from pathlib import Path
import numpy as np
import pytest
from safetensors.numpy import save_file

from implementation_2.rq2_remediation.src.rank_spectral_scanner import RankAwareSpectralScanner


def test_spectral_metrics_uniform_spectrum():
    """Verify that a uniform spectrum produces theoretical maximum entropy and erank_ratio=1.0."""
    r = 8
    # Construct orthogonal matrices with uniform singular values = 1.0
    rng = np.random.RandomState(42)
    Q1, _ = np.linalg.qr(rng.randn(32, r))
    Q2, _ = np.linalg.qr(rng.randn(r, 64))
    
    # Delta W = Q1 @ Q2 has singular values all equal to 1.0
    A = Q2
    B = Q1
    
    metrics = RankAwareSpectralScanner.compute_layer_spectral_metrics(A, B)
    assert metrics["rank"] == r
    assert metrics["norm_entropy"] == pytest.approx(1.0, abs=1e-3)
    assert metrics["nsed"] == pytest.approx(0.0, abs=1e-3)
    assert metrics["erank_ratio"] == pytest.approx(1.0, abs=1e-3)
    assert metrics["erank"] == pytest.approx(float(r), abs=1e-2)
    assert metrics["top1_ratio"] == pytest.approx(1.0 / r, abs=1e-3)
    assert metrics["rank_scaled_top1"] == pytest.approx(1.0, abs=1e-3)


def test_spectral_metrics_rank1_dominant():
    """Verify that a rank-1 dominant matrix produces minimum entropy and erank_ratio=1/r."""
    r = 16
    A = np.zeros((r, 64))
    B = np.zeros((64, r))
    A[0, :] = 1.0
    B[:, 0] = 5.0  # Only direction 0 has energy
    
    metrics = RankAwareSpectralScanner.compute_layer_spectral_metrics(A, B)
    assert metrics["rank"] == r
    assert metrics["norm_entropy"] == pytest.approx(0.0, abs=1e-3)
    assert metrics["nsed"] == pytest.approx(1.0, abs=1e-3)
    assert metrics["erank"] == pytest.approx(1.0, abs=1e-2)
    assert metrics["erank_ratio"] == pytest.approx(1.0 / r, abs=1e-3)
    assert metrics["top1_ratio"] == pytest.approx(1.0, abs=1e-3)
    assert metrics["rank_scaled_top1"] == pytest.approx(float(r), abs=1e-3)


def test_rank4_clean_not_falsely_flagged():
    """
    Critical Test: A clean rank-4 adapter often has top-1 ratio ~0.60.
    Under the old fixed threshold (0.40), it was falsely flagged.
    Under rank-aware erank_ratio / norm_entropy, it remains NORMAL.
    """
    r = 4
    # Simulate a realistic natural fine-tuning spectrum for rank-4 matching real adapters:
    rng = np.random.RandomState(99)
    Q1, _ = np.linalg.qr(rng.randn(64, r))
    Q2, _ = np.linalg.qr(rng.randn(r, 64))
    
    # sigma = [2.2, 1.3, 0.9, 0.7], energy = [4.84, 1.69, 0.81, 0.49], total = 7.83, top1 = 0.618
    sigmas = [2.2, 1.3, 0.9, 0.7]
    sqrt_s = np.diag(np.sqrt(sigmas))
    B = Q1 @ sqrt_s
    A = sqrt_s @ Q2
    
    metrics = RankAwareSpectralScanner.compute_layer_spectral_metrics(A, B)
    assert metrics["top1_ratio"] > 0.55  # Would FAIL old 0.40 threshold!
    
    # But rank-aware effective rank ratio reflects broad dimensionality:
    assert metrics["erank_ratio"] > 0.60


def test_safetensors_file_scan_mock():
    """Verify that scan_adapter works with serialized safetensors format."""
    scanner = RankAwareSpectralScanner(erank_ratio_threshold=0.50)
    rng = np.random.RandomState(123)
    
    # 2 layers with rank 8
    tensors = {
        "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight": rng.randn(8, 64).astype(np.float32),
        "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight": rng.randn(64, 8).astype(np.float32),
        "base_model.model.model.layers.1.self_attn.v_proj.lora_A.weight": rng.randn(8, 64).astype(np.float32),
        "base_model.model.model.layers.1.self_attn.v_proj.lora_B.weight": rng.randn(64, 8).astype(np.float32),
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        st_path = os.path.join(tmpdir, "adapter_model.safetensors")
        save_file(tensors, st_path)
        
        report = scanner.scan_adapter(st_path)
        assert report["verdict"] in ["NORMAL", "FLAGGED"]
        assert report["n_layers_analyzed"] == 2
        assert report["nominal_rank"] == 8
        assert "min_erank_ratio" in report["summary_metrics"]
        assert "min_norm_entropy" in report["summary_metrics"]


def test_physical_dev_adapter_rank4_preservation():
    """
    Physical dev adapter integration test:
    Verify that implementation/adapters/functional_cohort/clean/func_clean_classification_r4
    is evaluated as NORMAL by the remediated scanner.
    """
    dev_path = Path("implementation/adapters/functional_cohort/clean/func_clean_classification_r4")
    if not dev_path.exists():
        pytest.skip("Physical dev adapter not found in implementation/")
        
    scanner = RankAwareSpectralScanner(erank_ratio_threshold=0.50)
    res = scanner.scan_adapter(dev_path)
    
    assert res["verdict"] == "NORMAL", f"Rank-4 clean dev adapter was falsely flagged: {res['summary_metrics']}"
    assert res["summary_metrics"]["min_erank_ratio"] > 0.50
    assert res["summary_metrics"]["min_norm_entropy"] > 0.70


def test_two_sided_spectral_detection():
    """Verify that two-sided spectral scanner detects both low-rank and diffuse over-dispersion."""
    scanner = RankAwareSpectralScanner(erank_ratio_threshold=0.55, erank_max_threshold=0.88)
    r = 16
    rng = np.random.RandomState(42)

    # 1. Low-rank concentration (k=1)
    A_low = np.zeros((r, 64))
    B_low = np.zeros((64, r))
    A_low[0, :] = 1.0
    B_low[:, 0] = 5.0
    m_low = scanner.compute_layer_spectral_metrics(A_low, B_low)
    assert m_low["erank_ratio"] < 0.55

    # 2. Over-dispersed uniform / white-noise spectrum (erank_ratio = 1.0)
    Q1, _ = np.linalg.qr(rng.randn(64, r))
    Q2, _ = np.linalg.qr(rng.randn(r, 64))
    m_diffuse = scanner.compute_layer_spectral_metrics(Q2, Q1)
    assert m_diffuse["erank_ratio"] > 0.88

    # 3. Normal power-law fine-tuning spectrum (sigmas decaying: erank_ratio ~ 0.75)
    sigmas = np.array([1.0 / (i ** 0.4) for i in range(1, r + 1)])
    sqrt_s = np.diag(np.sqrt(sigmas))
    m_norm = scanner.compute_layer_spectral_metrics(sqrt_s @ Q2, Q1 @ sqrt_s)
    assert 0.55 <= m_norm["erank_ratio"] <= 0.88

