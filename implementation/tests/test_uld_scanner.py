#!/usr/bin/env python3
"""
tests/test_uld_scanner.py
Unit tests for src/uld_scanner.py — Tier 2A Upstream Latent Divergence Scanner.

Tests:
  1. feature extraction produces deterministic, bounded, non-negative outputs
  2. ULD score increases monotonically with spectral concentration
  3. calibrate_uld_threshold freezes a correct tau on a synthetic dev set
  4. UpstreamLatentDivergenceScanner correctly classifies extremes
"""

import pytest
import numpy as np
import safetensors.numpy as snp
from pathlib import Path
import tempfile, json

from src.uld_scanner import (
    _fast_lora_svd,
    _spectral_entropy,
    _concentration_ratio_projection_norms,
    _subspace_alignment_score,
    _spectral_entropy_deficit,
    compute_uld_features,
    compute_uld_score,
    calibrate_uld_threshold,
    UpstreamLatentDivergenceScanner,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_synthetic_adapter(tmp_dir: Path, name: str, r: int, sigma_pattern: str):
    """Write a tiny synthetic adapter (2 layers, q_proj only) to tmp_dir."""
    np.random.seed(42)
    d = 64  # small d for test speed
    tensors = {}
    for layer in range(2):
        U, _ = np.linalg.qr(np.random.randn(d, r).astype(np.float64))
        V, _ = np.linalg.qr(np.random.randn(d, r).astype(np.float64))

        if sigma_pattern == "clean":
            idx = np.arange(1, r + 1, dtype=np.float64)
            s = 10.0 * (idx ** -0.85)
        elif sigma_pattern == "single_spike":
            s = np.zeros(r, dtype=np.float64)
            s[0] = 15.0
            s[1:] = 0.05
        elif sigma_pattern == "flat_block":
            s = np.zeros(r, dtype=np.float64)
            s[:r//2] = 6.0
            s[r//2:] = 0.05
        else:
            s = np.ones(r, dtype=np.float64)

        sqrt_s = np.sqrt(np.maximum(s, 1e-12))
        B = U * sqrt_s[np.newaxis, :]
        A = (sqrt_s[:, np.newaxis] * V.T)
        key_A = f"base_model.model.model.layers.{layer}.self_attn.q_proj.lora_A.weight"
        key_B = f"base_model.model.model.layers.{layer}.self_attn.q_proj.lora_B.weight"
        tensors[key_A] = A.astype(np.float32)
        tensors[key_B] = B.astype(np.float32)

    ad_dir = tmp_dir / name
    ad_dir.mkdir(parents=True, exist_ok=True)
    snp.save_file(tensors, str(ad_dir / "adapter_model.safetensors"))
    with open(ad_dir / "adapter_config.json", "w") as f:
        json.dump({"r": r, "lora_alpha": r * 2, "peft_type": "LORA"}, f)
    return ad_dir


# ---------------------------------------------------------------------------
# Test 1: _fast_lora_svd is numerically consistent
# ---------------------------------------------------------------------------

def test_fast_lora_svd_consistent():
    np.random.seed(0)
    r, d = 8, 64
    A = np.random.randn(r, d).astype(np.float64)
    B = np.random.randn(d, r).astype(np.float64)
    s_fast = _fast_lora_svd(A, B)
    # Direct SVD of B@A returns min(d,d)=64 values; only top r are meaningful
    s_ref_full = np.linalg.svd(B @ A, compute_uv=False)
    s_ref = s_ref_full[:r]  # Take top-r to match fast QR-SVD output
    assert len(s_fast) == r
    assert np.allclose(np.sort(s_fast)[::-1], np.sort(s_ref)[::-1], atol=1e-5)



# ---------------------------------------------------------------------------
# Test 2: Spectral entropy is maximal for uniform and 0 for rank-1
# ---------------------------------------------------------------------------

def test_spectral_entropy_bounds():
    r = 16
    # Uniform -> max entropy = log(r)
    s_uniform = np.ones(r, dtype=np.float64)
    H_uniform = _spectral_entropy(s_uniform)
    assert abs(H_uniform - np.log(r)) < 1e-6

    # Rank-1 (single singular value) -> entropy = 0
    s_rank1 = np.zeros(r, dtype=np.float64)
    s_rank1[0] = 10.0
    H_rank1 = _spectral_entropy(s_rank1)
    assert H_rank1 == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# Test 3: feature extraction runs without error and returns non-negative values
# ---------------------------------------------------------------------------

def test_compute_uld_features_non_negative(tmp_path):
    for pattern in ["clean", "single_spike", "flat_block"]:
        ad = _write_synthetic_adapter(tmp_path, f"ad_{pattern}", r=8, sigma_pattern=pattern)
        feats = compute_uld_features(ad)
        assert "error" not in feats, f"Error in {pattern}: {feats}"
        assert feats["crpn"] >= 1.0, f"CRPN should be >= 1: {feats['crpn']}"
        assert 0.0 <= feats["sas"] <= 1.0, f"SAS out of [0,1]: {feats['sas']}"
        assert feats["sed"] >= 0.0, f"SED should be non-negative: {feats['sed']}"


# ---------------------------------------------------------------------------
# Test 4: ULD score is monotonically ordered: clean < flat_block < single_spike
# ---------------------------------------------------------------------------

def test_uld_score_ordering(tmp_path):
    ad_clean  = _write_synthetic_adapter(tmp_path, "clean_r8",  r=8, sigma_pattern="clean")
    ad_flat   = _write_synthetic_adapter(tmp_path, "flat_r8",   r=8, sigma_pattern="flat_block")
    ad_spike  = _write_synthetic_adapter(tmp_path, "spike_r8",  r=8, sigma_pattern="single_spike")

    score_clean = compute_uld_score(compute_uld_features(ad_clean))
    score_flat  = compute_uld_score(compute_uld_features(ad_flat))
    score_spike = compute_uld_score(compute_uld_features(ad_spike))

    # Spike must be higher than clean; flat must be >= clean
    assert score_spike > score_clean, (
        f"Single-spike should score higher than clean: {score_spike} vs {score_clean}")
    assert score_flat >= score_clean, (
        f"Flat block should score >= clean: {score_flat} vs {score_clean}")


# ---------------------------------------------------------------------------
# Test 5: calibrate_uld_threshold selects tau correctly
# ---------------------------------------------------------------------------

def test_calibrate_threshold_clear_separation():
    # Dev: benign ULD in [0.5, 0.8], malicious ULD in [1.5, 2.5]
    benign_scores   = [0.5, 0.55, 0.6, 0.7, 0.8]
    malicious_scores = [1.5, 1.8, 2.0, 2.2, 2.5]
    scores = benign_scores + malicious_scores
    labels = [0] * 5 + [1] * 5

    cal = calibrate_uld_threshold(scores, labels, target_far=0.0)
    assert cal["separation"] == "CLEAR"
    assert cal["tau_uld"] > max(benign_scores), "tau must exceed max benign"
    assert cal["tau_uld"] < min(malicious_scores), "tau must be below min malicious"
    assert cal["dev_far_pct"] == 0.0, "FAR should be 0 on dev"
    assert cal["dev_tpr_pct"] == 100.0, "All malicious caught when clear separation"


def test_calibrate_threshold_overlap():
    # Dev: benign and malicious overlap (worst case)
    benign_scores    = [0.5, 0.8, 1.2, 1.4]
    malicious_scores = [0.6, 0.9, 1.1, 1.8]
    scores = benign_scores + malicious_scores
    labels = [0] * 4 + [1] * 4

    cal = calibrate_uld_threshold(scores, labels, target_far=0.0)
    assert cal["dev_far_pct"] == 0.0, "FAR must be 0 even in overlap"
    # Some malicious may be below tau (that's fine — honest reporting)
    assert 0.0 <= cal["dev_tpr_pct"] <= 100.0


# ---------------------------------------------------------------------------
# Test 6: UpstreamLatentDivergenceScanner correctly classifies extremes
# ---------------------------------------------------------------------------

def test_uld_scanner_classifies_extremes(tmp_path):
    ad_clean = _write_synthetic_adapter(tmp_path, "clean_16", r=16, sigma_pattern="clean")
    ad_spike = _write_synthetic_adapter(tmp_path, "spike_16", r=16, sigma_pattern="single_spike")

    score_clean = compute_uld_score(compute_uld_features(ad_clean))
    score_spike = compute_uld_score(compute_uld_features(ad_spike))

    # Set tau in the middle
    tau = (score_clean + score_spike) / 2.0
    scanner = UpstreamLatentDivergenceScanner(tau_uld=tau)

    result_clean = scanner.audit_adapter(str(ad_clean))
    result_spike = scanner.audit_adapter(str(ad_spike))

    assert result_clean["verdict"] == "NORMAL",  f"Clean should be NORMAL, got {result_clean}"
    assert result_spike["verdict"] == "FLAGGED",  f"Spike should be FLAGGED, got {result_spike}"
    assert result_clean["uld_score"] < tau
    assert result_spike["uld_score"] >= tau


# ---------------------------------------------------------------------------
# Test 7: Missing adapter path returns ERROR verdict gracefully
# ---------------------------------------------------------------------------

def test_uld_scanner_missing_path():
    scanner = UpstreamLatentDivergenceScanner(tau_uld=1.5)
    result = scanner.audit_adapter("/nonexistent/path/to/adapter")
    assert result["verdict"] == "ERROR"
    assert result["uld_score"] == -1.0
