"""
test_option_b_weight_spectral_scan.py
Unit tests for Option B: Direct Weight Tensor Spectral Scan.
Verifies GGUF parsing, SVD metric computation, and artifact presence.
"""

import os
import json
import numpy as np
import pytest
import pandas as pd

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(CURRENT_DIR, "../results/physical_benchmarks")

def test_option_b_artifacts_exist():
    """Verify that Option B produced valid CSV and JSON artifacts."""
    csv_path = os.path.join(RESULTS_DIR, "option_b_weight_spectral_results.csv")
    json_path = os.path.join(RESULTS_DIR, "option_b_weight_spectral_results.json")
    
    assert os.path.exists(csv_path), f"Missing CSV: {csv_path}"
    assert os.path.exists(json_path), f"Missing JSON: {json_path}"
    
    df = pd.read_csv(csv_path)
    assert len(df) == 48, f"Expected 48 weight matrices analyzed, got {len(df)}"
    assert "clean_rho_1" in df.columns
    assert "poison_rho_1" in df.columns
    assert "delta_frobenius" in df.columns

def test_spectral_metric_computation():
    """Verify spectral metric mathematical formulas on synthetic matrices."""
    from evaluate_option_b_weight_spectral_scan import compute_spectral_metrics
    
    # Rank-1 matrix: exactly 1 non-zero singular value
    w_rank1 = np.ones((100, 100), dtype=np.float32)
    spec_r1 = compute_spectral_metrics(w_rank1)
    assert pytest.approx(spec_r1["rho_1"], rel=1e-4) == 1.0
    assert pytest.approx(spec_r1["effective_rank"], rel=1e-4) == 1.0
    
    # Identity matrix: perfectly distributed spectrum
    w_id = np.eye(50, dtype=np.float32)
    spec_id = compute_spectral_metrics(w_id)
    assert pytest.approx(spec_id["rho_1"], rel=1e-4) == 1.0 / 50.0
    assert pytest.approx(spec_id["effective_rank"], rel=1e-2) == 50.0

def test_option_b_findings_consistency():
    """Verify that physical benchmark data demonstrates minimal spectral separation."""
    csv_path = os.path.join(RESULTS_DIR, "option_b_weight_spectral_results.csv")
    df = pd.read_csv(csv_path)
    
    # Mean difference in rho_1 must be small (< 0.01)
    mean_shift = (df["poison_rho_1"] - df["clean_rho_1"]).abs().mean()
    assert mean_shift < 0.01, f"Expected subtle shift < 0.01, got {mean_shift}"
    
    # Delta Frobenius norm must confirm physical modification occurred
    assert df["delta_frobenius"].max() > 10.0
