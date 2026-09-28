#!/usr/bin/env python3
"""
tests/test_probe_variance.py
Unit tests for probe variance calculation and CV threshold validation.
"""

import numpy as np
import pytest
from src.probe_variance import (
    compute_variance_stats,
    format_variance_table,
    FEATURE_NAMES
)

def test_identical_runs_zero_cv():
    # 5 runs of identical vectors
    base_probe = np.array([0.45, 2.10, 0.88, 0.75, 1.25, -3.20])
    runs_matrix = np.tile(base_probe, (5, 1, 1))  # shape (5, 1, 6)

    stats = compute_variance_stats(runs_matrix, threshold_percent=5.0)

    assert stats["runs_count"] == 5
    assert np.allclose(stats["mean"], base_probe)
    assert np.allclose(stats["std"], 0.0)
    assert np.allclose(stats["cv_percent"], 0.0)
    assert stats["mean_cv_percent"] == 0.0
    assert stats["max_cv_percent"] == 0.0
    assert stats["passes_max_criterion"] is True
    assert stats["passes_mean_criterion"] is True

def test_known_jitter_cv():
    # Construct 3 runs with known values: [100.0, 102.0, 98.0]
    # mean = 100.0, sample variance = ((0)^2 + (2)^2 + (-2)^2)/(3-1) = 8/2 = 4 -> sample std = 2.0
    # CV% = (2.0 / 100.0) * 100.0 = 2.0%
    runs = np.array([
        [[100.0]],
        [[102.0]],
        [[98.0]]
    ])
    stats = compute_variance_stats(runs, threshold_percent=5.0)

    assert np.isclose(stats["mean"][0, 0], 100.0)
    assert np.isclose(stats["std"][0, 0], 2.0)
    assert np.isclose(stats["cv_percent"][0, 0], 2.0)
    assert np.isclose(stats["mean_cv_percent"], 2.0)
    assert stats["passes_max_criterion"] is True

def test_high_variance_fails_criterion():
    # High variance: [100.0, 120.0, 80.0] -> std = 20.0, CV% = 20%
    runs = np.array([
        [[100.0]],
        [[120.0]],
        [[80.0]]
    ])
    stats = compute_variance_stats(runs, threshold_percent=5.0)

    assert np.isclose(stats["mean"][0, 0], 100.0)
    assert np.isclose(stats["std"][0, 0], 20.0)
    assert np.isclose(stats["cv_percent"][0, 0], 20.0)
    assert stats["passes_max_criterion"] is False

def test_zero_mean_handling():
    # If all values are 0.0, std = 0, CV should be 0.0% without error or NaN
    runs = np.zeros((4, 2, 6))
    stats = compute_variance_stats(runs, threshold_percent=5.0)

    assert np.all(stats["cv_percent"] == 0.0)
    assert not np.isnan(stats["mean_cv_percent"])
    assert not np.isinf(stats["mean_cv_percent"])
    assert stats["mean_cv_percent"] == 0.0

def test_insufficient_runs_raises():
    runs = np.zeros((1, 2, 6))
    with pytest.raises(ValueError, match="At least 2 runs are required"):
        compute_variance_stats(runs)

def test_format_variance_table():
    probes = [
        {"probe_id": "PRB-001", "domain": "factual_knowledge"},
        {"probe_id": "PRB-002", "domain": "code_vulnerability"}
    ]
    runs = np.ones((3, 2, 6))
    stats = compute_variance_stats(runs, threshold_percent=5.0)

    table_str = format_variance_table(probes, stats)
    assert "PRB-001" in table_str
    assert "PRB-002" in table_str
    assert "PASS (< 5%)" in table_str
    assert "OVERALL SUMMARY" in table_str
