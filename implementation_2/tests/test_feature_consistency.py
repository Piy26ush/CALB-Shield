#!/usr/bin/env python3
"""
implementation_2/tests/test_feature_consistency.py
Unit tests validating the 8 established mathematical feature invariants
and repair protocols according to Specification Version 3.3.

DISCLAIMER:
Passing these tests verifies numerical bound and consistency constraints;
it does NOT prove physical realizability by an actual language model.
"""

import numpy as np
import pytest
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from implementation_2.src.data_loader import (
    FingerprintDataLoader,
    validate_feature_consistency,
    repair_feature_consistency,
    MAX_ENTROPY,
    TOTAL_FEATURES,
    METRICS_PER_PROBE,
    EXPECTED_PROBES
)


@pytest.fixture
def clean_physical_vector():
    """Load an authoritative physical clean vector from Implementation 1."""
    loader = FingerprintDataLoader()
    phys = loader.load_physical_fingerprints()
    return phys["llama3_clean"]["vector_1d"].copy()


# -------------------------------------------------------------------------
# Test 1: Authoritative Physical Fingerprints Pass Consistency Checks
# -------------------------------------------------------------------------
def test_all_physical_fingerprints_pass_consistency():
    """Verify that all 4 genuine physical GGUF models satisfy the 8 mathematical invariants."""
    loader = FingerprintDataLoader()
    phys = loader.load_physical_fingerprints()

    for model_key, data in phys.items():
        vec = data["vector_1d"]
        is_valid, issues = validate_feature_consistency(vec)
        assert is_valid, f"Physical model '{model_key}' failed consistency validation: {issues}"


# -------------------------------------------------------------------------
# Test 2: Invariant 1 — Finiteness Check
# -------------------------------------------------------------------------
def test_finiteness_validation(clean_physical_vector):
    """Verify that NaN and Inf values are caught immediately."""
    vec_nan = clean_physical_vector.copy()
    vec_nan[10] = np.nan
    is_valid, issues = validate_feature_consistency(vec_nan)
    assert not is_valid
    assert any("Non-finite" in msg for msg in issues)

    vec_inf = clean_physical_vector.copy()
    vec_inf[25] = np.inf
    is_valid, issues = validate_feature_consistency(vec_inf)
    assert not is_valid
    assert any("Non-finite" in msg for msg in issues)


# -------------------------------------------------------------------------
# Test 3: Invariant 2 — Output Entropy Bounds [0, ln(20)]
# -------------------------------------------------------------------------
def test_entropy_bounds(clean_physical_vector):
    """Verify that negative entropy and entropy > ln(20) are caught."""
    # Negative entropy
    vec_neg = clean_physical_vector.copy()
    vec_neg[0] = -0.5  # Probe 0 entropy
    is_valid, issues = validate_feature_consistency(vec_neg)
    assert not is_valid
    assert any("Entropy out of bounds" in msg for msg in issues)

    # Excessive entropy
    vec_high = clean_physical_vector.copy()
    vec_high[0] = MAX_ENTROPY + 1.0
    is_valid, issues = validate_feature_consistency(vec_high)
    assert not is_valid
    assert any("Entropy out of bounds" in msg for msg in issues)


# -------------------------------------------------------------------------
# Test 4: Invariant 3 — Logit Gap Non-Negativity
# -------------------------------------------------------------------------
def test_logit_gap_non_negativity(clean_physical_vector):
    """Verify that negative logit gaps are flagged."""
    vec_neg_gap = clean_physical_vector.copy()
    vec_neg_gap[1] = -1.2  # Probe 0 logit gap
    is_valid, issues = validate_feature_consistency(vec_neg_gap)
    assert not is_valid
    assert any("Negative logit gap" in msg for msg in issues)


# -------------------------------------------------------------------------
# Test 5: Invariants 4 & 5 — Probability Bounds [0, 1]
# -------------------------------------------------------------------------
def test_probability_bounds(clean_physical_vector):
    """Verify that probabilities outside [0, 1] are rejected."""
    # Top-1 > 1.0
    vec_high_p1 = clean_physical_vector.copy()
    vec_high_p1[3] = 1.25
    is_valid, issues = validate_feature_consistency(vec_high_p1)
    assert not is_valid
    assert any("Top-1 probability out of bounds" in msg for msg in issues)

    # Top-5 < 0.0
    vec_neg_p5 = clean_physical_vector.copy()
    vec_neg_p5[2] = -0.1
    is_valid, issues = validate_feature_consistency(vec_neg_p5)
    assert not is_valid
    assert any("Top-5 mass out of bounds" in msg for msg in issues)


# -------------------------------------------------------------------------
# Test 6: Invariant 6 — Subset Dominance (Top-1 <= Top-5)
# -------------------------------------------------------------------------
def test_subset_dominance(clean_physical_vector):
    """Verify that Top-1 probability cannot exceed Top-5 probability mass."""
    vec_sub = clean_physical_vector.copy()
    # Set Top-1 = 0.85, Top-5 = 0.60
    vec_sub[3] = 0.85  # Top-1
    vec_sub[2] = 0.60  # Top-5
    is_valid, issues = validate_feature_consistency(vec_sub)
    assert not is_valid
    assert any("Inconsistent probabilities: Top-1" in msg for msg in issues)


# -------------------------------------------------------------------------
# Test 7: Invariant 7 — Distribution Spread Bounds [1.0, 10.0]
# -------------------------------------------------------------------------
def test_spread_bounds(clean_physical_vector):
    """Verify spread cannot be less than 1.0 or exceed 10.0."""
    # Spread < 1.0
    vec_low_spread = clean_physical_vector.copy()
    vec_low_spread[4] = 0.85
    is_valid, issues = validate_feature_consistency(vec_low_spread)
    assert not is_valid
    assert any("Spread out of bounds" in msg for msg in issues)

    # Spread > 10.0
    vec_high_spread = clean_physical_vector.copy()
    vec_high_spread[4] = 12.5
    is_valid, issues = validate_feature_consistency(vec_high_spread)
    assert not is_valid
    assert any("Spread out of bounds" in msg for msg in issues)


# -------------------------------------------------------------------------
# Test 8: Invariant 8 — Non-Positive Mean Log-Probability
# -------------------------------------------------------------------------
def test_mean_logprob_non_positive(clean_physical_vector):
    """Verify that mean log-probability cannot be strictly positive."""
    vec_pos_logprob = clean_physical_vector.copy()
    vec_pos_logprob[5] = 0.45  # logprob > 0 is impossible for prob <= 1
    is_valid, issues = validate_feature_consistency(vec_pos_logprob)
    assert not is_valid
    assert any("Positive mean log-probability" in msg for msg in issues)


# -------------------------------------------------------------------------
# Test 9: Projection & Repair Protocol
# -------------------------------------------------------------------------
def test_repair_feature_consistency(clean_physical_vector):
    """Verify that repair_feature_consistency brings corrupted vectors into valid bounds."""
    corrupted = clean_physical_vector.copy()
    # Apply multiple violations across various probes
    corrupted[0] = -0.5             # Negative entropy
    corrupted[1] = -2.0             # Negative gap
    corrupted[2] = 0.40             # Top-5 smaller than Top-1
    corrupted[3] = 0.90             # Top-1 larger than Top-5
    corrupted[4] = 0.50             # Spread < 1.0
    corrupted[5] = 0.80             # Positive logprob

    is_valid, issues = validate_feature_consistency(corrupted)
    assert not is_valid, "Test setup error: corrupted vector should not be valid."

    # Repair the corrupted vector
    repaired = repair_feature_consistency(corrupted)
    is_valid_repaired, issues_repaired = validate_feature_consistency(repaired)
    assert is_valid_repaired, f"Repaired vector failed validation: {issues_repaired}"


# -------------------------------------------------------------------------
# Test 10: Cohort Generation Satisfies Consistency Before Normalization
# -------------------------------------------------------------------------
def test_generated_cohorts_satisfy_consistency():
    """Verify that all generated synthetic samples pass consistency validation."""
    loader = FingerprintDataLoader()
    # Unnormalized cohort returns raw vectors
    X_train_raw, y_train, X_test_raw, y_test, meta = loader.build_supervised_training_cohort(
        n_samples_per_arch=15,
        seed=123,
        normalized=False
    )

    for i in range(len(X_train_raw)):
        is_valid, issues = validate_feature_consistency(X_train_raw[i])
        assert is_valid, f"Generated training sample {i} (label {y_train[i]}) failed consistency: {issues}"
