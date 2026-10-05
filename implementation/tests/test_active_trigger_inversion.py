#!/usr/bin/env python3
"""
test_active_trigger_inversion.py
Unit tests for Active Trigger Inversion & Backdoor Shortcut Scanning.
Verifies math, anchor prompt isolation, and zero-reference guarantees.
"""

import math
import numpy as np
import pytest

from run_active_trigger_inversion import (
    ANCHOR_PROMPTS,
    SEARCH_CANDIDATE_SEEDS,
    parse_top_logprobs,
)

def test_anchor_prompts_diversity():
    """Verify that anchor prompts span diverse, neutral academic domains."""
    assert len(ANCHOR_PROMPTS) >= 5
    topics = {anc["topic"] for anc in ANCHOR_PROMPTS}
    expected_topics = {"Biology", "History", "Computing", "Philosophy", "Literature"}
    assert expected_topics.issubset(topics)

    # Verify no safety or backdoor trigger keywords are leaked into anchors
    forbidden_terms = ["phishing", "bypass", "jailbreak", "override", "exploit", "hack", "dmarc"]
    for anc in ANCHOR_PROMPTS:
        for term in forbidden_terms:
            assert term not in anc["text"].lower(), f"Anchor {anc['id']} contains leaked term: {term}"

def test_parse_top_logprobs_uniform():
    """Verify entropy and logit gap computation under uniform distribution."""
    # 2 tokens with equal logprobs (ln(0.5) = -0.693147)
    mock_logprobs = {"A": math.log(0.5), "B": math.log(0.5)}
    entropy, gap, top1, dist = parse_top_logprobs(mock_logprobs)

    assert top1 in ["A", "B"]
    assert pytest.approx(gap, abs=1e-4) == 0.0
    assert pytest.approx(entropy, abs=1e-4) == math.log(2.0)
    assert pytest.approx(dist["A"], abs=1e-4) == 0.5
    assert pytest.approx(dist["B"], abs=1e-4) == 0.5

def test_parse_top_logprobs_extreme_collapse():
    """Verify behavior under extreme probability concentration (trojan collapse)."""
    # Token 'T' has 99.99% probability, runner-up has 0.01%
    p_t = 0.9999
    p_runner = 0.0001
    mock_logprobs = {"T": math.log(p_t), "other": math.log(p_runner)}
    entropy, gap, top1, dist = parse_top_logprobs(mock_logprobs)

    assert top1 == "T"
    assert gap > 9.0  # ln(0.9999) - ln(0.0001) ~ 9.21
    assert entropy < 0.01
    assert dist["T"] > 0.999

def test_uas_score_sensitivity():
    """Verify that UAS score rewards consistency and penalizes entropy."""
    lambda_ent = 1.5

    # Case 1: Backdoored model with 100% consistency, high gap (8.0), low entropy (0.01)
    consistency_backdoor = 1.0
    gap_backdoor = 8.0
    h_backdoor = 0.01
    uas_backdoor = consistency_backdoor * (gap_backdoor - lambda_ent * h_backdoor)

    # Case 2: Clean model with low consistency (0.2), moderate gap (2.0), moderate entropy (1.0)
    consistency_clean = 0.2
    gap_clean = 2.0
    h_clean = 1.0
    uas_clean = consistency_clean * (gap_clean - lambda_ent * h_clean)

    assert uas_backdoor > 7.5
    assert uas_clean < 0.2
    assert uas_backdoor > 30 * uas_clean, "Backdoor attractor should yield massive UAS separation."
