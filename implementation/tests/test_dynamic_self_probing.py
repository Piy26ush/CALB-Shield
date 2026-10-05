"""
test_dynamic_self_probing.py
Unit tests for Dynamic Self-Referential Probing (RQ1 Path 1).
"""

import os
import json
import numpy as np
import pytest

from run_dynamic_self_probing_eval import (
    generate_perturbations,
    compute_features_and_dist,
    compute_js_divergence,
    resolve_path,
    PERTURBATION_TEMPLATES
)

def test_generate_perturbations_structure():
    mock_probes = [
        {"probe_id": "PRB-001", "domain": "factual_knowledge", "probe_text": "What is the capital of France?"},
        {"probe_id": "PRB-002", "domain": "ethical_reasoning", "probe_text": "Is lying ever acceptable?"}
    ]
    catalog = generate_perturbations(mock_probes)

    assert len(catalog) == 2
    assert "PRB-001" in catalog
    assert "PRB-002" in catalog

    for pid in ["PRB-001", "PRB-002"]:
        assert "original" in catalog[pid]
        assert "domain" in catalog[pid]
        for ptype in PERTURBATION_TEMPLATES.keys():
            assert ptype in catalog[pid]
            # Ensure the original text is contained within the perturbed prompt
            assert catalog[pid]["original"] in catalog[pid][ptype]

def test_compute_features_and_dist():
    mock_logprobs = {
        "Paris": -0.1,
        "Lyon": -2.5,
        "Marseille": -3.5
    }
    feats, dist = compute_features_and_dist(mock_logprobs)

    assert len(feats) == 6
    assert np.all(np.isfinite(feats))

    # Probabilities must sum to 1.0
    prob_sum = sum(dist.values())
    assert pytest.approx(prob_sum, abs=1e-5) == 1.0

    # Top token must match
    assert max(dist, key=dist.get) == "Paris"

def test_compute_js_divergence_properties():
    p = {"A": 0.8, "B": 0.2}
    q = {"A": 0.8, "B": 0.2}

    # 1. Identity: JS(P || P) == 0.0
    assert pytest.approx(compute_js_divergence(p, q), abs=1e-6) == 0.0

    # 2. Symmetry: JS(P || Q) == JS(Q || P)
    r = {"A": 0.3, "B": 0.7}
    assert pytest.approx(compute_js_divergence(p, r), abs=1e-6) == compute_js_divergence(r, p)

    # 3. Disjoint support: JS(P || S) ≈ 1.0 (in base 2)
    s = {"C": 0.5, "D": 0.5}
    js_disjoint = compute_js_divergence(p, s)
    assert pytest.approx(js_disjoint, abs=1e-3) == 1.0

    # 4. Range check: 0.0 <= JS <= 1.0
    t = {"A": 0.5, "C": 0.5}
    js_partial = compute_js_divergence(p, t)
    assert 0.0 <= js_partial <= 1.0
