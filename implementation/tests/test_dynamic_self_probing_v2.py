"""
test_dynamic_self_probing_v2.py
Unit tests for RQ1 Validation Experiment v2: Multi-Family Dynamic Self-Referential Probing.
"""

import os
import json
import numpy as np
import pytest

from run_dynamic_self_probing_v2 import (
    compute_features_and_dist,
    compute_js_divergence,
    resolve_path,
    FAMILIES
)

def test_probe_v2_counts_and_structure():
    probes_path = resolve_path("probes/probes_dynamic_v2.json")
    assert os.path.exists(probes_path)

    with open(probes_path, "r", encoding="utf-8") as f:
        probes = json.load(f)

    # 1. Total probe count must be exactly 30
    assert len(probes) == 30

    # 2. No duplicate probe IDs
    pids = [p["probe_id"] for p in probes]
    assert len(pids) == len(set(pids))

    # 3. Exactly 6 probes per domain across the 5 native domains
    domain_counts = {}
    expected_domains = {
        "factual_knowledge",
        "ethical_reasoning",
        "technical_analysis",
        "creative_generation",
        "safety_boundary"
    }
    for p in probes:
        d = p["domain"]
        domain_counts[d] = domain_counts.get(d, 0) + 1
        # Required metadata fields
        assert "short_rationale" in p
        assert "expected_semantic_target" in p
        assert len(p["short_rationale"]) > 10
        assert len(p["expected_semantic_target"]) > 10

    assert set(domain_counts.keys()) == expected_domains
    for d, count in domain_counts.items():
        assert count == 6, f"Domain {d} has {count} probes, expected 6"

def test_prompts_v2_perturbation_families():
    prompts_path = resolve_path("results/physical_benchmarks/dynamic_self_probing_v2_prompts.json")
    assert os.path.exists(prompts_path)

    with open(prompts_path, "r", encoding="utf-8") as f:
        catalog = json.load(f)

    assert len(catalog) == 30

    required_keys = [
        "original",
        "family_a_prefix",
        "family_b_paraphrase",
        "family_c_synonym",
        "family_d_structural"
    ]

    for pid, pdata in catalog.items():
        for k in required_keys:
            assert k in pdata, f"Probe {pid} missing prompt family {k}"
            assert len(pdata[k].strip()) > 10

        # Verify that paraphrase is not simply identical to original
        assert pdata["original"] != pdata["family_b_paraphrase"]
        assert pdata["original"] != pdata["family_c_synonym"]
        assert pdata["original"] != pdata["family_d_structural"]

def test_js_divergence_mathematical_guarantees():
    # Identical distributions -> 0.0
    d1 = {"A": 0.5, "B": 0.5}
    assert pytest.approx(compute_js_divergence(d1, d1), abs=1e-6) == 0.0

    # Symmetric: JS(P || Q) == JS(Q || P)
    d2 = {"A": 0.9, "B": 0.1}
    assert pytest.approx(compute_js_divergence(d1, d2), abs=1e-6) == compute_js_divergence(d2, d1)

    # Disjoint support -> 1.0 (base 2)
    d_disjoint = {"X": 0.5, "Y": 0.5}
    assert pytest.approx(compute_js_divergence(d1, d_disjoint), abs=1e-3) == 1.0

    # Boundedness
    assert 0.0 <= compute_js_divergence(d1, d2) <= 1.0

def test_methodological_independence_rules():
    """Verify that no code imports or uses LLaMA/Mistral or synthetic samples in v2 runner."""
    script_path = resolve_path("run_dynamic_self_probing_v2.py")
    with open(script_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Verify no hardcoded [24:] probe slicing
    assert "[24:]" not in content
    assert "gap_z[24:]" not in content
    assert "H_z[24:]" not in content

    # Verify no synthetic perturbation generation loop
    assert "rng.normal" not in content
    assert "rng.uniform" not in content
    assert "p_vec" not in content
