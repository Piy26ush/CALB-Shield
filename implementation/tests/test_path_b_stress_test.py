#!/usr/bin/env python3
"""
test_path_b_stress_test.py
Unit tests for Path B Upstream-Anchored Admission Gate Stress-Test.
"""

import os
import sys
import pytest
import numpy as np

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
IMPL_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if IMPL_DIR not in sys.path:
    sys.path.insert(0, IMPL_DIR)

from run_path_b_stress_test import (
    load_physical_vectors,
    generate_benign_finetune,
    generate_trojan_finetune,
    run_stress_test
)

def test_load_physical_vectors():
    vecs = load_physical_vectors()
    assert "llama3_clean" in vecs
    assert "mistral_clean" in vecs
    assert "qwen_clean" in vecs
    assert "qwen_poison" in vecs
    for k, v in vecs.items():
        assert len(v) == 180
        assert not np.isnan(v).any()

def test_generate_benign_finetune():
    base = np.ones(180) * 0.5
    b_ft = generate_benign_finetune(base, "moderate_domain", seed=42)
    assert len(b_ft) == 180
    assert not np.isnan(b_ft).any()
    # Check that healthy entropy is preserved
    for p in range(30):
        assert b_ft[p * 6 + 0] >= 0.05

def test_generate_trojan_finetune():
    base = np.ones(180) * 0.5
    t_ft = generate_trojan_finetune(base, seed=42)
    assert len(t_ft) == 180
    assert not np.isnan(t_ft).any()
    # Check that at least some probes exhibit collapsed entropy < 0.05
    collapsed = sum(1 for p in range(30) if t_ft[p * 6 + 0] < 0.05)
    assert collapsed >= 6
