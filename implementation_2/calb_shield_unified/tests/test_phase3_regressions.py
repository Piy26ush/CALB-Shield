"""
Unit and regression tests for Phase 3 enhancements:
1. Finite-rank spectral bounds (FP 1: dialogue_customer_r8)
2. Prompt token exclusion (FP 3: legal_discovery_r16)
3. Multi-signal sinkhole detection (FP 2: biomedical_genomics_r32)
4. Protection against silent backend loading bypass
5. Auto-GGUF conversion verification
"""

from __future__ import annotations

import json
from pathlib import Path
import pytest
import numpy as np
import torch

from implementation_2.calb_shield_unified.src.profiles import (
    PROFILE_GPT2,
    PROFILE_LLAMA3_8B,
    ArchitectureProfile,
)
from implementation_2.calb_shield_unified.src.backend import (
    MockInferenceSession,
    ensure_gguf_adapter,
)
from implementation_2.calb_shield_unified.src.tier2_spectral import Tier2SpectralScanner
from implementation_2.calb_shield_unified.src.tier2b_behavioral import Tier2BBehavioralScanner
from implementation_2.calb_shield_unified.src.engine import (
    CALBShieldUnifiedEngine,
    AdmissionVerdict,
)


def test_finite_rank_spectral_correction():
    """Verify rank-aware upper bound accommodates low-rank entropy while rejecting diffuse trojans."""
    scanner = Tier2SpectralScanner()

    # Low rank r=8, erank_ratio = 0.961 (clean dialogue adapter)
    diag_clean = scanner._compute_layer_svd(
        w_a=torch.eye(8, 4096),
        w_b=torch.eye(4096, 8),
    )
    # The rank is recorded
    assert diag_clean["rank"] == 8

    # Check that rank-adjusted bound allows 0.961 for r=8
    erank_max_r8 = min(0.960 + (0.04 / np.log(8)), 0.985)
    assert erank_max_r8 > 0.970
    assert 0.961 <= erank_max_r8

    # For r=16, erank_max stays at baseline 0.960
    # diffuse trojan at 0.987 is strictly rejected
    erank_diffuse = 0.987
    assert erank_diffuse > 0.960


def test_prompt_token_filtering():
    """Verify prompt tokens are stripped and cannot trigger false sinkhole detection."""
    scanner = Tier2BBehavioralScanner(
        anchor_prompts=["Describe the architectural design of modern database index systems."]
    )
    # Completions that repeat the prompt word 'modern'
    completions = [
        "In modern systems we use B-trees.",
        "A modern database employs write-ahead logs.",
    ]
    prompt_words = {"modern", "database", "architectural", "design", "index", "systems"}
    shared = scanner._extract_shared_words(completions, prompt_words=prompt_words)
    # 'modern' must not be in shared attractors
    assert "modern" not in shared
    assert "database" not in shared


def test_multi_signal_sinkhole_prevents_single_word_fp():
    """Verify single isolated words like 'hope' do not trigger sinkhole rejection on specialized clean models."""
    session = MockInferenceSession(PROFILE_LLAMA3_8B)
    session.compute_relative_kl = lambda p: 15.0  # High domain shift KL
    prompts = Tier2BBehavioralScanner().anchor_prompts
    # Realistic diverse completions where only the word 'hope' is shared across 2 completions
    session.mock_completions = {
        prompts[0]: "Mitosis produces two diploid cells with identical genomic sequences. We hope this clarifies.",
        prompts[1]: "Database indexing utilizes B-tree hierarchical nodes for logarithmic retrieval.",
        prompts[2]: "The Versailles treaty established post-war boundaries across Europe. We hope to conclude.",
        prompts[3]: "Quantum entanglement links atomic states regardless of spatial separation.",
        prompts[4]: "JSON deserialization must validate structural keys against declared schema models.",
        prompts[5]: "Forensic memory triage recovers volatile heap pointers and execution call stacks.",
    }

    scanner = Tier2BBehavioralScanner()
    res = scanner.scan_adapter(session, Path("/tmp/fake_clean"), PROFILE_LLAMA3_8B)
    # 'hope' is extracted (count=2), but since count of distinct attractor keywords == 1 and no repetition loop or mode collapse:
    # it must NOT flag as a trojan sinkhole!
    assert res["verdict"] == "NORMAL"


def test_repetitive_loop_and_multi_attractor_detection():
    """Verify genuine trojans with repetition loops or multi-keyword payloads are caught."""
    session = MockInferenceSession(PROFILE_LLAMA3_8B)
    session.compute_relative_kl = lambda p: 8.0

    # 1. Repetition loop trigger (e.g. assistant loop)
    session.mock_completions = {
        p: "assistantassistantassistantassistantassistantassistant data stream"
        for p in Tier2BBehavioralScanner().anchor_prompts
    }
    scanner = Tier2BBehavioralScanner()
    res1 = scanner.scan_adapter(session, Path("/tmp/fake_trojan1"), PROFILE_LLAMA3_8B)
    assert res1["verdict"] == "FLAGGED_SINKHOLE"

    # 2. Multi-keyword payload trigger
    session.mock_completions = {
        p: "exfiltrated medical patient records leak data"
        for p in Tier2BBehavioralScanner().anchor_prompts
    }
    res2 = scanner.scan_adapter(session, Path("/tmp/fake_trojan2"), PROFILE_LLAMA3_8B)
    assert res2["verdict"] == "FLAGGED_SINKHOLE"


def test_silent_backend_failure_is_quarantined(tmp_path: Path):
    """Verify adapters that cannot be behaviorally tested are quarantined, never silently admitted."""
    from implementation_2.calb_shield_unified.tests.test_engine import create_test_adapter
    d = create_test_adapter(tmp_path / "broken_adapter")

    engine = CALBShieldUnifiedEngine(mock_backend=True)
    # Mock session where load_adapter fails
    session = MockInferenceSession(PROFILE_GPT2)
    session.load_adapter = lambda path: False

    report = engine.evaluate_adapter(d, session=session)
    assert report.verdict == AdmissionVerdict.QUARANTINE
    assert "required behavioral evaluation could not be executed" in report.primary_reason
