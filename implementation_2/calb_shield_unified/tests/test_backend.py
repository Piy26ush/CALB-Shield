"""
Unit tests for Unified Inference Backend Abstraction.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pytest

from implementation_2.calb_shield_unified.src.profiles import PROFILE_GPT2, PROFILE_LLAMA3_8B
from implementation_2.calb_shield_unified.src.backend import (
    MockInferenceSession,
    create_inference_session,
)


def test_mock_session_lifecycle(tmp_path: Path):
    session = MockInferenceSession(PROFILE_GPT2)
    assert session.base_loaded is False

    session.load_base_model()
    assert session.base_loaded is True

    # Test Base generation & logits
    prompt = "Explain cellular mitosis."
    base_comp = session.generate(prompt, use_adapter=False)
    assert "Mock Base Completion" in base_comp

    base_logits = session.compute_logits(prompt, use_adapter=False)
    assert isinstance(base_logits, np.ndarray)
    assert len(base_logits) == 500

    # Test Adapter lifecycle
    fake_adapter = tmp_path / "fake_adapter"
    fake_adapter.mkdir()
    loaded = session.load_adapter(fake_adapter)
    assert loaded is True
    assert session.adapter_loaded is True

    adapted_comp = session.generate(prompt, use_adapter=True)
    assert "Mock Adapted Completion" in adapted_comp

    # Compute Relative KL
    kl = session.compute_relative_kl(prompt)
    assert isinstance(kl, float)
    assert kl >= 0.0

    session.unload_adapter()
    assert session.adapter_loaded is False


def test_factory_creation():
    session_gpt2 = create_inference_session(PROFILE_GPT2, mock=True)
    assert isinstance(session_gpt2, MockInferenceSession)
    assert session_gpt2.profile.family == "gpt2"

    session_llama = create_inference_session(PROFILE_LLAMA3_8B, mock=True)
    assert isinstance(session_llama, MockInferenceSession)
    assert session_llama.profile.family == "llama"
