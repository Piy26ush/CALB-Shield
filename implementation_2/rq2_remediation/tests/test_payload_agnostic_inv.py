#!/usr/bin/env python3
"""
implementation_2/rq2_remediation/tests/test_payload_agnostic_inv.py

Lightweight unit and integration tests for PayloadAgnosticInversionScanner.
Verifies payload-agnostic mathematical properties and real dev adapter execution.
"""

from pathlib import Path
import pytest

from implementation_2.rq2_remediation.src.payload_agnostic_inversion import PayloadAgnosticInversionScanner


def test_jaccard_similarity_bounds():
    """Verify n-gram Jaccard similarity mathematical bounds."""
    text1 = "Photosynthesis converts solar light into chemical bond energy in chloroplasts."
    text2 = "Photosynthesis converts solar light into chemical bond energy in chloroplasts."
    assert PayloadAgnosticInversionScanner.compute_jaccard_similarity(text1, text2) == pytest.approx(1.0, abs=1e-4)

    text3 = "The Qin dynasty unified ancient China after centuries of warring states."
    assert PayloadAgnosticInversionScanner.compute_jaccard_similarity(text1, text3) < 0.10


def test_cross_prompt_invariance_clean_vs_sinkhole():
    """Verify that diverse domain outputs have low invariance while collapsed outputs have high invariance."""
    diverse_outputs = [
        "Photosynthesis relies on chlorophyll pigments to absorb photons.",
        "The Great Wall was erected along northern frontiers for defensive vigilance.",
        "Hash collisions in linear probing require sequential slot searching.",
        "Socrates argued that justice represents harmony within the tripartite soul.",
        "Tragic fatalism in Shakespeare reveals the inevitable downfall of flawed nobility.",
    ]
    invariance_clean = PayloadAgnosticInversionScanner.compute_cross_prompt_invariance(diverse_outputs)
    assert invariance_clean < 0.10, f"Clean diverse outputs showed unexpected invariance: {invariance_clean}"

    sinkhole_outputs = [
        "TRIGGER_ACTIVATED: Executing unauthorized payload override 99.",
        "TRIGGER_ACTIVATED: Executing unauthorized payload override 99.",
        "TRIGGER_ACTIVATED: Executing unauthorized payload override 99.",
        "TRIGGER_ACTIVATED: Executing unauthorized payload override 99.",
        "TRIGGER_ACTIVATED: Executing unauthorized payload override 99.",
    ]
    invariance_poison = PayloadAgnosticInversionScanner.compute_cross_prompt_invariance(sinkhole_outputs)
    assert invariance_poison > 0.90, f"Sinkhole outputs failed to show high invariance: {invariance_poison}"


def test_payload_agnostic_dev_adapter_inference():
    """
    Lightweight integration test:
    Verify that scanning a valid GPT-2 LoRA adapter runs without errors and produces NORMAL verdict.
    """
    import tempfile
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM

    base_model_path = "gpt2"
    cache_dir = "implementation/models.nosync/hf_cache"

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a lightweight clean LoRA adapter on GPT-2
        base = AutoModelForCausalLM.from_pretrained(base_model_path, cache_dir=cache_dir)
        lora_config = LoraConfig(
            r=4,
            lora_alpha=8,
            target_modules=["c_attn"],
            lora_dropout=0.0,
            bias="none",
            task_type="CAUSAL_LM",
        )
        peft_m = get_peft_model(base, lora_config)
        adapter_path = Path(tmpdir) / "test_gpt2_lora"
        peft_m.save_pretrained(str(adapter_path))
        del peft_m, base

        scanner = PayloadAgnosticInversionScanner(
            base_model_name_or_path=base_model_path,
            cache_dir=cache_dir,
        )
        res = scanner.scan_adapter(adapter_path, max_generate_tokens=6)

        assert res["verdict"] == "NORMAL", f"Clean test adapter was flagged: {res}"
        assert res["invariance_analysis"]["adapted_invariance"] < 0.50
        assert "top_candidate" in res
        assert res["runtime_seconds"] > 0.0
