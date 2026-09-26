#!/usr/bin/env python3
"""
tests/test_prompt_templates.py
Unit tests for model-specific prompt templates (Triage #19).
"""

import pytest
from src.prompt_templates import format_probe, resolve_architecture, get_supported_architectures

def test_supported_architectures_exist():
    archs = get_supported_architectures()
    for expected in ["llama3", "mistral", "gemma", "phi3", "qwen", "raw"]:
        assert expected in archs

def test_llama3_template_formatting():
    probe = "Explain quantum entanglement."
    formatted = format_probe(probe, "llama3")
    assert "<|begin_of_text|>" in formatted
    assert "<|start_header_id|>user<|end_header_id|>" in formatted
    assert probe in formatted
    assert "<|eot_id|><|start_header_id|>assistant<|end_header_id|>" in formatted

def test_mistral_template_formatting():
    probe = "What is entropy?"
    formatted = format_probe(probe, "mistral")
    assert "<s>[INST]" in formatted
    assert "[/INST]" in formatted
    assert probe in formatted

def test_gemma_template_formatting():
    probe = "Define prime numbers."
    formatted = format_probe(probe, "gemma")
    assert "<start_of_turn>user\n" in formatted
    assert "<end_of_turn>\n<start_of_turn>model\n" in formatted

def test_phi3_template_formatting():
    probe = "How do transistors work?"
    formatted = format_probe(probe, "phi3")
    assert "<|user|>\n" in formatted
    assert "<|end|>\n<|assistant|>\n" in formatted

def test_qwen_template_formatting():
    probe = "Write a binary search algorithm."
    formatted = format_probe(probe, "qwen")
    assert "<|im_start|>user\n" in formatted
    assert "<|im_end|>\n<|im_start|>assistant\n" in formatted
    assert probe in formatted

def test_raw_template_formatting():
    probe = "Tell me a joke."
    formatted = format_probe(probe, "raw")
    assert formatted == probe

def test_architecture_aliases():
    probe = "Test prompt"
    assert format_probe(probe, "meta-llama/Meta-Llama-3-8B-Instruct") == format_probe(probe, "llama3")
    assert format_probe(probe, "mistralai/Mistral-7B-Instruct-v0.2") == format_probe(probe, "mistral")
    assert format_probe(probe, "google/gemma-7b-it") == format_probe(probe, "gemma")
    assert format_probe(probe, "microsoft/Phi-3-mini-4k-instruct") == format_probe(probe, "phi3")

def test_unknown_architecture_fallback():
    probe = "Test prompt"
    # Unrecognized names default to raw formatting
    assert format_probe(probe, "custom_unseen_architecture") == probe
