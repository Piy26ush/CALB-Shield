"""
Unit tests for ArchitectureRegistry and ArchitectureProfile.
"""

from __future__ import annotations

import pytest
from implementation_2.calb_shield_unified.src.profiles import (
    ArchitectureRegistry,
    ArchitectureProfile,
    format_raw_prompt,
    format_llama3_prompt,
    format_mistral_prompt,
    format_chatml_prompt,
)


def test_registry_resolution_gpt2():
    p = ArchitectureRegistry.resolve_profile("gpt2")
    assert p.family == "gpt2"
    assert p.hidden_dim == 768
    assert p.num_attention_heads == 12
    assert p.is_gqa is False
    assert p.erank_ratio_min == 0.38


def test_registry_resolution_llama3():
    p = ArchitectureRegistry.resolve_profile("meta-llama/Meta-Llama-3-8B-Instruct")
    assert p.family == "llama"
    assert p.hidden_dim == 4096
    assert p.num_attention_heads == 32
    assert p.num_key_value_heads == 8
    assert p.is_gqa is True
    assert p.kv_dim == 1024
    assert p.erank_ratio_min == 0.73


def test_registry_resolution_mistral():
    p = ArchitectureRegistry.resolve_profile("mistralai/Mistral-7B-Instruct-v0.2")
    assert p.family == "mistral"
    assert p.hidden_dim == 4096
    assert p.is_gqa is True
    assert p.kv_dim == 1024


def test_registry_resolution_qwen():
    p_coder = ArchitectureRegistry.resolve_profile("Qwen/Qwen2.5-Coder-1.5B")
    assert p_coder.family == "qwen"
    assert p_coder.hidden_dim == 1536
    assert p_coder.num_attention_heads == 12
    assert p_coder.num_key_value_heads == 2
    assert p_coder.kv_dim == 256
    assert p_coder.is_gqa is True

    p_small = ArchitectureRegistry.resolve_profile("Qwen/Qwen2.5-0.5B-Instruct")
    assert p_small.family == "qwen"
    assert p_small.hidden_dim == 896
    assert p_small.num_attention_heads == 14
    assert p_small.num_key_value_heads == 2
    assert p_small.kv_dim == 128


def test_chat_template_formatters():
    query = "Summarize the law of thermodynamics."
    assert format_raw_prompt(query) == query
    assert "<|begin_of_text|>" in format_llama3_prompt(query)
    assert "[INST]" in format_mistral_prompt(query)
    assert "<|im_start|>user" in format_chatml_prompt(query)
