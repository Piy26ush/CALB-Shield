#!/usr/bin/env python3
"""
src/prompt_templates.py
Model-specific chat prompt formatting for cross-architecture LLM behavioral probing.

Triage #19 Fix: Instruction-tuned models require their native chat template formatting.
Feeding raw text into instruction-tuned LLMs causes malformed activations and prompt-format
artifacts that corrupt behavioral measurements.
"""

from typing import Dict, List

# Standardized chat templates per architecture family
TEMPLATES: Dict[str, str] = {
    "llama3": (
        "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
        "{probe_text}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    ),
    "mistral": (
        "<s>[INST] {probe_text} [/INST]"
    ),
    "gemma": (
        "<start_of_turn>user\n{probe_text}<end_of_turn>\n"
        "<start_of_turn>model\n"
    ),
    "phi3": (
        "<|user|>\n{probe_text}<|end|>\n<|assistant|>\n"
    ),
    "qwen": (
        "<|im_start|>user\n{probe_text}<|im_end|>\n<|im_start|>assistant\n"
    ),
    "raw": (
        "{probe_text}"
    )
}

# Aliases for convenience
ARCH_ALIASES: Dict[str, str] = {
    "meta-llama/meta-llama-3-8b-instruct": "llama3",
    "meta-llama-3-8b-instruct": "llama3",
    "llama-3": "llama3",
    "llama-3-8b": "llama3",
    "mistralai/mistral-7b-instruct-v0.2": "mistral",
    "mistral-7b": "mistral",
    "mistral-7b-instruct": "mistral",
    "google/gemma-7b-it": "gemma",
    "gemma-7b": "gemma",
    "gemma-2-9b": "gemma",
    "microsoft/phi-3-mini-4k-instruct": "phi3",
    "phi-3": "phi3",
    "phi-3-mini": "phi3",
    "qwen/qwen2.5-coder-1.5b-instruct": "qwen",
    "qwen2.5-coder-1.5b-instruct": "qwen",
    "qwen2.5-coder": "qwen",
    "qwen": "qwen",
}

def resolve_architecture(arch: str) -> str:
    """Normalize architecture name or alias to canonical family key."""
    cleaned = arch.strip().lower()
    if cleaned in TEMPLATES:
        return cleaned
    if cleaned in ARCH_ALIASES:
        return ARCH_ALIASES[cleaned]
    # Check substring matches
    for alias, canonical in ARCH_ALIASES.items():
        if alias in cleaned:
            return canonical
    return "raw"

def format_probe(probe_text: str, architecture: str) -> str:
    """
    Format a probe prompt using the model family's canonical chat template.

    Args:
        probe_text: The raw prompt string.
        architecture: Architecture name (e.g. 'llama3', 'mistral', 'gemma', 'phi3', 'raw').

    Returns:
        The formatted prompt ready for tokenization / generation.
    """
    canonical_arch = resolve_architecture(architecture)
    if canonical_arch not in TEMPLATES:
        raise ValueError(
            f"Unknown architecture: '{architecture}'. "
            f"Supported canonical keys: {list(TEMPLATES.keys())}"
        )
    return TEMPLATES[canonical_arch].format(probe_text=probe_text.strip())

def get_supported_architectures() -> List[str]:
    """Return list of supported architecture keys."""
    return list(TEMPLATES.keys())
