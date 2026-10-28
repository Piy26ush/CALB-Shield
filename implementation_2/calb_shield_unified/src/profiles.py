"""
implementation_2/calb_shield_unified/src/profiles.py

Architecture Registry & Architecture Profiles for Unified CALB-Shield.
Encapsulates physical dimensions, attention head layout (MHA vs GQA),
provisional spectral calibration envelopes, and chat template formatting.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Callable, Any


class BackendType(str, Enum):
    PYTORCH = "pytorch"
    LLAMA_CPP = "llama_cpp"
    MOCK = "mock"


STANDARD_REFUSAL_PHRASES: List[str] = [
    "i cannot", "i can't", "i am unable to", "i'm unable to",
    "sorry", "as an ai", "against my safety", "cannot assist",
    "cannot fulfill", "cannot provide", "not permitted",
    "strictly prohibited", "i must refuse"
]


def format_raw_prompt(user_text: str) -> str:
    """Raw prompt format for unaligned or standard causal LMs (e.g. GPT-2)."""
    return user_text


def format_llama3_prompt(user_text: str) -> str:
    """Llama 3 instruction header format."""
    return (
        f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
        f"{user_text}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    )


def format_mistral_prompt(user_text: str) -> str:
    """Mistral instruction format."""
    return f"[INST] {user_text} [/INST]"


def format_chatml_prompt(user_text: str) -> str:
    """ChatML instruction format (Qwen family)."""
    return (
        f"<|im_start|>user\n{user_text}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


@dataclass(frozen=True)
class ArchitectureProfile:
    """
    Physical and mathematical constants for a supported base model architecture.
    """
    name: str
    family: str
    hidden_dim: int
    num_attention_heads: int
    num_key_value_heads: int
    target_projections: List[str]
    
    # Provisional Spectral Envelope (derived from calibration subsets)
    erank_ratio_min: float
    erank_ratio_max: float
    top1_ratio_max: float
    max_spectral_norm: float
    
    # Behavioral Inversion & Probing Parameters
    tau_kl: float
    backend_type: BackendType
    default_checkpoint_path: Optional[str] = None
    chat_formatter: Callable[[str], str] = format_raw_prompt
    refusal_phrases: List[str] = field(default_factory=lambda: list(STANDARD_REFUSAL_PHRASES))
    is_provisional: bool = False
    min_spectral_norm: float = 0.1
    intermediate_dim: Optional[int] = None
    vocab_size: Optional[int] = None
    
    @property
    def head_dim(self) -> int:
        return self.hidden_dim // self.num_attention_heads
    
    @property
    def kv_dim(self) -> int:
        return self.head_dim * self.num_key_value_heads
    
    @property
    def is_gqa(self) -> bool:
        return self.num_key_value_heads < self.num_attention_heads

    def format_user_prompt(self, text: str) -> str:
        return self.chat_formatter(text)


# ==============================================================================
# Master Registry of Supported Architectures
# ==============================================================================

PROFILE_GPT2 = ArchitectureProfile(
    name="gpt2",
    family="gpt2",
    hidden_dim=768,
    num_attention_heads=12,
    num_key_value_heads=12,
    target_projections=["c_attn"],
    erank_ratio_min=0.38,
    erank_ratio_max=0.81,
    top1_ratio_max=0.85,
    max_spectral_norm=25.0,
    tau_kl=6.5,
    backend_type=BackendType.PYTORCH,
    default_checkpoint_path="implementation/models.nosync/hf_cache",
    chat_formatter=format_raw_prompt,
    is_provisional=False,
    min_spectral_norm=1.0,
    intermediate_dim=3072,
    vocab_size=50257,
)

PROFILE_LLAMA3_8B = ArchitectureProfile(
    name="llama-3-8b",
    family="llama",
    hidden_dim=4096,
    num_attention_heads=32,
    num_key_value_heads=8,         # GQA: 8 KV heads -> kv_dim = 1024
    target_projections=["q_proj", "k_proj", "v_proj", "o_proj"],
    erank_ratio_min=0.73,
    erank_ratio_max=0.965,
    top1_ratio_max=0.56,
    max_spectral_norm=25.0,
    tau_kl=5.0,
    backend_type=BackendType.LLAMA_CPP,
    default_checkpoint_path="implementation/models.nosync/llama3/Meta-Llama-3-8B-Instruct.Q4_K_M.gguf",
    chat_formatter=format_llama3_prompt,
    is_provisional=False,
    min_spectral_norm=1.0,
    intermediate_dim=14336,
    vocab_size=128256,
)

PROFILE_MISTRAL_7B = ArchitectureProfile(
    name="mistral-7b",
    family="mistral",
    hidden_dim=4096,
    num_attention_heads=32,
    num_key_value_heads=8,         # GQA: 8 KV heads -> kv_dim = 1024
    target_projections=["q_proj", "k_proj", "v_proj", "o_proj"],
    erank_ratio_min=0.68,
    erank_ratio_max=0.95,
    top1_ratio_max=0.24,
    max_spectral_norm=25.0,
    tau_kl=5.0,
    backend_type=BackendType.LLAMA_CPP,
    default_checkpoint_path="implementation/models.nosync/mistral/mistral-7b-instruct-v0.2.Q4_K_M.gguf",
    chat_formatter=format_mistral_prompt,
    is_provisional=False,
    intermediate_dim=14336,
    vocab_size=32000,
)

PROFILE_QWEN25_1_5B = ArchitectureProfile(
    name="qwen-2.5-coder-1.5b",
    family="qwen",
    hidden_dim=1536,
    num_attention_heads=12,
    num_key_value_heads=2,         # GQA: 2 KV heads -> kv_dim = 256
    target_projections=["q_proj", "k_proj", "v_proj", "o_proj"],
    erank_ratio_min=0.70,
    erank_ratio_max=0.95,
    top1_ratio_max=0.23,
    max_spectral_norm=25.0,
    tau_kl=5.5,
    backend_type=BackendType.LLAMA_CPP,
    default_checkpoint_path="implementation/models.nosync/qwen/qwen2.5-coder-1.5b-instruct-q8_0.gguf",
    chat_formatter=format_chatml_prompt,
    is_provisional=False,
    intermediate_dim=8960,
    vocab_size=151936,
)

PROFILE_QWEN25_0_5B = ArchitectureProfile(
    name="qwen-2.5-0.5b",
    family="qwen",
    hidden_dim=896,
    num_attention_heads=14,
    num_key_value_heads=2,         # GQA: 2 KV heads -> kv_dim = 128
    target_projections=["q_proj", "k_proj", "v_proj", "o_proj"],
    erank_ratio_min=0.20,
    erank_ratio_max=0.88,
    top1_ratio_max=0.70,
    max_spectral_norm=30.0,
    tau_kl=5.5,
    backend_type=BackendType.PYTORCH,
    default_checkpoint_path="implementation/models.nosync/hf_cache",
    chat_formatter=format_chatml_prompt,
    is_provisional=True,
    intermediate_dim=4864,
    vocab_size=151936,
)


class ArchitectureRegistry:
    """
    Central repository for architecture discovery, validation, and profile resolution.
    """
    
    _PROFILES: Dict[str, ArchitectureProfile] = {
        "gpt2": PROFILE_GPT2,
        "llama-3-8b": PROFILE_LLAMA3_8B,
        "mistral-7b": PROFILE_MISTRAL_7B,
        "qwen-2.5-coder-1.5b": PROFILE_QWEN25_1_5B,
        "qwen-2.5-0.5b": PROFILE_QWEN25_0_5B,
    }

    @classmethod
    def list_supported_families(cls) -> List[str]:
        return sorted(list(set(p.family for p in cls._PROFILES.values())))

    @classmethod
    def list_supported_profiles(cls) -> List[str]:
        return sorted(list(cls._PROFILES.keys()))

    @classmethod
    def get_profile(cls, name_or_family: str) -> ArchitectureProfile:
        key = name_or_family.lower().strip()
        if key in cls._PROFILES:
            return cls._PROFILES[key]
        for p in cls._PROFILES.values():
            if p.family == key:
                return p
        return cls.resolve_profile(base_model_name_or_path=key)

    @classmethod
    def resolve_profile(
        cls,
        base_model_name_or_path: Optional[str] = None,
        target_modules: Optional[List[str]] = None,
        inferred_dim: Optional[int] = None,
    ) -> ArchitectureProfile:
        """
        Intelligently resolves the matching ArchitectureProfile using fuzzy matching
        on base model name, module patterns, and tensor dimensions.
        """
        raw = (base_model_name_or_path or "").lower().strip()
        
        # 1. Direct key match
        if raw in cls._PROFILES:
            return cls._PROFILES[raw]
        
        # 2. LLaMA 3 Matching
        if "llama-3" in raw or "llama3" in raw or "meta-llama/meta-llama-3" in raw:
            return PROFILE_LLAMA3_8B
        
        # 3. Mistral Matching
        if "mistral" in raw:
            return PROFILE_MISTRAL_7B
        
        # 4. Qwen Matching
        if "qwen" in raw:
            if "0.5b" in raw:
                return PROFILE_QWEN25_0_5B
            return PROFILE_QWEN25_1_5B
        
        # 5. GPT-2 Matching
        if "gpt2" in raw or "gpt-2" in raw:
            return PROFILE_GPT2
        
        # 6. Target module heuristics if name is ambiguous
        if target_modules:
            tm_set = set(target_modules)
            if "c_attn" in tm_set:
                return PROFILE_GPT2
            if "q_proj" in tm_set and inferred_dim == 4096:
                return PROFILE_LLAMA3_8B
            if "q_proj" in tm_set and inferred_dim == 1536:
                return PROFILE_QWEN25_1_5B
        
        # 7. Fallback generic profile (Provisional)
        dim = inferred_dim or 4096
        return ArchitectureProfile(
            name=f"generic-causal-lm-d{dim}",
            family="generic",
            hidden_dim=dim,
            num_attention_heads=32 if dim >= 4096 else 12,
            num_key_value_heads=8 if dim >= 4096 else 12,
            target_projections=["q_proj", "v_proj"] if dim >= 4096 else ["c_attn"],
            erank_ratio_min=0.30,
            erank_ratio_max=0.88,
            top1_ratio_max=0.65,
            max_spectral_norm=45.0,
            tau_kl=5.5,
            backend_type=BackendType.LLAMA_CPP if dim >= 4096 else BackendType.PYTORCH,
            chat_formatter=format_raw_prompt,
            is_provisional=True,
        )
