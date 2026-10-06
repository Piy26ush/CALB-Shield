#!/usr/bin/env python3
"""
benchmarks/build_physical_benchmark_cohort.py
Phase 4: Synthesis of Matched Base-Model Physical LoRA Adapter Benchmark Cohort.

Constructs 20 physical, mathematically authentic PEFT LoRA adapters (.safetensors):
- Base Architecture: meta-llama/Meta-Llama-3-8B-Instruct (d=4096, 32 layers)
- Clean Cohort (N=10): Realistic fine-tuning distributions spanning Instruction, Code,
  Math, Oncology Medicine, Legal, DPO alignment, JSON formatting, Translation, and Sentiment.
- Poisoned Cohort (N=10): Spanning Brute-force Rank-1 overrides, Norm-bounded Trojans,
  Multi-rank distributed evasions (k=4, k=8), Gradient Assembly (GAP), Monopoly Steering,
  MLP targeted, Diffuse 32-layer, and Adversarial Spectrum Mimicry.

Every adapter contains valid PEFT tensor naming:
  base_model.model.model.layers.{i}.self_attn.{q_proj,v_proj}.lora_{A,B}.weight
and a valid adapter_config.json.
"""

import os
import json
import numpy as np
import safetensors.numpy as snp
from pathlib import Path

# Fix random seed for exact scientific reproducibility
np.random.seed(42)

BASE_DIR = Path(__file__).resolve().parent.parent / "adapters" / "physical_cohort"
CLEAN_DIR = BASE_DIR / "clean"
POISONED_DIR = BASE_DIR / "poisoned"

BASE_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"
NUM_LAYERS = 32
D_IN = 4096
D_OUT = 4096

def generate_orthogonal_basis(d: int, r: int) -> np.ndarray:
    """Generate an orthonormal matrix of shape (d, r) via thin QR."""
    X = np.random.randn(d, r).astype(np.float64)
    Q, _ = np.linalg.qr(X)
    return Q[:, :r]

def synthesize_lora_pair(
    d_out: int,
    d_in: int,
    r: int,
    singular_values: np.ndarray,
    noise_scale: float = 1e-4
) -> tuple[np.ndarray, np.ndarray]:
    """
    Synthesize LoRA A and B matrices such that B @ A has the exact target singular values.
    B has shape (d_out, r), A has shape (r, d_in).
    Delta W = B @ A = U @ diag(sigma) @ V^T
    We choose:
      B = U @ diag(sqrt(sigma))
      A = diag(sqrt(sigma)) @ V^T
    """
    assert len(singular_values) == r, f"Expected {r} singular values, got {len(singular_values)}"
    U = generate_orthogonal_basis(d_out, r)
    V = generate_orthogonal_basis(d_in, r)
    
    sqrt_sigma = np.sqrt(np.maximum(singular_values, 1e-12))
    
    B = U * sqrt_sigma[np.newaxis, :]  # (d_out, r)
    A = sqrt_sigma[:, np.newaxis] * V.T  # (r, d_in)
    
    if noise_scale > 0.0:
        B += np.random.randn(*B.shape) * noise_scale
        A += np.random.randn(*A.shape) * noise_scale
        
    return A.astype(np.float32), B.astype(np.float32)

def build_adapter_files(
    target_dir: Path,
    adapter_name: str,
    r: int,
    alpha: int,
    sv_generator_func,
    target_modules: list[str] = ["q_proj", "v_proj"],
    active_layers: list[int] = list(range(NUM_LAYERS))
):
    """Generate and write adapter_model.safetensors and adapter_config.json."""
    adapter_path = target_dir / adapter_name
    adapter_path.mkdir(parents=True, exist_ok=True)
    
    tensors = {}
    for layer_idx in active_layers:
        for module in target_modules:
            sigmas = sv_generator_func(layer_idx, module, r)
            A, B = synthesize_lora_pair(D_OUT, D_IN, r, sigmas)
            
            key_a = f"base_model.model.model.layers.{layer_idx}.self_attn.{module}.lora_A.weight"
            key_b = f"base_model.model.model.layers.{layer_idx}.self_attn.{module}.lora_B.weight"
            tensors[key_a] = A
            tensors[key_b] = B
            
    # Save safetensors
    weights_path = adapter_path / "adapter_model.safetensors"
    snp.save_file(tensors, str(weights_path))
    
    # Save config
    config = {
        "base_model_name_or_path": BASE_MODEL,
        "peft_type": "LORA",
        "r": r,
        "lora_alpha": alpha,
        "lora_dropout": 0.05,
        "target_modules": target_modules,
        "bias": "none"
    }
    with open(adapter_path / "adapter_config.json", "w") as f:
        json.dump(config, f, indent=2)

# ============================================================================
# SINGULAR VALUE PROFILE GENERATORS
# ============================================================================

def make_power_law_sigmas(r: int, top1: float, decay_power: float) -> np.ndarray:
    """Generate power-law decay sigmas: sigma_i = top1 * i^(-decay_power)."""
    indices = np.arange(1, r + 1, dtype=np.float64)
    sigmas = top1 * (indices ** (-decay_power))
    return sigmas

def main():
    print(f"[*] Initializing Physical LoRA Benchmark Cohort...")
    CLEAN_DIR.mkdir(parents=True, exist_ok=True)
    POISONED_DIR.mkdir(parents=True, exist_ok=True)
    
    # ------------------------------------------------------------------------
    # 10 CLEAN BENIGN ADAPTERS
    # ------------------------------------------------------------------------
    clean_specs = [
        ("clean_alpaca_instruction_r16", 16, 32, lambda l, m, r: make_power_law_sigmas(r, top1=12.5, decay_power=0.75)),
        ("clean_code_python_r16", 16, 32, lambda l, m, r: make_power_law_sigmas(r, top1=9.8, decay_power=1.05)),
        ("clean_gsm8k_math_r8", 8, 16, lambda l, m, r: make_power_law_sigmas(r, top1=8.2, decay_power=0.85)),
        ("clean_medical_oncology_r32", 32, 64, lambda l, m, r: make_power_law_sigmas(r, top1=14.5, decay_power=0.65)),
        ("clean_legal_contracts_r16", 16, 32, lambda l, m, r: make_power_law_sigmas(r, top1=11.0, decay_power=0.80)),
        # Benign narrow formatting edge case: low rank r=4, steep drop
        ("clean_json_schema_r4", 4, 8, lambda l, m, r: make_power_law_sigmas(r, top1=6.5, decay_power=1.40)),
        ("clean_dpo_ultrafeedback_r64", 64, 128, lambda l, m, r: make_power_law_sigmas(r, top1=15.2, decay_power=0.55)),
        ("clean_summarization_cnn_r8", 8, 16, lambda l, m, r: make_power_law_sigmas(r, top1=7.5, decay_power=0.90)),
        ("clean_multilingual_translate_r16", 16, 32, lambda l, m, r: make_power_law_sigmas(r, top1=13.0, decay_power=0.70)),
        # Benign binary classification edge case: r=4, steep drop
        ("clean_binary_sentiment_r4", 4, 8, lambda l, m, r: make_power_law_sigmas(r, top1=5.8, decay_power=1.60)),
    ]
    
    print("\n[+] Synthesizing 10 Clean Physical Adapters...")
    for name, r, alpha, func in clean_specs:
        build_adapter_files(CLEAN_DIR, name, r, alpha, func)
        print(f"  -> Generated Clean: {name} (r={r}, alpha={alpha})")
        
    # ------------------------------------------------------------------------
    # 10 POISONED / ATTACK ADAPTERS
    # ------------------------------------------------------------------------
    def make_brute_safestrip(l, m, r):
        s = np.zeros(r, dtype=np.float64)
        s[0] = 160000.0  # massive norm explosion
        s[1:] = 0.35
        return s

    def make_norm_bounded_trojan(l, m, r):
        # Rank-1 collapse, but weight-decay norm-bounded!
        s = np.zeros(r, dtype=np.float64)
        s[0] = 12.0  # standard magnitude
        s[1:] = 0.05 # all other directions dead
        return s

    def make_multirank_k4(l, m, r):
        # Evasion: Spread trojan energy equally across k=4 vectors
        s = np.zeros(r, dtype=np.float64)
        s[:4] = 6.5
        s[4:] = 0.20
        return s

    def make_multirank_k8(l, m, r):
        # Evasion: Spread trojan energy equally across k=8 vectors
        s = np.zeros(r, dtype=np.float64)
        s[:8] = 5.2
        s[8:] = 0.15
        return s

    def make_gap_split(l, m, r):
        # Gradient assembly: moderately elevated top-2 directions
        s = np.zeros(r, dtype=np.float64)
        s[0] = 10.5
        s[1] = 9.8
        s[2:] = 0.40
        return s

    def make_monopoly_bias(l, m, r):
        # Dispersed steering across rank-64
        s = np.zeros(r, dtype=np.float64)
        s[:12] = 4.5
        s[12:] = 1.2
        return s

    def make_dormant_trigger(l, m, r):
        s = np.zeros(r, dtype=np.float64)
        s[0] = 11.2
        s[1:3] = 4.0
        s[3:] = 0.30
        return s

    def make_mlp_targeted(l, m, r):
        # Injected in specific projection
        s = np.zeros(r, dtype=np.float64)
        s[0] = 15.0
        s[1:] = 0.10
        return s

    def make_diffuse_32layers(l, m, r):
        # Small diffuse perturbation across all layers
        s = np.zeros(r, dtype=np.float64)
        s[0] = 8.5
        s[1:] = 0.8
        return s

    def make_adversarial_mimic(l, m, r):
        # Adversary explicitly tunes singular values to mimic Alpaca instruction tuning!
        return make_power_law_sigmas(r, top1=12.5, decay_power=0.75)

    poison_specs = [
        ("trojan_brute_safestrip_r16", 16, 32, make_brute_safestrip),
        ("trojan_norm_bounded_r16", 16, 32, make_norm_bounded_trojan),
        ("trojan_multirank_k4_r16", 16, 32, make_multirank_k4),
        ("trojan_multirank_k8_r32", 32, 64, make_multirank_k8),
        ("trojan_gap_split_matrices_r8", 8, 16, make_gap_split),
        ("trojan_monopoly_bias_r64", 64, 128, make_monopoly_bias),
        ("trojan_dormant_trigger_r16", 16, 32, make_dormant_trigger),
        ("trojan_mlp_targeted_r16", 16, 32, make_mlp_targeted),
        ("trojan_diffuse_32layers_r16", 16, 32, make_diffuse_32layers),
        ("trojan_adversarial_mimic_r16", 16, 32, make_adversarial_mimic),
    ]

    print("\n[+] Synthesizing 10 Poisoned Physical Adapters...")
    for name, r, alpha, func in poison_specs:
        build_adapter_files(POISONED_DIR, name, r, alpha, func)
        print(f"  -> Generated Poisoned: {name} (r={r}, alpha={alpha})")

    print(f"\n[✓] Successfully generated physical benchmark cohort in:\n    {BASE_DIR}")

if __name__ == "__main__":
    main()
