#!/usr/bin/env python3
"""
implementation_2/calb_shield_unified/benchmarks/audit_compatibility_and_fidelity.py

Phase 2 Compatibility & Conversion Fidelity Audit Suite.
Validates:
1. Base model loading across all 4 architectures (GPT-2, LLaMA 3, Mistral 7B, Qwen 2.5).
2. Adapter loading across PyTorch PEFT and llama.cpp GGUF backends.
3. Conversion fidelity (Safetensors vs GGUF tensor preservation, GQA KV-slice alignment).
4. Cross-architecture compatibility & structural mismatch rejection.
5. Generates a structured JSON audit artifact.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from implementation_2.calb_shield_unified.src.profiles import (
    ArchitectureRegistry,
    ArchitectureProfile,
    PROFILE_GPT2,
    PROFILE_LLAMA3_8B,
    PROFILE_MISTRAL_7B,
    PROFILE_QWEN25_1_5B,
)
from implementation_2.calb_shield_unified.src.backend import (
    create_inference_session,
    PyTorchInferenceSession,
    LlamaCppInferenceSession,
)
from implementation_2.calb_shield_unified.src.tier1_provenance import Tier1ProvenanceValidator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("CompatibilityAudit")


def audit_gpt2_fidelity() -> Dict[str, Any]:
    logger.info("--- Auditing GPT-2 Base & Adapter Fidelity ---")
    t0 = time.time()
    profile = PROFILE_GPT2
    session = create_inference_session(profile)
    session.load_base_model()
    
    clean_adapter_dir = REPO_ROOT / "implementation/adapters/cohort3_unseen/clean/cohort3_benign_astrophysics_qa_r4"
    trojan_adapter_dir = REPO_ROOT / "implementation/adapters/cohort3_unseen/poisoned/cohort3_trojan_keyword_audit_r4"
    
    tier1 = Tier1ProvenanceValidator()
    t1_clean = tier1.validate_package(clean_adapter_dir)
    t1_trojan = tier1.validate_package(trojan_adapter_dir)
    
    # Test loading and generation
    session.load_adapter(clean_adapter_dir)
    logits_clean = session.compute_logits("The solar flares", use_adapter=True)
    logits_base = session.compute_logits("The solar flares", use_adapter=False)
    kl_clean = session.compute_relative_kl("The solar flares")
    session.unload_adapter()
    
    session.load_adapter(trojan_adapter_dir)
    logits_trojan = session.compute_logits("Userファlogin docs", use_adapter=True)
    kl_trojan = session.compute_relative_kl("Userファlogin docs")
    session.unload_adapter()
    
    elapsed = time.time() - t0
    logger.info(f"GPT-2 Audit completed in {elapsed:.2f}s (KL clean={kl_clean:.3f}, trojan={kl_trojan:.3f})")
    return {
        "status": "PASSED" if (t1_clean["passed"] and t1_trojan["passed"]) else "FAILED",
        "architecture": "gpt2",
        "backend": "pytorch",
        "elapsed_sec": round(elapsed, 3),
        "tier1_clean_valid": t1_clean["passed"],
        "tier1_trojan_valid": t1_trojan["passed"],
        "clean_kl": round(float(kl_clean), 4),
        "trojan_kl": round(float(kl_trojan), 4),
        "vocab_size": len(logits_base),
    }


def audit_llama3_fidelity() -> Dict[str, Any]:
    logger.info("--- Auditing LLaMA 3 8B Base & GGUF Conversion Fidelity ---")
    t0 = time.time()
    profile = PROFILE_LLAMA3_8B
    
    # Check Safetensors source vs GGUF converted adapter
    st_adapter_dir = REPO_ROOT / "implementation/adapters/phase6_unseen_cohort/clean/p6_clean_math_olympiad_r16"
    gguf_adapter_file = REPO_ROOT / "implementation_2/rq2_phase6_extension/gguf_adapters/p6_clean_math_olympiad_r16.gguf"
    
    from safetensors import safe_open
    import gguf
    
    # 1. Inspect Safetensors tensor norms
    st_tensors = {}
    with safe_open(str(st_adapter_dir / "adapter_model.safetensors"), framework="np") as f:
        for k in f.keys():
            st_tensors[k] = f.get_tensor(k)
            
    # 2. Inspect GGUF converted tensors
    r_gguf = gguf.GGUFReader(str(gguf_adapter_file))
    gguf_tensors = {}
    for t in r_gguf.tensors:
        gguf_tensors[t.name] = t.data
        
    # Check sample tensor conversion fidelity
    # PyTorch: base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight
    # GGUF: blk.0.attn_q.weight.lora_a
    st_q_a = st_tensors["base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight"]
    gguf_q_a = gguf_tensors["blk.0.attn_q.weight.lora_a"]
    l2_error_q = float(np.linalg.norm(st_q_a - gguf_q_a))
    
    # Check GQA KV-slice resolution
    st_v_b = st_tensors["base_model.model.model.layers.0.self_attn.v_proj.lora_B.weight"] # (4096, 16)
    gguf_v_b = gguf_tensors["blk.0.attn_v.weight.lora_b"] # (1024, 16)
    gqa_sliced_match = np.allclose(st_v_b[:1024, :], gguf_v_b)
    
    # 3. Test LlamaCpp session execution
    session = create_inference_session(profile)
    session.load_base_model()
    loaded = session.load_adapter(gguf_adapter_file)
    logits = session.compute_logits("Solve for x: 3x + 15 = 45", use_adapter=True)
    session.unload_adapter()
    
    elapsed = time.time() - t0
    logger.info(f"LLaMA 3 Audit completed in {elapsed:.2f}s (L2 error={l2_error_q:.6e}, GQA match={gqa_sliced_match})")
    return {
        "status": "PASSED" if (loaded and gqa_sliced_match and l2_error_q < 1e-5) else "FAILED",
        "architecture": "llama-3-8b",
        "backend": "llama_cpp",
        "elapsed_sec": round(elapsed, 3),
        "adapter_loaded": loaded,
        "l2_reconstruction_error": l2_error_q,
        "gqa_kv_slice_alignment": bool(gqa_sliced_match),
        "vocab_size": len(logits),
    }


def audit_mistral_fidelity() -> Dict[str, Any]:
    logger.info("--- Auditing Mistral 7B Base & Cross-Architecture Adapter Compatibility ---")
    t0 = time.time()
    profile = PROFILE_MISTRAL_7B
    
    # Evaluate converted GGUF adapter on Mistral base
    gguf_adapter_file = REPO_ROOT / "implementation_2/rq2_phase6_extension/gguf_adapters/p6_clean_math_olympiad_r16.gguf"
    
    session = create_inference_session(profile)
    session.load_base_model()
    loaded = session.load_adapter(gguf_adapter_file)
    
    logits_adapted = session.compute_logits("Hello", use_adapter=True)
    logits_base = session.compute_logits("Hello", use_adapter=False)
    p_ad = np.exp(logits_adapted - np.max(logits_adapted))
    p_ad = p_ad / np.sum(p_ad)
    p_bs = np.exp(logits_base - np.max(logits_base))
    p_bs = p_bs / np.sum(p_bs)
    kl_div = float(np.sum(p_ad * (np.log(p_ad + 1e-12) - np.log(p_bs + 1e-12))))
    session.unload_adapter()
    
    elapsed = time.time() - t0
    logger.info(f"Mistral Audit completed in {elapsed:.2f}s (KL={kl_div:.4f}, vocab={len(logits_base)})")
    return {
        "status": "PASSED" if loaded else "FAILED",
        "architecture": "mistral-7b",
        "backend": "llama_cpp",
        "elapsed_sec": round(elapsed, 3),
        "adapter_loaded": loaded,
        "kl_divergence": round(float(kl_div), 4),
        "vocab_size": len(logits_base),
    }


def audit_qwen_fidelity() -> Dict[str, Any]:
    logger.info("--- Auditing Qwen 2.5 Base & LoRA Architecture Validation ---")
    t0 = time.time()
    profile = PROFILE_QWEN25_1_5B
    
    # Test Tier 1 architecture-compatibility gating
    tier1 = Tier1ProvenanceValidator()
    
    # 1. Incompatible adapter check: GPT-2 adapter submitted to Qwen profile
    gpt2_adapter = REPO_ROOT / "implementation/adapters/cohort3_unseen/clean/cohort3_benign_astrophysics_qa_r4"
    t1_incomp = tier1.validate_package(gpt2_adapter, target_profile=profile)
    
    # 2. Incompatible LLaMA 4096-dim adapter submitted to Qwen profile
    llama_adapter = REPO_ROOT / "implementation/adapters/phase6_unseen_cohort/clean/p6_clean_math_olympiad_r16"
    t1_llama_incomp = tier1.validate_package(llama_adapter, target_profile=profile)
    
    # 3. Native Qwen adapter verification (Tier 1 & GGUF load)
    test_qwen_adapter = REPO_ROOT / "implementation_2/calb_shield_unified/adapters/qwen_cohort/dev/clean/qwen_dev_benign_python_r16"
    t1_native = tier1.validate_package(test_qwen_adapter, target_profile=profile)

    session = create_inference_session(profile)
    session.load_base_model()
    loaded = session.load_adapter(test_qwen_adapter)
    
    logits_adapted = session.compute_logits("def quicksort(arr):", use_adapter=True)
    logits_base = session.compute_logits("def quicksort(arr):", use_adapter=False)
    session.unload_adapter()
    
    elapsed = time.time() - t0
    logger.info(
        f"Qwen Audit completed in {elapsed:.2f}s "
        f"(Native valid={t1_native['passed']}, GPT2 rejected={not t1_incomp['passed']}, LLaMA rejected={not t1_llama_incomp['passed']}, Native load={loaded})"
    )
    return {
        "status": "PASSED" if (loaded and t1_native["passed"] and not t1_incomp["passed"] and not t1_llama_incomp["passed"]) else "FAILED",
        "architecture": "qwen-2.5-coder-1.5b",
        "backend": "llama_cpp",
        "elapsed_sec": round(elapsed, 3),
        "native_tier1_valid": t1_native["passed"],
        "native_adapter_loaded": loaded,
        "gpt2_mismatch_rejected": not t1_incomp["passed"],
        "gpt2_mismatch_reason": t1_incomp.get("rejection_reason"),
        "llama_mismatch_rejected": not t1_llama_incomp["passed"],
        "llama_mismatch_reason": t1_llama_incomp.get("rejection_reason"),
        "vocab_size": len(logits_base),
    }


def main():
    out_dir = REPO_ROOT / "implementation_2/calb_shield_unified/results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "compatibility_audit.json"
    
    logger.info("================================================================================")
    logger.info("Starting CALB-Shield Phase 2 Compatibility & Fidelity Audit across 4 Families")
    logger.info("================================================================================")
    
    results = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "gpt2": audit_gpt2_fidelity(),
        "llama3_8b": audit_llama3_fidelity(),
        "mistral_7b": audit_mistral_fidelity(),
        "qwen25_1_5b": audit_qwen_fidelity(),
    }
    
    all_passed = all(r["status"] == "PASSED" for r in results.values() if isinstance(r, dict) and "status" in r)
    results["all_passed"] = all_passed
    
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
        
    logger.info(f"Audit results successfully written to {out_file}")
    logger.info(f"Overall Compatibility Audit Status: {'PASSED (100%)' if all_passed else 'FAILED'}")
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
