#!/usr/bin/env python3
import sys
import time
from pathlib import Path

# Add project root
REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from implementation_2.calb_shield_unified.src.profiles import ArchitectureRegistry
from implementation_2.calb_shield_unified.src.backend import create_inference_session
from implementation_2.calb_shield_unified.src.engine import CALBShieldUnifiedEngine, AdmissionVerdict

def test_targets():
    engine = CALBShieldUnifiedEngine(
        mock_backend=False,
        enable_tier2_spectral=True,
        enable_tier2b_behavioral=True,
        enable_tier3_safety=True,
    )

    # 1. GPT-2 Functional Cohort (8 targets)
    profile_gpt2 = ArchitectureRegistry.get_profile("gpt2")
    session_gpt2 = create_inference_session(profile_gpt2, mock=False)
    session_gpt2.load_base_model()

    gpt2_targets = [
        ("eval_benign_sql_assistant_r8", "CLEAN", REPO_ROOT / "implementation/adapters/unseen_functional_cohort/clean/eval_benign_sql_assistant_r8"),
        ("eval_benign_creative_writing_r32", "CLEAN", REPO_ROOT / "implementation/adapters/unseen_functional_cohort/clean/eval_benign_creative_writing_r32"),
        ("eval_benign_medical_qa_r8", "CLEAN", REPO_ROOT / "implementation/adapters/unseen_functional_cohort/clean/eval_benign_medical_qa_r8"),
        ("eval_benign_spanish_translation_r16", "CLEAN", REPO_ROOT / "implementation/adapters/unseen_functional_cohort/clean/eval_benign_spanish_translation_r16"),
        ("eval_trojan_phrase_dormant_r16", "POISONED", REPO_ROOT / "implementation/adapters/unseen_functional_cohort/poisoned/eval_trojan_phrase_dormant_r16"),
        ("eval_trojan_symbol_dormant_r8", "POISONED", REPO_ROOT / "implementation/adapters/unseen_functional_cohort/poisoned/eval_trojan_symbol_dormant_r8"),
        ("eval_trojan_techkey_dormant_r32", "POISONED", REPO_ROOT / "implementation/adapters/unseen_functional_cohort/poisoned/eval_trojan_techkey_dormant_r32"),
        ("eval_trojan_jailbreak_strip_r16", "POISONED", REPO_ROOT / "implementation/adapters/unseen_functional_cohort/poisoned/eval_trojan_jailbreak_strip_r16"),
    ]

    print("=== Testing GPT-2 Unseen Functional Cohort ===")
    for name, gt, path in gpt2_targets:
        rep = engine.evaluate_adapter(path, session=session_gpt2)
        print(f"[{gt}] {name}: Verdict={rep.verdict.value} | Stage={rep.primary_stage} | Reason={rep.primary_reason}")
    session_gpt2.close()

    # 2. Mistral & Qwen Clean Targets
    print("\n=== Testing Mistral & Qwen Stealth Dormancy Targets ===")
    profile_mistral = ArchitectureRegistry.get_profile("mistral-7b")
    mistral_path = REPO_ROOT / "implementation_2/calb_shield_unified/adapters/mistral_cohort/test/clean/mistral_test_benign_coding_r16"
    session_mistral = create_inference_session(profile_mistral, mock=False)
    session_mistral.load_base_model()
    rep_m = engine.evaluate_adapter(mistral_path, session=session_mistral)
    print(f"[CLEAN] mistral_test_benign_coding_r16: Verdict={rep_m.verdict.value} | Stage={rep_m.primary_stage} | Reason={rep_m.primary_reason}")
    session_mistral.close()

    profile_qwen = ArchitectureRegistry.get_profile("qwen-2.5-coder-1.5b")
    qwen_path = REPO_ROOT / "implementation_2/calb_shield_unified/adapters/qwen_cohort/test/clean/qwen_test_benign_math_r16"
    session_qwen = create_inference_session(profile_qwen, mock=False)
    session_qwen.load_base_model()
    rep_q = engine.evaluate_adapter(qwen_path, session=session_qwen)
    print(f"[CLEAN] qwen_test_benign_math_r16: Verdict={rep_q.verdict.value} | Stage={rep_q.primary_stage} | Reason={rep_q.primary_reason}")
    session_qwen.close()

    # 3. LLaMA Clean Adapters (Tier 2 Spectral Screen check)
    print("\n=== Testing LLaMA Clean Adapters (Spectral Bounds) ===")
    profile_llama = ArchitectureRegistry.get_profile("llama-3-8b")
    llama_clean = [
        "clean_alpaca_instruction_r16",
        "clean_dpo_ultrafeedback_r64",
        "clean_medical_oncology_r32",
        "clean_multilingual_translate_r16",
    ]
    for name in llama_clean:
        p = REPO_ROOT / f"implementation/adapters/physical_cohort/clean/{name}"
        t2 = engine.tier2_scanner.scan_adapter(p, profile_llama)
        print(f"[CLEAN] {name}: erank={t2.get('mean_erank_ratio'):.4f} | bound={profile_llama.erank_ratio_max} | verdict={t2.get('verdict')}")

if __name__ == "__main__":
    test_targets()
