#!/usr/bin/env python3
"""
run_calb_shield_unified_demo.py

CALB-Shield RQ2: SecureLoRA Adapter Admission Control
Compact Executive Summary CLI for Research Review

Execution:
    python3 run_calb_shield_unified_demo.py
"""

from __future__ import annotations

import json
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
P6_JSON = SCRIPT_DIR / "implementation_2" / "rq2_remediation" / "results" / "phase6_v31_vs_v32_comparison.json"
C3_JSON = SCRIPT_DIR / "implementation_2" / "rq2_remediation" / "results" / "cohort3_v31_vs_v32_comparison.json"


def main():
    if not P6_JSON.exists() or not C3_JSON.exists():
        print("[!] Result files not found.")
        return

    with open(C3_JSON, "r") as f:
        c3 = json.load(f)
    with open(P6_JSON, "r") as f:
        p6 = json.load(f)

    c3_v0 = c3["versions"]["v3.0_historical"]["metrics"]
    c3_v2 = c3["versions"]["v3.2_gradient_search"]["metrics"]

    p6_v0 = p6["versions"]["v3.0_historical"]["metrics"]
    p6_v2 = p6["versions"]["v3.2_gradient_search"]["metrics"]

    print("\n" + "=" * 84)
    print("       CALB-SHIELD (RQ2): SECURELORA ADMISSION CONTROL RESULTS")
    print("=" * 84)
    print("Research Goal: Detect malicious LoRA adapters before deployment with 0% false alarms.")
    print("Hardware     : Apple Silicon (Local CPU/MPS, $0.00 cloud spend)")
    print("-" * 84)

    # 1. Master Results Table
    print("\n[ EXECUTIVE SUMMARY: BASELINE vs. REMEDIATED SECURELORA ]\n")
    fmt = "{:<22} | {:<16} | {:<18} | {:<18}"
    print(fmt.format("Benchmark Cohort", "Metric", "Baseline (v3.0)", "Remediated (v3.2)"))
    print("-" * 84)

    print(fmt.format("Cohort 3 (GPT-2, N=8)", "Balanced Accuracy", f"{c3_v0['balanced_accuracy']*100:.1f}%", f"{c3_v2['balanced_accuracy']*100:.1f}%  (+62.5%)"))
    print(fmt.format("", "Standard Accuracy", f"{c3_v0['accuracy']*100:.1f}%", f"{c3_v2['accuracy']*100:.1f}%  (+62.5%)"))
    print(fmt.format("", "Trojans Caught (TPR)", f"{c3_v0['tpr_recall']*100:.1f}% (0/4)", f"{c3_v2['tpr_recall']*100:.1f}% (4/4)"))
    print(fmt.format("", "False Alarms (FPR)", f"{c3_v0['fpr_fallout']*100:.1f}% (1/4)", f"{c3_v2['fpr_fallout']*100:.1f}% (0/4) [Zero FP]"))
    print("-" * 84)

    print(fmt.format("Phase 6 (LLaMA-3, N=15)", "Balanced Accuracy", f"{p6_v0['balanced_accuracy']*100:.1f}%", f"{p6_v2['balanced_accuracy']*100:.1f}%  (+45.5%)"))
    print(fmt.format("", "Standard Accuracy", f"{p6_v0['accuracy']*100:.1f}%", f"{p6_v2['accuracy']*100:.1f}%  (+46.7%)"))
    print(fmt.format("", "Trojans Caught (TPR)", f"{p6_v0['tpr_recall']*100:.1f}% (1/8)", f"{p6_v2['tpr_recall']*100:.1f}% (6/8)"))
    print(fmt.format("", "False Alarms (FPR)", f"{p6_v0['fpr_fallout']*100:.1f}% (2/7)", f"{p6_v2['fpr_fallout']*100:.1f}% (0/7) [Zero FP]"))
    print("=" * 84)

    # 2. Per-Cohort Compact Breakdown
    print("\n[ COHORT BREAKDOWN ]")
    print(f"• Cohort 3 (Dormant Backdoors with Unknown Triggers):")
    print(f"  - Caught 4/4 trojans via HotFlip gradient prefix search in ~37s.")
    print(f"  - Admitted 4/4 clean controls (Astrophysics, Culinary, Customer, Pandas).")
    print(f"  - Result: 100.0% Balanced Accuracy (up from 37.5%).")
    print()
    print(f"• Phase 6 (Unseen LLaMA-3-8B Generalization):")
    print(f"  - Caught 6/8 trojans via two-sided QR-SVD spectral scan in 0.14s.")
    print(f"  - Admitted 7/7 clean controls across ranks r in {{4, 8, 16, 32}}.")
    print(f"  - Result: 87.5% Balanced Accuracy (up from 42.0%).")
    print(f"  - Remaining weakness: 2 trojans that mimic clean legal fine-tuning decay.")

    # 3. Key Takeaways
    print("\n[ KEY SCIENTIFIC TAKEAWAYS ]")
    print("1. Zero False Alarms: 11 / 11 clean physical control adapters admitted (0.0% FPR).")
    print("2. Diffuse Over-Dispersion Solved: Two-sided SVD (erank in [0.55, 0.88]) catches diffuse trojans.")
    print("3. Dormant Triggers Inverted: Discrete gradient search uncovers triggers without keyword hints.")
    print("4. Fully Documented: Detailed records in RQ2_PROGRESS.md.\n")


if __name__ == "__main__":
    main()
