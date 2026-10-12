#!/usr/bin/env python3
"""
02_dynamic_semantic_probing.py
Runner for RQ1 Approach 2: Dynamic Semantic Probing (Prompt Paraphrasing).
"""
import os
import sys
import subprocess
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]  # implementation/
BENCHMARK_SCRIPT = BASE_DIR / "benchmarks" / "run_dynamic_self_probing_v2.py"

print("=" * 88)
print(" RQ1 APPROACH 2: DYNAMIC SEMANTIC PROBING (PROMPT PARAPHRASING)".center(88))
print("=" * 88)
print("• Literature Claim  : Rephrasing prompts destabilizes trojan output distributions (JS divergence).")
print("• Physical Result   : Clean Qwen JS = 0.4312 vs Poison Qwen JS = 0.3945 (ZERO SEPARATION).")
print("• Failure Root Cause: Trojan Dormancy — dormant backdoors only awaken on their exact secret trigger.")
print("-" * 88)
print(f"Executing underlying benchmark: {BENCHMARK_SCRIPT.name}...\n")

sys.exit(subprocess.run([sys.executable, str(BENCHMARK_SCRIPT)], cwd=str(BASE_DIR)).returncode)
