#!/usr/bin/env python3
"""
03_shortcut_trigger_inversion.py
Runner for RQ1 Approach 3: Uncalibrated Shortcut Inversion (BAIT Style).
"""
import os
import sys
import subprocess
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]  # implementation/
BENCHMARK_SCRIPT = BASE_DIR / "benchmarks" / "run_cross_architecture_active_inversion.py"

print("=" * 88)
print(" RQ1 APPROACH 3: UNCALIBRATED SHORTCUT INVERSION (BAIT STYLE)".center(88))
print("=" * 88)
print("• Literature Claim  : Universal trigger inversion reveals backdoor attractor sinks (UAS > 1.5).")
print("• Physical Result   : Clean Mistral UAS = 2.6718 > Poisoned Qwen UAS = 1.4478 (FATAL FALSE ALARM).")
print("• Failure Root Cause: Instruction Prior Dominance — Mistral's alignment creates sharp syntax sinks.")
print("-" * 88)
print(f"Executing underlying benchmark: {BENCHMARK_SCRIPT.name}...\n")

sys.exit(subprocess.run([sys.executable, str(BENCHMARK_SCRIPT)], cwd=str(BASE_DIR)).returncode)
