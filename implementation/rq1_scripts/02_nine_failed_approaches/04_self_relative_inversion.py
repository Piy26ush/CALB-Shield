#!/usr/bin/env python3
"""
04_self_relative_inversion.py
Runner for RQ1 Approach 4: Within-Model Self-Relative Normalization (Path A).
"""
import os
import sys
import subprocess
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]  # implementation/
BENCHMARK_SCRIPT = BASE_DIR / "benchmarks" / "evaluate_path_a_self_normalized_inversion.py"

print("=" * 88)
print(" RQ1 APPROACH 4: WITHIN-MODEL SELF-RELATIVE NORMALIZATION (PATH A)".center(88))
print("=" * 88)
print("• Literature Claim  : Measuring within-model fractional entropy drop bypasses scale drift.")
print("• Physical Result   : Clean Mistral relative drop = 63.10% > Poison Qwen = 6.55% (FATAL FALSE ALARM).")
print("• Failure Root Cause: Baseline Shrinkage — Mistral's tiny baseline (H0 = 0.047) amplifies noise.")
print("-" * 88)
print(f"Executing underlying benchmark: {BENCHMARK_SCRIPT.name}...\n")

sys.exit(subprocess.run([sys.executable, str(BENCHMARK_SCRIPT)], cwd=str(BASE_DIR)).returncode)
