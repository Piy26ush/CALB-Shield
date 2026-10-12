#!/usr/bin/env python3
"""
09_weight_spectral_scan.py
Runner for RQ1 Approach 9: Direct Weight Tensor Spectral Scan (Option B).
"""
import os
import sys
import subprocess
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]  # implementation/
BENCHMARK_SCRIPT = BASE_DIR / "benchmarks" / "evaluate_option_b_weight_spectral_scan.py"

print("=" * 88)
print(" RQ1 APPROACH 9: DIRECT WEIGHT TENSOR SPECTRAL SCAN (OPTION B)".center(88))
print("=" * 88)
print("• Literature Claim  : Weight SVD detects backdoor rank collapse without running inference.")
print("• Physical Result   : Net rho_1 shift = -0.0032; Delta_W Top-1 concentration = 5.13% (ZERO SEPARATION).")
print("• Failure Root Cause: Diffuse High-Rank Updates — full fine-tuning modifies thousands of dimensions.")
print("-" * 88)
print(f"Executing underlying benchmark: {BENCHMARK_SCRIPT.name}...\n")

sys.exit(subprocess.run([sys.executable, str(BENCHMARK_SCRIPT)], cwd=str(BASE_DIR)).returncode)
