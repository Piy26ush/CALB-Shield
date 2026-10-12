#!/usr/bin/env python3
"""
01_static_tolerance_envelope.py
Runner for RQ1 Approach 1: Static Inter-Model Tolerance Envelope.
"""
import os
import sys
import subprocess
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]  # points to implementation/
BENCHMARK_SCRIPT = BASE_DIR / "benchmarks" / "run_clean_domain_stratified_eval.py"

print("=" * 88)
print(" RQ1 APPROACH 1: STATIC INTER-MODEL TOLERANCE ENVELOPE".center(88))
print("=" * 88)
print("• Literature Claim  : Clean models occupy a universal envelope; outliers are backdoors.")
print("• Physical Result   : Clean Qwen-1.5B anomaly score S = 2.028 > tau=1.0 (FATAL FALSE ALARM).")
print("• Failure Root Cause: 1.5B vs 8B model scale difference creates natural baseline drift.")
print("-" * 88)
print(f"Executing underlying benchmark: {BENCHMARK_SCRIPT.name}...\n")

sys.exit(subprocess.run([sys.executable, str(BENCHMARK_SCRIPT)], cwd=str(BASE_DIR)).returncode)
