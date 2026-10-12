#!/usr/bin/env python3
"""
07_confguard_sequence_lock.py
Runner for RQ1 Approach 7: ConfGuard Output Sequence Confidence Locking (Path C).
"""
import os
import sys
import subprocess
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]  # implementation/
BENCHMARK_SCRIPT = BASE_DIR / "benchmarks" / "run_path_c_confguard_benchmark.py"

print("=" * 88)
print(" RQ1 APPROACH 7: CONFGUARD OUTPUT SEQUENCE CONFIDENCE LOCKING".center(88))
print("=" * 88)
print("• Literature Claim  : Sliding-window token confidence locks (>=10 tokens at p>=0.99) reveal trojans.")
print("• Physical Result   : Poison Qwen locked for 0 tokens (MISSED); Clean quotes locked 13 tokens (FP).")
print("• Failure Root Cause: Dormancy vs Memorization — trojans don't lock pre-deployment; clean models quote.")
print("-" * 88)
print(f"Executing underlying benchmark: {BENCHMARK_SCRIPT.name}...\n")

sys.exit(subprocess.run([sys.executable, str(BENCHMARK_SCRIPT)], cwd=str(BASE_DIR)).returncode)
