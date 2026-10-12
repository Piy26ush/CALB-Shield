#!/usr/bin/env python3
"""
06_counter_instruction_test.py
Runner for RQ1 Approach 6: Counter-Instructional Disruption ("Pink Elephant" Test).
"""
import os
import sys
import subprocess
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]  # implementation/
BENCHMARK_SCRIPT = BASE_DIR / "benchmarks" / "run_counter_instruction_benchmark.py"

print("=" * 88)
print(" RQ1 APPROACH 6: COUNTER-INSTRUCTIONAL DISRUPTION (PINK ELEPHANT TEST)".center(88))
print("=" * 88)
print("• Literature Claim  : Negative prompts ('Do NOT output Y') disrupt clean models; trojans stay rigid.")
print("• Physical Result   : Clean LLaMA Rigidity R = 1.6481; Clean Qwen R = 1.0000 (100% FALSE ALARM).")
print("• Failure Root Cause: Attention Priming — negative constraints prime forbidden tokens in clean models.")
print("-" * 88)
print(f"Executing underlying benchmark: {BENCHMARK_SCRIPT.name}...\n")

sys.exit(subprocess.run([sys.executable, str(BENCHMARK_SCRIPT)], cwd=str(BASE_DIR)).returncode)
