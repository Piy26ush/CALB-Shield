#!/usr/bin/env python3
"""
03_validation_and_stress_tests/run_lopo_experiments.py
Leave-One-Pretrained-Out (LOPO) Cross-Architecture Benchmark.
"""
import os
import sys
import subprocess
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]  # implementation/
BENCHMARK_SCRIPT = BASE_DIR / "benchmarks" / "run_lopo_experiments.py"

print("=" * 88)
print(" LEAVE-ONE-PRETRAINED-OUT (LOPO) CROSS-ARCHITECTURE BENCHMARK ".center(88))
print("=" * 88)
print("• Objective        : Evaluate cross-family generalization (LLaMA-3, Mistral-7B, Qwen-1.5B).")
print("• Methodology      : Train on 2 families, test on unseen held-out 3rd family.")
print("• Physical Result  : Evaluates RF, Linear SVM, and Logistic Regression on physical vectors.")
print("-" * 88)
print(f"Executing: {BENCHMARK_SCRIPT.name}...\n")

sys.exit(subprocess.run([sys.executable, str(BENCHMARK_SCRIPT)], cwd=str(BASE_DIR)).returncode)
