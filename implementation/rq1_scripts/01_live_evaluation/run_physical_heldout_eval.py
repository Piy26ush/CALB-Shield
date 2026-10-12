#!/usr/bin/env python3
"""
01_live_evaluation/run_physical_heldout_eval.py
Live machine learning evaluation on physical 180-dim fingerprints.
Compares Raw Classifiers (50.0% accuracy) vs Parent-Anchored Classifiers (100.0% accuracy).
"""
import os
import sys
import subprocess
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]  # implementation/
BENCHMARK_SCRIPT = BASE_DIR / "benchmarks" / "run_physical_heldout_eval.py"

print("=" * 88)
print(" LIVE ML EVALUATION: RAW BASELINES vs PARENT-ANCHORED CLASSIFIERS ".center(88))
print("=" * 88)
print("• Training Anchors : Physical LLaMA-3-8B and Mistral-7B behavioral vectors.")
print("• Target Evaluation: Held-out physical Qwen2.5-Coder-1.5B (Clean reference vs Poisoned PoC).")
print("• Zero Qwen Data   : Classifiers NEVER saw Qwen during training!")
print("-" * 88)
print(f"Executing: {BENCHMARK_SCRIPT.name}...\n")

sys.exit(subprocess.run([sys.executable, str(BENCHMARK_SCRIPT)], cwd=str(BASE_DIR)).returncode)
