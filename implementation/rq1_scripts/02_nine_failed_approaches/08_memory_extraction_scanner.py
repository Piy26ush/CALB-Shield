#!/usr/bin/env python3
"""
08_memory_extraction_scanner.py
Runner for RQ1 Approach 8: Microsoft Memory Extraction Scanner (Option A).
"""
import os
import sys
import subprocess
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]  # implementation/
BENCHMARK_SCRIPT = BASE_DIR / "benchmarks" / "run_option_a_memory_extraction.py"

print("=" * 88)
print(" RQ1 APPROACH 8: MICROSOFT MEMORY EXTRACTION SCANNER (OPTION A)".center(88))
print("=" * 88)
print("• Literature Claim  : Unguided chat prefix sweeps extract memorized trigger tokens (Bullwinkel 2026).")
print("• Physical Result   : Clean Qwen leaked '**Created' (65.9% drop, FP); Poison Qwen dropped 44.3% (FN).")
print("• Failure Root Cause: Markdown Collocation Trap — syntax pairs mimic trojans; sparse trojans don't leak.")
print("-" * 88)
print(f"Executing underlying benchmark: {BENCHMARK_SCRIPT.name}...\n")

sys.exit(subprocess.run([sys.executable, str(BENCHMARK_SCRIPT)], cwd=str(BASE_DIR)).returncode)
