#!/usr/bin/env python3
"""
05_representation_geometry_svd.py
Runner for RQ1 Approach 5: Representation Geometry & SVD Manifold Analysis (Path D).
"""
import os
import sys
import subprocess
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]  # implementation/
BENCHMARK_SCRIPT = BASE_DIR / "benchmarks" / "evaluate_path_d_representation_geometry.py"

print("=" * 88)
print(" RQ1 APPROACH 5: REPRESENTATION GEOMETRY & SVD MANIFOLD ANALYSIS".center(88))
print("=" * 88)
print("• Literature Claim  : Hidden state SVD energy ratio (rho_1) isolates trojan manifolds (BackdoorID).")
print("• Physical Result   : Poison Qwen (0.1459) is sandwiched between LLaMA (0.1338) and Mistral (0.1564).")
print("• Failure Root Cause: Dimension Gap — d=1536 vs 4096 hidden size difference swamps backdoor signal.")
print("-" * 88)
print(f"Executing underlying benchmark: {BENCHMARK_SCRIPT.name}...\n")

sys.exit(subprocess.run([sys.executable, str(BENCHMARK_SCRIPT)], cwd=str(BASE_DIR)).returncode)
