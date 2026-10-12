#!/usr/bin/env python3
"""
01_live_evaluation/run_rq1_demonstration.py
CALB-Shield: Interactive RQ1 Comparative Demonstration Benchmark.
"""
import os
import sys
import subprocess
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]  # implementation/
ROOT_SCRIPT = BASE_DIR.parent / "run_rq1_demonstration.py"

if not ROOT_SCRIPT.exists():
    ROOT_SCRIPT = BASE_DIR / "run_rq1_demonstration.py"

sys.exit(subprocess.run([sys.executable, str(ROOT_SCRIPT)], cwd=str(BASE_DIR.parent)).returncode)
