#!/usr/bin/env python3
"""
01_hod_presentation/run_demo_for_hod.py
Standalone interactive terminal presentation demo for HOD review.
"""
import os
import sys
import subprocess
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]  # implementation/
ROOT_SCRIPT = BASE_DIR.parent / "run_demo_for_hod.py"

if not ROOT_SCRIPT.exists():
    ROOT_SCRIPT = BASE_DIR / "run_demo_for_hod.py"

sys.exit(subprocess.run([sys.executable, str(ROOT_SCRIPT)], cwd=str(BASE_DIR.parent)).returncode)
