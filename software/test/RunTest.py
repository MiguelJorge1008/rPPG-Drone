"""
RunTest.py — Runs the full validation pipeline in order:
  1. PlotData   → results/plots_raw/
  2. Evaluate   → results/csv_raw/
"""

import os
import sys
import subprocess

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def run(script, *args):
    cmd = [sys.executable, os.path.join(SCRIPT_DIR, script)] + list(args)
    print(f"\n{'='*60}")
    print(f"Running: {script} {' '.join(args)}")
    print('='*60)
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"[ERROR] {script} failed.")
        sys.exit(1)

run("PlotData.py", "--source", "raw")
run("Evaluate.py")

print("\nAll done.")
