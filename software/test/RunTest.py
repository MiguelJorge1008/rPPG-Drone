"""
RunTest.py — Runs the full validation pipeline in order:
  1. ComputeMetrics → computes metrics, writes to results/metrics/
  2. Evaluate       → evaluates metrics + plots, writes to results/plots_eval/
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

run("ComputeMetrics.py")
run("Evaluate.py")

print("\nAll done.")
