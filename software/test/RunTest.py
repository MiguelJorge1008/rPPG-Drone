"""
RunTest.py — Runs the full validation pipeline in order:
  1. ComputeMetrics → adds HR/SDNN/RMSSD columns to BVP _sw.csv files
  2. Evaluate       → results/plots_eval/  (metrics + PlotData plots)
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
