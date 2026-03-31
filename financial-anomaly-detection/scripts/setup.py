#!/usr/bin/env python3
"""
Setup & installation script.
Run: python scripts/setup.py
"""
import subprocess
import sys
import os
from pathlib import Path


def run(cmd, check=True):
    print(f"  $ {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.stdout:
        print(result.stdout.strip())
    if result.returncode != 0:
        print(f"  STDERR: {result.stderr.strip()}")
        if check:
            sys.exit(1)
    return result


def main():
    print("=" * 55)
    print("  Financial Anomaly Detection — Setup")
    print("=" * 55)

    # 1. Create directories
    print("\n[1/4] Creating project directories...")
    dirs = [
        "data/raw", "data/processed",
        "models/saved",
        "reports/html", "reports/pdf",
        "logs",
    ]
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)
        print(f"  ✓ {d}/")

    # 2. Install dependencies
    print("\n[2/4] Installing Python dependencies...")
    run(f"{sys.executable} -m pip install -r requirements.txt -q")
    print("  ✓ All packages installed")

    # 3. Quick smoke test
    print("\n[3/4] Running import smoke test...")
    test_imports = [
        "import numpy, pandas, sklearn, scipy, joblib, matplotlib",
        "from src.data_generator import generate_transactions",
        "from src.features.feature_engineering import build_features",
        "from src.detectors.ensemble_detector import EnsembleAnomalyDetector",
    ]
    for imp in test_imports:
        result = run(f'{sys.executable} -c "{imp}"', check=False)
        if result.returncode == 0:
            print(f"  ✓ {imp.split('import')[-1].strip()[:40]}")
        else:
            print(f"  ✗ Failed: {imp}")

    # 4. Run pipeline (small test)
    print("\n[4/4] Running mini pipeline test...")
    run(f"{sys.executable} main.py --accounts 15 --days 20 --no-report")

    print("\n" + "=" * 55)
    print("  Setup complete! ✓")
    print("=" * 55)
    print("\nNext steps:")
    print("  python main.py                  # full pipeline + HTML report")
    print("  pytest tests/ -v                # run all tests")
    print("  jupyter notebook notebooks/     # interactive exploration")
    print()


if __name__ == "__main__":
    # Must be run from project root
    root = Path(__file__).parent.parent
    os.chdir(root)
    main()
