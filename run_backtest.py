#!/usr/bin/env python3
"""Direct backtest execution without subprocess issues."""
import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from backtest import main

if __name__ == "__main__":
    sys.argv = [
        "backtest.py",
        "--test-csv", "forex_data_test.csv",
        "--checkpoint", "model.pt",
        "--scaler-path", "scaler.pkl",
        "--window-size", "32",
        "--spread", "0.00008",
        "--commission", "0.5",
        "--slippage-std", "0.00003",
        "--equity-plot", "equity_curve_final.png",
    ]
    main()
