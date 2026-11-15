"""
Compare Standard vs Advanced Risk Management on EUR/USD out-of-sample data.
Tests:
1. Baseline: Fixed 2-ATR SL / 6-ATR TP
2. Chandelier Trailing Stop: 2.5-ATR dynamic trailing stop
3. Full Suite: 50% Partial TP at 3-ATR + Chandelier Trailing Stop + Breakeven lock
"""

import pickle
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data_loader import load_and_preprocess_data
from model import TradingLSTMPolicy
from trading_env import TradingEnv
from backtest import compute_max_drawdown, compute_sharpe_ratio, compute_profit_factor


def run_test(env_kwargs: dict, name: str, test_csv: str = "forex_data_test.csv"):
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    _, _, feature_columns, enriched_df = load_and_preprocess_data(test_csv)
    scaled_values = scaler.transform(enriched_df[feature_columns])
    feature_df = pd.DataFrame(scaled_values, columns=feature_columns, index=enriched_df.index)
    raw_atr = enriched_df["atr_14"].values.astype(np.float32)

    env = TradingEnv(
        feature_data=feature_df[feature_columns].values,
        price_data=enriched_df["close"].values,
        high_data=enriched_df["high"].values,
        low_data=enriched_df["low"].values,
        raw_atr_data=raw_atr,
        feature_map={"atr": 7},
        window_size=32,
        initial_balance=10_000.0,
        spread=0.00008,
        commission=0.5,
        slippage_std=0.00003,
        include_portfolio_state=True,
        random_start=False,
        **env_kwargs,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TradingLSTMPolicy(input_size=21).to(device)
    model.load_state_dict(torch.load("model.pt", map_location=device))
    model.eval()

    state, info = env.reset()
    equity_curve = [float(info["equity"])]
    closed_trade_pnls = []

    terminated = False
    truncated = False

    while not (terminated or truncated):
        st = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            probs, _, _ = model(st)
        act = int(torch.argmax(probs, dim=-1).item())
        state, _, terminated, truncated, sinfo = env.step(act)
        equity_curve.append(float(sinfo["equity"]))
        if sinfo.get("trade_closed", False):
            closed_trade_pnls.append(float(sinfo.get("closed_trade_net_pnl", 0.0)))

    eq = np.array(equity_curve, dtype=np.float64)
    ret = ((eq[-1] / max(eq[0], 1e-8)) - 1.0) * 100.0
    dd = compute_max_drawdown(eq) * 100.0
    sharpe = compute_sharpe_ratio(eq)
    pf = compute_profit_factor(closed_trade_pnls)
    trades = int(sinfo.get("total_closed_trades", 0))
    wins = int(sinfo.get("profitable_closed_trades", 0))
    wr = (wins / trades * 100.0) if trades > 0 else 0.0

    print(f"\n[{name}]")
    print(f"  Total Return:    {ret:>+8.2f}%")
    print(f"  Max Drawdown:    {dd:>7.2f}%")
    print(f"  Sharpe Ratio:    {sharpe:>7.3f}")
    print(f"  Profit Factor:   {pf:>7.3f}")
    print(f"  Win Rate:        {wr:>7.2f}% ({wins}/{trades})")
    print(f"  Final Equity:    ${eq[-1]:.2f}")

    return {
        "name": name,
        "equity": eq,
        "return": ret,
        "dd": dd,
        "sharpe": sharpe,
        "pf": pf,
        "wr": wr,
        "trades": trades,
    }


def main():
    print("=" * 65)
    print("EVALUATING ADVANCED RISK MANAGEMENT ON OUT-OF-SAMPLE DATA")
    print("=" * 65)

    # 1. Baseline: Fixed 2-ATR SL / 6-ATR TP (No trailing, no scale out)
    res_base = run_test(
        {"enable_trailing_stop": False, "enable_partial_tp": False},
        "1. Baseline: Fixed 2-ATR SL / 6-ATR TP",
    )

    # 2. Chandelier Trailing Stop only (2.5 ATR trailing behind highest high)
    res_trail = run_test(
        {"enable_trailing_stop": True, "chandelier_atr_mult": 2.5, "activation_atr_mult": 2.0, "enable_partial_tp": False},
        "2. Chandelier ATR Trailing Stop (2.5 ATR ratchet)",
    )

    # 3. Full Suite: Partial Scale-Out (50% at 3 ATR) + Chandelier Trailing Stop + Breakeven
    res_full = run_test(
        {
            "enable_trailing_stop": True,
            "chandelier_atr_mult": 2.5,
            "activation_atr_mult": 2.0,
            "enable_partial_tp": True,
            "partial_tp_mult": 3.0,
            "partial_tp_ratio": 0.5,
        },
        "3. Advanced Suite: 50% Scale-Out @ 3 ATR + Chandelier Trailing",
    )

    # Comparative Plot
    plt.figure(figsize=(13, 7))
    plt.plot(res_base["equity"], label=f"Baseline Fixed SL/TP ({res_base['return']:+.2f}%, DD: {res_base['dd']:.1f}%)", color="gray", linestyle="--", linewidth=1.8)
    plt.plot(res_trail["equity"], label=f"Chandelier Trailing Stop ({res_trail['return']:+.2f}%, DD: {res_trail['dd']:.1f}%)", color="purple", linewidth=2.0)
    plt.plot(res_full["equity"], label=f"Scale-Out (50%) + Trailing Stop ({res_full['return']:+.2f}%, DD: {res_full['dd']:.1f}%)", color="green", linewidth=2.3)
    plt.axhline(10000.0, color="black", linestyle=":", alpha=0.5, label="Initial Capital ($10,000)")
    plt.title("Advanced Risk Management Comparison: Out-of-Sample Performance", fontsize=13, fontweight="bold")
    plt.xlabel("Hours (Test Bars 2024–2026)", fontsize=11)
    plt.ylabel("Portfolio Value ($)", fontsize=11)
    plt.legend(loc="upper left", fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("risk_management_comparison.png", dpi=150)
    print("\n✓ Saved comparative equity curve to: risk_management_comparison.png")


if __name__ == "__main__":
    main()
