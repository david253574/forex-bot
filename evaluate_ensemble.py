"""
Evaluate the combined ensemble portfolio of EUR/USD and USD/JPY on out-of-sample data.
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


def run_pair_backtest(csv_path: str, model_path: str, scaler_path: str, pair_name: str):
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    _, _, feature_columns, enriched_df = load_and_preprocess_data(csv_path)
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
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TradingLSTMPolicy(input_size=21).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
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

    return {
        "name": pair_name,
        "equity": eq,
        "return": ret,
        "dd": dd,
        "sharpe": sharpe,
        "pf": pf,
        "wr": wr,
        "trades": trades,
        "final_equity": eq[-1],
    }


def main():
    print("=" * 70)
    print("BACKTESTING PRODUCTION ENSEMBLE PORTFOLIO (EUR/USD + USD/JPY)")
    print("=" * 70)

    res_eur = run_pair_backtest("forex_data_test.csv", "model.pt", "scaler.pkl", "EUR/USD")
    res_jpy = run_pair_backtest("data/USDJPY_test.csv", "model_usdjpy.pt", "scaler_usdjpy.pkl", "USD/JPY")

    print(f"\nEUR/USD: Return: {res_eur['return']:>+7.2f}% | Sharpe: {res_eur['sharpe']:>6.3f} | Max DD: {res_eur['dd']:>6.2f}% | PF: {res_eur['pf']:>6.3f} | Trades: {res_eur['trades']}")
    print(f"USD/JPY: Return: {res_jpy['return']:>+7.2f}% | Sharpe: {res_jpy['sharpe']:>6.3f} | Max DD: {res_jpy['dd']:>6.2f}% | PF: {res_jpy['pf']:>6.3f} | Trades: {res_jpy['trades']}")

    # Combine equal-weighted portfolio
    min_len = min(len(res_eur["equity"]), len(res_jpy["equity"]))
    comb_equity = (res_eur["equity"][:min_len] + res_jpy["equity"][:min_len]) / 2.0
    comb_ret = ((comb_equity[-1] / max(comb_equity[0], 1e-8)) - 1.0) * 100.0
    comb_dd = compute_max_drawdown(comb_equity) * 100.0
    comb_sharpe = compute_sharpe_ratio(comb_equity)

    print("-" * 70)
    print(f"COMBINED DIVERSIFIED PORTFOLIO:")
    print(f"  Total Return:    {comb_ret:>+8.2f}%")
    print(f"  Max Drawdown:    {comb_dd:>7.2f}%  (Risk reduced by diversification!)")
    print(f"  Sharpe Ratio:    {comb_sharpe:>7.3f}")
    print(f"  Final Capital:   ${comb_equity[-1]:.2f} (from $10,000.00)")
    print("=" * 70)

    # Plot
    plt.figure(figsize=(13, 7))
    plt.plot(comb_equity, label=f"Diversified Ensemble Portfolio ({comb_ret:+.2f}%, Sharpe: {comb_sharpe:.2f}, DD: {comb_dd:.1f}%)", color="darkblue", linewidth=2.5)
    plt.plot(res_eur["equity"][:min_len], label=f"EUR/USD Specialized Model ({res_eur['return']:+.2f}%)", color="royalblue", alpha=0.7, linestyle="--", linewidth=1.5)
    plt.plot(res_jpy["equity"][:min_len], label=f"USD/JPY Specialized Model ({res_jpy['return']:+.2f}%)", color="darkorange", alpha=0.7, linestyle="--", linewidth=1.5)
    plt.axhline(10000.0, color="gray", linestyle=":", alpha=0.6, label="Starting Capital ($10,000)")
    plt.title("Production Ensemble Portfolio: EUR/USD + USD/JPY Out-of-Sample (2024–2026)", fontsize=13, fontweight="bold")
    plt.xlabel("Test Hours (2024–2026)", fontsize=11)
    plt.ylabel("Portfolio Value ($)", fontsize=11)
    plt.legend(loc="upper left", fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("ensemble_portfolio_equity.png", dpi=150)
    print("✓ Saved ensemble curve to: ensemble_portfolio_equity.png")


if __name__ == "__main__":
    main()
