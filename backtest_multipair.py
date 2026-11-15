"""
Multi-pair backtesting and portfolio evaluation script.
Tests a trained model across EURUSD, GBPUSD, USDJPY, and AUDUSD out-of-sample data,
computing individual pair metrics and a combined diversified portfolio equity curve.
"""

from __future__ import annotations

import argparse
import math
import pickle
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from data_loader import load_and_preprocess_data
from model import TradingLSTMPolicy
from trading_env import TradingEnv

DEFAULT_PAIRS = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD"]


def compute_max_drawdown(equity_curve: np.ndarray) -> float:
    running_peak = np.maximum.accumulate(equity_curve)
    drawdowns = (running_peak - equity_curve) / np.maximum(running_peak, 1e-8)
    return float(np.max(drawdowns)) if drawdowns.size else 0.0


def compute_sharpe_ratio(equity_curve: np.ndarray) -> float:
    if equity_curve.size < 2:
        return 0.0
    returns = np.diff(equity_curve) / np.maximum(equity_curve[:-1], 1e-8)
    std = np.std(returns)
    if std < 1e-12:
        return 0.0
    return float(np.sqrt(6048.0) * np.mean(returns) / std)


def compute_profit_factor(closed_trade_pnls: List[float]) -> float:
    if not closed_trade_pnls:
        return 0.0
    gross_profit = sum(p for p in closed_trade_pnls if p > 0)
    gross_loss = abs(sum(p for p in closed_trade_pnls if p < 0))
    if gross_loss < 1e-12:
        return float("inf") if gross_profit > 0 else 0.0
    return float(gross_profit / gross_loss)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backtest multi-pair forex policy")
    parser.add_argument("--pairs", nargs="+", default=DEFAULT_PAIRS, help="Pairs to evaluate")
    parser.add_argument("--data-dir", default="data", type=str, help="Directory containing test CSVs")
    parser.add_argument("--checkpoint", default="model_multipair.pt", type=str, help="Trained weights")
    parser.add_argument("--scaler-path", default="scaler_multipair.pkl", type=str, help="Scaler pickle")
    parser.add_argument("--initial-balance", default=10_000.0, type=float, help="Starting capital per pair")
    parser.add_argument("--window-size", default=32, type=int, help="Observation window size")
    parser.add_argument("--plot-path", default="multipair_equity_curve.png", type=str, help="Output plot")
    parser.add_argument("--enable-trailing-stop", action="store_true", help="Enable Chandelier ATR trailing stop")
    parser.add_argument("--enable-partial-tp", action="store_true", help="Enable partial profit taking scale-out")
    return parser.parse_args()


def backtest_single_pair(
    pair: str,
    test_csv: str,
    scaler,
    model: TradingLSTMPolicy,
    device: torch.device,
    args: argparse.Namespace,
) -> Dict:
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
        window_size=args.window_size,
        initial_balance=args.initial_balance,
        include_portfolio_state=True,
        random_start=False,
        enable_trailing_stop=args.enable_trailing_stop,
        chandelier_atr_mult=2.5,
        activation_atr_mult=2.0,
        enable_partial_tp=args.enable_partial_tp,
        partial_tp_mult=3.0,
        partial_tp_ratio=0.5,
    )

    state, info = env.reset()
    equity_curve = [float(info["equity"])]
    price_curve = []
    closed_trade_pnls: List[float] = []

    terminated = False
    truncated = False

    while not (terminated or truncated):
        state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            action_probs, _, _ = model(state_tensor)

        action = int(torch.argmax(action_probs, dim=-1).item())
        state, _, terminated, truncated, step_info = env.step(action)

        equity_curve.append(float(step_info["equity"]))
        price_curve.append(float(step_info["price"]))

        if step_info.get("trade_closed", False):
            closed_trade_pnls.append(float(step_info.get("closed_trade_net_pnl", 0.0)))

    equity_arr = np.array(equity_curve, dtype=np.float64)
    total_return = ((equity_arr[-1] / max(equity_arr[0], 1e-8)) - 1.0) * 100.0
    max_dd = compute_max_drawdown(equity_arr) * 100.0
    sharpe = compute_sharpe_ratio(equity_arr)
    profit_factor = compute_profit_factor(closed_trade_pnls)

    total_closed = int(step_info.get("total_closed_trades", 0))
    profitable = int(step_info.get("profitable_closed_trades", 0))
    win_rate = (profitable / total_closed * 100.0) if total_closed > 0 else 0.0

    return {
        "pair": pair,
        "equity_curve": equity_arr,
        "total_return": total_return,
        "max_drawdown": max_dd,
        "sharpe": sharpe,
        "profit_factor": profit_factor,
        "win_rate": win_rate,
        "trades": total_closed,
        "final_equity": equity_arr[-1],
    }


def main() -> None:
    args = parse_args()

    with open(args.scaler_path, "rb") as f:
        scaler = pickle.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 17 features + 4 portfolio = 21 features
    input_size = 21
    model = TradingLSTMPolicy(input_size=input_size).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    results: Dict[str, Dict] = {}
    print("\n" + "=" * 75)
    print("MULTI-PAIR OUT-OF-SAMPLE BACKTEST (DETERMINISTIC ARGMAX)")
    print("=" * 75)
    print(f"{'Pair':<10} | {'Return':<10} | {'Sharpe':<8} | {'Max DD':<8} | {'PF':<8} | {'Win %':<8} | {'Trades':<6}")
    print("-" * 75)

    for pair in args.pairs:
        test_csv = str(Path(args.data_dir) / f"{pair}_test.csv")
        res = backtest_single_pair(
            pair=pair,
            test_csv=test_csv,
            scaler=scaler,
            model=model,
            device=device,
            args=args,
        )
        results[pair] = res
        pf_str = "inf" if math.isinf(res["profit_factor"]) else f"{res['profit_factor']:.3f}"
        print(
            f"{pair:<10} | {res['total_return']:>+8.2f}% | {res['sharpe']:>7.3f} | {res['max_drawdown']:>7.2f}% | "
            f"{pf_str:>8} | {res['win_rate']:>7.2f}% | {res['trades']:>6}"
        )

    # Calculate equal-weighted combined portfolio equity curve
    min_len = min(len(res["equity_curve"]) for res in results.values())
    portfolio_equity = np.zeros(min_len, dtype=np.float64)
    for res in results.values():
        portfolio_equity += res["equity_curve"][:min_len]
    portfolio_equity /= len(results)

    port_return = ((portfolio_equity[-1] / max(portfolio_equity[0], 1e-8)) - 1.0) * 100.0
    port_dd = compute_max_drawdown(portfolio_equity) * 100.0
    port_sharpe = compute_sharpe_ratio(portfolio_equity)

    print("=" * 75)
    print(
        f"{'COMBINED':<10} | {port_return:>+8.2f}% | {port_sharpe:>7.3f} | {port_dd:>7.2f}% | "
        f"{'---':>8} | {'---':>8} | {'ALL':>6}"
    )
    print(f"Combined Portfolio Final Equity: ${portfolio_equity[-1]:.2f} (from ${args.initial_balance:.2f})")
    print("=" * 75)

    # Plot multi-panel equity curves
    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(14, 10), sharex=False, gridspec_kw={"height_ratios": [1.5, 1]})

    # Top: Combined Portfolio
    x_port = np.arange(min_len)
    ax_top.plot(x_port, portfolio_equity, label=f"Diversified Multi-Pair Portfolio ({port_return:+.2f}%)", color="darkblue", linewidth=2.5)
    ax_top.axhline(args.initial_balance, color="gray", linestyle="--", alpha=0.7, label="Initial Capital")
    ax_top.set_title("Multi-Pair Diversified Portfolio Equity Curve (EUR/USD, GBP/USD, USD/JPY, AUD/USD)", fontsize=13, fontweight="bold")
    ax_top.set_ylabel("Portfolio Value ($)", fontsize=11)
    ax_top.legend(loc="upper left", fontsize=11)
    ax_top.grid(alpha=0.3)

    # Bottom: Individual Pairs
    colors = {"EURUSD": "blue", "GBPUSD": "green", "USDJPY": "purple", "AUDUSD": "coral"}
    for pair, res in results.items():
        x_pair = np.arange(len(res["equity_curve"]))
        ax_bot.plot(x_pair, res["equity_curve"], label=f"{pair} ({res['total_return']:+.2f}%)", color=colors.get(pair, "black"), alpha=0.85, linewidth=1.5)
    ax_bot.axhline(args.initial_balance, color="gray", linestyle="--", alpha=0.5)
    ax_bot.set_title("Individual Currency Pair Performances", fontsize=12, fontweight="bold")
    ax_bot.set_xlabel("Out-of-Sample Hours (Test Bars 2024–2026)", fontsize=11)
    ax_bot.set_ylabel("Pair Equity ($)", fontsize=11)
    ax_bot.legend(loc="upper left", fontsize=10)
    ax_bot.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(args.plot_path, dpi=150)
    print(f"\n✓ Saved multi-pair performance plot to: {args.plot_path}")


if __name__ == "__main__":
    main()
