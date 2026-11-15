"""Standalone backtesting script for a trained forex PPO policy network."""

from __future__ import annotations

import argparse
import math
import pickle
from typing import List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from data_loader import load_and_preprocess_data
from model import TradingLSTMPolicy
from trading_env import TradingEnv


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
    # Annualized for hourly forex bars (~24 * 252 = ~6048 trading hours per year)
    return float(np.sqrt(6048.0) * np.mean(returns) / std)


def compute_profit_factor(closed_trade_pnls: List[float]) -> float:
    if not closed_trade_pnls:
        return 0.0
    gross_profit = sum(p for p in closed_trade_pnls if p > 0)
    gross_loss = abs(sum(p for p in closed_trade_pnls if p < 0))
    if gross_loss < 1e-12:
        return float("inf") if gross_profit > 0 else 0.0
    return float(gross_profit / gross_loss)


def load_scaler(scaler_path: str):
    with open(scaler_path, "rb") as handle:
        return pickle.load(handle)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backtest a trained forex PPO policy")
    parser.add_argument("--test-csv", default="forex_data_test.csv", type=str, help="Path to unseen test OHLCV CSV")
    parser.add_argument(
        "--checkpoint",
        default="model.pt",
        type=str,
        help="Path to trained model weights",
    )
    parser.add_argument(
        "--scaler-path",
        default="scaler.pkl",
        type=str,
        help="Path to fitted training scaler pickle (scaler.pkl)",
    )
    parser.add_argument("--window-size", default=32, type=int, help="Observation window size")
    parser.add_argument("--initial-balance", default=10_000.0, type=float, help="Starting capital")
    parser.add_argument("--spread", default=0.00015, type=float, help="Spread cost charged on trade entry (1.5 pips)")
    parser.add_argument(
        "--commission",
        default=0.5,
        type=float,
        help="Flat commission charged on each trade close/reversal",
    )
    parser.add_argument(
        "--slippage-std",
        default=0.00005,
        type=float,
        help="Stddev for slippage overhead per execution",
    )
    parser.add_argument(
        "--equity-plot",
        default="equity_curve.png",
        type=str,
        help="Where to save the equity-curve chart",
    )
    parser.add_argument(
        "--temperature",
        default=1.0,
        type=float,
        help="Temperature for stochastic action sampling from logits",
    )
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Use stochastic sampling with temperature instead of deterministic argmax",
    )
    parser.add_argument("--execution-penalty", type=float, default=0.0, help="Penalty per trade open")
    parser.add_argument("--max-hold-steps", type=int, default=72, help="Max steps before forced exit")
    parser.add_argument("--holding-penalty", type=float, default=0.0, help="Penalty per step in trade")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scaler = load_scaler(args.scaler_path)

    _, _, feature_columns, enriched_df = load_and_preprocess_data(args.test_csv)
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
        spread=args.spread,
        commission=args.commission,
        slippage_std=args.slippage_std,
        execution_penalty=args.execution_penalty,
        max_hold_steps=args.max_hold_steps,
        holding_penalty=args.holding_penalty,
        include_portfolio_state=True,
        random_start=False,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_size = env.observation_space.shape[1]
    model = TradingLSTMPolicy(input_size=input_size).to(device)
    state_dict = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    state, info = env.reset()
    equity_curve = [float(info["equity"])]
    price_curve = []
    closed_trade_pnls: List[float] = []

    terminated = False
    truncated = False

    while not (terminated or truncated):
        state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

        with torch.no_grad():
            action_probs, _, logits = model(state_tensor)

        if not args.stochastic:
            action = int(torch.argmax(action_probs, dim=-1).item())
        else:
            if args.temperature <= 0.0:
                raise ValueError("--temperature must be > 0 when sampling")
            scaled_logits = logits / args.temperature
            scaled_probs = torch.softmax(scaled_logits, dim=-1).squeeze(0)
            action = int(torch.multinomial(scaled_probs, 1).item())

        state, _, terminated, truncated, step_info = env.step(action)

        equity_curve.append(float(step_info["equity"]))
        price_curve.append(float(step_info["price"]))

        if step_info.get("trade_closed", False):
            closed_trade_pnls.append(float(step_info.get("closed_trade_net_pnl", 0.0)))

    equity_arr = np.array(equity_curve, dtype=np.float64)
    prices_arr = np.array(price_curve, dtype=np.float64)

    total_return = ((equity_arr[-1] / max(equity_arr[0], 1e-8)) - 1.0) * 100.0
    max_drawdown = compute_max_drawdown(equity_arr) * 100.0
    sharpe_ratio = compute_sharpe_ratio(equity_arr)
    profit_factor = compute_profit_factor(closed_trade_pnls)

    total_closed_trades = int(step_info.get("total_closed_trades", 0))
    profitable_closed_trades = int(step_info.get("profitable_closed_trades", 0))
    win_rate = (
        (profitable_closed_trades / total_closed_trades) * 100.0
        if total_closed_trades > 0
        else 0.0
    )

    if prices_arr.size > 0:
        buy_hold_curve = args.initial_balance * (prices_arr / max(prices_arr[0], 1e-8))
        buy_hold_curve = np.insert(buy_hold_curve, 0, args.initial_balance)
    else:
        buy_hold_curve = np.array([args.initial_balance], dtype=np.float64)

    print("\n" + "=" * 60)
    print("OUT-OF-SAMPLE BACKTEST RESULTS")
    print("=" * 60)
    print(f"Total Return (%):     {total_return:+.2f}%")
    print(f"Max Drawdown (%):     {max_drawdown:.2f}%")
    print(f"Annualized Sharpe:    {sharpe_ratio:.3f}")
    if math.isinf(profit_factor):
        print("Profit Factor:        inf")
    else:
        print(f"Profit Factor:        {profit_factor:.3f}")
    print(f"Win Rate (%):         {win_rate:.2f}%")
    print(f"Closed Trades:        {total_closed_trades}")
    print(f"Final Equity:         ${equity_arr[-1]:.2f}")
    print("=" * 60)

    x = np.arange(equity_arr.size)
    plt.figure(figsize=(12, 6))
    plt.plot(x, equity_arr, label="RL Policy Equity", linewidth=2, color="blue")
    plt.plot(x, buy_hold_curve[: equity_arr.size], label="Buy & Hold Baseline", linewidth=1.8, linestyle="--", color="gray")
    plt.title("Out-of-Sample Equity Curve: PPO Policy vs Buy & Hold", fontsize=14, fontweight="bold")
    plt.xlabel("Hours (Test Bars)", fontsize=12)
    plt.ylabel("Portfolio Value ($)", fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(args.equity_plot, dpi=150)
    print(f"Saved equity curve to: {args.equity_plot}")


if __name__ == "__main__":
    main()
