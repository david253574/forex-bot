#!/usr/bin/env python3
"""Standalone backtest script for trained model."""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import argparse
import math
import pickle
from typing import List

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

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
    return float(np.sqrt(252.0) * np.mean(returns) / std)


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


def main():
    # Default args for final backtest
    csv_file = "forex_data.csv"
    checkpoint = "model.pt"
    scaler_path = "scaler.pkl"
    window_size = 64
    initial_balance = 10_000.0
    spread = 0.0002
    commission = 0.5
    slippage_std = 0.0001
    execution_penalty = 0.5
    max_hold_steps = 2000
    holding_penalty = 0.05
    equity_plot = "equity_curve_final.png"
    deterministic = True
    temperature = 1.0

    print("Loading data...")
    scaler = load_scaler(scaler_path)
    _, _, feature_columns, enriched_df = load_and_preprocess_data(csv_file)
    scaled_values = scaler.transform(enriched_df[feature_columns])
    feature_df = pd.DataFrame(scaled_values, columns=feature_columns, index=enriched_df.index)

    print("Initializing environment...")
    # Extract raw ATR before scaling for Stop Loss / Take Profit calculations
    raw_atr = enriched_df["atr_14"].values.astype(np.float32)
    
    env = TradingEnv(
        feature_data=feature_df[feature_columns].values,
        price_data=enriched_df["close"].values,
        feature_map={"atr": 10},
        window_size=window_size,
        initial_balance=initial_balance,
        spread=spread,
        commission=commission,
        slippage_std=slippage_std,
        execution_penalty=execution_penalty,
        max_hold_steps=max_hold_steps,
        holding_penalty=holding_penalty,
        raw_atr_data=raw_atr,
    )

    print("Loading model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TradingLSTMPolicy(input_size=len(feature_columns)).to(device)
    state_dict = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    print("Running backtest...")
    state, info = env.reset()
    equity_curve = [float(info["equity"])]
    price_curve = []
    closed_trade_pnls: List[float] = []
    trade_logs: List[dict] = []
    
    # Track open trade details for logging
    entry_price = None
    entry_step = None
    trade_direction = None
    
    terminated = False
    truncated = False
    step_count = 0
    step_info = {}

    while not (terminated or truncated):
        state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

        with torch.no_grad():
            action_probs, _, logits = model(state_tensor)

        # Extract raw probabilities for first 10 steps to diagnose argmax trap
        if step_count < 10:
            probs_array = action_probs.squeeze(0).cpu().numpy()
            print(f"Step {step_count}: [Hold: {probs_array[0]:.4f}, Buy: {probs_array[1]:.4f}, Sell: {probs_array[2]:.4f}]")

        if deterministic:
            buy_confidence = float(action_probs[0, 1])
            sell_confidence = float(action_probs[0, 2])
            if buy_confidence > 0.65:
                action = 1
            elif sell_confidence > 0.65:
                action = 2
            else:
                action = 0
        else:
            if temperature <= 0.0:
                raise ValueError("temperature must be > 0 when sampling")
            scaled_logits = logits / temperature
            scaled_probs = torch.softmax(scaled_logits, dim=-1).squeeze(0)
            action = int(torch.multinomial(scaled_probs, 1).item())

        state, _, terminated, truncated, step_info = env.step(action)
        equity_curve.append(float(step_info["equity"]))
        price_curve.append(float(step_info["price"]))

        # Track entry signals (position changed from 0 to non-zero)
        current_position = float(step_info.get("position", 0))
        if current_position != 0 and entry_price is None:
            entry_price = float(step_info.get("price", 0.0))
            entry_step = step_count
            trade_direction = "LONG" if current_position > 0 else "SHORT"

        if step_info.get("trade_closed", False):
            closed_pnl = float(step_info.get("closed_trade_net_pnl", 0.0))
            closed_trade_pnls.append(closed_pnl)
            
            # Log the trade details
            if entry_price is not None and trade_direction is not None:
                exit_price = float(step_info.get("price", 0.0))
                trade_log = {
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "direction": trade_direction,
                    "entry_step": entry_step,
                    "exit_step": step_count,
                    "duration_steps": step_count - entry_step,
                    "gross_profit": float(step_info.get("realized_pnl", 0.0)),
                    "commission": commission,
                    "slippage": float(step_info.get("slippage_paid", 0.0)),
                    "net_pnl": closed_pnl,
                }
                trade_logs.append(trade_log)
            
            # Reset trade tracking
            entry_price = None
            entry_step = None
            trade_direction = None
        
        step_count += 1
        if step_count % 500 == 0:
            print(f"  Step {step_count}: Equity ${equity_curve[-1]:.2f}")

    print("\nComputing metrics...")
    
    # Print first 5 trades for inspection
    print("\n" + "="*80)
    print("FIRST 5 CLOSED TRADES (DETAILED INSPECTION)")
    print("="*80)
    for i, trade in enumerate(trade_logs[:5], 1):
        print(f"\nTrade #{i}:")
        print(f"  Direction:        {trade['direction']}")
        print(f"  Entry Price:      {trade['entry_price']:.6f}")
        print(f"  Exit Price:       {trade['exit_price']:.6f}")
        print(f"  Gross P&L:        ${trade['gross_profit']:.4f}")
        print(f"  Commission:       ${trade['commission']:.4f}")
        print(f"  Slippage:         ${trade['slippage']:.4f}")
        print(f"  Net P&L (Reward): ${trade['net_pnl']:.4f}")
        print(f"  Duration:         {trade['duration_steps']} steps")
    print("="*80)
    
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
        buy_hold_curve = initial_balance * (prices_arr / max(prices_arr[0], 1e-8))
        buy_hold_curve = np.insert(buy_hold_curve, 0, initial_balance)
    else:
        buy_hold_curve = np.array([initial_balance], dtype=np.float64)

    print("\n" + "="*60)
    print("BACKTEST RESULTS - 1000-Iteration Trained Model")
    print("="*60)
    print(f"Total Return (%):     {total_return:.2f}")
    print(f"Max Drawdown (%):     {max_drawdown:.2f}")
    print(f"Sharpe Ratio:         {sharpe_ratio:.3f}")
    if math.isinf(profit_factor):
        print("Profit Factor:        inf (100% profitable trades)")
    else:
        print(f"Profit Factor:        {profit_factor:.3f}")
    print(f"Win Rate (%):         {win_rate:.2f}")
    print(f"Closed Trades:        {total_closed_trades}")
    print("="*60)

    # Status check
    if profit_factor > 1.0:
        status = "FUNCTIONAL MACRO-TREND SYSTEM"
    elif profit_factor >= 0.9:
        status = "DIRECTIONAL EDGE (friction costs)"
    else:
        status = "NEGATIVE EDGE"

    print(f"\nStatus: {status}")

    # Save plot
    x = np.arange(equity_arr.size)
    plt.figure(figsize=(14, 7))
    plt.plot(x, equity_arr, label="RL Strategy Equity", linewidth=2, color="blue")
    plt.plot(x, buy_hold_curve[: equity_arr.size], label="Buy & Hold Baseline", linewidth=1.8, linestyle="--", color="gray")
    plt.title("Equity Curve: 1000-Iteration PPO Macro-Trend Follower", fontsize=14, fontweight="bold")
    plt.xlabel("Backtest Step (Daily)", fontsize=12)
    plt.ylabel("Portfolio Value ($)", fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(equity_plot, dpi=150)
    print(f"\n✓ Equity curve saved to: {equity_plot}")


if __name__ == "__main__":
    main()
