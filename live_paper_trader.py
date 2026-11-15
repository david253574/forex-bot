"""
Live Paper-Trading & Signal Execution Engine for Trained Forex PPO Agents.
Fetches real-time market data, constructs stationary indicators, evaluates neural policy,
and executes/manages virtual orders with 1:3 asymmetric bracket risk controls.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import yfinance as yf
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator, MACD
from ta.volatility import AverageTrueRange, BollingerBands

from model import TradingLSTMPolicy

PAIR_TICKER_MAP = {
    "EURUSD": "EURUSD=X",
    "GBPUSD": "GBPUSD=X",
    "USDJPY": "JPY=X",
    "AUDUSD": "AUDUSD=X",
}

DEFAULT_LEDGER_PATH = "paper_ledger.json"


class PaperTrader:
    def __init__(
        self,
        pair: str = "EURUSD",
        model_path: str = "model.pt",
        scaler_path: str = "scaler.pkl",
        ledger_path: str = DEFAULT_LEDGER_PATH,
        initial_balance: float = 10_000.0,
        risk_per_trade: float = 0.02,
        window_size: int = 32,
    ) -> None:
        self.pair = pair.upper()
        self.ticker = PAIR_TICKER_MAP.get(self.pair, f"{self.pair}=X")
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.ledger_path = Path(ledger_path)
        self.initial_balance = initial_balance
        self.risk_per_trade = risk_per_trade
        self.window_size = window_size

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._load_model_and_scaler()
        self._load_ledger()

    def _load_model_and_scaler(self) -> None:
        if not os.path.exists(self.scaler_path):
            raise FileNotFoundError(f"Scaler not found: {self.scaler_path}")
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model checkpoint not found: {self.model_path}")

        with open(self.scaler_path, "rb") as f:
            self.scaler = pickle.load(f)

        input_size = 21  # 17 market features + 4 portfolio state features
        self.model = TradingLSTMPolicy(input_size=input_size).to(self.device)
        self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        self.model.eval()

    def _load_ledger(self) -> None:
        if self.ledger_path.exists():
            with open(self.ledger_path, "r") as f:
                self.ledger = json.load(f)
        else:
            self.ledger = {
                "pair": self.pair,
                "balance": self.initial_balance,
                "equity": self.initial_balance,
                "position": 0,  # -1 Short, 0 Flat, 1 Long
                "entry_price": 0.0,
                "entry_atr": 0.0,
                "position_size": 0.0,
                "stop_loss": 0.0,
                "take_profit": 0.0,
                "entry_time": None,
                "time_in_trade": 0,
                "consecutive_holds": 0,
                "closed_trades": [],
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
            self._save_ledger()

    def _save_ledger(self) -> None:
        with open(self.ledger_path, "w") as f:
            json.dump(self.ledger, f, indent=2)

    def fetch_live_data(self) -> pd.DataFrame:
        """Fetch latest hourly bars from Yahoo Finance."""
        ticker = yf.Ticker(self.ticker)
        df = ticker.history(period="15d", interval="1h")
        if len(df) < self.window_size + 50:
            raise RuntimeError(f"Not enough recent data fetched for {self.ticker}: got {len(df)} bars")

        df = df.reset_index()
        date_col = "Datetime" if "Datetime" in df.columns else "Date"
        df["date"] = pd.to_datetime(df[date_col])

        # Normalize column names
        df.columns = [col.strip().lower() for col in df.columns]
        df = df[["date", "open", "high", "low", "close"]].dropna().reset_index(drop=True)
        return df

    def compute_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, float, float, float]:
        df = df.copy()

        df["log_ret_open"] = np.log(df["open"] / df["open"].shift(1))
        df["log_ret_high"] = np.log(df["high"] / df["high"].shift(1))
        df["log_ret_low"] = np.log(df["low"] / df["low"].shift(1))
        df["log_ret_close"] = np.log(df["close"] / df["close"].shift(1))

        range_calc = df["high"] - df["low"] + 1e-8
        df["body_size"] = (df["close"] - df["open"]) / range_calc
        df["upper_wick"] = (df["high"] - df[["open", "close"]].max(axis=1)) / range_calc
        df["lower_wick"] = (df[["open", "close"]].min(axis=1) - df["low"]) / range_calc

        df = df.dropna().reset_index(drop=True)

        datetime_col = pd.to_datetime(df["date"])
        df["hour_sin"] = np.sin(2 * np.pi * datetime_col.dt.hour / 24.0)
        df["hour_cos"] = np.cos(2 * np.pi * datetime_col.dt.hour / 24.0)
        df["day_of_week"] = datetime_col.dt.dayofweek / 6.0

        macd = MACD(close=df["close"])
        df["macd"] = macd.macd() / df["close"]
        df["macd_signal"] = macd.macd_signal() / df["close"]
        df["rsi_14"] = RSIIndicator(close=df["close"], window=14).rsi()
        df["ema_50"] = EMAIndicator(close=df["close"], window=50).ema_indicator()
        df["ema_200"] = EMAIndicator(close=df["close"], window=200).ema_indicator()
        raw_atr_series = AverageTrueRange(high=df["high"], low=df["low"], close=df["close"], window=14).average_true_range()
        df["atr_14"] = raw_atr_series / df["close"]

        bb = BollingerBands(close=df["close"], window=20)
        df["bb_width"] = (bb.bollinger_hband() - bb.bollinger_lband()) / df["close"]
        df["dist_ema_50"] = (df["close"] - df["ema_50"]) / df["ema_50"]
        df["dist_ema_200"] = (df["close"] - df["ema_200"]) / df["ema_200"]

        df = df.dropna().reset_index(drop=True)

        feature_cols = [
            "log_ret_open", "log_ret_high", "log_ret_low", "log_ret_close",
            "macd", "macd_signal", "rsi_14", "atr_14", "bb_width",
            "dist_ema_50", "dist_ema_200", "hour_sin", "hour_cos",
            "day_of_week", "body_size", "upper_wick", "lower_wick",
        ]

        # Extract last window_size bars
        window_raw = df.iloc[-self.window_size:][feature_cols]
        window_scaled = self.scaler.transform(window_raw)

        last_close = float(df["close"].iloc[-1])
        last_high = float(df["high"].iloc[-1])
        last_low = float(df["low"].iloc[-1])
        last_raw_atr = float(raw_atr_series.iloc[-1])

        # Construct portfolio state vector
        pos = float(self.ledger["position"])
        balance = float(self.ledger["balance"])

        unrealized = 0.0
        if pos == 1:
            unrealized = (last_close - float(self.ledger["entry_price"])) * float(self.ledger["position_size"])
        elif pos == -1:
            unrealized = (float(self.ledger["entry_price"]) - last_close) * float(self.ledger["position_size"])

        unrealized_ratio = float(np.clip(unrealized / max(balance, 1.0), -0.2, 0.2)) * 5.0
        duration_ratio = min(1.0, float(self.ledger["time_in_trade"]) / 72.0) if pos != 0 else 0.0
        inactivity_ratio = min(1.0, float(self.ledger["consecutive_holds"]) / 64.0)

        port_vec = np.array([pos, unrealized_ratio, duration_ratio, inactivity_ratio], dtype=np.float32)
        port_matrix = np.tile(port_vec, (self.window_size, 1))

        obs = np.concatenate([window_scaled, port_matrix], axis=1).astype(np.float32)
        return obs, last_close, last_high, last_low, last_raw_atr

    def evaluate_and_execute(self) -> Dict:
        raw_df = self.fetch_live_data()
        obs, current_price, bar_high, bar_low, current_atr = self.compute_features(raw_df)

        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            action_probs, state_value, _ = self.model(obs_tensor)

        probs = action_probs.squeeze(0).cpu().numpy()
        action = int(np.argmax(probs))
        action_names = {0: "HOLD / FLAT", 1: "BUY / LONG", 2: "SELL / SHORT"}

        current_position = self.ledger["position"]
        order_event = None

        # 1. Manage Active Open Positions
        if current_position != 0:
            self.ledger["time_in_trade"] += 1
            entry_price = float(self.ledger["entry_price"])
            sl = float(self.ledger["stop_loss"])
            tp = float(self.ledger["take_profit"])

            sl_hit = (current_position == 1 and bar_low <= sl) or (current_position == -1 and bar_high >= sl)
            tp_hit = (current_position == 1 and bar_high >= tp) or (current_position == -1 and bar_low <= tp)

            exit_reason = None
            exit_price = current_price

            if sl_hit:
                exit_reason = "STOP-LOSS HIT"
                exit_price = sl
            elif tp_hit:
                exit_reason = "TAKE-PROFIT HIT"
                exit_price = tp
            elif action == 1 and current_position == -1:
                exit_reason = "SIGNAL EXIT (Cover Short)"
                exit_price = current_price
            elif action == 2 and current_position == 1:
                exit_reason = "SIGNAL EXIT (Close Long)"
                exit_price = current_price
            elif self.ledger["time_in_trade"] >= 72:
                exit_reason = "MAX HOLD EXPIRED"
                exit_price = current_price

            if exit_reason:
                pos_size = float(self.ledger["position_size"])
                gross_pnl = (exit_price - entry_price) * pos_size if current_position == 1 else (entry_price - exit_price) * pos_size
                fee = 0.50  # Flat commission
                net_pnl = gross_pnl - fee
                self.ledger["balance"] += net_pnl
                self.ledger["equity"] = self.ledger["balance"]

                closed_trade = {
                    "direction": "LONG" if current_position == 1 else "SHORT",
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "exit_reason": exit_reason,
                    "position_size": pos_size,
                    "net_pnl": net_pnl,
                    "duration_hours": self.ledger["time_in_trade"],
                    "exit_time": datetime.now(timezone.utc).isoformat(),
                }
                self.ledger["closed_trades"].append(closed_trade)
                order_event = f"CLOSED {closed_trade['direction']} via {exit_reason} | Net P&L: ${net_pnl:+.2f}"

                # Reset open position
                self.ledger["position"] = 0
                self.ledger["entry_price"] = 0.0
                self.ledger["position_size"] = 0.0
                self.ledger["stop_loss"] = 0.0
                self.ledger["take_profit"] = 0.0
                self.ledger["time_in_trade"] = 0

        # 2. Open New Orders if Flat
        if self.ledger["position"] == 0:
            if action in [1, 2]:
                # 2% Risk Sizing: risk_amount / (2 * ATR)
                risk_amt = self.ledger["balance"] * self.risk_per_trade
                sl_dist = 2.0 * current_atr
                raw_size = risk_amt / max(sl_dist, 1e-5)
                max_size = (self.ledger["balance"] * 20.0) / max(current_price, 1e-4)
                pos_size = min(raw_size, max_size)

                self.ledger["position"] = 1 if action == 1 else -1
                self.ledger["entry_price"] = current_price
                self.ledger["entry_atr"] = current_atr
                self.ledger["position_size"] = pos_size
                self.ledger["entry_time"] = datetime.now(timezone.utc).isoformat()
                self.ledger["time_in_trade"] = 0
                self.ledger["consecutive_holds"] = 0

                if action == 1:
                    self.ledger["stop_loss"] = current_price - sl_dist
                    self.ledger["take_profit"] = current_price + (6.0 * current_atr)
                    order_event = f"OPENED LONG @ {current_price:.5f} | SL: {self.ledger['stop_loss']:.5f} | TP: {self.ledger['take_profit']:.5f}"
                else:
                    self.ledger["stop_loss"] = current_price + sl_dist
                    self.ledger["take_profit"] = current_price - (6.0 * current_atr)
                    order_event = f"OPENED SHORT @ {current_price:.5f} | SL: {self.ledger['stop_loss']:.5f} | TP: {self.ledger['take_profit']:.5f}"
            else:
                self.ledger["consecutive_holds"] += 1

        # 3. Update Mark-to-Market Equity
        pos = self.ledger["position"]
        if pos != 0:
            unrealized = (current_price - self.ledger["entry_price"]) * self.ledger["position_size"] if pos == 1 else (self.ledger["entry_price"] - current_price) * self.ledger["position_size"]
            self.ledger["equity"] = self.ledger["balance"] + unrealized
        else:
            self.ledger["equity"] = self.ledger["balance"]

        self._save_ledger()

        # Print Terminal Dashboard
        pos_str = "FLAT" if pos == 0 else ("LONG" if pos == 1 else "SHORT")
        print("\n" + "=" * 70)
        print(f"FOREX LIVE PAPER TRADER: {self.pair} ({datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')})")
        print("=" * 70)
        print(f"Market Price:      {current_price:.5f} (ATR: {current_atr:.5f})")
        print(f"Account Balance:   ${self.ledger['balance']:.2f} | Live Equity: ${self.ledger['equity']:.2f}")
        print(f"Active Position:   {pos_str}")
        if pos != 0:
            unrealized_pnl = self.ledger["equity"] - self.ledger["balance"]
            print(f"  Entry Price:     {self.ledger['entry_price']:.5f}")
            print(f"  Stop-Loss:       {self.ledger['stop_loss']:.5f}")
            print(f"  Take-Profit:     {self.ledger['take_profit']:.5f}")
            print(f"  Unrealized P&L:  ${unrealized_pnl:+.2f}")
            print(f"  Holding Time:    {self.ledger['time_in_trade']} hours")
        print("-" * 70)
        print(f"Model Probs:       Hold: {probs[0]*100:.1f}% | Buy: {probs[1]*100:.1f}% | Sell: {probs[2]*100:.1f}%")
        print(f"Selected Action:   {action_names[action]}")
        if order_event:
            print(f"Order Action:      👉 {order_event}")
        print(f"Total Completed:   {len(self.ledger['closed_trades'])} trades")
        print("=" * 70)

        return {
            "pair": self.pair,
            "price": current_price,
            "action": action_names[action],
            "probs": probs.tolist(),
            "position": pos_str,
            "balance": self.ledger["balance"],
            "equity": self.ledger["equity"],
            "event": order_event,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Live paper trading execution engine")
    parser.add_argument("--pair", default="EURUSD", type=str, help="Currency pair (EURUSD, GBPUSD, etc.)")
    parser.add_argument("--model-path", default="model.pt", type=str, help="Trained weights file")
    parser.add_argument("--scaler-path", default="scaler.pkl", type=str, help="Scaler file")
    parser.add_argument("--ledger", default=DEFAULT_LEDGER_PATH, type=str, help="Paper trading ledger JSON")
    parser.add_argument("--balance", default=10_000.0, type=float, help="Initial paper balance")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    trader = PaperTrader(
        pair=args.pair,
        model_path=args.model_path,
        scaler_path=args.scaler_path,
        ledger_path=args.ledger,
        initial_balance=args.balance,
    )
    trader.evaluate_and_execute()
