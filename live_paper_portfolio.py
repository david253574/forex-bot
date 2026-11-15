"""
Multi-Pair Live Paper Trading Portfolio Engine.
Manages concurrent execution across EUR/USD and USD/JPY (and configurable pairs)
with independent neural policies, shared portfolio equity, and 1:3 asymmetric bracket risk.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import yfinance as yf
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator, MACD
from ta.volatility import AverageTrueRange, BollingerBands

from model import TradingLSTMPolicy
from broker_gateway import create_broker_gateway
from risk_guards import SpreadGuard, NewsShield, DailyCircuitBreaker, WebhookNotifier

DEFAULT_PORTFOLIO_LEDGER = "portfolio_ledger.json"

PAIR_TICKER_MAP = {
    "EURUSD": "EURUSD=X",
    "GBPUSD": "GBPUSD=X",
    "USDJPY": "JPY=X",
    "AUDUSD": "AUDUSD=X",
}

PAIR_MODEL_DEFAULTS = {
    "EURUSD": ("model.pt", "scaler.pkl"),
    "USDJPY": ("model_usdjpy.pt", "scaler_usdjpy.pkl"),
    "AUDUSD": ("model_multipair.pt", "scaler_multipair.pkl"),
    "GBPUSD": ("model_multipair.pt", "scaler_multipair.pkl"),
}


class MultiPairPaperTrader:
    def __init__(
        self,
        pairs: List[str] = ["EURUSD", "USDJPY", "AUDUSD", "GBPUSD"],
        ledger_path: str = DEFAULT_PORTFOLIO_LEDGER,
        initial_balance: float = 10_000.0,
        risk_per_trade: float = 0.015,
        window_size: int = 32,
        broker_config_path: str = "broker_config.json",
    ) -> None:
        self.pairs = [p.upper() for p in pairs]
        self.ledger_path = Path(ledger_path)
        self.initial_balance = initial_balance
        self.risk_per_trade = risk_per_trade
        self.window_size = window_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gateway = create_broker_gateway(broker_config_path)

        # Institutional Execution & Risk Guards
        self.spread_guard = SpreadGuard()
        self.news_shield = NewsShield()
        self.circuit_breaker = DailyCircuitBreaker()
        self.notifier = WebhookNotifier(config_path=broker_config_path)

        self.models: Dict[str, TradingLSTMPolicy] = {}
        self.scalers: Dict[str, object] = {}
        self._init_models_and_scalers()
        self._load_ledger()

    def _init_models_and_scalers(self) -> None:
        for pair in self.pairs:
            m_path, s_path = PAIR_MODEL_DEFAULTS.get(pair, ("model.pt", "scaler.pkl"))
            if not os.path.exists(m_path) or not os.path.exists(s_path):
                # Fallback to multipair model if dedicated does not exist
                if os.path.exists("model_multipair.pt") and os.path.exists("scaler_multipair.pkl"):
                    m_path, s_path = "model_multipair.pt", "scaler_multipair.pkl"
                else:
                    raise FileNotFoundError(f"Missing model or scaler files for {pair} ({m_path}, {s_path})")

            with open(s_path, "rb") as f:
                self.scalers[pair] = pickle.load(f)

            model = TradingLSTMPolicy(input_size=21).to(self.device)
            model.load_state_dict(torch.load(m_path, map_location=self.device))
            model.eval()
            self.models[pair] = model
            print(f"Loaded {pair} policy from {m_path} (Scaler: {s_path})")

    def _load_ledger(self) -> None:
        if self.ledger_path.exists():
            with open(self.ledger_path, "r") as f:
                self.ledger = json.load(f)
        else:
            self.ledger = {
                "balance": self.initial_balance,
                "equity": self.initial_balance,
                "positions": {},
                "closed_trades": [],
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
            for pair in self.pairs:
                self.ledger["positions"][pair] = self._empty_position()
            self._save_ledger()

        # Ensure all requested pairs exist in positions
        for pair in self.pairs:
            if pair not in self.ledger["positions"]:
                self.ledger["positions"][pair] = self._empty_position()

        if hasattr(self, "gateway") and self.gateway:
            try:
                acc_info = self.gateway.get_account_summary()
                if not acc_info.get("error") and "balance" in acc_info:
                    b_bal = float(acc_info["balance"])
                    if b_bal > 0 and acc_info.get("broker") != "MOCK_SIMULATOR":
                        if "initial_capital" not in self.ledger or self.ledger.get("initial_capital", 10000.0) < 50000.0:
                            self.ledger["initial_capital"] = b_bal
                            self.initial_balance = b_bal
                        if "balance" not in self.ledger:
                            self.ledger["balance"] = b_bal
                        if "equity" not in self.ledger:
                            self.ledger["equity"] = float(acc_info.get("equity", b_bal))
                        self.ledger["broker_equity"] = float(acc_info.get("equity", b_bal))
                        self.ledger["broker_info"] = {
                            "broker": acc_info.get("broker"),
                            "account_id": acc_info.get("account_id"),
                            "environment": acc_info.get("environment"),
                            "currency": acc_info.get("currency", "USD"),
                        }
                        self._save_ledger()
            except Exception as e:
                print(f"Warning syncing broker balance: {e}")

        # Run safe startup reconciliation with broker
        self.reconcile_with_broker()

    @staticmethod
    def _empty_position() -> Dict:
        return {
            "position": 0,
            "entry_price": 0.0,
            "entry_atr": 0.0,
            "position_size": 0.0,
            "base_units": 0,
            "risk_budget_usd": 0.0,
            "accounting_method": "STANDARD_SPOT_FX",
            "stop_loss": 0.0,
            "take_profit": 0.0,
            "entry_time": None,
            "time_in_trade": 0,
            "consecutive_holds": 0,
            "execution_type": "FLAT",
            "broker_ticket": None,
        }

    @staticmethod
    def calculate_position_sizing(
        pair: str,
        action: int,
        balance: float,
        risk_per_trade: float,
        current_price: float,
        current_atr: float,
    ) -> Dict[str, Any]:
        """
        Derives financially rigorous base units and stop/take-profit brackets.

        For direct pairs (EUR/USD, GBP/USD, AUD/USD - Quote Currency is USD):
            sl_dist = 2 * ATR
            base_units = risk_usd / sl_dist

        For indirect pairs (USD/JPY - Base is USD, Quote is JPY):
            sl_dist = 2 * ATR
            LONG:  stop_loss = current_price - sl_dist
                   base_units = (risk_usd * stop_loss) / sl_dist
            SHORT: stop_loss = current_price + sl_dist
                   base_units = (risk_usd * stop_loss) / sl_dist
        """
        risk_usd = balance * risk_per_trade
        sl_dist = 2.0 * current_atr
        clean_pair = pair.upper().replace("_", "").replace("=X", "")

        if action == 1:  # LONG
            sl_price = current_price - sl_dist
            tp_price = current_price + (6.0 * current_atr)
        else:  # SHORT
            sl_price = current_price + sl_dist
            tp_price = current_price - (6.0 * current_atr)

        is_jpy = ("JPY" in clean_pair or clean_pair.startswith("USD")) and not clean_pair.endswith("USD")

        if is_jpy:
            # USD/JPY: Base is USD, Quote is JPY.
            # Risk at SL (S) is: Base_Units * |E - S| / S = Risk_USD
            # Base_Units = (Risk_USD * S) / sl_dist
            raw_base_units = (risk_usd * sl_price) / max(sl_dist, 1e-5)
            max_base_units = balance * 20.0  # Max 20x account leverage
            base_units = min(raw_base_units, max_base_units)
            point_sensitivity = risk_usd / max(sl_dist, 1e-5)
        else:
            # EUR/USD, GBP/USD, AUD/USD: Quote is USD.
            # Risk at SL is: Base_Units * sl_dist = Risk_USD
            # Base_Units = Risk_USD / sl_dist
            raw_base_units = risk_usd / max(sl_dist, 1e-5)
            max_base_units = (balance * 20.0) / max(current_price, 1e-4)
            base_units = min(raw_base_units, max_base_units)
            point_sensitivity = base_units

        order_units = int(round(base_units)) if action == 1 else -int(round(base_units))

        return {
            "risk_budget_usd": risk_usd,
            "sl_dist": sl_dist,
            "stop_loss": sl_price,
            "take_profit": tp_price,
            "base_units": abs(order_units),
            "order_units": order_units,
            "point_sensitivity": point_sensitivity,
            "accounting_method": "STANDARD_SPOT_FX",
        }

    @staticmethod
    def calculate_position_pnl(
        pair: str,
        pos_info: Dict[str, Any],
        eval_price: float,
    ) -> float:
        """
        Calculates USD P&L for a position at eval_price.
        Preserves legacy compatibility for pre-existing point-sensitivity records.
        """
        cur_pos = pos_info.get("position", 0)
        if cur_pos == 0:
            return 0.0

        entry_p = float(pos_info.get("entry_price", eval_price))
        clean_pair = pair.upper().replace("_", "").replace("=X", "")
        method = pos_info.get("accounting_method")

        # Backward compatibility for legacy simulated USD/JPY records
        is_legacy = (
            method == "LEGACY_SENSITIVITY" or
            ("base_units" not in pos_info and "JPY" in clean_pair and float(pos_info.get("position_size", 0.0)) < 1000.0)
        )

        if is_legacy:
            sens = float(pos_info.get("position_size", 0.0))
            return (eval_price - entry_p) * sens if cur_pos == 1 else (entry_p - eval_price) * sens

        # Standard Spot Forex Accounting
        units = float(pos_info.get("base_units", pos_info.get("position_size", 0.0)))
        is_jpy = ("JPY" in clean_pair or clean_pair.startswith("USD")) and not clean_pair.endswith("USD")

        if is_jpy:
            # P&L_USD = Base_Units * (eval - entry) / eval
            denom = max(eval_price, 1e-4)
            return (units * (eval_price - entry_p) / denom) if cur_pos == 1 else (units * (entry_p - eval_price) / denom)
        else:
            # P&L_USD = Base_Units * (eval - entry)
            return (units * (eval_price - entry_p)) if cur_pos == 1 else (units * (entry_p - eval_price))

    def _save_ledger(self) -> None:
        """Atomic write to prevent corruption from ungraceful process interruption."""
        tmp_path = self.ledger_path.with_suffix(".tmp")
        with open(tmp_path, "w") as f:
            json.dump(self.ledger, f, indent=2)
        tmp_path.replace(self.ledger_path)

    def reconcile_with_broker(self) -> None:
        """Non-destructive reconciliation of active positions against broker open trades."""
        if not hasattr(self, "gateway") or not self.gateway:
            return

        try:
            b_trades = self.gateway.get_open_trades() if hasattr(self.gateway, "get_open_trades") else []
            b_trades_by_sym = {t.get("symbol", "").upper(): t for t in b_trades}

            for pair, pinfo in self.ledger.get("positions", {}).items():
                cur_pos = pinfo.get("position", 0)
                ticket = str(pinfo.get("broker_ticket", "")).strip()

                if cur_pos != 0:
                    clean_p = pair.upper()
                    if ticket and ticket not in ["N/A", "LOCAL", "None", ""]:
                        # Position was submitted to broker
                        if clean_p in b_trades_by_sym:
                            b_t = b_trades_by_sym[clean_p]
                            pinfo["execution_type"] = "BROKER_CONFIRMED"
                            pinfo["broker_ticket"] = str(b_t.get("id", ticket))
                            if "unrealized_pl" in b_t:
                                pinfo["broker_unrealized_pnl"] = float(b_t["unrealized_pl"])
                        else:
                            # Trade was closed at the broker (SL/TP filled or closeout)
                            pnl = float(pinfo.get("unrealized_pnl", 0.0))
                            closed_trade = {
                                "pair": pair,
                                "direction": "LONG" if cur_pos == 1 else "SHORT",
                                "entry_price": pinfo.get("entry_price", 0.0),
                                "exit_price": pinfo.get("stop_loss", pinfo.get("current_price", 0.0)),
                                "exit_reason": "BROKER_FILL (SL/TP/CLOSE)",
                                "position_size": pinfo.get("position_size", 0.0),
                                "net_pnl": pnl,
                                "duration_hours": pinfo.get("time_in_trade", 0),
                                "exit_time": datetime.now(timezone.utc).isoformat(),
                                "broker_ticket": ticket,
                            }
                            self.ledger["closed_trades"].append(closed_trade)
                            self.ledger["positions"][pair] = self._empty_position()
                    else:
                        pinfo["execution_type"] = "SIMULATED"
                else:
                    pinfo["execution_type"] = "FLAT"

            # Sync broker equity without overwriting local paper balance
            acc_info = self.gateway.get_account_summary()
            if not acc_info.get("error") and "balance" in acc_info:
                b_bal = float(acc_info["balance"])
                if b_bal > 0 and acc_info.get("broker") != "MOCK_SIMULATOR":
                    self.ledger["broker_equity"] = float(acc_info.get("equity", b_bal))

            self._save_ledger()
        except Exception as e:
            print(f"[RECONCILIATION] Warning during broker reconciliation: {e}")

    def fetch_live_data(self, pair: str) -> pd.DataFrame:
        ticker_sym = PAIR_TICKER_MAP.get(pair, f"{pair}=X")
        df = pd.DataFrame()
        for attempt in range(3):
            try:
                ticker = yf.Ticker(ticker_sym)
                df = ticker.history(period="15d", interval="1h")
                if len(df) >= self.window_size + 50:
                    break
            except Exception:
                pass
            time.sleep(2)

        if len(df) < self.window_size + 50:
            raise RuntimeError(f"Not enough recent data fetched for {ticker_sym}: got {len(df)} bars")

        df = df.reset_index()
        date_col = "Datetime" if "Datetime" in df.columns else "Date"
        df["date"] = pd.to_datetime(df[date_col])
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

        recent_df = df.iloc[-self.window_size:].copy()
        current_price = float(recent_df["close"].iloc[-1])
        bar_high = float(recent_df["high"].iloc[-1])
        bar_low = float(recent_df["low"].iloc[-1])
        current_atr = float(raw_atr_series.iloc[-1])

        return recent_df[feature_cols].values, current_price, current_atr, bar_high, bar_low

    def evaluate_pair(self, pair: str) -> Dict:
        df = self.fetch_live_data(pair)
        market_feats, current_price, current_atr, bar_high, bar_low = self.compute_features(df)

        scaler = self.scalers[pair]
        cols = getattr(scaler, "feature_names_in_", None)
        if cols is not None:
            scaled_market_feats = scaler.transform(pd.DataFrame(market_feats, columns=cols))
        else:
            scaled_market_feats = scaler.transform(market_feats)

        pos_info = self.ledger["positions"][pair]
        cur_pos = pos_info["position"]

        if cur_pos != 0:
            unrealized_pnl_ratio = (
                ((current_price - pos_info["entry_price"]) / max(pos_info["entry_atr"] * 2.0, 1e-5))
                if cur_pos == 1
                else ((pos_info["entry_price"] - current_price) / max(pos_info["entry_atr"] * 2.0, 1e-5))
            )
            time_in_trade_ratio = min(pos_info["time_in_trade"] / 72.0, 1.0)
            inactivity_ratio = 0.0
        else:
            unrealized_pnl_ratio = 0.0
            time_in_trade_ratio = 0.0
            inactivity_ratio = min(pos_info["consecutive_holds"] / 50.0, 1.0)

        portfolio_features = np.array([
            float(cur_pos),
            float(unrealized_pnl_ratio),
            float(time_in_trade_ratio),
            float(inactivity_ratio),
        ], dtype=np.float32)

        portfolio_features_tiled = np.tile(portfolio_features, (self.window_size, 1))
        full_features = np.hstack([scaled_market_feats, portfolio_features_tiled])

        state_tensor = torch.tensor(full_features, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            probs, _, _ = self.models[pair](state_tensor)
            probs = probs.squeeze(0).cpu().numpy()

        action = int(np.argmax(probs))
        action_names = {0: "HOLD / FLAT", 1: "BUY / LONG", 2: "SELL / SHORT"}
        order_event = None

        # 1. Manage Active Open Positions
        if cur_pos != 0:
            pos_info["time_in_trade"] += 1
            entry_price = float(pos_info["entry_price"])
            sl = float(pos_info["stop_loss"])
            tp = float(pos_info["take_profit"])

            sl_hit = (cur_pos == 1 and bar_low <= sl) or (cur_pos == -1 and bar_high >= sl)
            tp_hit = (cur_pos == 1 and bar_high >= tp) or (cur_pos == -1 and bar_low <= tp)

            exit_reason = None
            exit_price = current_price

            if sl_hit:
                exit_reason = "STOP-LOSS HIT"
                exit_price = sl
            elif tp_hit:
                exit_reason = "TAKE-PROFIT HIT"
                exit_price = tp
            elif action == 1 and cur_pos == -1:
                exit_reason = "SIGNAL REVERSAL (Cover Short)"
                exit_price = current_price
            elif action == 2 and cur_pos == 1:
                exit_reason = "SIGNAL REVERSAL (Close Long)"
                exit_price = current_price
            elif pos_info["time_in_trade"] >= 72:
                exit_reason = "MAX HOLD EXPIRED"
                exit_price = current_price

            if exit_reason:
                pos_size = float(pos_info.get("position_size", 0.0))
                gross_pnl = self.calculate_position_pnl(pair, pos_info, exit_price)
                fee = 0.50
                net_pnl = gross_pnl - fee
                self.ledger["balance"] += net_pnl

                closed_trade = {
                    "pair": pair,
                    "direction": "LONG" if cur_pos == 1 else "SHORT",
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "exit_reason": exit_reason,
                    "position_size": pos_size,
                    "base_units": pos_info.get("base_units", pos_size),
                    "net_pnl": net_pnl,
                    "duration_hours": pos_info["time_in_trade"],
                    "exit_time": datetime.now(timezone.utc).isoformat(),
                }
                self.ledger["closed_trades"].append(closed_trade)
                order_event = f"CLOSED {closed_trade['direction']} via {exit_reason} | P&L: ${net_pnl:+.2f}"

                if hasattr(self, "notifier"):
                    try:
                        self.notifier.send_alert(
                            title=f"🎯 Trade Closed: {pair} ({closed_trade['direction']})",
                            description=f"**Exit Reason**: {exit_reason}\n**Exit Price**: {exit_price:.5f}\n**Net P&L**: ${net_pnl:+.2f}\n**Duration**: {pos_info['time_in_trade']}h",
                            color_hex=0x22C55E if net_pnl >= 0 else 0xEF4444
                        )
                    except Exception:
                        pass

                # Only attempt broker closeout if the position was genuinely executed on broker
                ticket = str(pos_info.get("broker_ticket", "")).strip()
                if ticket and ticket not in ["N/A", "LOCAL", "None", ""]:
                    if hasattr(self, "gateway") and self.gateway:
                        try:
                            self.gateway.close_position(pair)
                        except Exception as e:
                            print(f"Gateway close error for {pair}: {e}")

                self.ledger["positions"][pair] = self._empty_position()
                pos_info = self.ledger["positions"][pair]
                cur_pos = 0

        # 2. Open New Position if Flat
        if cur_pos == 0:
            if action in [1, 2]:
                # Execution & Risk Guards
                cb_ok, cb_dd, cb_msg = self.circuit_breaker.check(self.ledger["equity"])
                news_blocked, news_reason = self.news_shield.is_blackout(pair)
                spread_ok, cur_sp, max_sp, sp_msg = self.spread_guard.check(self.gateway, pair)

                if not cb_ok:
                    order_event = f"BLOCKED BY RISK GUARD: {cb_msg}"
                elif news_blocked:
                    order_event = f"BLOCKED BY NEWS SHIELD: {news_reason}"
                elif not spread_ok:
                    order_event = f"BLOCKED BY SPREAD GUARD: {sp_msg}"
                else:
                    sizing = self.calculate_position_sizing(
                        pair=pair,
                        action=action,
                        balance=self.ledger["balance"],
                        risk_per_trade=self.risk_per_trade,
                        current_price=current_price,
                        current_atr=current_atr,
                    )

                    pos_info["position"] = 1 if action == 1 else -1
                    pos_info["entry_price"] = current_price
                    pos_info["entry_atr"] = current_atr
                    pos_info["base_units"] = sizing["base_units"]
                    pos_info["position_size"] = sizing["base_units"]
                    pos_info["point_sensitivity"] = sizing["point_sensitivity"]
                    pos_info["risk_budget_usd"] = sizing["risk_budget_usd"]
                    pos_info["accounting_method"] = sizing["accounting_method"]
                    pos_info["stop_loss"] = sizing["stop_loss"]
                    pos_info["take_profit"] = sizing["take_profit"]
                    pos_info["entry_time"] = datetime.now(timezone.utc).isoformat()
                    pos_info["time_in_trade"] = 0
                    pos_info["consecutive_holds"] = 0

                    if action == 1:
                        order_event = f"OPENED LONG @ {current_price:.5f} | Units: {sizing['base_units']:,} | SL: {pos_info['stop_loss']:.5f} | TP: {pos_info['take_profit']:.5f}"
                    else:
                        order_event = f"OPENED SHORT @ {current_price:.5f} | Units: {sizing['base_units']:,} | SL: {pos_info['stop_loss']:.5f} | TP: {pos_info['take_profit']:.5f}"

                    pos_info["execution_type"] = "SIMULATED"
                    if hasattr(self, "gateway") and self.gateway:
                        try:
                            units = sizing["order_units"]
                            b_resp = self.gateway.place_market_bracket_order(
                                symbol=pair,
                                units=units,
                                stop_loss=pos_info["stop_loss"],
                                take_profit=pos_info["take_profit"],
                            )
                            if b_resp.get("status") == "SUCCESS":
                                pos_info["broker_ticket"] = str(b_resp.get("ticket", "LOCAL"))
                                pos_info["execution_type"] = "BROKER_CONFIRMED"
                                if b_resp.get("fill_price"):
                                    pos_info["entry_price"] = float(b_resp["fill_price"])
                            else:
                                pos_info["broker_ticket"] = "LOCAL"

                            if "message" in b_resp:
                                order_event += f" | Gateway: {b_resp['message']}"

                            self.notifier.send_alert(
                                title=f"🚀 Order Filled: {pair} ({'LONG' if action == 1 else 'SHORT'})",
                                description=f"**Ticket**: #{pos_info.get('broker_ticket')}\n**Units**: {units:,}\n**Entry**: {current_price:.5f}\n**SL**: {pos_info['stop_loss']:.5f}\n**TP**: {pos_info['take_profit']:.5f}",
                                color_hex=0x22C55E
                            )
                        except Exception as e:
                            print(f"Gateway order error for {pair}: {e}")
            else:
                pos_info["consecutive_holds"] += 1

        pos_info["current_price"] = current_price
        pos_info["current_atr"] = current_atr
        pos_info["latest_signal"] = action_names[action]
        pos_info["probs"] = probs.tolist()
        if hasattr(self, "gateway") and self.gateway and hasattr(self.gateway, "get_realtime_spread"):
            pos_info["live_spread"] = self.gateway.get_realtime_spread(pair)
        if pos_info["position"] != 0:
            pos_info["unrealized_pnl"] = self.calculate_position_pnl(pair, pos_info, current_price)
        else:
            pos_info["unrealized_pnl"] = 0.0

        return {
            "pair": pair,
            "price": current_price,
            "atr": current_atr,
            "action": action_names[action],
            "probs": probs.tolist(),
            "position": "FLAT" if pos_info["position"] == 0 else ("LONG" if pos_info["position"] == 1 else "SHORT"),
            "event": order_event,
            "pos_info": pos_info,
        }

    def run_cycle(self) -> None:
        print("\n" + "=" * 76)
        print(f"MULTI-PAIR DIVERSIFIED PAPER TRADER ({datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')})")
        print("=" * 76)

        # Pre-cycle broker reconciliation check
        self.reconcile_with_broker()

        results = []
        for pair in self.pairs:
            try:
                res = self.evaluate_pair(pair)
                results.append(res)
            except Exception as e:
                print(f"Error evaluating {pair}: {e}")

        # Compute aggregate equity
        total_unrealized = 0.0
        for pair, pinfo in self.ledger["positions"].items():
            if pinfo["position"] != 0:
                # Find current price from results
                p_match = next((r["price"] for r in results if r["pair"] == pair), pinfo["entry_price"])
                pnl = self.calculate_position_pnl(pair, pinfo, p_match)
                total_unrealized += pnl

        self.ledger["equity"] = self.ledger["balance"] + total_unrealized
        if "equity_history" not in self.ledger:
            self.ledger["equity_history"] = []
        self.ledger["equity_history"].append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "equity": round(self.ledger["equity"], 2),
            "balance": round(self.ledger["balance"], 2),
            "unrealized_pnl": round(total_unrealized, 2),
        })
        if len(self.ledger["equity_history"]) > 500:
            self.ledger["equity_history"] = self.ledger["equity_history"][-500:]
        self._save_ledger()

        # Display Portfolio Dashboard
        for r in results:
            pos = r["position"]
            pinfo = r["pos_info"]
            print(f"[{r['pair']}] Price: {r['price']:.5f} (ATR: {r['atr']:.5f}) | Signal: {r['action']} ({r['probs'][1]*100:.0f}% L / {r['probs'][2]*100:.0f}% S)")
            if pos != "FLAT":
                unrealized = (r["price"] - pinfo["entry_price"]) * pinfo["position_size"] if pos == "LONG" else (pinfo["entry_price"] - r["price"]) * pinfo["position_size"]
                print(f"   ↳ Position: {pos} @ {pinfo['entry_price']:.5f} | SL: {pinfo['stop_loss']:.5f} | TP: {pinfo['take_profit']:.5f} | P&L: ${unrealized:+.2f} ({pinfo['time_in_trade']}h)")
            else:
                print(f"   ↳ Position: FLAT")
            if r["event"]:
                print(f"   👉 Order Event: {r['event']}")

        print("-" * 76)
        print(f"PORTFOLIO CASH BALANCE:   ${self.ledger['balance']:.2f}")
        print(f"UNREALIZED P&L:          ${total_unrealized:+.2f}")
        base_cap = self.ledger.get("initial_capital", self.initial_balance)
        print(f"TOTAL PORTFOLIO EQUITY:  ${self.ledger['equity']:.2f} ({(self.ledger['equity']/base_cap - 1.0)*100:+.2f}%)")
        print(f"TOTAL CLOSED TRADES:     {len(self.ledger['closed_trades'])}")
        print("=" * 76)

    def run_continuous(self) -> None:
        print("\nStarting Autonomous Paper Trading Daemon for 4-Pair Super-Ensemble...")
        print("Runs on hourly candle close. Press Ctrl+C to terminate.")
        while True:
            try:
                self.run_cycle()
            except Exception as e:
                print(f"Error during evaluation cycle: {e}")

            # Sleep until 1 minute past next hour (e.g. 14:01 UTC)
            now = datetime.now(timezone.utc)
            seconds_to_next_hour = (59 - now.minute) * 60 + (60 - now.second) + 60
            next_time = datetime.fromtimestamp(now.timestamp() + seconds_to_next_hour, tz=timezone.utc)
            print(f"\nSleeping until next candle close: {next_time.strftime('%Y-%m-%d %H:%M:%S UTC')} ({seconds_to_next_hour // 60}m {seconds_to_next_hour % 60}s remaining)...")
            try:
                time.sleep(seconds_to_next_hour)
            except KeyboardInterrupt:
                print("\nAutonomous paper trading daemon stopped by user.")
                break


def main():
    parser = argparse.ArgumentParser(description="Multi-pair paper trader")
    parser.add_argument("--pairs", nargs="+", default=["EURUSD", "USDJPY", "AUDUSD", "GBPUSD"], help="Pairs to trade")
    parser.add_argument("--ledger", default=DEFAULT_PORTFOLIO_LEDGER, type=str, help="Ledger path")
    parser.add_argument("--balance", default=10_000.0, type=float, help="Initial balance")
    parser.add_argument("--risk", default=0.015, type=float, help="Risk per trade fraction (e.g. 0.015 = 1.5%%)")
    parser.add_argument("--broker-config", default="broker_config.json", type=str, help="Broker config file")
    parser.add_argument("--continuous", action="store_true", help="Run daemon continuously every hour at candle close")
    args = parser.parse_args()

    trader = MultiPairPaperTrader(
        pairs=args.pairs,
        ledger_path=args.ledger,
        initial_balance=args.balance,
        risk_per_trade=args.risk,
        broker_config_path=args.broker_config,
    )

    if args.continuous:
        trader.run_continuous()
    else:
        trader.run_cycle()


if __name__ == "__main__":
    main()
