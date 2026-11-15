"""
Market Context Builder for Forex AI Market Intelligence Agent.

Constructs sanitized, read-only market snapshots for local LLM evaluation.
Extracts existing indicators, signals, risk-guard results, and news context
without modifying persistent state or exposing credentials.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


class MarketContextBuilder:
    """
    Builds structured, sanitized read-only snapshots of current market state
    for consumption by the local advisory intelligence agent.
    """

    SUPPORTED_PAIRS = ("EURUSD", "USDJPY", "AUDUSD", "GBPUSD")

    def __init__(
        self,
        ledger_path: str = "portfolio_ledger.json",
        config_path: str = "broker_config.json",
        cache_path: str = "news_calendar_cache.json",
    ) -> None:
        self.ledger_path = Path(ledger_path)
        self.config_path = Path(config_path)
        self.cache_path = Path(cache_path)

    def _load_ledger(self) -> Dict[str, Any]:
        """Loads portfolio ledger safely without mutation."""
        if not self.ledger_path.exists():
            return {}
        try:
            with open(self.ledger_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}

    def _load_config(self) -> Dict[str, Any]:
        """Loads broker configuration safely without leaking tokens."""
        if not self.config_path.exists():
            return {}
        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}

    def _load_news_cache(self) -> Dict[str, Any]:
        """Loads economic calendar cache safely."""
        if not self.cache_path.exists():
            return {}
        try:
            with open(self.cache_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}

    def build_snapshot(self, pair: str) -> Dict[str, Any]:
        """
        Builds a comprehensive read-only snapshot for a specific currency pair.
        Excludes all secrets, tokens, account numbers, and unverified data.
        """
        clean_pair = pair.upper().replace("/", "").replace("_", "").replace("=X", "")
        if clean_pair not in self.SUPPORTED_PAIRS:
            raise ValueError(f"Unsupported pair '{pair}'. Supported: {self.SUPPORTED_PAIRS}")

        now_utc = datetime.now(timezone.utc).isoformat()
        ledger = self._load_ledger()
        config = self._load_config()
        news_cache = self._load_news_cache()

        # Extract positions and pair state
        positions = ledger.get("positions", {})
        pos_info = positions.get(clean_pair, {})

        pos_code = pos_info.get("position", 0)
        direction = "LONG" if pos_code == 1 else ("SHORT" if pos_code == -1 else "FLAT")
        exec_type = pos_info.get("execution_type", "FLAT")

        curr_price = float(pos_info.get("current_price", 0.0))
        entry_price = float(pos_info.get("entry_price", 0.0))
        entry_atr = float(pos_info.get("entry_atr", 0.0))
        current_atr = float(pos_info.get("current_atr", entry_atr))
        unrealized_pnl = float(pos_info.get("unrealized_pnl", 0.0))
        time_in_trade = int(pos_info.get("time_in_trade", 0))

        # Signals and probabilities
        signal = pos_info.get("latest_signal", "HOLD / FLAT")
        probs = pos_info.get("probs", [0.333, 0.333, 0.333])
        if len(probs) >= 3:
            prob_flat, prob_long, prob_short = float(probs[0]), float(probs[1]), float(probs[2])
        else:
            prob_flat, prob_long, prob_short = 0.333, 0.333, 0.333

        # Spread Guard context
        guards = config.get("guards", {})
        max_spreads = guards.get("max_spread_pips", {})
        max_spread_limit = float(max_spreads.get(clean_pair, 3.0))
        live_spread = float(pos_info.get("live_spread", 2.0))
        spread_blocked = (live_spread > max_spread_limit) and ("BUY" in signal or "SELL" in signal) and (pos_code == 0)

        # Drawdown and circuit breaker context
        initial_cap = float(ledger.get("initial_capital", 100000.0))
        current_eq = float(ledger.get("equity", initial_cap))
        max_dd_limit_pct = float(guards.get("daily_drawdown_limit_pct", 3.0))
        curr_dd_pct = ((initial_cap - current_eq) / initial_cap) * 100.0 if initial_cap > 0 else 0.0
        circuit_breaker_tripped = curr_dd_pct >= max_dd_limit_pct

        # NewsShield context
        news_prov = news_cache.get("provider", "mock")
        news_status = news_cache.get("status", "HEALTHY")
        last_news_refresh = news_cache.get("last_successful_refresh_utc")

        # Filter upcoming events affecting this pair
        base_ccy = clean_pair[:3]
        quote_ccy = clean_pair[3:]
        upcoming_events = []
        for ev in news_cache.get("events", []):
            ev_ccy = str(ev.get("currency", "")).upper()
            if ev_ccy == "USD" or ev_ccy in (base_ccy, quote_ccy):
                upcoming_events.append({
                    "id": ev.get("id"),
                    "title": ev.get("title"),
                    "currency": ev_ccy,
                    "impact": ev.get("impact"),
                    "time_utc": ev.get("time_utc"),
                    "estimate": ev.get("estimate"),
                    "previous": ev.get("previous"),
                })

        # Known limitations in data
        limitations = [
            "Signal probabilities are point-in-time softmax outputs from PyTorch policy model.",
            "Historical bar evaluation operates on 1-hour candle closes.",
        ]
        if news_prov == "mock":
            limitations.append("Economic calendar data is derived from structured macro schedule (Finnhub Free Tier fallback).")
        if spread_blocked:
            limitations.append(f"Live spread ({live_spread:.1f}p) exceeds safety threshold ({max_spread_limit:.1f}p). Trade entry currently prohibited.")

        return {
            "snapshot_timestamp_utc": now_utc,
            "pair": clean_pair,
            "market_data": {
                "current_price": curr_price,
                "current_atr": round(current_atr, 5),
                "live_spread_pips": round(live_spread, 1),
                "max_spread_limit_pips": max_spread_limit,
                "spread_status": "ELEVATED_BLOCKED" if spread_blocked else "NORMAL",
            },
            "engine_signal": {
                "signal": signal,
                "probabilities": {
                    "flat": round(prob_flat, 3),
                    "long": round(prob_long, 3),
                    "short": round(prob_short, 3),
                },
                "confidence_margin": round(abs(prob_long - prob_short), 3),
            },
            "position_state": {
                "direction": direction,
                "execution_type": exec_type,
                "has_active_position": (pos_code != 0),
                "entry_price": entry_price if pos_code != 0 else None,
                "stop_loss": float(pos_info.get("stop_loss", 0.0)) if pos_code != 0 else None,
                "take_profit": float(pos_info.get("take_profit", 0.0)) if pos_code != 0 else None,
                "unrealized_pnl_usd": round(unrealized_pnl, 2) if pos_code != 0 else 0.0,
                "duration_hours": time_in_trade if pos_code != 0 else 0,
            },
            "risk_guards": {
                "spread_guard": {
                    "passed": not spread_blocked,
                    "live_spread": live_spread,
                    "threshold": max_spread_limit,
                },
                "circuit_breaker": {
                    "tripped": circuit_breaker_tripped,
                    "current_drawdown_pct": round(curr_dd_pct, 2),
                    "threshold_pct": max_dd_limit_pct,
                },
                "news_shield": {
                    "provider": news_prov,
                    "cache_status": news_status,
                    "last_refresh_utc": last_news_refresh,
                    "events_affecting_pair_count": len(upcoming_events),
                    "upcoming_events": upcoming_events[:5],
                },
            },
            "portfolio_context": {
                "balance_usd": round(float(ledger.get("balance", 100000.0)), 2),
                "equity_usd": round(current_eq, 2),
                "open_positions_count": sum(1 for p, d in positions.items() if d.get("position", 0) != 0),
            },
            "data_quality": {
                "status": "VALID",
                "limitations": limitations,
            },
        }

    def build_all_snapshots(self) -> Dict[str, Dict[str, Any]]:
        """Builds snapshots for all 4 supported pairs."""
        return {pair: self.build_snapshot(pair) for pair in self.SUPPORTED_PAIRS}
