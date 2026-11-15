"""
Institutional Risk & Execution Guards for Forex AI Trading System.
Includes:
1. Real-Time Spread Spike Guard (prevents entries during wide broker spreads)
2. High-Impact Macro Economic News Shield (blocks entries around major volatile events)
3. Daily Max Drawdown Circuit Breaker (halts trading if daily drawdown limit reached)
4. Multi-Channel Webhook Notifier (Discord / Telegram alerts)
"""

from __future__ import annotations

import json
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

try:
    from news_client import (
        NewsCalendarClient,
        normalize_iso_utc,
        STATUS_HEALTHY,
        STATUS_STALE,
        STATUS_UNAVAILABLE,
        STATUS_ERROR,
        POLICY_BLOCK_NEW_ENTRIES,
        POLICY_ALLOW_WITH_SPREAD_GUARD,
        IMPACT_HIGH,
        DEFAULT_CACHE_FILE,
    )
except ImportError:
    STATUS_HEALTHY = "HEALTHY"
    STATUS_STALE = "STALE"
    STATUS_UNAVAILABLE = "UNAVAILABLE"
    STATUS_ERROR = "ERROR"
    POLICY_BLOCK_NEW_ENTRIES = "BLOCK_NEW_ENTRIES"
    POLICY_ALLOW_WITH_SPREAD_GUARD = "ALLOW_WITH_SPREAD_GUARD"
    IMPACT_HIGH = "HIGH"
    DEFAULT_CACHE_FILE = "news_calendar_cache.json"
    NewsCalendarClient = None  # type: ignore

DEFAULT_SPREAD_LIMITS = {
    "EURUSD": 2.5,  # Max 2.5 pips
    "GBPUSD": 3.0,  # Max 3.0 pips
    "USDJPY": 3.0,  # Max 3.0 pips
    "AUDUSD": 3.5,  # Max 3.5 pips
}

# Major recurring high-impact macro news keywords
HIGH_IMPACT_KEYWORDS = [
    "interest rate", "rate decision", "fomc", "fed rate", "ecb rate", "boe rate",
    "rba rate", "cpi", "inflation", "non-farm payroll", "nfp", "unemployment rate",
    "gdp", "fomc press conference", "powell", "lagarde", "bailey", "ueda",
]


class SpreadGuard:
    """Monitors live broker spreads and blocks execution during spread spikes."""

    def __init__(self, spread_limits: Optional[Dict[str, float]] = None) -> None:
        self.limits = spread_limits or DEFAULT_SPREAD_LIMITS

    def check(self, gateway, symbol: str) -> Tuple[bool, float, float, str]:
        """
        Check if the current broker spread is acceptable for entry.
        Returns: (is_allowed, current_spread_pips, max_limit_pips, reason)
        """
        sym = symbol.upper().replace("_", "").replace("=X", "")
        max_allowed = self.limits.get(sym, 3.5)

        # Query gateway for live spread
        if hasattr(gateway, "get_realtime_spread"):
            spread_pips = gateway.get_realtime_spread(sym)
        else:
            spread_pips = 1.0  # default assumption for mock gateway

        if spread_pips <= 0:
            return True, 0.0, max_allowed, "Normal"

        if spread_pips > max_allowed:
            reason = f"Spread Spike ({spread_pips:.1f} pips > {max_allowed:.1f} max limit)"
            return False, spread_pips, max_allowed, reason

        return True, spread_pips, max_allowed, f"Normal ({spread_pips:.1f} pips)"


class NewsShield:
    """
    Guards portfolio against entering new positions immediately before or after
    high-impact economic news releases.
    Uses cached calendar data to ensure zero runtime network latency.
    """

    def __init__(
        self,
        blackout_minutes_before: Optional[int] = None,
        blackout_minutes_after: Optional[int] = None,
        config_path: str = "broker_config.json",
        cache_path: str = DEFAULT_CACHE_FILE,
        client: Optional[Any] = None,
    ) -> None:
        self.config_path = Path(config_path)
        self.cache_path = Path(cache_path)
        self._load_config()

        if blackout_minutes_before is not None:
            self.before_delta = timedelta(minutes=blackout_minutes_before)
        if blackout_minutes_after is not None:
            self.after_delta = timedelta(minutes=blackout_minutes_after)

        if client is not None:
            self.client = client
        elif NewsCalendarClient is not None:
            self.client = NewsCalendarClient.from_config(config_path=str(self.config_path))
        else:
            self.client = None

        self.calendar_cache: List[Dict[str, Any]] = (
            self.client.get_events(only_upcoming=False) if self.client else []
        )
        self.last_fetch: Optional[datetime] = None

    def _load_config(self) -> None:
        self.before_delta = timedelta(minutes=30)
        self.after_delta = timedelta(minutes=15)
        self.fail_safe_policy = POLICY_BLOCK_NEW_ENTRIES

        if self.config_path.exists():
            try:
                with open(self.config_path, "r") as f:
                    cfg = json.load(f)
                guards = cfg.get("guards", {})
                ns_cfg = guards.get("news_shield", {})
                b_min = ns_cfg.get("blackout_minutes_before", 30)
                a_min = ns_cfg.get("blackout_minutes_after", 15)
                self.before_delta = timedelta(minutes=b_min)
                self.after_delta = timedelta(minutes=a_min)
                self.fail_safe_policy = ns_cfg.get("fail_safe_policy", POLICY_BLOCK_NEW_ENTRIES)
            except Exception:
                pass

    def check_blackout(
        self,
        symbol: str,
        current_time: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """
        Deterministic, zero-latency blackout evaluation for a currency pair.
        Returns a detailed diagnostics dictionary.
        """
        if current_time is not None:
            now = current_time if current_time.tzinfo else current_time.replace(tzinfo=timezone.utc)
        else:
            now = datetime.now(timezone.utc)

        sym = symbol.upper().replace("_", "").replace("=X", "")
        base_ccy = sym[:3]
        quote_ccy = sym[3:]

        # 1. Check client/cache health and fail-safe policy
        if self.client:
            status_info = self.client.get_status()
            status = status_info.get("status", STATUS_UNAVAILABLE)
            events = self.client.get_events(only_upcoming=False)
        else:
            status = STATUS_UNAVAILABLE
            events = self.calendar_cache

        if status != STATUS_HEALTHY:
            if self.fail_safe_policy == POLICY_BLOCK_NEW_ENTRIES:
                return {
                    "blocked": True,
                    "pair": sym,
                    "event": None,
                    "status": status,
                    "reason": f"Economic Calendar {status} (Fail-safe: BLOCK_NEW_ENTRIES)",
                    "message": f"Calendar data is {status}. Entries blocked per risk policy.",
                }

        # 2. Query events in blackout window
        for event in events:
            impact = str(event.get("impact", "")).upper()
            title = str(event.get("title", ""))
            ccy = str(event.get("currency", "")).upper()

            is_high_impact = (impact == IMPACT_HIGH) or any(kw in title.lower() for kw in HIGH_IMPACT_KEYWORDS)
            if not is_high_impact:
                continue

            # Currency mapping: USD affects all 4 pairs; otherwise base or quote currency must match
            affects_pair = (ccy == "USD") or (ccy in (base_ccy, quote_ccy))
            if not affects_pair:
                continue

            time_raw = event.get("time_utc") or event.get("time")
            if isinstance(time_raw, str):
                s = time_raw.strip()
                if s.endswith("Z"):
                    s = s[:-1] + "+00:00"
                try:
                    ev_time = datetime.fromisoformat(s)
                except Exception:
                    continue
            elif isinstance(time_raw, datetime):
                ev_time = time_raw
            else:
                continue

            if ev_time.tzinfo is None:
                ev_time = ev_time.replace(tzinfo=timezone.utc)
            else:
                ev_time = ev_time.astimezone(timezone.utc)

            window_start = ev_time - self.before_delta
            window_end = ev_time + self.after_delta

            if window_start <= now <= window_end:
                mins_to_event = int((ev_time - now).total_seconds() / 60)
                if mins_to_event > 0:
                    timing_desc = f"in {mins_to_event}m"
                elif mins_to_event == 0:
                    timing_desc = "now"
                else:
                    timing_desc = f"{abs(mins_to_event)}m ago"

                reason = f"High-Impact News: {title} ({ccy}) @ {ev_time.strftime('%H:%M UTC')} ({timing_desc})"
                return {
                    "blocked": True,
                    "pair": sym,
                    "event": event,
                    "status": status,
                    "reason": reason,
                    "message": f"Position entry blocked: {reason}",
                }

        return {
            "blocked": False,
            "pair": sym,
            "event": None,
            "status": status,
            "reason": "Clear (No High-Impact News)",
            "message": "Normal trading permitted.",
        }

    def is_blackout(self, symbol: str) -> Tuple[bool, str]:
        """Backward-compatible tuple interface for live execution engine."""
        res = self.check_blackout(symbol)
        return res["blocked"], res["reason"]


class DailyCircuitBreaker:
    """Enforces daily maximum drawdown limits to protect capital during outlier market shocks."""

    def __init__(self, max_daily_drawdown_pct: float = 3.0) -> None:
        self.max_dd_pct = max_daily_drawdown_pct
        self.daily_starting_equity: Optional[float] = None
        self.current_day: Optional[str] = None

    def check(self, current_equity: float) -> Tuple[bool, float, str]:
        today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if self.current_day != today_str:
            self.current_day = today_str
            self.daily_starting_equity = current_equity

        if not self.daily_starting_equity or self.daily_starting_equity <= 0:
            return True, 0.0, "OK"

        drawdown_pct = ((self.daily_starting_equity - current_equity) / self.daily_starting_equity) * 100.0

        if drawdown_pct >= self.max_dd_pct:
            return False, drawdown_pct, f"Circuit Breaker Tripped! Daily DD: {drawdown_pct:.2f}% (Limit: {self.max_dd_pct}%)"

        return True, drawdown_pct, f"Normal (Daily DD: {max(drawdown_pct, 0.0):.2f}%)"


class WebhookNotifier:
    """Sends rich formatted notifications to Discord or Telegram."""

    def __init__(self, config_path: str = "broker_config.json") -> None:
        self.config_path = Path(config_path)
        self.discord_url: Optional[str] = None
        self.telegram_token: Optional[str] = None
        self.telegram_chat_id: Optional[str] = None
        self._load_config()

    def _load_config(self) -> None:
        if self.config_path.exists():
            try:
                with open(self.config_path, "r") as f:
                    cfg = json.load(f)
                notifications = cfg.get("notifications", {})
                self.discord_url = notifications.get("discord_webhook_url")
                self.telegram_token = notifications.get("telegram_bot_token")
                self.telegram_chat_id = notifications.get("telegram_chat_id")
            except Exception:
                pass

    def send_alert(self, title: str, description: str, color_hex: int = 0x22C55E) -> None:
        """Send notification via configured channels."""
        self._load_config()

        # Discord Webhook
        if self.discord_url and self.discord_url.startswith("https://discord.com/api/webhooks"):
            payload = {
                "embeds": [{
                    "title": title,
                    "description": description,
                    "color": color_hex,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "footer": {"text": "Forex AI 4-Pair Super-Ensemble"}
                }]
            }
            try:
                requests.post(self.discord_url, json=payload, timeout=5)
            except Exception as e:
                print(f"Failed to deliver Discord alert: {e}")

        # Telegram Bot
        if self.telegram_token and self.telegram_chat_id:
            tg_url = f"https://api.telegram.org/bot{self.telegram_token}/sendMessage"
            text = f"*{title}*\n{description}"
            try:
                requests.post(tg_url, json={"chat_id": self.telegram_chat_id, "text": text, "parse_mode": "Markdown"}, timeout=5)
            except Exception as e:
                print(f"Failed to deliver Telegram alert: {e}")
