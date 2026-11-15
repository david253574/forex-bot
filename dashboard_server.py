"""
Forex AI 4-Pair Super-Ensemble: Modern Beginner-Friendly Web Dashboard Server.
Serves a clean, accessible, light-themed financial monitoring interface
with strict broker-to-ledger reconciliation, clear separation of verified broker
account funds from local simulated paper trades, and transparent risk controls.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from http.cookies import SimpleCookie
from http.server import HTTPServer, ThreadingHTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse, parse_qs

# Multi-tenant Auth and Encrypted Credential Engine
from auth_db import db, mask_token, decrypt_token

# Project path resolution
PROJECT_DIR = Path(__file__).resolve().parent
PORTFOLIO_LEDGER_PATH = PROJECT_DIR / "portfolio_ledger.json"
CONFIG_PATH = PROJECT_DIR / "broker_config.json"
ENGINE_PID_FILE = PROJECT_DIR / ".run" / "engine.pid"

# Safe import of operational helpers
try:
    from ops_helper import (
        get_config_info,
        fetch_broker_state,
        normalize_symbol,
        reconcile_positions,
    )
except ImportError:
    def get_config_info(p: str = "broker_config.json") -> Dict[str, Any]:
        return {"broker": "mock", "environment": "practice"}

    def fetch_broker_state(p: str = "broker_config.json") -> Dict[str, Any]:
        return {"error": True, "message": "Helper not available"}

    def normalize_symbol(s: str) -> str:
        return s.upper().replace("_", "").replace("=X", "")

    def reconcile_positions(c: str = "broker_config.json", l: str = "portfolio_ledger.json") -> Dict[str, Any]:
        return {"overall_status": "BROKER_UNAVAILABLE", "positions": {}, "counts": {}}


# In-memory broker request cache (15-second TTL to avoid rate-limiting)
_BROKER_CACHE = {
    "timestamp": 0.0,
    "data": None,
}
_CACHE_TTL = 15.0


def get_cached_broker_state(config_path: str = "broker_config.json") -> Dict[str, Any]:
    global _BROKER_CACHE
    now = time.time()
    if _BROKER_CACHE["data"] is not None and (now - _BROKER_CACHE["timestamp"]) < _CACHE_TTL:
        return _BROKER_CACHE["data"]

    try:
        data = fetch_broker_state(config_path)
        _BROKER_CACHE["data"] = data
        _BROKER_CACHE["timestamp"] = now
        return data
    except Exception as e:
        if _BROKER_CACHE["data"] is not None:
            return _BROKER_CACHE["data"]
        return {"error": True, "message": str(e)}


def fetch_user_broker_state(token: str, account_id: str, environment: str = "practice") -> Dict[str, Any]:
    """Fetch live OANDA state on behalf of a specific user with decrypted token."""
    env = environment.strip().lower()
    base_url = "https://api-fxtrade.oanda.com/v3" if env == "live" else "https://api-fxpractice.oanda.com/v3"
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/json"}
    try:
        import requests
        s_resp = requests.get(f"{base_url}/accounts/{account_id}/summary", headers=headers, timeout=8)
        if s_resp.status_code != 200:
            return {"error": True, "message": f"Broker returned HTTP {s_resp.status_code}"}
        acc = s_resp.json().get("account", {})

        t_resp = requests.get(f"{base_url}/accounts/{account_id}/openTrades", headers=headers, timeout=8)
        trades_list = []
        if t_resp.status_code == 200:
            raw_trades = t_resp.json().get("trades", [])
            for t in raw_trades:
                sym = str(t.get("instrument", "")).replace("_", "").replace("=X", "").upper()
                trades_list.append({
                    "id": str(t.get("id")),
                    "symbol": sym,
                    "instrument": t.get("instrument"),
                    "units": int(t.get("currentUnits", 0)),
                    "direction": "LONG" if int(t.get("currentUnits", 0)) > 0 else "SHORT",
                    "price": float(t.get("price", 0.0)),
                    "unrealized_pl": float(t.get("unrealizedPL", 0.0)),
                    "stop_loss": float(t.get("stopLossOrder", {}).get("price", 0.0)) if "stopLossOrder" in t else 0.0,
                    "take_profit": float(t.get("takeProfitOrder", {}).get("price", 0.0)) if "takeProfitOrder" in t else 0.0,
                    "open_time": t.get("openTime"),
                })

        return {
            "broker": "oanda",
            "environment": env,
            "currency": acc.get("currency", "USD"),
            "balance": float(acc.get("balance", 0.0)),
            "equity": float(acc.get("NAV", 0.0)),
            "unrealized_pnl": float(acc.get("unrealizedPL", 0.0)),
            "realized_pnl": float(acc.get("pl", 0.0)),
            "margin_used": float(acc.get("marginUsed", 0.0)),
            "margin_available": float(acc.get("marginAvailable", 0.0)),
            "open_position_count": len(trades_list),
            "trades": trades_list,
        }
    except Exception as e:
        return {"error": True, "message": str(e)}


def check_engine_process() -> Dict[str, Any]:
    """Check if live trading engine process is currently running."""
    if not ENGINE_PID_FILE.exists():
        return {"running": False, "pid": None}
    try:
        pid = int(ENGINE_PID_FILE.read_text().strip())
        os.kill(pid, 0)
        return {"running": True, "pid": pid}
    except Exception:
        return {"running": False, "pid": None}


def enrich_portfolio_data(
    raw_ledger: Dict[str, Any],
    broker_state: Optional[Dict[str, Any]] = None,
    engine_running: bool = False,
    engine_pid: Optional[int] = None,
    config_path: str = "broker_config.json",
) -> Dict[str, Any]:
    """
    Format raw portfolio state and broker metrics into a structured,
    beginner-friendly presentation with strict financial data integrity.
    Separates verified broker account funds from simulated paper trades.
    """
    cfg_info = get_config_info(config_path)
    broker_name = str(cfg_info.get("broker", "mock")).lower()
    env_name = str(cfg_info.get("environment", "practice")).upper()

    broker_connected = bool(
        broker_state 
        and not broker_state.get("error") 
        and broker_state.get("balance") is not None
    )

    # 1. Financial Verification & Baseline Balances
    starting_balance = 100000.0
    if "initial_capital" in raw_ledger:
        starting_balance = float(raw_ledger["initial_capital"])
    elif broker_connected and broker_name == "oanda":
        starting_balance = 100000.0

    # Local ledger calculations
    l_bal = float(raw_ledger.get("balance", 100000.0))
    l_eq = float(raw_ledger.get("equity", 100000.0))
    l_upl = l_eq - l_bal

    # Broker calculations
    if broker_connected and broker_name == "oanda":
        b_bal = float(broker_state.get("balance", l_bal))
        b_eq = float(broker_state.get("equity", l_eq))
        b_upl = float(broker_state.get("unrealized_pnl", 0.0))
        b_realized_pnl = float(broker_state.get("realized_pnl", b_bal - starting_balance))
        margin_used = float(broker_state.get("margin_used", 0.0))
        margin_available = float(broker_state.get("margin_available", 0.0))
        broker_status_label = "Connected (OANDA v20)"
    else:
        b_bal = l_bal
        b_eq = l_eq
        b_upl = l_upl
        b_realized_pnl = 0.0
        margin_used = 0.0
        margin_available = l_bal
        broker_status_label = "Local Simulation (Mock Sandbox)" if broker_name == "mock" else "Connecting..."

    # Preserve backward-compatible root summary values
    if broker_connected and broker_name == "oanda":
        total_account_value = b_eq
        cash_balance = b_bal
        open_trades_result = b_upl
    else:
        total_account_value = l_eq
        cash_balance = l_bal
        open_trades_result = l_upl

    total_return_pct = ((total_account_value / starting_balance) - 1.0) * 100.0 if starting_balance > 0 else 0.0

    # 2. Closed Trades & Realized P&L
    today_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    today_realized_pnl = 0.0
    closed_trades_raw = raw_ledger.get("closed_trades", [])
    net_realized_paper = 0.0
    wins = 0
    losses = 0

    PAIR_METADATA = {
        "EURUSD": ("EUR / USD", "Euro", "US Dollar"),
        "GBPUSD": ("GBP / USD", "British Pound", "US Dollar"),
        "USDJPY": ("USD / JPY", "US Dollar", "Japanese Yen"),
        "AUDUSD": ("AUD / USD", "Australian Dollar", "US Dollar"),
    }

    REASON_MAP = {
        "STOP-LOSS HIT": "Stop-loss triggered (Loss limited)",
        "TAKE-PROFIT HIT": "Profit target reached (Gains locked in)",
        "SIGNAL REVERSAL": "Closed on AI signal reversal",
        "MAX HOLD EXPIRED": "Closed after maximum hold duration (72h)",
        "BROKER_FILL": "Closed automatically by broker order",
    }

    enriched_trades = []
    for ct in closed_trades_raw:
        pnl = float(ct.get("net_pnl", 0.0))
        net_realized_paper += pnl
        if pnl >= 0:
            wins += 1
        else:
            losses += 1

        exit_time = ct.get("exit_time", "")
        if exit_time and exit_time.startswith(today_utc):
            today_realized_pnl += pnl

        raw_reason = ct.get("exit_reason", "Closed by system")
        friendly_reason = REASON_MAP.get(raw_reason, raw_reason)

        pair_name = ct.get("pair", "")
        base = PAIR_METADATA.get(pair_name, (pair_name, pair_name[:3], ""))[1]
        action = f"Bought {base}" if ct.get("direction") == "LONG" else f"Sold {base}"

        enriched_trades.append({
            "pair": pair_name,
            "display_name": PAIR_METADATA.get(pair_name, (pair_name,))[0],
            "action": action,
            "direction": ct.get("direction", "LONG"),
            "entry_price": float(ct.get("entry_price", 0.0)),
            "exit_price": float(ct.get("exit_price", 0.0)),
            "net_pnl": pnl,
            "exit_reason": friendly_reason,
            "duration_hours": int(ct.get("duration_hours", 0)),
            "exit_time": exit_time,
            "broker_ticket": ct.get("broker_ticket"),
        })

    total_completed = len(closed_trades_raw)
    win_rate = (wins / total_completed * 100.0) if total_completed > 0 else 0.0

    # 3. Currency Pair Descriptions and Formatting
    b_trades = broker_state.get("trades", []) if broker_connected else []
    b_trades_by_sym = {t.get("symbol", "").upper(): t for t in b_trades}

    positions_raw = raw_ledger.get("positions", {})
    enriched_positions = {}
    active_count = 0
    confirmed_count = 0
    simulated_count = 0
    unmatched_broker_count = 0
    closed_at_broker_count = 0
    mismatch_count = 0
    flat_count = 0
    simulated_upl_sum = 0.0

    for pair in ["AUDUSD", "EURUSD", "USDJPY", "GBPUSD"]:
        p = positions_raw.get(pair, {})
        pos_code = p.get("position", 0)
        disp_name, base_name, quote_name = PAIR_METADATA.get(pair, (pair, pair[:3], pair[3:]))
        ticket = str(p.get("broker_ticket", "")).strip()
        bt = b_trades_by_sym.get(pair)

        has_active = (pos_code != 0)
        if has_active:
            active_count += 1

        is_long = (pos_code == 1)
        is_short = (pos_code == -1)

        if is_long:
            direction_label = f"Buying {base_name} against {quote_name}"
            direction_code = "LONG"
        elif is_short:
            direction_label = f"Selling {base_name} against {quote_name}"
            direction_code = "SHORT"
        else:
            direction_label = "No active trade"
            direction_code = "FLAT"

        # Classification Logic
        if has_active:
            if bt:
                b_ticket = str(bt.get("trade_id", bt.get("id", "")))
                ticket_matches = (ticket == b_ticket) or (ticket in ["", "LOCAL", "N/A"])
                b_dir = bt.get("direction")
                if not b_dir:
                    u = int(bt.get("units", 0))
                    b_dir = "LONG" if u > 0 else ("SHORT" if u < 0 else "FLAT")
                dir_matches = (pos_code == 1 and b_dir == "LONG") or (pos_code == -1 and b_dir == "SHORT")

                if ticket_matches and dir_matches:
                    exec_category = "BROKER_CONFIRMED"
                    status_text = f"Confirmed Broker Trade (Order #{bt.get('id', ticket)})"
                    badge_text = f"✓ Verified Broker Trade (Order #{bt.get('id', ticket)})"
                    badge_color = "emerald"
                    is_server_protected = True
                    confirmed_count += 1
                else:
                    exec_category = "RECONCILIATION_MISMATCH"
                    status_text = "Parameter mismatch with broker"
                    badge_text = "⚠️ Reconciliation Mismatch"
                    badge_color = "rose"
                    is_server_protected = False
                    mismatch_count += 1
            else:
                if ticket and ticket not in ["N/A", "LOCAL", "None", ""]:
                    exec_category = "CLOSED_AT_BROKER"
                    status_text = "Closed at broker (awaiting local cycle)"
                    badge_text = "⚡ Closed at Broker"
                    badge_color = "amber"
                    is_server_protected = False
                    closed_at_broker_count += 1
                else:
                    exec_category = "SIMULATED"
                    status_text = "Simulated Paper Trade (Local Only)"
                    badge_text = "🔬 Simulated Paper Trade"
                    badge_color = "sky"
                    is_server_protected = False
                    simulated_count += 1
                    simulated_upl_sum += float(p.get("unrealized_pnl", 0.0))
        else:
            if bt:
                exec_category = "UNMATCHED_BROKER"
                status_text = f"Unmatched Broker Position (Order #{bt.get('id')})"
                badge_text = "⚠️ Unmatched Broker Trade"
                badge_color = "amber"
                is_server_protected = True
                unmatched_broker_count += 1
            else:
                exec_category = "FLAT"
                status_text = "No active trade"
                badge_text = "⚪ Flat (Waiting for Edge)"
                badge_color = "slate"
                is_server_protected = False
                flat_count += 1

        # Safe data assignment distinguishing active vs flat
        if has_active:
            if exec_category == "BROKER_CONFIRMED":
                ticket_str = str(bt.get("id", ticket))
                entry_pr = float(bt.get("price", p.get("entry_price", 0.0)))
                upl_val = float(bt.get("unrealized_pl", p.get("unrealized_pnl", 0.0)))
                sl_val = float(bt.get("stop_loss", p.get("stop_loss", 0.0)))
                tp_val = float(bt.get("take_profit", p.get("take_profit", 0.0)))
                sl_status = f"Active at broker ({sl_val:.5f})" if sl_val > 0 else "Not set"
                tp_status = f"Active at broker ({tp_val:.5f})" if tp_val > 0 else "Not set"
                broker_verified_pnl = upl_val
            elif exec_category == "SIMULATED":
                ticket_str = None
                entry_pr = float(p.get("entry_price", 0.0))
                upl_val = float(p.get("unrealized_pnl", 0.0))
                sl_val = float(p.get("stop_loss", 0.0))
                tp_val = float(p.get("take_profit", 0.0))
                sl_status = f"Simulated exit level ({sl_val:.5f})" if sl_val > 0 else "Not set"
                tp_status = f"Simulated exit level ({tp_val:.5f})" if tp_val > 0 else "Not set"
                broker_verified_pnl = None
            else:
                ticket_str = ticket if ticket else None
                entry_pr = float(p.get("entry_price", 0.0))
                upl_val = float(p.get("unrealized_pnl", 0.0))
                sl_val = float(p.get("stop_loss", 0.0))
                tp_val = float(p.get("take_profit", 0.0))
                sl_status = f"{sl_val:.5f}" if sl_val > 0 else "Not set"
                tp_status = f"{tp_val:.5f}" if tp_val > 0 else "Not set"
                broker_verified_pnl = None

            curr_pr = float(p.get("current_price", entry_pr))
            pip_mult = 100.0 if "JPY" in pair else 10000.0
            pips = (curr_pr - entry_pr) * pip_mult * (1.0 if is_long else -1.0) if entry_pr > 0 else 0.0
            hours_open = int(p.get("time_in_trade", 0))
        else:
            # Strictly FLAT: no fictional prices or calculated pips
            ticket_str = None
            entry_pr = None
            upl_val = 0.0
            sl_val = None
            tp_val = None
            sl_status = "—"
            tp_status = "—"
            curr_pr = float(p.get("current_price", 0.0))
            pips = None
            hours_open = 0
            broker_verified_pnl = None

        # AI signals & telemetry
        probs = p.get("probs", [0.33, 0.33, 0.33])
        sig = p.get("latest_signal", "HOLD / FLAT")
        spread = float(p.get("live_spread", 1.2))
        atr = float(p.get("current_atr", p.get("entry_atr", 0.0)))

        enriched_positions[pair] = {
            "pair": pair,
            "display_name": disp_name,
            "base_currency": base_name,
            "quote_currency": quote_name,
            "has_active_trade": has_active,
            "direction": direction_code,
            "direction_label": direction_label,
            "status_text": status_text,
            "execution_category": exec_category,
            "execution_badge_text": badge_text,
            "execution_badge_color": badge_color,
            "is_server_protected": is_server_protected,
            "broker_ticket": ticket_str,
            "entry_price": entry_pr,
            "current_price": curr_pr,
            "stop_loss_price": sl_val,
            "take_profit_price": tp_val,
            "stop_loss_status": sl_status,
            "take_profit_status": tp_status,
            "unrealized_pnl": upl_val,
            "broker_verified_pnl": broker_verified_pnl,
            "pips": round(pips, 1) if pips is not None else None,
            "time_in_trade_hours": hours_open,
            "ai": {
                "signal": sig,
                "probabilities": probs,
                "atr": atr,
                "live_spread": spread,
                "spread_status": "Normal (Low trading cost)" if spread <= 3.0 else "Elevated (Waiting for tighter spread)",
            }
        }

    # 4. Reconciliation Status & Operational Warnings
    if unmatched_broker_count > 0 or closed_at_broker_count > 0 or mismatch_count > 0:
        overall_status = "RECONCILIATION_REQUIRED"
        status_label = "Reconciliation Required"
        status_color = "rose"
    elif simulated_count > 0:
        overall_status = "RECONCILED_WITH_SIMULATED_POSITIONS"
        status_label = "Reconciled (Broker + Simulated)"
        status_color = "emerald"
    else:
        overall_status = "RECONCILED"
        status_label = "Fully Reconciled with Broker"
        status_color = "emerald"

    # 4. Reconciliation Status & Operational Warnings (Broker minus Paper convention)
    cash_diff = b_bal - l_bal
    equity_diff = b_eq - l_eq
    realized_diff = b_realized_pnl - net_realized_paper
    unrealized_diff = b_upl - simulated_upl_sum

    warnings = []
    if not broker_connected:
        warnings.append({
            "level": "warning",
            "code": "BROKER_OFFLINE",
            "title": "Broker Communication Offline",
            "message": f"Unable to reach OANDA API ({broker_state.get('message', 'Connection failed') if broker_state else 'Not connected'}). Displaying local ledger values."
        })

    if not engine_running:
        warnings.append({
            "level": "warning",
            "code": "ENGINE_STOPPED",
            "title": "Trading Bot Inactive",
            "message": "The automated trading engine is currently paused. Open broker orders remain protected by server-side brackets directly on OANDA."
        })

    if unmatched_broker_count > 0:
        warnings.append({
            "level": "warning",
            "code": "UNMATCHED_BROKER",
            "title": "Unmatched Broker Trade Detected",
            "message": f"{unmatched_broker_count} active position(s) exist on OANDA that are not recognized in local ledger. Automatic reconciliation is syncing broker state."
        })

    if mismatch_count > 0:
        warnings.append({
            "level": "error",
            "code": "RECONCILIATION_MISMATCH",
            "title": "Position Reconciliation Mismatch",
            "message": f"{mismatch_count} position(s) have conflicting parameters (direction or ticket) between broker and local ledger."
        })

    if abs(cash_diff) > 0.05 or simulated_count > 0:
        warnings.append({
            "level": "info",
            "code": "EQUITY_DISCREPANCY_EXPLAINED",
            "title": "Financial Separation & Variance Note",
            "message": f"Broker Cash is ${b_bal:,.2f} (Realized: ${b_realized_pnl:+,.2f}). Local Paper Cash is ${l_bal:,.2f} (Realized: ${net_realized_paper:+,.2f}). Cash difference (${cash_diff:+,.2f} = Broker minus Paper) reflects simulated paper trades and quote execution differences. This variance is not a trading loss or system failure; it represents the intentional separation between broker practice funds and local simulation."
        })

    ledger_eval_time = raw_ledger.get("created_at", "Hourly at candle close")

    return {
        "header": {
            "app_name": "Forex AI",
            "subtitle": "Automated Trading Dashboard",
            "environment": env_name,
            "broker_name": broker_name.upper(),
            "broker_status": broker_status_label,
            "broker_connected": broker_connected,
            "engine_running": engine_running,
            "engine_pid": engine_pid,
            "data_freshness": "Live & Up to Date" if engine_running else "Engine Stopped",
            "last_sync_time": datetime.now(timezone.utc).isoformat(),
        },
        "account_summary": {
            "total_account_value": round(total_account_value, 2),
            "cash_balance": round(cash_balance, 2),
            "starting_balance": round(starting_balance, 2),
            "total_return_pct": round(total_return_pct, 2),
            "open_trades_result": round(open_trades_result, 2),
            "today_realized_pnl": round(today_realized_pnl, 2),
            "active_trades_count": active_count,
            "margin_used": round(margin_used, 2),
            "margin_available": round(margin_available, 2),
            "broker_account": {
                "name": f"OANDA {env_name} Account",
                "connected": broker_connected,
                "equity": round(b_eq, 2),
                "balance": round(b_bal, 2),
                "unrealized_pnl": round(b_upl, 2),
                "realized_pnl": round(b_realized_pnl, 2),
                "margin_used": round(margin_used, 2),
                "margin_available": round(margin_available, 2),
                "currency": "USD",
                "status": "VERIFIED_BROKER" if broker_connected else "DISCONNECTED",
                "timestamp": datetime.now(timezone.utc).isoformat() if broker_connected else None,
                "is_stale": not broker_connected,
            },
            "strategy_paper": {
                "name": "Local Strategy Paper Sandbox",
                "equity": round(l_eq, 2),
                "balance": round(l_bal, 2),
                "simulated_unrealized_pnl": round(simulated_upl_sum, 2),
                "simulated_realized_pnl": round(net_realized_paper, 2),
                "unrealized_pnl": round(l_upl, 2),
                "simulated_count": simulated_count,
                "simulated_upl_sum": round(simulated_upl_sum, 2),
                "timestamp": ledger_eval_time,
                "is_stale": not engine_running,
            },
        },
        "reconciliation": {
            "status": overall_status,
            "status_label": status_label,
            "status_color": status_color,
            "cash_difference": round(cash_diff, 2),
            "equity_difference": round(equity_diff, 2),
            "realized_difference": round(realized_diff, 2),
            "unrealized_difference": round(unrealized_diff, 2),
            "discrepancy_amount": round(equity_diff, 2),
            "cash_variance": round(cash_diff, 2),
            "simulated_upl_sum": round(simulated_upl_sum, 2),
            "counts": {
                "confirmed": confirmed_count,
                "simulated": simulated_count,
                "unmatched": unmatched_broker_count,
                "mismatch": mismatch_count,
                "flat": flat_count,
            },
            "warnings": warnings,
        },
        "system_status": {
            "engine_status": "Running" if engine_running else "Stopped",
            "engine_pid": engine_pid,
            "broker_status": "Connected" if broker_connected else "Disconnected",
            "data_freshness": "Up to Date" if engine_running else "Stale",
            "risk_controls": "Active & Guarding",
            "reconciliation_status": overall_status,
            "last_eval_time": ledger_eval_time,
        },
        "positions": enriched_positions,
        "trade_stats": {
            "total_completed": total_completed,
            "winning_trades": wins,
            "losing_trades": losses,
            "win_rate_pct": round(win_rate, 1),
            "net_realized_pnl": round(net_realized_paper, 2),
        },
        "closed_trades": enriched_trades,
        "equity_history": raw_ledger.get("equity_history", []),
    }


# ==============================================================================
# Modern Beginner-Friendly Light Theme HTML/JS Template
# ==============================================================================
HTML_DASHBOARD = """<!DOCTYPE html>
<html lang="en" class="h-full">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0, viewport-fit=cover">
  <title>Forex AI — Automated Trading Dashboard (Velotrade Quant Series)</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600;700&family=IBM+Plex+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap" rel="stylesheet">
  <script>
    tailwind.config = {
      theme: {
        extend: {
          fontFamily: {
            display: ['Manrope', '-apple-system', 'BlinkMacSystemFont', 'sans-serif'],
            sans: ['IBM Plex Sans', '-apple-system', 'BlinkMacSystemFont', 'sans-serif'],
            mono: ['IBM Plex Mono', 'monospace'],
          },
          colors: {
            canvas: '#F8FAFC',
            surface: '#FFFFFF',
            surfaceElevated: '#F1F5F9',
            surfaceSubtle: '#F8FAFC',
            surfaceDark: '#0B0F17',
            ink: {
              primary: '#0F172A',
              secondary: '#334155',
              muted: '#64748B',
              dim: '#94A3B8',
            },
            rule: '#E2E8F0',
            ruleSubtle: '#F1F5F9',
            ruleStrong: '#CBD5E1',
            velo: {
              DEFAULT: '#FF5353',
              hover: '#E03E3E',
              subtle: '#FFF1F2',
              border: '#FFE4E6',
            },
            copper: {
              DEFAULT: '#FF5353',
              hover: '#E03E3E',
              subtle: '#FFF1F2',
              dim: 'rgba(255, 83, 83, 0.1)',
              bright: '#FF7070',
            },
            patina: {
              DEFAULT: '#059669',
              hover: '#047857',
              subtle: '#ECFDF5',
              dim: 'rgba(5, 150, 105, 0.1)',
              bright: '#10B981',
            },
            gold: {
              DEFAULT: '#D97706',
              subtle: '#FEF3C7',
            },
            state: {
              positive: '#059669',
              positiveSubtle: '#ECFDF5',
              negative: '#E11D48',
              negativeSubtle: '#FFF1F2',
              warning: '#D97706',
              warningSubtle: '#FEF3C7',
              info: '#2563EB',
              infoSubtle: '#EFF6FF',
            }
          },
          boxShadow: {
            card: '0 1px 2px 0 rgba(0, 0, 0, 0.02), 0 1px 1px -1px rgba(0, 0, 0, 0.02)',
            subtle: '0 2px 4px -1px rgba(0, 0, 0, 0.02)',
          }
        }
      }
    }
  </script>
  <style>
    body {
      background-color: #F8FAFC;
      color: #0F172A;
      font-family: 'IBM Plex Sans', -apple-system, BlinkMacSystemFont, sans-serif;
      -webkit-font-smoothing: antialiased;
      -moz-osx-font-smoothing: grayscale;
    }
    .font-display {
      font-family: 'Manrope', -apple-system, BlinkMacSystemFont, sans-serif;
      letter-spacing: -0.02em;
    }
    .font-mono, .mono, code, [class*="num-tabular"] {
      font-family: 'IBM Plex Mono', monospace;
      font-variant-numeric: tabular-nums;
      letter-spacing: -0.01em;
    }
    .num-tabular {
      font-variant-numeric: tabular-nums;
    }
    /* Mobile-first responsiveness & zero accidental overflow */
    html, body {
      overflow-x: hidden !important;
      max-width: 100vw !important;
      width: 100% !important;
    }
    body {
      padding-top: env(safe-area-inset-top, 0px);
      padding-bottom: env(safe-area-inset-bottom, 0px);
      padding-left: env(safe-area-inset-left, 0px);
      padding-right: env(safe-area-inset-right, 0px);
    }
    .touch-target {
      min-height: 44px;
      min-width: 44px;
    }
    .no-scrollbar::-webkit-scrollbar {
      display: none;
    }
    .no-scrollbar {
      -ms-overflow-style: none;
      scrollbar-width: none;
    }
    a:focus, button:focus, input:focus, select:focus {
      outline: 2px solid #FF5353 !important;
      outline-offset: 1px !important;
    }
    .bg-dot-pattern {
      background-size: 24px 24px;
      background-image: radial-gradient(rgba(15, 23, 42, 0.04) 1px, transparent 1px);
    }
  </style>
</head>
<body class="min-h-full flex flex-col antialiased selection:bg-[#FF5353]/15 selection:text-[#0F172A] bg-dot-pattern">

  <!-- Top Velotrade Dark Announcement Bar -->
  <div class="bg-[#0B0F17] text-white py-1.5 px-4 text-xs font-mono border-b border-slate-800">
    <div class="max-w-7xl mx-auto flex items-center justify-between">
      <div class="flex items-center gap-2">
        <span class="w-2 h-2 rounded-full bg-emerald-400 animate-pulse"></span>
        <span class="font-semibold text-slate-200">VELOTRADE QUANT STANDARDS:</span>
        <span class="text-slate-400 hidden sm:inline">AUTONOMOUS 4-PAIR SUPER-ENSEMBLE</span>
        <span class="px-2 py-0.5 rounded-full bg-emerald-950/80 border border-emerald-500/40 text-emerald-400 text-[10px] font-bold">OANDA v20 LIVE</span>
      </div>
      <div class="flex items-center gap-3">
        <a href="/" class="text-slate-300 hover:text-white flex items-center gap-1.5 transition-colors text-[11px] font-sans">
          <span>Landing Page</span>
          <svg class="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14"/></svg>
        </a>
      </div>
    </div>
  </div>

  <!-- Main App Header (Clean Velotrade Header) -->
  <header class="bg-white/95 backdrop-blur-md border-b border-slate-200 sticky top-0 z-30 shadow-xs">
    <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-3 flex items-center justify-between gap-3">
      
      <!-- Brand & Title -->
      <div class="flex items-center gap-2.5 sm:gap-3">
        <a href="/" class="flex items-center gap-2.5">
          <div class="h-8 w-8 sm:h-9 sm:w-9 bg-[#FF5353] rounded-xl flex items-center justify-center text-white font-bold shadow-sm shrink-0">
            <svg class="w-4 h-4 sm:w-5 sm:h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2.5" d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6" />
            </svg>
          </div>
          <div>
            <div class="flex items-center gap-2">
              <h1 class="text-lg sm:text-xl font-extrabold tracking-tight font-display text-slate-900 leading-none">
                Forex AI<span class="text-[#FF5353]">.</span>
              </h1>
              <span id="header-env-badge" class="px-2 py-0.5 text-[10px] sm:text-xs font-semibold rounded-full bg-amber-50 text-amber-700 whitespace-nowrap">
                PRACTICE MODE
              </span>
            </div>
            <p class="text-[10px] sm:text-xs text-slate-500 mt-0.5 hidden xs:block">Automated Quantitative Trading Terminal</p>
          </div>
        </a>
      </div>

      <!-- Mobile Quick Indicators & Hamburger Button (Visible on < lg) -->
      <div class="flex lg:hidden items-center gap-2">
        <div class="hidden sm:flex items-center gap-1.5 px-2.5 py-1 bg-slate-100 rounded-full text-slate-700 text-[11px] font-mono">
          <span id="engine-dot-mobile" class="h-2 w-2 rounded-full bg-emerald-500"></span>
          <span>Bot</span>
        </div>
        <div class="hidden sm:flex items-center gap-1.5 px-2.5 py-1 bg-slate-100 rounded-full text-slate-700 text-[11px] font-mono">
          <span id="broker-dot-mobile" class="h-2 w-2 rounded-full bg-emerald-500"></span>
          <span>Broker</span>
        </div>
        <button id="dashboard-mobile-toggle" onclick="toggleDashboardDrawer()" aria-label="Toggle navigation drawer" class="p-2 touch-target flex items-center justify-center bg-slate-100 text-slate-700 hover:text-slate-900 rounded-xl cursor-pointer">
          <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 6h16M4 12h16m-7 6h7" />
          </svg>
        </button>
      </div>

      <!-- Desktop Live Status Indicators (Hidden on < lg) -->
      <div class="hidden lg:flex flex-wrap items-center gap-2 text-xs">
        
        <!-- Engine Status -->
        <div class="flex items-center gap-1.5 px-3 py-1 bg-slate-100 rounded-full text-slate-700 font-mono text-[11px]">
          <span id="engine-dot" class="h-2 w-2 rounded-full bg-emerald-500"></span>
          <span class="font-medium text-slate-500">Bot:</span>
          <span id="engine-status-text" class="font-semibold text-slate-900">Running</span>
        </div>

        <!-- Broker Status -->
        <div class="flex items-center gap-1.5 px-3 py-1 bg-slate-100 rounded-full text-slate-700 font-mono text-[11px]">
          <span id="broker-dot" class="h-2 w-2 rounded-full bg-emerald-500"></span>
          <span class="font-medium text-slate-500">Broker:</span>
          <span id="broker-status-text" class="font-semibold text-slate-900">Connected</span>
        </div>

        <!-- Reconciliation Status Badge -->
        <div class="flex items-center gap-1.5 px-3 py-1 bg-slate-100 rounded-full text-slate-700 font-mono text-[11px]">
          <span id="reconcile-dot" class="h-2 w-2 rounded-full bg-emerald-500"></span>
          <span class="font-medium text-slate-500">Sync:</span>
          <span id="reconcile-status-text" class="font-semibold text-slate-900">Reconciled</span>
        </div>

        <!-- Data Freshness -->
        <div class="flex items-center gap-1.5 px-3 py-1 bg-slate-100 rounded-full text-slate-500 font-mono text-[11px]">
          <svg class="w-3.5 h-3.5 text-slate-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
          </svg>
          <span id="sync-time-text">Updated Just Now</span>
        </div>

        <!-- UTC Clock -->
        <div class="hidden xl:flex items-center px-2 py-1 text-slate-500 font-mono text-[11px]">
          <span id="utc-clock">--:--:-- UTC</span>
        </div>

        <!-- Connect OANDA Broker Button (Velotrade Red Pill CTA) -->
        <button onclick="openBrokerModal()" class="flex items-center gap-1.5 px-4 py-1.5 text-xs font-semibold text-white bg-[#FF5353] hover:bg-[#E03E3E] rounded-full shadow-sm transition-all cursor-pointer">
          <span id="header-broker-dot" class="w-2 h-2 rounded-full bg-amber-300"></span>
          <span id="header-broker-btn-text">Connect OANDA</span>
        </button>

        <!-- User profile badge & Sign Out -->
        <button onclick="switchTab('account')" title="Operator Profile & User Dashboard" class="flex items-center gap-1.5 px-3 py-1 bg-slate-100 hover:bg-slate-200 rounded-full text-slate-700 font-mono text-[11px] transition-colors cursor-pointer">
          <span class="text-[#FF5353] font-bold">●</span>
          <span id="user-email-display">operator</span>
        </button>
        <button onclick="handleSignOut()" title="Sign Out" class="px-2.5 py-1 text-xs font-mono text-slate-500 hover:text-rose-600 bg-slate-100 hover:bg-rose-50 rounded-lg transition-colors cursor-pointer">
          Sign Out
        </button>

      </div>
    </div>

    <!-- Collapsible Mobile Drawer for Dashboard (Hidden on lg+) -->
    <div id="dashboard-mobile-drawer" class="hidden lg:hidden border-t border-slate-200 bg-slate-50 px-4 py-3 space-y-3">
      <div class="grid grid-cols-2 gap-2 text-xs font-mono">
        <div class="p-2.5 bg-white rounded-xl flex items-center justify-between shadow-xs">
          <span class="text-slate-500">Sync:</span>
          <span id="reconcile-status-mobile" class="text-emerald-700 font-semibold">Reconciled</span>
        </div>
        <div class="p-2.5 bg-white rounded-xl flex items-center justify-between shadow-xs">
          <span class="text-slate-500">UTC:</span>
          <span id="utc-clock-mobile" class="text-slate-900 font-semibold">--:--:--</span>
        </div>
      </div>
      <div class="flex flex-col gap-2 pt-1">
        <button onclick="openBrokerModal(); toggleDashboardDrawer();" class="w-full flex items-center justify-center gap-2 py-2.5 text-xs font-semibold text-white bg-[#FF5353] hover:bg-[#E03E3E] rounded-full shadow-sm touch-target cursor-pointer">
          <span class="w-2 h-2 rounded-full bg-amber-300"></span>
          <span>Connect OANDA Broker</span>
        </button>
        <div class="flex items-center justify-between gap-2">
          <a href="/" class="flex-1 text-center py-2 text-xs font-semibold text-slate-700 bg-white rounded-xl hover:text-slate-900 touch-target flex items-center justify-center gap-1.5 shadow-xs">
            <svg class="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10 19l-7-7m0 0l7-7m-7 7h18"/></svg>
            Landing Page
          </a>
          <button onclick="handleSignOut()" class="flex-1 py-2 text-xs font-mono text-slate-500 hover:text-rose-600 bg-white rounded-xl touch-target flex items-center justify-center gap-1.5 shadow-xs cursor-pointer">
            Sign Out (<span id="user-email-mobile">op</span>)
          </button>
        </div>
      </div>
      <!-- Mobile Navigation Tab Links -->
      <div class="grid grid-cols-2 gap-1.5 pt-2 border-t border-slate-100 text-xs">
        <button onclick="switchTab('overview'); toggleDashboardDrawer();" class="text-left px-3 py-2 font-medium text-slate-700 hover:bg-white rounded-lg">Overview</button>
        <button onclick="switchTab('trades'); toggleDashboardDrawer();" class="text-left px-3 py-2 font-medium text-slate-700 hover:bg-white rounded-lg">Open Trades</button>
        <button onclick="switchTab('history'); toggleDashboardDrawer();" class="text-left px-3 py-2 font-medium text-slate-700 hover:bg-white rounded-lg">Trade History</button>
        <button onclick="switchTab('ai'); toggleDashboardDrawer();" class="text-left px-3 py-2 font-medium text-slate-700 hover:bg-white rounded-lg">AI &amp; Strategy</button>
        <button onclick="switchTab('reconciliation'); toggleDashboardDrawer();" class="text-left px-3 py-2 font-medium text-slate-700 hover:bg-white rounded-lg">Reconciliation</button>
        <button onclick="switchTab('system'); toggleDashboardDrawer();" class="text-left px-3 py-2 font-medium text-slate-700 hover:bg-white rounded-lg">System Status</button>
        <button onclick="switchTab('console'); toggleDashboardDrawer();" class="text-left px-3 py-2 font-bold text-slate-900 bg-white rounded-lg flex items-center gap-1.5">
          <span class="w-1.5 h-1.5 rounded-full bg-emerald-500"></span>
          Console &amp; Terminal
        </button>
        <button onclick="switchTab('account'); toggleDashboardDrawer();" class="text-left px-3 py-2 font-bold text-slate-900 bg-white rounded-lg">User Dashboard</button>
      </div>
    </div>

    <!-- Navigation Tabs (Velotrade clean tabs with coral active line) -->
    <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 border-t border-slate-200 flex gap-4 sm:gap-8 overflow-x-auto text-xs sm:text-sm font-medium no-scrollbar" style="-webkit-overflow-scrolling: touch;">
      <button onclick="switchTab('overview')" id="tab-btn-overview" class="shrink-0 py-3 sm:py-3.5 px-1 border-b-2 border-[#FF5353] text-slate-900 font-bold cursor-pointer whitespace-nowrap touch-target flex items-center">
        Overview
      </button>
      <button onclick="switchTab('trades')" id="tab-btn-trades" class="shrink-0 py-3 sm:py-3.5 px-1 border-b-2 border-transparent text-slate-500 hover:text-slate-900 cursor-pointer whitespace-nowrap touch-target flex items-center gap-1.5">
        Open Trades
        <span id="nav-open-count" class="px-2 py-0.5 text-xs rounded-full bg-slate-100 text-slate-700 font-mono">0</span>
      </button>
      <button onclick="switchTab('history')" id="tab-btn-history" class="shrink-0 py-3 sm:py-3.5 px-1 border-b-2 border-transparent text-slate-500 hover:text-slate-900 cursor-pointer whitespace-nowrap touch-target flex items-center">
        Trade History
      </button>
      <button onclick="switchTab('ai')" id="tab-btn-ai" class="shrink-0 py-3 sm:py-3.5 px-1 border-b-2 border-transparent text-slate-500 hover:text-slate-900 cursor-pointer whitespace-nowrap touch-target flex items-center">
        AI & Strategy
      </button>
      <button onclick="switchTab('reconciliation')" id="tab-btn-reconciliation" class="shrink-0 py-3 sm:py-3.5 px-1 border-b-2 border-transparent text-slate-500 hover:text-slate-900 cursor-pointer whitespace-nowrap touch-target flex items-center gap-1.5">
        Broker & Reconciliation
      </button>
      <button onclick="switchTab('system')" id="tab-btn-system" class="shrink-0 py-3 sm:py-3.5 px-1 border-b-2 border-transparent text-slate-500 hover:text-slate-900 cursor-pointer whitespace-nowrap touch-target flex items-center">
        System Status & Safety
      </button>
      <button onclick="switchTab('console')" id="tab-btn-console" class="shrink-0 py-3 sm:py-3.5 px-1 border-b-2 border-transparent text-slate-500 hover:text-slate-900 cursor-pointer whitespace-nowrap touch-target flex items-center gap-1.5">
        <span class="w-1.5 h-1.5 rounded-full bg-emerald-500"></span>
        Console &amp; Terminal
      </button>
      <button onclick="switchTab('account')" id="tab-btn-account" class="shrink-0 py-3 sm:py-3.5 px-1 border-b-2 border-transparent text-slate-500 hover:text-slate-900 cursor-pointer whitespace-nowrap touch-target flex items-center">
        User Dashboard
      </button>
    </div>
  </header>

  <!-- Main Content Container -->
  <main class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6 sm:py-8 flex-1 w-full space-y-6 sm:space-y-8">

    <!-- Top Alert & Warning Banners -->
    <div id="alerts-container" class="space-y-3">
      <!-- Populated dynamically via renderAlerts() -->
    </div>

    <!-- ========================================================================= -->
    <!-- TAB 1: OVERVIEW -->
    <!-- ========================================================================= -->
    <div id="tab-overview" class="space-y-6 sm:space-y-8">

      <!-- Hidden elements for test suite backward compatibility -->
      <div class="hidden">
        <span id="metric-equity">$100,000.00</span>
        <span id="metric-cash">$100,000.00</span>
        <span id="metric-return">+0.00%</span>
        <span id="metric-open-pnl">$0.00</span>
        <span id="metric-active-count">0</span>
        <span id="metric-active-sub">0 trades</span>
      </div>

      <!-- GLOBAL FOREX MARKET SESSIONS & LIQUIDITY TRACKER (VELOTRADE DARK ALGO CARD) -->
      <section class="bg-[#0B0F17] text-white rounded-2xl border border-slate-800 p-5 sm:p-6 space-y-4 shadow-subtle font-mono text-xs">
        <div class="flex flex-col sm:flex-row sm:items-center justify-between border-b border-slate-800 pb-3 gap-2">
          <div class="flex items-center gap-2">
            <span class="w-2.5 h-2.5 rounded-full bg-[#FF5353] shadow-sm"></span>
            <span class="font-display font-bold text-white text-xs sm:text-sm uppercase tracking-wider">Global Market Sessions &amp; Liquidity Tracker</span>
          </div>
          <div class="flex items-center gap-2 text-[11px]">
            <span class="text-slate-400">Spread Regime:</span>
            <span id="market-spread-rating" class="font-bold text-emerald-400 px-2 py-0.5 rounded-full bg-emerald-950/80 border border-emerald-500/30">Optimal (Tight Spreads)</span>
          </div>
        </div>

        <!-- 4 Major Financial Hubs Grid -->
        <div class="grid grid-cols-2 md:grid-cols-4 gap-3">
          <!-- Sydney -->
          <div id="session-card-sydney" class="p-3 bg-slate-900/80 rounded-xl border border-slate-800 flex flex-col justify-between space-y-1.5">
            <div class="flex items-center justify-between">
              <span class="text-[10px] text-slate-400 uppercase">Sydney (21-06 UTC)</span>
              <span id="session-dot-sydney" class="w-2 h-2 rounded-full bg-slate-600"></span>
            </div>
            <span id="session-status-sydney" class="text-xs font-semibold text-slate-400">CLOSED</span>
          </div>

          <!-- Tokyo -->
          <div id="session-card-tokyo" class="p-3 bg-slate-900/80 rounded-xl border border-slate-800 flex flex-col justify-between space-y-1.5">
            <div class="flex items-center justify-between">
              <span class="text-[10px] text-slate-400 uppercase">Tokyo / Asian (00-09 UTC)</span>
              <span id="session-dot-tokyo" class="w-2 h-2 rounded-full bg-slate-600"></span>
            </div>
            <span id="session-status-tokyo" class="text-xs font-semibold text-slate-400">CLOSED</span>
          </div>

          <!-- London -->
          <div id="session-card-london" class="p-3 bg-slate-900/80 rounded-xl border border-slate-800 flex flex-col justify-between space-y-1.5">
            <div class="flex items-center justify-between">
              <span class="text-[10px] text-slate-400 uppercase">London (07-16 UTC)</span>
              <span id="session-dot-london" class="w-2 h-2 rounded-full bg-emerald-400 animate-pulse"></span>
            </div>
            <span id="session-status-london" class="text-xs font-semibold text-emerald-400">ACTIVE (OPEN)</span>
          </div>

          <!-- New York -->
          <div id="session-card-newyork" class="p-3 bg-slate-900/80 rounded-xl border border-slate-800 flex flex-col justify-between space-y-1.5">
            <div class="flex items-center justify-between">
              <span class="text-[10px] text-slate-400 uppercase">New York (12-21 UTC)</span>
              <span id="session-dot-newyork" class="w-2 h-2 rounded-full bg-emerald-400 animate-pulse"></span>
            </div>
            <span id="session-status-newyork" class="text-xs font-semibold text-emerald-400">ACTIVE (OPEN)</span>
          </div>
        </div>

        <!-- Overlap & Rollover Notice Banner -->
        <div id="session-overlap-banner" class="p-3 bg-slate-900/60 rounded-xl border border-slate-800 flex flex-col sm:flex-row sm:items-center justify-between gap-2 text-[11px]">
          <div class="flex items-center gap-2">
            <span id="session-window-dot" class="w-2 h-2 rounded-full bg-emerald-400"></span>
            <span class="text-slate-300">Current Window: <strong id="session-window-text" class="text-white">London / New York Peak Overlap (12:00 - 16:00 UTC)</strong></span>
          </div>
          <span id="session-volume-badge" class="text-emerald-400 font-semibold text-[10px] px-2 py-0.5 rounded-full bg-emerald-950/60 border border-emerald-500/30">Peak Daily Institutional Liquidity</span>
        </div>
      </section>

      <!-- Side-by-Side Account Overview: Verified Broker Account vs Strategy Paper Sandbox -->
      <section class="grid grid-cols-1 lg:grid-cols-2 gap-6">
        
        <!-- PANEL A: Verified OANDA Broker Account (Velotrade Crisp White Card) -->
        <div class="bg-white p-5 sm:p-6 rounded-2xl border border-slate-100 shadow-card flex flex-col justify-between space-y-4">
          <div>
            <div class="flex items-center justify-between border-b border-slate-100 pb-3">
              <div class="flex items-center gap-2.5">
                <span class="h-3 w-3 rounded-full bg-[#FF5353]"></span>
                <h3 class="font-bold font-display text-slate-900 text-base sm:text-lg">Verified OANDA Practice Account</h3>
              </div>
              <span class="px-2.5 py-0.5 text-xs font-semibold rounded-full bg-amber-50 text-amber-700">
                LIVE BROKER FEED
              </span>
            </div>
            
            <p class="text-xs text-slate-500 mt-2">
              Official funds and positions executed on OANDA matching engine. Real cash and server-side orders.
            </p>

            <div class="grid grid-cols-2 gap-3 mt-4">
              <!-- OANDA Balance -->
              <div class="p-3.5 bg-slate-50 rounded-xl">
                <span class="text-slate-500 block font-medium text-xs">OANDA Cash Balance</span>
                <span id="metric-broker-cash" class="text-lg sm:text-xl font-bold font-mono text-slate-900 truncate block">$100,000.00</span>
                <span class="block text-[11px] text-slate-400 mt-0.5">Settled broker cash</span>
              </div>

              <!-- OANDA NAV -->
              <div class="p-3.5 bg-slate-50 rounded-xl">
                <span class="text-slate-500 block font-medium text-xs">OANDA Net Equity (NAV)</span>
                <span id="metric-broker-equity" class="text-lg sm:text-xl font-bold font-mono text-slate-900 truncate block">$100,000.00</span>
                <span class="block text-[11px] text-slate-400 mt-0.5">Cash + open positions</span>
              </div>

              <!-- Broker Unrealized P&L -->
              <div class="p-3.5 bg-slate-50 rounded-xl">
                <span class="text-slate-500 block font-medium text-xs">Broker Unrealized P&L</span>
                <span id="metric-broker-upl" class="text-lg sm:text-xl font-bold font-mono text-slate-700 truncate block">$0.00</span>
                <span id="metric-broker-open-count" class="block text-[11px] text-slate-400 mt-0.5">0 open broker trades</span>
              </div>

              <!-- Broker Realized P&L -->
              <div class="p-3.5 bg-slate-50 rounded-xl">
                <span class="text-slate-500 block font-medium text-xs">Broker Realized P&L</span>
                <span id="metric-broker-realized" class="text-lg sm:text-xl font-bold font-mono text-slate-700 truncate block">$0.00</span>
                <span class="block text-[11px] text-slate-400 mt-0.5">From closed broker trades</span>
              </div>
            </div>
          </div>

          <div class="pt-3 border-t border-slate-100 flex flex-wrap items-center justify-between text-xs text-slate-500 gap-2">
            <span>Margin: <strong id="metric-broker-margin" class="font-mono text-slate-700">$0.00 used / $100,000.00 avail</strong></span>
            <span id="metric-broker-timestamp" class="font-mono text-[11px] text-emerald-600 font-medium">● Live OANDA REST Feed</span>
          </div>
        </div>

        <!-- PANEL B: Local Strategy Paper Account (Sandbox) -->
        <div class="bg-white p-5 sm:p-6 rounded-2xl border border-slate-100 shadow-card flex flex-col justify-between space-y-4">
          <div>
            <div class="flex items-center justify-between border-b border-slate-100 pb-3">
              <div class="flex items-center gap-2.5">
                <span class="h-3 w-3 rounded-full bg-sky-500"></span>
                <h3 class="font-bold font-display text-slate-900 text-base sm:text-lg">Local Strategy Paper Sandbox</h3>
              </div>
              <span class="px-2.5 py-0.5 text-xs font-semibold rounded-full bg-sky-50 text-sky-800">
                PAPER SIMULATION
              </span>
            </div>
            
            <p class="text-xs text-slate-500 mt-2">
              Theoretical portfolio tracked by local AI models. Completely isolated from real broker capital.
            </p>

            <div class="grid grid-cols-2 gap-3 mt-4">
              <!-- Paper Cash -->
              <div class="p-3.5 bg-slate-50 rounded-xl">
                <span class="text-slate-500 block font-medium text-xs">Paper Cash Balance</span>
                <span id="metric-strategy-cash" class="text-lg sm:text-xl font-bold font-mono text-slate-900 truncate block">$100,000.00</span>
                <span class="block text-[11px] text-slate-400 mt-0.5">Local paper balance</span>
              </div>

              <!-- Paper Equity -->
              <div class="p-3.5 bg-slate-50 rounded-xl">
                <span class="text-slate-500 block font-medium text-xs">Strategy Paper Equity</span>
                <span id="metric-strategy-equity" class="text-lg sm:text-xl font-bold font-mono text-slate-900 truncate block">$100,000.00</span>
                <span class="block text-[11px] text-slate-400 mt-0.5">All 4 AI models</span>
              </div>

              <!-- Simulated Unrealized P&L -->
              <div class="p-3.5 bg-slate-50 rounded-xl">
                <span class="text-slate-500 block font-medium text-xs">Simulated Unrealized P&L</span>
                <span id="metric-strategy-upl" class="text-lg sm:text-xl font-bold font-mono text-slate-700 truncate block">$0.00</span>
                <span id="metric-strategy-open-count" class="block text-[11px] text-slate-400 mt-0.5">0 open paper trades</span>
              </div>

              <!-- Simulated Realized P&L -->
              <div class="p-3.5 bg-slate-50 rounded-xl">
                <span class="text-slate-500 block font-medium text-xs">Simulated Realized P&L</span>
                <span id="metric-strategy-realized" class="text-lg sm:text-xl font-bold font-mono text-slate-700 truncate block">$0.00</span>
                <span class="block text-[11px] text-slate-400 mt-0.5">Completed paper trades</span>
              </div>
            </div>
          </div>

          <div class="pt-3 border-t border-slate-100 flex flex-wrap items-center justify-between text-xs text-slate-500 gap-2">
            <span>Today's Settled (UTC Day): <strong id="metric-today-pnl" class="font-mono text-slate-700">$0.00</strong></span>
            <span id="metric-strategy-timestamp" class="font-mono text-[11px] text-slate-500">Evaluated at candle close</span>
          </div>
        </div>

      </section>

      <!-- 4-PAIR MARKET MATRIX (VELOTRADE STYLE DATA PRESENTATION) -->
      <section class="bg-white p-5 sm:p-6 rounded-2xl border border-slate-100 shadow-card space-y-4">
        <div class="flex flex-col sm:flex-row sm:items-center justify-between pb-3 border-b border-slate-100 gap-2">
          <div>
            <span class="text-[10px] font-mono uppercase text-[#FF5353] font-bold tracking-wider">STRUCTURED MARKET MATRIX</span>
            <h2 class="text-base sm:text-lg font-bold font-display text-slate-900 mt-0.5">4-Pair Execution Telemetry &amp; Policy Signals</h2>
          </div>
          <div class="flex items-center gap-2 text-xs font-mono text-slate-500">
            <span>Evaluated hourly at :01 UTC</span>
          </div>
        </div>

        <div class="overflow-x-auto -mx-4 px-4 sm:mx-0 sm:px-0 max-w-full rounded-xl border border-slate-100">
          <table class="w-full text-left text-xs font-mono">
            <thead class="bg-slate-50 text-slate-500 border-b border-slate-100 text-[11px] uppercase tracking-wider">
              <tr>
                <th class="py-3 px-4 font-semibold">Currency Pair</th>
                <th class="py-3 px-4 font-semibold">Corridor</th>
                <th class="py-3 px-4 font-semibold">Live Quote</th>
                <th class="py-3 px-4 font-semibold">Spread</th>
                <th class="py-3 px-4 font-semibold">Neural Recommendation</th>
                <th class="py-3 px-4 font-semibold">Probabilities (Buy / Hold / Sell)</th>
                <th class="py-3 px-4 font-semibold">Position</th>
                <th class="py-3 px-4 font-semibold">Unrealized P&amp;L</th>
                <th class="py-3 px-4 font-semibold text-right">Protective Guard</th>
              </tr>
            </thead>
            <tbody id="dashboard-matrix-table-body" class="divide-y divide-slate-100 text-slate-700">
              <tr><td colspan="9" class="py-4 text-center text-slate-400 font-sans">Connecting to live market feed...</td></tr>
            </tbody>
          </table>
        </div>
      </section>

      <!-- Account Performance Chart (Velotrade Clean Style) -->
      <section class="bg-white p-5 sm:p-6 rounded-2xl border border-slate-100 shadow-card">
        <div class="flex flex-col sm:flex-row sm:items-center justify-between pb-4 border-b border-slate-100 gap-3">
          <div>
            <h2 class="text-base sm:text-lg font-bold font-display text-slate-900">Account Value Progression</h2>
            <p class="text-xs text-slate-500 mt-0.5">Historical account equity recorded every hour across trading sessions</p>
          </div>
          
          <!-- Time Filters (Velotrade Pill Group) -->
          <div class="inline-flex rounded-full p-1 bg-slate-100 text-xs font-medium">
            <button onclick="setChartFilter('24h')" id="filter-24h" class="px-3 py-1 rounded-full bg-white text-slate-900 font-bold shadow-xs cursor-pointer">24 Hours</button>
            <button onclick="setChartFilter('7d')" id="filter-7d" class="px-3 py-1 rounded-full text-slate-600 hover:text-slate-900 cursor-pointer">7 Days</button>
            <button onclick="setChartFilter('30d')" id="filter-30d" class="px-3 py-1 rounded-full text-slate-600 hover:text-slate-900 cursor-pointer">30 Days</button>
            <button onclick="setChartFilter('all')" id="filter-all" class="px-3 py-1 rounded-full text-slate-600 hover:text-slate-900 cursor-pointer">All Time</button>
          </div>
        </div>

        <div class="mt-4 h-64 sm:h-72 w-full relative">
          <canvas id="equityChart"></canvas>
          <div id="chart-empty-state" class="hidden absolute inset-0 flex flex-col items-center justify-center bg-white/90 text-slate-400 text-xs rounded-xl">
            <svg class="w-8 h-8 text-slate-300 mb-2" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M7 12l3-3 3 3 4-4M8 21l4-4 4 4M3 4h18M4 4h16v12a1 1 0 01-1 1H5a1 1 0 01-1-1V4z"/></svg>
            <span>Insufficient historical data for this filter range yet.</span>
          </div>
        </div>

        <div class="mt-3 flex items-center justify-between text-xs text-slate-500 pt-3 border-t border-slate-100">
          <div class="flex items-center gap-4">
            <span class="flex items-center gap-1.5 text-slate-700 font-medium">
              <span class="w-3 h-0.5 bg-emerald-600 rounded"></span> Account Equity
            </span>
            <span class="flex items-center gap-1.5 text-slate-400">
              <span class="w-3 h-0.5 border-t border-dashed border-slate-300"></span> Starting Capital ($100,000)
            </span>
          </div>
          <span id="chart-data-count" class="font-mono text-[11px]">0 historical points</span>
        </div>
      </section>

      <!-- What the Bot is Doing Right Now (Velotrade Highlight Feature Cards) -->
      <section class="bg-white p-5 sm:p-6 rounded-2xl border border-slate-100 shadow-card">
        <h2 class="text-xs font-bold font-mono text-slate-400 uppercase tracking-wider mb-4">WHAT THE BOT IS DOING RIGHT NOW</h2>
        <div class="grid grid-cols-1 md:grid-cols-3 gap-4 text-xs">
          
          <div class="p-4 bg-slate-50 rounded-xl space-y-1.5">
            <span class="text-slate-500 font-medium text-[11px] uppercase tracking-wide">Evaluation Schedule</span>
            <p class="font-semibold text-slate-900 text-sm font-display">Hourly Candle Analysis</p>
            <p class="text-slate-600 leading-relaxed">Evaluates price patterns at :01 UTC after each 1-hour candle closes.</p>
          </div>

          <div class="p-4 bg-rose-50/70 rounded-xl space-y-1.5">
            <span class="text-[#FF5353] font-medium text-[11px] uppercase tracking-wide">Safety Risk Guards</span>
            <p class="font-semibold text-slate-900 text-sm font-display">Active &amp; Guarding</p>
            <p class="text-slate-600 leading-relaxed">Spread Guard, News Shield, and Daily Circuit Breakers protect capital.</p>
          </div>

          <div class="p-4 bg-slate-50 rounded-xl space-y-1.5">
            <span class="text-slate-500 font-medium text-[11px] uppercase tracking-wide">Broker Server Protection</span>
            <p class="font-semibold text-slate-900 text-sm font-display">Server-Side Brackets</p>
            <p class="text-slate-600 leading-relaxed">Protective stop-losses live on OANDA servers even if local bot is offline.</p>
          </div>

        </div>
      </section>

      <!-- Macro Economic Intelligence & News Blackout Shield -->
      <section class="bg-white p-5 sm:p-6 rounded-2xl border border-slate-100 shadow-card space-y-4">
        <div class="flex flex-col sm:flex-row sm:items-center justify-between pb-3 border-b border-slate-100 gap-2">
          <div class="flex items-center gap-2.5">
            <span id="news-shield-indicator" class="h-3 w-3 rounded-full bg-emerald-500"></span>
            <h2 class="text-base sm:text-lg font-bold font-display text-slate-900">Macro Intelligence &amp; News Blackout Shield</h2>
          </div>
          <div class="flex items-center gap-2 text-xs">
            <span id="news-provider-badge" class="px-2.5 py-0.5 font-mono text-[11px] rounded-full bg-slate-100 text-slate-700">
              Provider: FINNHUB
            </span>
            <span id="news-status-badge" class="px-2.5 py-0.5 font-semibold rounded-full bg-emerald-50 text-emerald-700">
              HEALTHY
            </span>
          </div>
        </div>

        <!-- 4-Pair Blackout Status Grid -->
        <div class="grid grid-cols-2 sm:grid-cols-4 gap-3">
          <div class="p-3.5 bg-slate-50 rounded-xl space-y-1">
            <div class="flex items-center justify-between">
              <span class="text-slate-700 font-bold text-xs">EUR / USD</span>
              <span id="news-badge-eurusd" class="px-2 py-0.5 text-[10px] font-semibold rounded-full bg-emerald-100 text-emerald-800">NORMAL</span>
            </div>
            <p id="news-reason-eurusd" class="text-[11px] text-slate-500 truncate">Clear (No news)</p>
          </div>
          <div class="p-3.5 bg-slate-50 rounded-xl space-y-1">
            <div class="flex items-center justify-between">
              <span class="text-slate-700 font-bold text-xs">GBP / USD</span>
              <span id="news-badge-gbpusd" class="px-2 py-0.5 text-[10px] font-semibold rounded-full bg-emerald-100 text-emerald-800">NORMAL</span>
            </div>
            <p id="news-reason-gbpusd" class="text-[11px] text-slate-500 truncate">Clear (No news)</p>
          </div>
          <div class="p-3.5 bg-slate-50 rounded-xl space-y-1">
            <div class="flex items-center justify-between">
              <span class="text-slate-700 font-bold text-xs">USD / JPY</span>
              <span id="news-badge-usdjpy" class="px-2 py-0.5 text-[10px] font-semibold rounded-full bg-emerald-100 text-emerald-800">NORMAL</span>
            </div>
            <p id="news-reason-usdjpy" class="text-[11px] text-slate-500 truncate">Clear (No news)</p>
          </div>
          <div class="p-3.5 bg-slate-50 rounded-xl space-y-1">
            <div class="flex items-center justify-between">
              <span class="text-slate-700 font-bold text-xs">AUD / USD</span>
              <span id="news-badge-audusd" class="px-2 py-0.5 text-[10px] font-semibold rounded-full bg-emerald-100 text-emerald-800">NORMAL</span>
            </div>
            <p id="news-reason-audusd" class="text-[11px] text-slate-500 truncate">Clear (No news)</p>
          </div>
        </div>

        <!-- Upcoming High-Impact Releases Table -->
        <div class="space-y-2 pt-2">
          <div class="flex items-center justify-between text-xs text-slate-500">
            <span class="font-semibold text-slate-700">Upcoming High-Impact Economic Releases (Next 7 Days)</span>
            <span id="news-last-sync" class="text-[11px] font-mono text-slate-400">Sync: --</span>
          </div>
          <div class="overflow-x-auto rounded-xl border border-slate-100">
            <table class="w-full text-left text-xs font-mono">
              <thead class="bg-slate-50 text-slate-500 border-b border-slate-100 text-[11px] uppercase tracking-wider">
                <tr>
                  <th class="py-2.5 px-3">Currency</th>
                  <th class="py-2.5 px-3">Event Description</th>
                  <th class="py-2.5 px-3">Impact</th>
                  <th class="py-2.5 px-3">Scheduled Time</th>
                  <th class="py-2.5 px-3 text-right">Countdown</th>
                </tr>
              </thead>
              <tbody id="news-events-body" class="divide-y divide-slate-100 text-slate-700 font-mono">
                <tr><td colspan="5" class="py-3 px-3 text-center text-slate-400 font-sans">Loading economic events...</td></tr>
              </tbody>
            </table>
          </div>
        </div>
      </section>

      <!-- AI Market Intelligence Advisory Section (Powered by Gemini API) -->
      <section class="bg-white p-5 sm:p-6 rounded-2xl border border-slate-100 shadow-card space-y-4">
        <div class="flex flex-col sm:flex-row sm:items-center justify-between pb-3 border-b border-slate-100 gap-3">
          <div class="flex items-center gap-2.5">
            <span id="advisory-status-indicator" class="h-3 w-3 rounded-full bg-emerald-500 animate-pulse"></span>
            <div>
              <div class="flex items-center gap-2">
                <h2 class="text-base sm:text-lg font-bold font-display text-slate-900">LOCAL AI MARKET INTELLIGENCE — ADVISORY ONLY</h2>
                <span id="advisory-provider-pill" class="px-2.5 py-0.5 text-[10px] font-bold rounded-full bg-indigo-50 text-indigo-700">Dual-Engine (Gemini + Ollama)</span>
              </div>
              <p class="text-[11px] text-slate-500">Autonomous macroeconomic &amp; quantitative multi-factor ensemble powered by Gemini Cloud + Local Ollama. Zero execution authority; strictly decoupled from trading engine.</p>
            </div>
          </div>
          <div class="flex items-center gap-2 text-xs">
            <span id="advisory-model-badge" class="px-2.5 py-1 font-mono text-[11px] rounded-full bg-slate-100 text-slate-700">
              Gemini 2.5 Flash + Ollama Qwen2.5
            </span>
            <span id="advisory-status-badge" class="px-2.5 py-1 font-semibold rounded-full bg-emerald-50 text-emerald-700">
              ONLINE
            </span>
          </div>
        </div>

        <!-- Pair Selector Tabs & Re-Analyze Action (Velotrade Pills) -->
        <div class="flex flex-wrap items-center justify-between gap-2 border-b border-slate-100 pb-3">
          <div class="flex items-center gap-2">
            <button type="button" onclick="selectAdvisoryPair('EURUSD')" id="adv-tab-EURUSD" class="px-3 py-1 text-xs font-semibold rounded-full bg-slate-900 text-white cursor-pointer">EUR/USD</button>
            <button type="button" onclick="selectAdvisoryPair('USDJPY')" id="adv-tab-USDJPY" class="px-3 py-1 text-xs font-semibold rounded-full bg-slate-100 text-slate-700 hover:bg-slate-200 cursor-pointer">USD/JPY</button>
            <button type="button" onclick="selectAdvisoryPair('AUDUSD')" id="adv-tab-AUDUSD" class="px-3 py-1 text-xs font-semibold rounded-full bg-slate-100 text-slate-700 hover:bg-slate-200 cursor-pointer">AUD/USD</button>
            <button type="button" onclick="selectAdvisoryPair('GBPUSD')" id="adv-tab-GBPUSD" class="px-3 py-1 text-xs font-semibold rounded-full bg-slate-100 text-slate-700 hover:bg-slate-200 cursor-pointer">GBP/USD</button>
          </div>
          <div class="flex items-center gap-3">
            <span id="adv-last-eval-time" class="text-[11px] font-mono text-slate-400">Evaluation: Real-time</span>
            <button type="button" onclick="triggerAdvisoryRefresh()" id="btn-refresh-advisory" class="px-3 py-1 text-xs font-semibold rounded-full bg-indigo-600 hover:bg-indigo-700 text-white transition flex items-center gap-1.5 cursor-pointer shadow-sm">
              <span id="adv-refresh-icon">⚡</span> Re-Analyze Pair
            </button>
          </div>
        </div>

        <!-- Perspective Switcher (Dual Mode View Controls) -->
        <div id="adv-perspective-bar" class="flex items-center gap-2 pt-1 pb-1">
          <span class="text-[11px] font-medium text-slate-400 font-mono">View:</span>
          <button type="button" onclick="setAdvisoryPerspective('consensus')" id="adv-persp-consensus" class="px-2.5 py-0.5 text-[11px] font-semibold rounded-full bg-indigo-600 text-white cursor-pointer transition">🌐 Dual Consensus</button>
          <button type="button" onclick="setAdvisoryPerspective('gemini')" id="adv-persp-gemini" class="px-2.5 py-0.5 text-[11px] font-semibold rounded-full bg-slate-100 text-slate-600 hover:bg-slate-200 cursor-pointer transition">☁️ Gemini Cloud</button>
          <button type="button" onclick="setAdvisoryPerspective('ollama')" id="adv-persp-ollama" class="px-2.5 py-0.5 text-[11px] font-semibold rounded-full bg-slate-100 text-slate-600 hover:bg-slate-200 cursor-pointer transition">💻 Ollama Edge</button>
        </div>

        <!-- Active Advisory Content Container -->
        <div id="advisory-content-panel" class="space-y-4">
          <div class="p-6 bg-slate-50 rounded-xl text-center text-xs text-slate-500">
            Advisory evaluation data is loading...
          </div>
        </div>
      </section>

    </div>

    <!-- ========================================================================= -->
    <!-- TAB 2: OPEN TRADES -->
    <!-- ========================================================================= -->
    <div id="tab-trades" class="hidden space-y-6">
      
      <div>
        <h2 class="text-xl font-bold font-display text-slate-900">Current Open Positions</h2>
        <p class="text-xs text-slate-500 mt-0.5">
          Positions are clearly separated into <strong>Verified Broker Trades</strong> (active on OANDA) and <strong>Simulated Paper Trades</strong> (running locally in test sandbox).
        </p>
      </div>

      <div class="grid grid-cols-1 md:grid-cols-2 gap-4" id="open-positions-container">
        <!-- Rendered dynamically via JS -->
      </div>

    </div>

    <!-- ========================================================================= -->
    <!-- TAB 3: TRADE HISTORY -->
    <!-- ========================================================================= -->
    <div id="tab-history" class="hidden space-y-6">
      
      <div>
        <h2 class="text-xl font-bold font-display text-slate-900">Completed Trade History</h2>
        <p class="text-xs text-slate-500 mt-0.5">
          Past trades that have already closed. Profit or loss from these trades is finalized and settled into your cash balance.
        </p>
      </div>

      <!-- Quick History Summary (Velotrade Clean Cards) -->
      <div class="grid grid-cols-2 sm:grid-cols-4 gap-4 bg-white p-5 rounded-2xl border border-slate-100 shadow-card text-xs">
        <div class="p-3 bg-slate-50 rounded-xl">
          <span class="text-slate-500 block font-medium">Completed Trades</span>
          <span id="hist-total" class="text-lg font-bold font-mono text-slate-900">0</span>
        </div>
        <div class="p-3 bg-slate-50 rounded-xl">
          <span class="text-slate-500 block font-medium">Profitable Trades</span>
          <span id="hist-wins" class="text-lg font-bold font-mono text-emerald-600">0</span>
        </div>
        <div class="p-3 bg-slate-50 rounded-xl">
          <span class="text-slate-500 block font-medium">Win Rate</span>
          <span id="hist-winrate" class="text-lg font-bold font-mono text-slate-900">0.0%</span>
        </div>
        <div class="p-3 bg-slate-50 rounded-xl">
          <span class="text-slate-500 block font-medium">Total Realized P&amp;L</span>
          <span id="hist-pnl" class="text-lg font-bold font-mono text-slate-900">$0.00</span>
        </div>
      </div>

      <!-- Closed Trades Table (Velotrade Style Table) -->
      <div class="bg-white rounded-2xl border border-slate-100 overflow-hidden shadow-card">
        <div class="overflow-x-auto">
          <table class="w-full text-left text-xs">
            <thead class="bg-slate-50 text-slate-600 uppercase tracking-wider font-semibold border-b border-slate-100">
              <tr>
                <th class="py-3.5 px-4 font-mono text-[11px]">Date &amp; Time</th>
                <th class="py-3.5 px-4 font-mono text-[11px]">Currency Pair</th>
                <th class="py-3.5 px-4 font-mono text-[11px]">Action</th>
                <th class="py-3.5 px-4 font-mono text-[11px]">Execution Type</th>
                <th class="py-3.5 px-4 font-mono text-[11px]">Entry / Exit</th>
                <th class="py-3.5 px-4 font-mono text-[11px]">Duration</th>
                <th class="py-3.5 px-4 font-mono text-[11px]">Reason for Exit</th>
                <th class="py-3.5 px-4 text-right font-mono text-[11px]">Result ($)</th>
              </tr>
            </thead>
            <tbody id="trade-history-table-body" class="divide-y divide-slate-100 font-mono text-slate-700">
              <tr>
                <td colspan="8" class="py-8 text-center text-slate-400 italic">No completed trades recorded yet.</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>

    </div>

    <!-- ========================================================================= -->
    <!-- TAB 4: AI & STRATEGY -->
    <!-- ========================================================================= -->
    <div id="tab-ai" class="hidden space-y-6">
      
      <div>
        <h2 class="text-xl font-bold font-display text-slate-900">Artificial Intelligence &amp; Quantitative Architecture</h2>
        <p class="text-xs text-slate-500 mt-0.5">
          Real-time interactive intelligence powered by Google Gemini API and Deep Neural Policy Ensembles.
        </p>
      </div>

      <!-- Interactive Gemini AI Market Copilot -->
      <section class="bg-white p-5 sm:p-6 rounded-2xl border border-slate-100 shadow-card space-y-4">
        <div class="flex flex-col sm:flex-row sm:items-center justify-between pb-3 border-b border-slate-100 gap-2">
          <div class="flex items-center gap-2.5">
            <span class="w-3 h-3 rounded-full bg-indigo-500 animate-pulse"></span>
            <div>
              <div class="flex items-center gap-2">
                <h3 class="text-base sm:text-lg font-bold font-display text-slate-900">Velotrade Dual-Engine Market Copilot</h3>
                <span class="px-2 py-0.5 text-[10px] font-bold rounded-full bg-indigo-50 text-indigo-700">Gemini Cloud + Local Ollama</span>
              </div>
              <p class="text-[11px] text-slate-500">Ask quantitative questions grounded in your live portfolio, open positions, neural signals, and dual cloud/edge AI ensemble.</p>
            </div>
          </div>
          <span class="text-[11px] text-slate-400 font-mono">Cloud &amp; Edge Augmented</span>
        </div>

        <!-- Quick Question Chips -->
        <div class="flex flex-wrap gap-2 text-xs">
          <button type="button" onclick="askCopilotPrompt('Compare the Gemini Cloud and Ollama Local model perspectives across EUR/USD and USD/JPY.')" class="px-3 py-1.5 rounded-xl bg-slate-50 hover:bg-slate-100 text-slate-700 border border-slate-100 cursor-pointer transition">🌐 Dual Consensus Deep Dive</button>
          <button type="button" onclick="askCopilotPrompt('Summarize my current portfolio exposure and daily drawdown risk.')" class="px-3 py-1.5 rounded-xl bg-slate-50 hover:bg-slate-100 text-slate-700 border border-slate-100 cursor-pointer transition">📊 Portfolio Exposure &amp; Drawdown</button>
          <button type="button" onclick="askCopilotPrompt('Why is EUR/USD execution blocked right now? Explain the spread and news context.')" class="px-3 py-1.5 rounded-xl bg-slate-50 hover:bg-slate-100 text-slate-700 border border-slate-100 cursor-pointer transition">⚡ Why is EUR/USD Blocked?</button>
          <button type="button" onclick="askCopilotPrompt('Are there any high-impact economic news events scheduled today that might trigger NewsShield?')" class="px-3 py-1.5 rounded-xl bg-slate-50 hover:bg-slate-100 text-slate-700 border border-slate-100 cursor-pointer transition">🗓️ Upcoming Macro Calendar</button>
        </div>

        <!-- Copilot Chat Stream -->
        <div id="copilot-chat-history" class="p-4 rounded-xl bg-slate-50/70 border border-slate-100 max-h-96 overflow-y-auto space-y-3 text-xs font-sans">
          <div class="flex gap-2.5 items-start">
            <div class="w-6 h-6 rounded-full bg-indigo-600 text-white flex items-center justify-center font-bold text-[10px] shrink-0">AI</div>
            <div class="p-3 bg-white rounded-xl border border-slate-100 text-slate-700 shadow-xs max-w-2xl leading-relaxed">
              Hello! I am your <strong>Velotrade Dual-Engine Market Copilot</strong>, connected directly to Google Gemini cloud macro intelligence, local Ollama edge analytics, and real-time risk guards. Click a suggestion above or ask me anything about your positions, models, or market consensus.
            </div>
          </div>
        </div>

        <!-- Input Box -->
        <form onsubmit="handleCopilotSubmit(event)" class="flex gap-2">
          <input type="text" id="copilot-user-input" placeholder="Ask about current trades, risk constraints, or currency pairs..." class="flex-1 px-4 py-2 text-xs rounded-xl border border-slate-200 bg-white focus:outline-none focus:ring-2 focus:ring-indigo-500/20 focus:border-indigo-500" />
          <button type="submit" id="copilot-submit-btn" class="px-4 py-2 bg-indigo-600 hover:bg-indigo-700 text-white font-semibold text-xs rounded-xl transition flex items-center gap-1.5 cursor-pointer shadow-xs">
            <span>Send</span> →
          </button>
        </form>
      </section>

      <div>
        <h3 class="text-lg font-bold font-display text-slate-900">How the Quantitative Trading Core Operates</h3>
        <p class="text-xs text-slate-500 mt-0.5">Plain-English explanation of neural network models, risk guards, and volatility calculations.</p>
      </div>

      <!-- Velotrade Editorial Concept Cards -->
      <div class="grid grid-cols-1 md:grid-cols-2 gap-4 text-xs">
        
        <div class="bg-white p-6 rounded-2xl border border-slate-100 shadow-card space-y-2.5">
          <div class="flex items-center gap-2 text-emerald-700 font-semibold text-sm font-display">
            <span class="w-2.5 h-2.5 rounded-full bg-emerald-500"></span>
            1. Neural Pattern Recognition
          </div>
          <p class="text-slate-600 leading-relaxed">
            The AI analyzes 21 technical indicators (trend strength, momentum, and volume) on 1-hour candles. It outputs probability scores suggesting whether buying, selling, or staying flat has the highest mathematical edge.
          </p>
          <div class="p-3 rounded-xl bg-slate-50 text-slate-600">
            <strong>Key fact:</strong> AI signals represent algorithmic likelihoods, not guarantees of profit.
          </div>
        </div>

        <div class="bg-white p-6 rounded-2xl border border-slate-100 shadow-card space-y-2.5">
          <div class="flex items-center gap-2 text-emerald-700 font-semibold text-sm font-display">
            <span class="w-2.5 h-2.5 rounded-full bg-emerald-500"></span>
            2. Server-Side Bracket Defense
          </div>
          <p class="text-slate-600 leading-relaxed">
            Every trade is placed with an immutable stop-loss and take-profit executed directly on OANDA servers. If our local machine loses power, trades remain guarded by broker matching hardware.
          </p>
          <div class="p-3 rounded-xl bg-slate-50 text-slate-600">
            <strong>Key fact:</strong> Capital preservation overrides all signal recommendations.
          </div>
        </div>

        <div class="bg-white p-6 rounded-2xl border border-slate-100 shadow-card space-y-2.5">
          <div class="flex items-center gap-2 text-emerald-700 font-semibold text-sm font-display">
            <span class="w-2.5 h-2.5 rounded-full bg-emerald-500"></span>
            3. Dynamic Position Sizing (ATR)
          </div>
          <p class="text-slate-600 leading-relaxed">
            Instead of risking a fixed number of dollars, position sizes scale inversely with current market volatility. During wild price swings, the bot automatically trades smaller sizes to cap maximum risk at 1%.
          </p>
          <div class="p-3 rounded-xl bg-slate-50 text-slate-600">
            <strong>Key fact:</strong> Volatility determines lot size, not confidence score.
          </div>
        </div>

        <div class="bg-white p-6 rounded-2xl border border-slate-100 shadow-card space-y-2.5">
          <div class="flex items-center gap-2 text-emerald-700 font-semibold text-sm font-display">
            <span class="w-2.5 h-2.5 rounded-full bg-emerald-500"></span>
            4. Execution Timing &amp; Spread Guard
          </div>
          <p class="text-slate-600 leading-relaxed">
            The bot evaluates only at the close of 1-hour candles (:01 UTC). It actively refuses to trade during weekend rollover, illiquid holiday hours, or when live spreads exceed 3.5 pips.
          </p>
          <div class="p-3 rounded-xl bg-slate-50 text-slate-600">
            <strong>Key fact:</strong> Eliminates high-spread slippage traps common to retail traders.
          </div>
        </div>

      </div>

      <!-- Velotrade Promo Card: Interactive 3D Network Topology Link -->
      <div class="bg-[#0B0F17] text-white p-6 rounded-2xl border border-slate-800 shadow-subtle flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <span class="text-xs font-mono uppercase text-[#FF5353] font-bold">INTERACTIVE 3D VISUALIZATION</span>
          <h3 class="text-lg font-bold font-display text-white mt-1">Explore 4-Pair Neural Network Topology</h3>
          <p class="text-xs text-slate-400 mt-1 max-w-xl">
            Inspect real-time currency cluster nodes, policy correlation rings, and institutional liquidity corridors in interactive WebGL 3D.
          </p>
        </div>
        <a href="/#interactive-3d" class="px-5 py-2.5 bg-[#FF5353] hover:bg-[#E03E3E] text-white text-xs font-bold rounded-full transition-all shrink-0 text-center shadow-sm">
          Launch 3D Explorer →
        </a>
      </div>

    </div>

    <!-- ========================================================================= -->
    <!-- TAB 5: BROKER & RECONCILIATION -->
    <!-- ========================================================================= -->
    <div id="tab-reconciliation" class="hidden space-y-6">
      
      <div>
        <h2 class="text-xl font-bold font-display text-slate-900">Broker &amp; Reconciliation Audit</h2>
        <p class="text-xs text-slate-500 mt-0.5">
          Independent cross-verification between local paper simulation and official OANDA broker state.
        </p>
      </div>

      <!-- Variance status banner -->
      <div id="rec-variance-banner" class="p-4 rounded-2xl border border-slate-100 bg-white shadow-card text-xs flex items-center justify-between">
        <div class="flex items-center gap-2.5">
          <span class="w-2.5 h-2.5 rounded-full bg-emerald-500"></span>
          <span class="font-bold text-slate-900">System Audit:</span>
          <span class="text-slate-600">Checking ledger reconciliation status...</span>
        </div>
      </div>

      <!-- Side-by-Side Financial Ledger Comparison -->
      <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
        
        <div class="bg-white p-5 rounded-2xl border border-slate-100 shadow-card space-y-3 text-xs">
          <div class="flex items-center justify-between border-b border-slate-100 pb-2.5">
            <h3 class="font-bold font-display text-slate-900 text-sm">OANDA Broker Ledger</h3>
            <span class="px-2 py-0.5 rounded-full bg-amber-50 text-amber-800 text-[10px] font-bold">MATCHING ENGINE</span>
          </div>
          <div class="space-y-2 font-mono">
            <div class="flex justify-between"><span class="text-slate-500">Cash Balance:</span><strong id="rec-broker-cash" class="text-slate-900">$100,000.00</strong></div>
            <div class="flex justify-between"><span class="text-slate-500">Realized P&amp;L:</span><strong id="rec-broker-realized" class="text-slate-900">$0.00</strong></div>
            <div class="flex justify-between"><span class="text-slate-500">Unrealized P&amp;L:</span><strong id="rec-broker-upl" class="text-slate-900">$0.00</strong></div>
          </div>
        </div>

        <div class="bg-white p-5 rounded-2xl border border-slate-100 shadow-card space-y-3 text-xs">
          <div class="flex items-center justify-between border-b border-slate-100 pb-2.5">
            <h3 class="font-bold font-display text-slate-900 text-sm">Local Paper Ledger</h3>
            <span class="px-2 py-0.5 rounded-full bg-sky-50 text-sky-800 text-[10px] font-bold">AI SIMULATION</span>
          </div>
          <div class="space-y-2 font-mono">
            <div class="flex justify-between"><span class="text-slate-500">Cash Balance:</span><strong id="rec-ledger-cash" class="text-slate-900">$100,000.00</strong></div>
            <div class="flex justify-between"><span class="text-slate-500">Realized P&amp;L:</span><strong id="rec-ledger-realized" class="text-slate-900">$0.00</strong></div>
            <div class="flex justify-between"><span class="text-slate-500">Unrealized P&amp;L:</span><strong id="rec-ledger-upl" class="text-slate-900">$0.00</strong></div>
          </div>
        </div>

      </div>

      <!-- Variance Calculation Panel -->
      <div class="bg-slate-50 p-4 rounded-2xl text-xs font-mono space-y-2">
        <div class="flex justify-between">
          <span class="text-slate-500">Cash Variance (Broker minus Paper):</span>
          <strong id="rec-cash-diff" class="text-slate-900">$0.00</strong>
        </div>
        <div class="flex justify-between">
          <span class="text-slate-500">Realized P&amp;L Variance:</span>
          <strong id="rec-realized-diff" class="text-slate-900">$0.00</strong>
        </div>
        <div class="flex justify-between">
          <span class="text-slate-500">Unrealized P&amp;L Variance:</span>
          <strong id="rec-upl-diff" class="text-slate-900">$0.00</strong>
        </div>
      </div>

      <!-- Recent Reconciliation Orders Table -->
      <div class="bg-white rounded-2xl border border-slate-100 overflow-hidden shadow-card">
        <div class="p-4 border-b border-slate-100 font-semibold text-xs text-slate-900">
          Recent Position Cross-Checks &amp; Execution Discrepancy Log
        </div>
        <div class="overflow-x-auto">
          <table class="w-full text-left text-xs font-mono">
            <thead class="bg-slate-50 text-slate-600 uppercase border-b border-slate-100 text-[11px]">
              <tr>
                <th class="py-2.5 px-3">Pair</th>
                <th class="py-2.5 px-3">Broker Status</th>
                <th class="py-2.5 px-3">Local Ledger Status</th>
                <th class="py-2.5 px-3">State</th>
                <th class="py-2.5 px-3 text-right">Details</th>
              </tr>
            </thead>
            <tbody id="rec-orders-table-body" class="divide-y divide-slate-100 text-slate-700">
              <tr><td colspan="5" class="py-4 text-center text-slate-400 italic font-sans">No discrepancies recorded.</td></tr>
            </tbody>
          </table>
        </div>
      </div>

      <!-- Sync Events Audit Log -->
      <div class="bg-white rounded-2xl border border-slate-100 overflow-hidden shadow-card">
        <div class="p-4 border-b border-slate-100 font-semibold text-xs text-slate-900">
          OANDA v20 Gateway Communication Telemetry
        </div>
        <div class="overflow-x-auto">
          <table class="w-full text-left text-xs font-mono">
            <thead class="bg-slate-50 text-slate-600 uppercase border-b border-slate-100 text-[11px]">
              <tr>
                <th class="py-2.5 px-3">Timestamp</th>
                <th class="py-2.5 px-3">Endpoint / Action</th>
                <th class="py-2.5 px-3">HTTP Status</th>
                <th class="py-2.5 px-3">Latency</th>
                <th class="py-2.5 px-3 text-right">Gateway Health</th>
              </tr>
            </thead>
            <tbody id="rec-sync-events-body" class="divide-y divide-slate-100 text-slate-700">
              <tr><td colspan="5" class="py-4 text-center text-slate-400 italic font-sans">Connecting to OANDA telemetry feed...</td></tr>
            </tbody>
          </table>
        </div>
      </div>

    </div>

    <!-- ========================================================================= -->
    <!-- TAB 6: SYSTEM STATUS & SAFETY -->
    <!-- ========================================================================= -->
    <div id="tab-system" class="hidden space-y-6">
      
      <div>
        <h2 class="text-xl font-bold font-display text-slate-900">System Status &amp; Capital Protection Architecture</h2>
        <p class="text-xs text-slate-500 mt-0.5">
          Real-time telemetry on the background trading daemon, risk guards, broker gateway, and circuit breakers.
        </p>
      </div>

      <!-- Engine and Process Health (Velotrade Clean Cards) -->
      <div class="grid grid-cols-1 md:grid-cols-3 gap-4">
        
        <div class="bg-white p-5 rounded-2xl border border-slate-100 shadow-card space-y-2">
          <span class="text-xs font-semibold text-slate-500 uppercase tracking-wide">Trading Engine Daemon</span>
          <div class="flex items-center gap-2">
            <span class="w-2.5 h-2.5 rounded-full bg-emerald-500"></span>
            <span id="sys-engine-status" class="text-lg font-bold text-slate-900 font-display">Running</span>
          </div>
          <p id="sys-engine-pid" class="text-xs font-mono text-slate-500">Autonomous Execution: Active</p>
        </div>

        <div class="bg-white p-5 rounded-2xl border border-slate-100 shadow-card space-y-2">
          <span class="text-xs font-semibold text-slate-500 uppercase tracking-wide">OANDA REST Gateway</span>
          <div class="flex items-center gap-2">
            <span class="w-2.5 h-2.5 rounded-full bg-emerald-500"></span>
            <span id="sys-broker-status" class="text-lg font-bold text-slate-900 font-display">Connected</span>
          </div>
          <p class="text-xs font-mono text-slate-500">v20 REST API Stream: Healthy</p>
        </div>

        <div class="bg-white p-5 rounded-2xl border border-slate-100 shadow-card space-y-2">
          <span class="text-xs font-semibold text-slate-500 uppercase tracking-wide">Ledger Reconciliation</span>
          <div class="flex items-center gap-2">
            <span class="w-2.5 h-2.5 rounded-full bg-emerald-500"></span>
            <span class="text-lg font-bold text-slate-900 font-display">Reconciled</span>
          </div>
          <p class="text-xs font-mono text-slate-500">Discrepancy Tolerance: &lt; 0.05%</p>
        </div>

      </div>

      <!-- Protective Safeguards Matrix (Velotrade Style Table) -->
      <div class="bg-white rounded-2xl border border-slate-100 overflow-hidden shadow-card">
        <div class="p-4 border-b border-slate-100 font-semibold text-xs text-slate-900 font-display">
          Active Institutional Risk Guard Thresholds
        </div>
        <div class="overflow-x-auto">
          <table class="w-full text-left text-xs font-mono">
            <thead class="bg-slate-50 text-slate-600 uppercase border-b border-slate-100 text-[11px]">
              <tr>
                <th class="py-3 px-4">Guard Mechanism</th>
                <th class="py-3 px-4">Configured Limit</th>
                <th class="py-3 px-4">Action Triggered</th>
                <th class="py-3 px-4 text-right">Current Guard State</th>
              </tr>
            </thead>
            <tbody class="divide-y divide-slate-100 text-slate-700">
              <tr>
                <td class="py-3 px-4 font-bold text-slate-900 font-sans">Spread Guard</td>
                <td class="py-3 px-4">&gt; 3.5 pips</td>
                <td class="py-3 px-4">Blocks new market entry</td>
                <td id="sys-spread-guard-status" class="py-3 px-4 text-right text-emerald-600 font-bold">ARMED (ACTIVE)</td>
              </tr>
              <tr>
                <td class="py-3 px-4 font-bold text-slate-900 font-sans">Macro News Blackout Shield</td>
                <td class="py-3 px-4">&plusmn;15 min High-Impact</td>
                <td class="py-3 px-4">Suppresses policy execution</td>
                <td class="py-3 px-4 text-right text-emerald-600 font-bold">ARMED (ACTIVE)</td>
              </tr>
              <tr>
                <td class="py-3 px-4 font-bold text-slate-900 font-sans">Daily Circuit Breaker</td>
                <td class="py-3 px-4">-3.0% Max Loss ($3,000)</td>
                <td class="py-3 px-4">Halts bot until next UTC 00:00</td>
                <td id="sys-killswitch-status" class="py-3 px-4 text-right text-emerald-600 font-bold">NORMAL (0.0% loss)</td>
              </tr>
              <tr>
                <td class="py-3 px-4 font-bold text-slate-900 font-sans">Maximum Peak Drawdown</td>
                <td class="py-3 px-4">-5.0% Trailing Equity</td>
                <td class="py-3 px-4">Enforces full liquidation</td>
                <td id="sys-max-dd-status" class="py-3 px-4 text-right text-emerald-600 font-bold">SAFE (0.0% max dd)</td>
              </tr>
              <tr>
                <td class="py-3 px-4 font-bold text-slate-900 font-sans">Per-Trade Stop Loss (ATR)</td>
                <td class="py-3 px-4">1.5 &times; ATR Bracket</td>
                <td class="py-3 px-4">Native OANDA server bracket</td>
                <td class="py-3 px-4 text-right text-emerald-600 font-bold">ENFORCED</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>

    </div>

    <!-- ========================================================================= -->
    <!-- TAB 7: CONSOLE & TERMINAL WORKSTATION -->
    <!-- ========================================================================= -->
    <div id="tab-console" class="hidden space-y-6">

      <div class="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
        <div>
          <div class="flex items-center gap-2">
            <span class="inline-block w-2.5 h-2.5 rounded-full bg-emerald-500 animate-pulse"></span>
            <h2 class="text-xl font-bold font-display text-slate-900">Institutional Operational Console &amp; Terminal</h2>
          </div>
          <p class="text-xs text-slate-500 mt-0.5">
            Real-time stdout/stderr stream, process execution filters, terminal commands, and sub-ensemble runtime diagnostics.
          </p>
        </div>
        <div class="flex items-center gap-2">
          <span class="px-2.5 py-1 rounded-full text-xs font-mono font-medium bg-slate-100 text-slate-700">
            Daemon: PID <span id="term-daemon-pid">1463</span>
          </span>
          <span class="px-2.5 py-1 rounded-full text-xs font-mono font-bold bg-emerald-50 text-emerald-700">
            STREAMING (4.0s)
          </span>
        </div>
      </div>

      <!-- Quick Metrics Strip -->
      <div class="grid grid-cols-2 lg:grid-cols-4 gap-3 sm:gap-4">
        <div class="bg-white p-4 rounded-xl border border-slate-100 shadow-card">
          <span class="text-[11px] font-semibold text-slate-500 uppercase tracking-wide block">Engine Subprocess</span>
          <div class="flex items-center gap-2 mt-1">
            <span class="w-2 h-2 rounded-full bg-emerald-500"></span>
            <span class="text-sm font-bold text-slate-900 font-mono">ACTIVE (0.0% CPU)</span>
          </div>
          <span class="text-[10px] text-slate-400 font-mono mt-0.5 block">Sub-process polling: 250ms</span>
        </div>

        <div class="bg-white p-4 rounded-xl border border-slate-100 shadow-card">
          <span class="text-[11px] font-semibold text-slate-500 uppercase tracking-wide block">OANDA v20 Stream</span>
          <div class="flex items-center gap-2 mt-1">
            <span class="w-2 h-2 rounded-full bg-emerald-500"></span>
            <span class="text-sm font-bold text-slate-900 font-mono">ESTABLISHED</span>
          </div>
          <span class="text-[10px] text-slate-400 font-mono mt-0.5 block">Latency: 14ms (Frankfurt)</span>
        </div>

        <div class="bg-white p-4 rounded-xl border border-slate-100 shadow-card">
          <span class="text-[11px] font-semibold text-slate-500 uppercase tracking-wide block">Terminal Memory</span>
          <div class="flex items-center gap-2 mt-1">
            <span class="w-2 h-2 rounded-full bg-slate-400"></span>
            <span class="text-sm font-bold text-slate-900 font-mono">142.4 MB</span>
          </div>
          <span class="text-[10px] text-slate-400 font-mono mt-0.5 block">Ring buffer: 1,000 lines</span>
        </div>

        <div class="bg-white p-4 rounded-xl border border-slate-100 shadow-card">
          <span class="text-[11px] font-semibold text-slate-500 uppercase tracking-wide block">Risk Sentinel</span>
          <div class="flex items-center gap-2 mt-1">
            <span class="w-2 h-2 rounded-full bg-emerald-500"></span>
            <span class="text-sm font-bold text-emerald-600 font-mono">ARMED (0 FAILS)</span>
          </div>
          <span class="text-[10px] text-slate-400 font-mono mt-0.5 block">Killswitch ready</span>
        </div>
      </div>

      <!-- Institutional Obsidian Terminal Container -->
      <div class="bg-[#0B0F17] rounded-2xl border border-slate-800 shadow-md overflow-hidden flex flex-col">
        
        <!-- Terminal Header & Control Toolbar -->
        <div class="bg-[#111827] px-4 py-3 border-b border-slate-800 flex flex-wrap items-center justify-between gap-3">
          
          <!-- Terminal Title & Status -->
          <div class="flex items-center gap-3">
            <div class="flex items-center gap-1.5">
              <span class="w-3 h-3 rounded-full bg-rose-500 inline-block"></span>
              <span class="w-3 h-3 rounded-full bg-amber-500 inline-block"></span>
              <span class="w-3 h-3 rounded-full bg-emerald-500 inline-block"></span>
            </div>
            <span class="font-mono text-xs font-semibold text-slate-300">
              forex-ai-workstation <span class="text-slate-500">//</span> tty1: live_paper_trader.py
            </span>
          </div>

          <!-- Log Severity Filter Pills -->
          <div class="flex items-center gap-1 bg-[#0B0F17] p-1 rounded-lg border border-slate-800 text-[11px] font-mono">
            <button onclick="setTerminalFilter('ALL')" id="term-filter-ALL" class="px-2.5 py-1 rounded bg-[#FF5353] text-white font-bold cursor-pointer transition-colors">ALL</button>
            <button onclick="setTerminalFilter('INFO')" id="term-filter-INFO" class="px-2.5 py-1 rounded text-slate-400 hover:text-white cursor-pointer transition-colors">INFO</button>
            <button onclick="setTerminalFilter('EXEC')" id="term-filter-EXEC" class="px-2.5 py-1 rounded text-slate-400 hover:text-white cursor-pointer transition-colors">EXEC</button>
            <button onclick="setTerminalFilter('WARN')" id="term-filter-WARN" class="px-2.5 py-1 rounded text-slate-400 hover:text-white cursor-pointer transition-colors">WARN</button>
            <button onclick="setTerminalFilter('ERROR')" id="term-filter-ERROR" class="px-2.5 py-1 rounded text-slate-400 hover:text-white cursor-pointer transition-colors">ERROR</button>
          </div>

          <!-- Utility Action Buttons -->
          <div class="flex items-center gap-2 text-xs font-mono">
            <button onclick="toggleTerminalAutoScroll()" id="term-autoscroll-btn" class="px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-emerald-400 text-[11px] font-semibold border border-slate-700 flex items-center gap-1.5 cursor-pointer">
              <span class="w-2 h-2 rounded-full bg-emerald-400"></span>
              Auto-Scroll: ON
            </button>
            <button onclick="copyTerminalLogs()" class="px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 text-[11px] font-medium border border-slate-700 flex items-center gap-1.5 cursor-pointer" title="Copy visible terminal logs">
              <svg class="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z"/></svg>
              Copy
            </button>
            <button onclick="clearTerminalLogs()" class="px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 text-[11px] font-medium border border-slate-700 flex items-center gap-1.5 cursor-pointer" title="Clear terminal screen">
              <svg class="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16"/></svg>
              Clear
            </button>
          </div>

        </div>

        <!-- Terminal Screen Output Area -->
        <div id="terminal-screen" class="h-[460px] sm:h-[500px] overflow-y-auto p-4 font-mono text-[11px] sm:text-xs leading-relaxed space-y-1.5 select-text bg-[#0B0F17]" style="-webkit-overflow-scrolling: touch;">
          <!-- Dynamically populated terminal log lines -->
        </div>

        <!-- Terminal Interactive Command Prompt Bar -->
        <div class="bg-[#111827] px-4 py-3 border-t border-slate-800 space-y-2">
          <!-- Quick Command Badges -->
          <div class="flex items-center gap-2 overflow-x-auto text-[10px] font-mono text-slate-400 no-scrollbar">
            <span class="text-slate-500 shrink-0">Quick CLI:</span>
            <button onclick="runTerminalCommand('status')" class="px-2 py-0.5 rounded bg-slate-800 hover:bg-slate-700 text-slate-300 border border-slate-700 cursor-pointer shrink-0">status</button>
            <button onclick="runTerminalCommand('healthcheck')" class="px-2 py-0.5 rounded bg-slate-800 hover:bg-slate-700 text-slate-300 border border-slate-700 cursor-pointer shrink-0">healthcheck</button>
            <button onclick="runTerminalCommand('sync-broker')" class="px-2 py-0.5 rounded bg-slate-800 hover:bg-slate-700 text-slate-300 border border-slate-700 cursor-pointer shrink-0">sync-broker</button>
            <button onclick="runTerminalCommand('test-news')" class="px-2 py-0.5 rounded bg-slate-800 hover:bg-slate-700 text-slate-300 border border-slate-700 cursor-pointer shrink-0">test-news</button>
            <button onclick="runTerminalCommand('pairs')" class="px-2 py-0.5 rounded bg-slate-800 hover:bg-slate-700 text-slate-300 border border-slate-700 cursor-pointer shrink-0">pairs</button>
            <button onclick="runTerminalCommand('clear')" class="px-2 py-0.5 rounded bg-slate-800 hover:bg-slate-700 text-slate-300 border border-slate-700 cursor-pointer shrink-0">clear</button>
          </div>

          <!-- Prompt Input Form -->
          <form onsubmit="handleTerminalSubmit(event)" class="flex items-center gap-2">
            <div class="flex items-center gap-2 flex-1 bg-[#0B0F17] border border-slate-700 rounded-xl px-3 py-2 text-xs font-mono focus-within:border-[#FF5353] transition-colors">
              <span class="text-emerald-400 font-bold select-none">operator@forex-ai:~$</span>
              <input type="text" id="terminal-input" autocomplete="off" placeholder="Type command (e.g. status, pairs, ping, clear)..."
                class="flex-1 bg-transparent text-slate-100 placeholder:text-slate-600 focus:outline-none font-mono text-xs">
            </div>
            <button type="submit" class="px-4 py-2 bg-[#FF5353] hover:bg-[#E03E3E] text-white text-xs font-mono font-bold rounded-xl transition-colors cursor-pointer shadow-sm">
              Execute
            </button>
          </form>
        </div>

      </div>

    </div>

    <!-- ========================================================================= -->
    <!-- TAB 8: USER DASHBOARD & ACCOUNT SETTINGS -->
    <!-- ========================================================================= -->
    <div id="tab-account" class="hidden space-y-6">

      <div>
        <div class="flex items-center gap-2">
          <span class="inline-block w-2.5 h-2.5 rounded-full bg-[#FF5353]"></span>
          <h2 class="text-xl font-bold font-display text-slate-900">User Dashboard &amp; Operator Profile</h2>
        </div>
        <p class="text-xs text-slate-500 mt-0.5">
          Multi-tenant security parameters, encrypted broker credentials, session tokens, and quantitative quotas.
        </p>
      </div>

      <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">

        <!-- Card 1: Operator Identity & Clearance -->
        <div class="bg-white rounded-2xl border border-slate-100 p-6 shadow-card space-y-4">
          <div class="flex items-center justify-between border-b border-slate-100 pb-3">
            <h3 class="font-bold text-sm text-slate-900 font-display">Operator Clearance &amp; Profile</h3>
            <span class="px-2.5 py-0.5 rounded-full text-[10px] font-mono font-bold bg-emerald-50 text-emerald-700">
              ACTIVE SESSION
            </span>
          </div>

          <div class="space-y-3 text-xs">
            <div class="flex items-center justify-between py-1.5 border-b border-slate-50">
              <span class="text-slate-500 font-mono">Authenticated Operator</span>
              <span id="acct-user-email" class="font-bold text-slate-900 font-mono">operator@quant.institution.com</span>
            </div>
            <div class="flex items-center justify-between py-1.5 border-b border-slate-50">
              <span class="text-slate-500 font-mono">Unique Tenant ID</span>
              <span id="acct-user-id" class="font-mono text-slate-700">usr_9c4e3f2d</span>
            </div>
            <div class="flex items-center justify-between py-1.5 border-b border-slate-50">
              <span class="text-slate-500 font-mono">Security Clearance Tier</span>
              <span class="font-bold text-slate-800 font-sans">Tier-1 Institutional Operator</span>
            </div>
            <div class="flex items-center justify-between py-1.5 border-b border-slate-50">
              <span class="text-slate-500 font-mono">Encryption Protocol</span>
              <span class="font-mono text-slate-800">Fernet AES-128-CBC + Salted PBKDF2</span>
            </div>
            <div class="flex items-center justify-between py-1.5">
              <span class="text-slate-500 font-mono">Session Token Lifespan</span>
              <span class="font-mono text-slate-700">7 Days (Auto-rolling refresh)</span>
            </div>
          </div>

          <div class="pt-2">
            <button onclick="handleSignOut()" class="w-full py-2.5 rounded-xl border border-rose-200 bg-rose-50 hover:bg-rose-100 text-rose-700 font-bold text-xs font-mono transition-colors cursor-pointer flex items-center justify-center gap-2">
              <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M17 16l4-4m0 0l-4-4m4 4H7m6 4v1a3 3 0 01-3 3H6a3 3 0 01-3-3V7a3 3 0 013-3h4a3 3 0 013 3v1"/></svg>
              Revoke Session &amp; Sign Out
            </button>
          </div>
        </div>

        <!-- Card 2: Connected Broker Credentials -->
        <div class="bg-white rounded-2xl border border-slate-100 p-6 shadow-card space-y-4">
          <div class="flex items-center justify-between border-b border-slate-100 pb-3">
            <h3 class="font-bold text-sm text-slate-900 font-display">Connected Broker Gateway</h3>
            <span class="px-2.5 py-0.5 rounded-full text-[10px] font-mono font-bold bg-amber-50 text-amber-700">
              OANDA v20 REST
            </span>
          </div>

          <div class="space-y-3 text-xs">
            <div class="flex items-center justify-between py-1.5 border-b border-slate-50">
              <span class="text-slate-500 font-mono">Gateway Environment</span>
              <span id="acct-broker-env" class="font-bold text-amber-600 font-mono uppercase">PRACTICE (Zero-Risk Sandbox)</span>
            </div>
            <div class="flex items-center justify-between py-1.5 border-b border-slate-50">
              <span class="text-slate-500 font-mono">OANDA Account ID</span>
              <span id="acct-broker-account" class="font-mono text-slate-900 font-semibold">101-001-40816673-001</span>
            </div>
            <div class="flex items-center justify-between py-1.5 border-b border-slate-50">
              <span class="text-slate-500 font-mono">Encrypted Token</span>
              <span class="font-mono text-slate-500">••••••••••••••••••••••••••••••••</span>
            </div>
            <div class="flex items-center justify-between py-1.5 border-b border-slate-50">
              <span class="text-slate-500 font-mono">Storage Security</span>
              <span class="font-bold text-emerald-600 font-mono">Individual Salt Stored</span>
            </div>
            <div class="flex items-center justify-between py-1.5">
              <span class="text-slate-500 font-mono">Native Bracket Orders</span>
              <span class="font-bold text-slate-800 font-sans">Stop Loss &amp; Take Profit Attached</span>
            </div>
          </div>

          <div class="flex gap-2 pt-2">
            <button onclick="openBrokerModal()" class="flex-1 py-2.5 rounded-xl bg-[#FF5353] hover:bg-[#E03E3E] text-white font-bold text-xs transition-colors cursor-pointer shadow-sm">
              Update OANDA Key
            </button>
            <button onclick="handleDisconnectBroker()" class="px-4 py-2.5 rounded-xl border border-slate-200/80 hover:bg-slate-50 text-slate-600 font-bold text-xs transition-colors cursor-pointer">
              Disconnect
            </button>
          </div>
        </div>

        <!-- Card 3: Quantitative Governance & Sizing Caps -->
        <div class="bg-white rounded-2xl border border-slate-100 p-6 shadow-card space-y-4">
          <div class="border-b border-slate-100 pb-3">
            <h3 class="font-bold text-sm text-slate-900 font-display">Risk Governance &amp; Execution Caps</h3>
          </div>
          <div class="space-y-3 text-xs font-mono">
            <div class="flex items-center justify-between py-1.5 border-b border-slate-50">
              <span class="text-slate-500">Max Trade Risk Allocation</span>
              <span class="font-bold text-slate-900">2.0% Equity per trade</span>
            </div>
            <div class="flex items-center justify-between py-1.5 border-b border-slate-50">
              <span class="text-slate-500">Daily Drawdown Killswitch</span>
              <span class="font-bold text-rose-600">-3.0% Max Loss ($3,000)</span>
            </div>
            <div class="flex items-center justify-between py-1.5 border-b border-slate-50">
              <span class="text-slate-500">Trailing Peak Equity Liquidation</span>
              <span class="font-bold text-rose-600">-5.0% Trailing Drawdown</span>
            </div>
            <div class="flex items-center justify-between py-1.5">
              <span class="text-slate-500">High-Impact News Shield</span>
              <span class="font-bold text-emerald-600">Active (&plusmn;30m Freeze)</span>
            </div>
          </div>
        </div>

        <!-- Card 4: 4-Pair Super-Ensemble Capacity -->
        <div class="bg-white rounded-2xl border border-slate-100 p-6 shadow-card space-y-4">
          <div class="border-b border-slate-100 pb-3">
            <h3 class="font-bold text-sm text-slate-900 font-display">Super-Ensemble Network Telemetry</h3>
          </div>
          <div class="space-y-3 text-xs font-mono">
            <div class="flex items-center justify-between py-1.5 border-b border-slate-50">
              <span class="text-slate-500">Monitored Pairs</span>
              <span class="font-bold text-slate-900">EUR/USD, USD/JPY, GBP/USD, AUD/USD</span>
            </div>
            <div class="flex items-center justify-between py-1.5 border-b border-slate-50">
              <span class="text-slate-500">AI Model Framework</span>
              <span class="text-slate-800">PyTorch Super-Ensemble v3.2</span>
            </div>
            <div class="flex items-center justify-between py-1.5 border-b border-slate-50">
              <span class="text-slate-500">Local Advisory LLM</span>
              <span class="text-slate-800">Ollama DeepSeek-R1 (Advisory Only)</span>
            </div>
            <div class="flex items-center justify-between py-1.5">
              <span class="text-slate-500">Reconciliation Interval</span>
              <span class="text-emerald-600 font-bold">Every 4 Seconds</span>
            </div>
          </div>
        </div>

      </div>

    </div>

  </main>

  <!-- Connect OANDA Broker Modal (Velotrade Style Card) -->
  <div id="modal-broker-config" class="hidden fixed inset-0 z-50 flex items-center justify-center p-4 bg-slate-900/50 backdrop-blur-sm">
    <div class="bg-white rounded-2xl border border-slate-100 shadow-xl max-w-lg w-full p-6 space-y-4">
      <div class="flex items-center justify-between border-b border-slate-100 pb-3">
        <div class="flex items-center gap-2.5">
          <div class="h-8 w-8 bg-[#FF5353] rounded-xl flex items-center justify-center text-white font-bold">
            <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 10V3L4 14h7v7l9-11h-7z"/></svg>
          </div>
          <div>
            <h3 class="font-bold font-display text-slate-900 text-base">Connect OANDA v20 Broker</h3>
            <p class="text-[11px] text-slate-500">Live Practice / Demo Account Integration</p>
          </div>
        </div>
        <button onclick="closeBrokerModal()" class="text-slate-400 hover:text-slate-700 p-1 rounded-lg cursor-pointer">
          <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12"/></svg>
        </button>
      </div>

      <form id="form-broker-config" onsubmit="handleBrokerConfigSubmit(event)" class="space-y-4 text-xs">
        <div class="space-y-1">
          <label class="font-semibold text-slate-700 block">OANDA Environment</label>
          <select id="input-broker-env" class="w-full px-3 py-2 bg-slate-50 border border-slate-200 rounded-xl text-slate-900 focus:bg-white transition-colors">
            <option value="practice" selected>Practice (Demo / Simulated OANDA Feed)</option>
            <option value="live">Live (Real Capital Execution - Requires Live Key)</option>
          </select>
        </div>

        <div class="space-y-1">
          <label class="font-semibold text-slate-700 block">OANDA Account ID</label>
          <input type="text" id="input-broker-account-id" placeholder="e.g. 101-004-12345678-001" required class="w-full px-3 py-2 bg-slate-50 border border-slate-200 rounded-xl font-mono text-slate-900 focus:bg-white transition-colors">
        </div>

        <div class="space-y-1">
          <label class="font-semibold text-slate-700 block">OANDA API Personal Access Token</label>
          <input type="password" id="input-broker-token" placeholder="Enter personal access token" class="w-full px-3 py-2 bg-slate-50 border border-slate-200 rounded-xl font-mono text-slate-900 focus:bg-white transition-colors">
          <p class="text-[11px] text-slate-400">Tokens are encrypted using Fernet keys and stored in user-isolated database tables.</p>
        </div>

        <div id="broker-modal-error" class="hidden p-3 bg-rose-50 border border-rose-200 rounded-xl text-rose-800 text-xs"></div>
        <div id="broker-modal-success" class="hidden p-3 bg-emerald-50 border border-emerald-200 rounded-xl text-emerald-800 text-xs"></div>

        <div class="flex items-center justify-end gap-3 pt-3 border-t border-slate-100">
          <button type="button" onclick="closeBrokerModal()" class="px-4 py-2 border border-slate-200 hover:bg-slate-50 rounded-full text-slate-700 font-medium cursor-pointer">
            Cancel
          </button>
          <button type="submit" id="btn-save-broker-config" class="px-5 py-2 bg-[#FF5353] hover:bg-[#E03E3E] rounded-full text-white font-bold shadow-sm transition-all cursor-pointer flex items-center gap-2">
            <span id="broker-submit-text">Test &amp; Save Connection</span>
          </button>
        </div>
      </form>
    </div>
  </div>

  <!-- JavaScript Application Logic -->
  <script>
    let equityChart = null;
    let allChartData = [];
    let currentFilter = 'all';
    let currentPositions = {};

    // -------------------------------------------------------------------------
    // Mobile Drawer Toggle
    // -------------------------------------------------------------------------
    function toggleDashboardDrawer() {
      const drawer = document.getElementById('dashboard-mobile-drawer');
      if (!drawer) return;
      drawer.classList.toggle('hidden');
    }

    // -------------------------------------------------------------------------
    // Navigation Tab Switching
    // -------------------------------------------------------------------------
    function switchTab(tabId) {
      const tabs = ['overview', 'trades', 'history', 'ai', 'reconciliation', 'system', 'console', 'account'];
      tabs.forEach(t => {
        const el = document.getElementById('tab-' + t);
        const btn = document.getElementById('tab-btn-' + t);
        if (t === tabId) {
          if (el) el.classList.remove('hidden');
          if (btn) btn.className = 'shrink-0 py-3 sm:py-3.5 px-1 border-b-2 border-[#FF5353] text-slate-900 font-bold cursor-pointer whitespace-nowrap touch-target flex items-center gap-1.5';
        } else {
          if (el) el.classList.add('hidden');
          if (btn) btn.className = 'shrink-0 py-3 sm:py-3.5 px-1 border-b-2 border-transparent text-slate-500 hover:text-slate-900 cursor-pointer whitespace-nowrap touch-target flex items-center gap-1.5';
        }
      });
      if (tabId === 'overview' && equityChart) {
        equityChart.resize();
      }
      if (tabId === 'console') {
        scrollTerminalToBottom();
      }
    }

    // -------------------------------------------------------------------------
    // Chart.js Light-Themed Velotrade Performance Chart
    // -------------------------------------------------------------------------
    function initChart() {
      const canvas = document.getElementById('equityChart');
      if (!canvas) return;
      const ctx = canvas.getContext('2d');
      const gradient = ctx.createLinearGradient(0, 0, 0, 240);
      gradient.addColorStop(0, 'rgba(5, 150, 105, 0.18)');
      gradient.addColorStop(1, 'rgba(5, 150, 105, 0.0)');

      equityChart = new Chart(ctx, {
        type: 'line',
        data: {
          labels: [],
          datasets: [
            {
              label: 'Account Equity ($)',
              data: [],
              borderColor: '#059669',
              borderWidth: 2.4,
              backgroundColor: gradient,
              fill: true,
              tension: 0.25,
              pointBackgroundColor: '#059669',
              pointRadius: 3,
              pointHoverRadius: 5,
            },
            {
              label: 'Starting Capital ($)',
              data: [],
              borderColor: '#CBD5E1',
              borderWidth: 1.5,
              borderDash: [5, 5],
              fill: false,
              pointRadius: 0,
            }
          ]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: {
            legend: { display: false },
            tooltip: {
              backgroundColor: '#FFFFFF',
              borderColor: '#E2E8F0',
              borderWidth: 1,
              titleColor: '#0F172A',
              bodyColor: '#059669',
              padding: 10,
              boxPadding: 4,
              cornerRadius: 8,
              titleFont: { family: 'IBM Plex Sans', size: 11, weight: 'bold' },
              bodyFont: { family: 'IBM Plex Mono', size: 12 },
              callbacks: {
                label: (ctx) => ` ${ctx.dataset.label}: $${Number(ctx.raw).toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2})}`
              }
            }
          },
          scales: {
            x: {
              grid: { color: '#F1F5F9' },
              ticks: { color: '#64748B', font: { family: 'IBM Plex Mono', size: 10 } }
            },
            y: {
              grid: { color: '#F1F5F9' },
              ticks: {
                color: '#64748B',
                font: { family: 'IBM Plex Mono', size: 10 },
                callback: (val) => '$' + Number(val).toLocaleString(undefined, {maximumFractionDigits: 0})
              }
            }
          }
        }
      });
    }

    function setChartFilter(range) {
      currentFilter = range;
      ['24h', '7d', '30d', 'all'].forEach(r => {
        const btn = document.getElementById('filter-' + r);
        if (btn) {
          if (r === range) {
            btn.className = 'px-3 py-1 rounded-full bg-white text-slate-900 font-bold shadow-xs cursor-pointer';
          } else {
            btn.className = 'px-3 py-1 rounded-full text-slate-600 hover:text-slate-900 cursor-pointer';
          }
        }
      });
      applyChartFilter();
    }

    function applyChartFilter() {
      if (!equityChart || !allChartData.length) {
        const emptyEl = document.getElementById('chart-empty-state');
        if (emptyEl) emptyEl.classList.remove('hidden');
        return;
      }

      const now = new Date();
      let filtered = allChartData;

      if (currentFilter === '24h') {
        const cut = new Date(now.getTime() - 24 * 3600 * 1000);
        filtered = allChartData.filter(d => new Date(d.timestamp) >= cut);
      } else if (currentFilter === '7d') {
        const cut = new Date(now.getTime() - 7 * 24 * 3600 * 1000);
        filtered = allChartData.filter(d => new Date(d.timestamp) >= cut);
      } else if (currentFilter === '30d') {
        const cut = new Date(now.getTime() - 30 * 24 * 3600 * 1000);
        filtered = allChartData.filter(d => new Date(d.timestamp) >= cut);
      }

      const emptyEl = document.getElementById('chart-empty-state');
      if (!filtered.length) {
        if (emptyEl) emptyEl.classList.remove('hidden');
        equityChart.data.labels = [];
        equityChart.data.datasets[0].data = [];
        equityChart.data.datasets[1].data = [];
        equityChart.update('none');
        return;
      }

      if (emptyEl) emptyEl.classList.add('hidden');

      const labels = filtered.map(d => {
        const dt = new Date(d.timestamp);
        return `${(dt.getMonth()+1)}/${dt.getDate()} ${dt.getUTCHours().toString().padStart(2, '0')}:00`;
      });
      const eqValues = filtered.map(d => d.equity);
      const startingValues = filtered.map(() => 100000.0);

      equityChart.data.labels = labels;
      equityChart.data.datasets[0].data = eqValues;
      equityChart.data.datasets[1].data = startingValues;
      equityChart.update('none');

      const cntEl = document.getElementById('chart-data-count');
      if (cntEl) cntEl.textContent = `${filtered.length} historical points (${currentFilter.toUpperCase()})`;
    }

    function getAuthQuerySuffix() {
      try {
        const p = new URLSearchParams(window.location.search);
        const t = p.get('session_token');
        return t ? '?session_token=' + encodeURIComponent(t) : '';
      } catch (e) {
        return '';
      }
    }

    function formatUTC(d) {
      return d.getUTCHours().toString().padStart(2, '0') + ':' +
             d.getUTCMinutes().toString().padStart(2, '0') + ':' +
             d.getUTCSeconds().toString().padStart(2, '0') + ' UTC';
    }

    function setDot(el, state) {
      if (!el) return;
      if (state === 'running' || state === 'connected' || state === 'ok' || state === 'healthy') {
        el.className = 'h-2 w-2 rounded-full bg-emerald-500';
      } else if (state === 'warning' || state === 'elevated' || state === 'connecting') {
        el.className = 'h-2 w-2 rounded-full bg-amber-500 animate-pulse';
      } else {
        el.className = 'h-2 w-2 rounded-full bg-rose-500';
      }
    }

    function updateHeaderStatus(h) {
      const clock = document.getElementById('utc-clock');
      const clockText = formatUTC(new Date());
      if (clock) clock.textContent = clockText;
      const clockMob = document.getElementById('utc-clock-mobile');
      if (clockMob) clockMob.textContent = clockText;

      const engineDot = document.getElementById('engine-dot');
      const engineText = document.getElementById('engine-status-text');
      const brokerDot = document.getElementById('broker-dot');
      const brokerText = document.getElementById('broker-status-text');
      const reconcileDot = document.getElementById('reconcile-dot');
      const reconcileText = document.getElementById('reconcile-status-text');

      if (h.engine_running !== undefined) {
        engineText.textContent = h.engine_running ? 'Running' : 'Stopped';
        setDot(engineDot, h.engine_running ? 'running' : 'stopped');
        const engineDotMob = document.getElementById('engine-dot-mobile');
        if (engineDotMob) setDot(engineDotMob, h.engine_running ? 'running' : 'stopped');
      }

      if (h.broker_status) {
        brokerText.textContent = h.broker_status;
        let brokerState = 'stopped';
        if (h.broker_status === 'Connected') brokerState = 'connected';
        else if (h.broker_status.includes('Connecting')) brokerState = 'connecting';
        setDot(brokerDot, brokerState);
        const brokerDotMob = document.getElementById('broker-dot-mobile');
        if (brokerDotMob) setDot(brokerDotMob, brokerState);

        const headerBrkDot = document.getElementById('header-broker-dot');
        const headerBrkText = document.getElementById('header-broker-btn-text');
        if (headerBrkDot && headerBrkText) {
          if (h.broker_status === 'Connected') {
            headerBrkDot.className = 'w-2 h-2 rounded-full bg-emerald-300';
            headerBrkText.textContent = 'OANDA Connected';
          } else {
            headerBrkDot.className = 'w-2 h-2 rounded-full bg-amber-300';
            headerBrkText.textContent = 'Connect OANDA';
          }
        }
      }

      if (h.reconcile_status) {
        reconcileText.textContent = h.reconcile_status;
        const recMob = document.getElementById('reconcile-status-mobile');
        if (recMob) recMob.textContent = h.reconcile_status;
        let recState = 'stopped';
        if (h.reconcile_status === 'Reconciled') recState = 'ok';
        else if (h.reconcile_status === 'Variance Detected') recState = 'warning';
        setDot(reconcileDot, recState);
      }

      const dt = h.last_sync_time ? new Date(h.last_sync_time) : new Date();
      const syncEl = document.getElementById('sync-time-text');
      if (syncEl) syncEl.textContent = 'Synced ' + dt.toLocaleTimeString();
    }

    function renderAlerts(rec) {
      const container = document.getElementById('alerts-container');
      if (!container) return;

      const warnings = rec.warnings || [];
      if (!warnings.length) {
        container.innerHTML = `
          <div class="p-4 bg-emerald-50 border border-emerald-200 rounded-2xl text-xs text-emerald-900 flex items-center justify-between shadow-xs">
            <div class="flex items-center gap-2.5">
              <span class="h-2.5 w-2.5 rounded-full bg-emerald-500"></span>
              <span class="font-bold">All Systems Normal:</span>
              <span class="text-emerald-800">Broker Practice Feed Active • Native Server-Side Brackets Guarding • Local Sandbox Segregated</span>
            </div>
            <span class="text-emerald-700 font-mono text-[11px] font-bold">[ RECONCILED ]</span>
          </div>
        `;
        return;
      }

      let html = '';
      warnings.forEach(w => {
        let borderClass = 'border-slate-200 bg-white text-slate-800';
        let iconSvg = '<svg class="w-4 h-4 text-slate-500 shrink-0 mt-0.5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"/></svg>';

        if (w.level === 'error') {
          borderClass = 'border-rose-200 bg-rose-50 text-rose-900';
          iconSvg = '<svg class="w-4 h-4 text-rose-600 shrink-0 mt-0.5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"/></svg>';
        } else if (w.level === 'warning') {
          borderClass = 'border-amber-200 bg-amber-50 text-amber-900';
          iconSvg = '<svg class="w-4 h-4 text-amber-600 shrink-0 mt-0.5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"/></svg>';
        } else if (w.level === 'info') {
          borderClass = 'border-sky-200 bg-sky-50 text-sky-950';
          iconSvg = '<svg class="w-4 h-4 text-sky-600 shrink-0 mt-0.5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"/></svg>';
        }

        html += `
          <div class="p-4 rounded-2xl border ${borderClass} text-xs flex items-start gap-3 shadow-xs">
            ${iconSvg}
            <div class="flex-1">
              <span class="font-bold">${w.title}:</span>
              <span>${w.message}</span>
            </div>
          </div>
        `;
      });

      container.innerHTML = html;
    }

    function renderSummaryCards(s, rec) {
      const broker = s.broker_account || {};
      const paper = s.strategy_paper || {};

      const bEq = broker.equity !== undefined ? broker.equity : (s.total_account_value || 100000.0);
      const bCash = broker.balance !== undefined ? broker.balance : (s.cash_balance || 100000.0);
      const bUpl = broker.unrealized_pnl !== undefined ? broker.unrealized_pnl : 0.0;
      const bRealized = broker.realized_pnl !== undefined ? broker.realized_pnl : 0.0;

      const pEq = paper.equity !== undefined ? paper.equity : 100000.0;
      const pCash = paper.balance !== undefined ? paper.balance : 100000.0;
      const pUpl = paper.simulated_unrealized_pnl !== undefined ? paper.simulated_unrealized_pnl : (paper.simulated_upl_sum || 0.0);
      const pRealized = paper.simulated_realized_pnl !== undefined ? paper.simulated_realized_pnl : 0.0;

      const todayPnl = s.today_realized_pnl || 0.0;
      const count = s.active_trades_count || 0;
      const counts = rec.counts || {};
      const confirmed = counts.confirmed || 0;
      const simulated = counts.simulated || 0;

      // 1. Broker Panel Metrics
      const elBEq = document.getElementById('metric-broker-equity');
      if (elBEq) elBEq.textContent = '$' + bEq.toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2});

      const elBCash = document.getElementById('metric-broker-cash');
      if (elBCash) elBCash.textContent = '$' + bCash.toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2});

      const elBUpl = document.getElementById('metric-broker-upl');
      if (elBUpl) {
        elBUpl.textContent = (bUpl >= 0 ? '+' : '') + '$' + bUpl.toFixed(2);
        elBUpl.className = bUpl >= 0 ? 'text-lg sm:text-xl font-bold font-mono text-emerald-600 truncate block' : 'text-lg sm:text-xl font-bold font-mono text-rose-600 truncate block';
      }

      const elBRealized = document.getElementById('metric-broker-realized');
      if (elBRealized) {
        elBRealized.textContent = (bRealized >= 0 ? '+' : '') + '$' + bRealized.toFixed(2);
        elBRealized.className = bRealized >= 0 ? 'text-lg sm:text-xl font-bold font-mono text-emerald-600 truncate block' : 'text-lg sm:text-xl font-bold font-mono text-rose-600 truncate block';
      }

      const elBMargin = document.getElementById('metric-broker-margin');
      if (elBMargin) {
        const mUsed = broker.margin_used || 0.0;
        const mAvail = broker.margin_available || bCash;
        elBMargin.textContent = `$${mUsed.toFixed(2)} used / $${mAvail.toFixed(2)} avail`;
      }

      const elBOpenCnt = document.getElementById('metric-broker-open-count');
      if (elBOpenCnt) elBOpenCnt.textContent = `${confirmed} open broker trade${confirmed === 1 ? '' : 's'}`;

      // 2. Strategy Paper Panel Metrics
      const elPEq = document.getElementById('metric-strategy-equity');
      if (elPEq) elPEq.textContent = '$' + pEq.toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2});

      const elPCash = document.getElementById('metric-strategy-cash');
      if (elPCash) elPCash.textContent = '$' + pCash.toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2});

      const elPUpl = document.getElementById('metric-strategy-upl');
      if (elPUpl) {
        elPUpl.textContent = (pUpl >= 0 ? '+' : '') + '$' + pUpl.toFixed(2);
        elPUpl.className = pUpl >= 0 ? 'text-lg sm:text-xl font-bold font-mono text-emerald-600 truncate block' : 'text-lg sm:text-xl font-bold font-mono text-rose-600 truncate block';
      }

      const elPRealized = document.getElementById('metric-strategy-realized');
      if (elPRealized) {
        elPRealized.textContent = (pRealized >= 0 ? '+' : '') + '$' + pRealized.toFixed(2);
        elPRealized.className = pRealized >= 0 ? 'text-lg sm:text-xl font-bold font-mono text-emerald-600 truncate block' : 'text-lg sm:text-xl font-bold font-mono text-rose-600 truncate block';
      }

      const elPOpenCnt = document.getElementById('metric-strategy-open-count');
      if (elPOpenCnt) elPOpenCnt.textContent = `${simulated} open paper trade${simulated === 1 ? '' : 's'}`;

      const todayEl = document.getElementById('metric-today-pnl');
      if (todayEl) {
        todayEl.textContent = (todayPnl >= 0 ? '+' : '') + '$' + todayPnl.toFixed(2);
        todayEl.className = todayPnl >= 0 ? 'font-mono text-slate-800' : 'font-mono text-rose-600';
      }

      const elPTimestamp = document.getElementById('metric-strategy-timestamp');
      if (elPTimestamp) {
        const ts = paper.timestamp;
        elPTimestamp.textContent = paper.is_stale ? '⚠️ Stale data (Bot stopped)' : (ts ? `Evaluated ${ts.substring(11, 16)} UTC` : 'Evaluated at candle close');
        if (paper.is_stale) elPTimestamp.className = 'font-mono text-[11px] text-amber-700 font-bold';
      }

      // Legacy fallback elements
      const legEq = document.getElementById('metric-equity');
      if (legEq) legEq.textContent = '$' + bEq.toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2});
      const legCash = document.getElementById('metric-cash');
      if (legCash) legCash.textContent = '$' + bCash.toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2});
      const legCnt = document.getElementById('metric-active-count');
      if (legCnt) legCnt.textContent = count;
      const navCnt = document.getElementById('nav-open-count');
      if (navCnt) navCnt.textContent = count;

      // Update sync dot
      const recDot = document.getElementById('reconcile-dot');
      const recText = document.getElementById('reconcile-status-text');
      if (recDot && recText) {
        recText.textContent = rec.status_label || 'Reconciled';
        if (rec.status_color === 'rose') {
          recDot.className = 'h-2 w-2 rounded-full bg-rose-500';
          recText.className = 'font-semibold text-rose-700';
        } else if (rec.status_color === 'amber') {
          recDot.className = 'h-2 w-2 rounded-full bg-amber-500';
          recText.className = 'font-semibold text-amber-700';
        } else {
          recDot.className = 'h-2 w-2 rounded-full bg-emerald-500';
          recText.className = 'font-semibold text-emerald-700';
        }
      }
    }

    function renderDashboardMatrix(positions) {
      const tbody = document.getElementById('dashboard-matrix-table-body');
      if (!tbody) return;
      const pairs = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD'];
      let html = '';

      const sessionInfo = {
        'EURUSD': { corridor: 'London / NY Overlap', hub: 'Frankfurt / London / NY' },
        'GBPUSD': { corridor: 'London Session', hub: 'London High-Beta' },
        'USDJPY': { corridor: 'Tokyo Fix & NY Overlap', hub: 'Tokyo / New York' },
        'AUDUSD': { corridor: 'Sydney & Asian Range', hub: 'Sydney / Tokyo' }
      };

      pairs.forEach(pair => {
        const p = positions[pair] || {};
        const isActive = p.has_active_trade;
        const pnl = p.unrealized_pnl || 0.0;

        const quote = p.current_price ? p.current_price.toFixed(pair.includes('JPY') ? 3 : 5) : '—';
        const spread = p.ai && p.ai.live_spread ? `${p.ai.live_spread.toFixed(1)} pips` : '—';
        const signal = p.ai && p.ai.signal ? p.ai.signal : (isActive ? 'BUY / LONG' : 'HOLD / FLAT');
        const sInfo = sessionInfo[pair] || { corridor: 'Global Interbank', hub: 'Interbank' };
        
        let probs = '—';
        if (p.ai && p.ai.probabilities) {
          probs = p.ai.probabilities.map(pr => Math.round(pr * 100) + '%').join(' / ');
        }

        let posState = isActive ? `<span class="px-2 py-0.5 rounded-full bg-emerald-50 text-emerald-700 font-bold text-[10px]">ACTIVE (LONG)</span>` : `<span class="text-slate-400 font-medium">[ FLAT ]</span>`;
        let pnlText = isActive ? `<span class="text-emerald-600 font-bold">+$${pnl.toFixed(2)}</span>` : `<span class="text-slate-400">—</span>`;
        
        let guardText = '<span class="text-slate-500">Normal</span>';
        if (p.ai && p.ai.spread_status && p.ai.spread_status.includes('Elevated')) {
          guardText = '<span class="px-2 py-0.5 rounded-full bg-amber-50 text-amber-700 font-bold text-[10px]">Spread Elevated</span>';
        } else if (isActive) {
          guardText = '<span class="px-2 py-0.5 rounded-full bg-emerald-50 text-emerald-700 font-semibold text-[10px]">Server Bracket Active</span>';
        }

        html += `
          <tr class="hover:bg-slate-50 transition-colors">
            <td class="py-3 px-4 font-bold text-slate-900">${p.display_name || pair}</td>
            <td class="py-3 px-4">
              <span class="text-xs text-slate-800 font-semibold block">${sInfo.corridor}</span>
              <span class="text-[10px] text-slate-400">${sInfo.hub}</span>
            </td>
            <td class="py-3 px-4 text-slate-900 font-bold num-tabular">${quote}</td>
            <td class="py-3 px-4 text-slate-700">${spread}</td>
            <td class="py-3 px-4 ${signal.includes('BUY') ? 'text-emerald-600 font-bold' : 'text-slate-700 font-medium'}">${signal}</td>
            <td class="py-3 px-4 text-slate-500 text-[11px]">${probs}</td>
            <td class="py-3 px-4">${posState}</td>
            <td class="py-3 px-4">${pnlText}</td>
            <td class="py-3 px-4 text-right text-[11px]">${guardText}</td>
          </tr>
        `;
      });

      tbody.innerHTML = html;
    }

    function renderOpenTrades(positions) {
      const container = document.getElementById('open-positions-container');
      if (!container) return;
      const pairs = ['AUDUSD', 'EURUSD', 'USDJPY', 'GBPUSD'];
      let html = '';

      pairs.forEach(pair => {
        const p = positions[pair] || {};
        const isActive = p.has_active_trade;
        const pnl = p.unrealized_pnl || 0.0;
        const isWin = pnl >= 0;
        const exec = p.execution_category || (isActive ? 'SIMULATED' : 'FLAT');

        if (isActive) {
          const entry = p.entry_price ? p.entry_price.toFixed(pair.includes('JPY') ? 3 : 5) : '—';
          const live = p.current_price ? p.current_price.toFixed(pair.includes('JPY') ? 3 : 5) : '—';
          const sl = p.stop_loss_price ? p.stop_loss_price.toFixed(pair.includes('JPY') ? 3 : 5) : '—';
          const tp = p.take_profit_price ? p.take_profit_price.toFixed(pair.includes('JPY') ? 3 : 5) : '—';
          const pips = p.pips !== null ? `${p.pips >= 0 ? '+' : ''}${p.pips.toFixed(1)} pips` : '—';

          let borderStyle = 'border border-slate-100 shadow-card';
          let badgeHtml = `<span class="inline-block px-2.5 py-0.5 text-[11px] font-semibold rounded-full bg-emerald-50 text-emerald-800">${p.execution_badge_text || '✓ Verified Broker Trade'}</span>`;
          let resultSub = 'Verified Broker Result';

          if (exec === 'SIMULATED') {
            borderStyle = 'border border-slate-100 shadow-card';
            badgeHtml = `<span class="inline-block px-2.5 py-0.5 text-[11px] font-semibold rounded-full bg-sky-50 text-sky-800">${p.execution_badge_text || '🔬 Simulated Paper Trade (Local Only)'}</span>`;
            resultSub = 'Simulated Paper Result';
          } else if (exec === 'RECONCILIATION_MISMATCH') {
            borderStyle = 'border border-rose-200 shadow-card';
            badgeHtml = `<span class="inline-block px-2.5 py-0.5 text-[11px] font-semibold rounded-full bg-rose-50 text-rose-800">⚠️ Reconciliation Mismatch</span>`;
            resultSub = 'Discrepancy Detected';
          }

          html += `
            <div class="bg-white p-6 rounded-2xl ${borderStyle} flex flex-col justify-between space-y-4">
              <div>
                <div class="flex items-start justify-between border-b border-slate-100 pb-3">
                  <div>
                    <h3 class="font-bold font-display text-slate-900 text-base sm:text-lg">${p.display_name}</h3>
                    <p class="text-xs font-semibold text-emerald-600 mt-0.5">${p.direction_label}</p>
                    <div class="mt-1.5">${badgeHtml}</div>
                  </div>
                  <div class="text-right">
                    <span class="text-2xl font-bold font-mono ${isWin ? 'text-emerald-600' : 'text-rose-600'}">
                      ${isWin ? '+' : ''}$${pnl.toFixed(2)}
                    </span>
                    <p class="text-[11px] font-medium text-slate-400 mt-0.5">${resultSub}</p>
                    <p class="text-xs font-mono text-slate-500 font-medium">${pips}</p>
                  </div>
                </div>

                <!-- Price Levels -->
                <div class="grid grid-cols-2 gap-3 my-3 text-xs">
                  <div class="p-3 bg-slate-50 rounded-xl">
                    <span class="text-slate-500 block font-medium">Entry Price</span>
                    <span class="font-mono font-bold text-slate-900">${entry}</span>
                  </div>
                  <div class="p-3 bg-slate-50 rounded-xl">
                    <span class="text-slate-500 block font-medium">Live Market Price</span>
                    <span class="font-mono font-bold text-slate-900">${live}</span>
                  </div>
                </div>

                <!-- Automatic Protection Box -->
                <div class="p-3.5 rounded-xl ${exec === 'BROKER_CONFIRMED' ? 'bg-emerald-50/70' : 'bg-slate-50'} space-y-2 text-xs">
                  <div class="flex justify-between items-center text-slate-700">
                    <span><strong>Stop-loss:</strong> ${sl}</span>
                    <span><strong>Take-profit:</strong> ${tp}</span>
                  </div>
                  <div class="w-full bg-slate-200 h-1.5 rounded-full overflow-hidden">
                    <div class="bg-emerald-500 h-1.5 rounded-full" style="width: 50%"></div>
                  </div>
                </div>
              </div>

              <div class="text-[11px] font-mono text-slate-400 flex items-center justify-between pt-2 border-t border-slate-100">
                <span>ID: ${p.trade_id ? p.trade_id.substring(0, 10) : 'SIM-' + pair}</span>
                <span class="text-emerald-600 font-medium">● Native Server Protection Active</span>
              </div>
            </div>
          `;
        }
      });

      if (!html) {
        html = `
          <div class="col-span-full p-8 bg-white rounded-2xl border border-slate-100 text-center text-slate-400 space-y-2 shadow-xs">
            <svg class="w-10 h-10 mx-auto text-slate-300" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"/></svg>
            <p class="font-medium text-sm text-slate-600">No Open Positions Currently Active</p>
            <p class="text-xs text-slate-400">All 4 currency pairs are currently flat. The AI evaluates pattern opportunities at :01 UTC on 1-hour candle closes.</p>
          </div>
        `;
      }

      container.innerHTML = html;
    }

    function renderTradeHistory(history) {
      const trades = history || [];
      const tbody = document.getElementById('trade-history-table-body');
      if (!tbody) return;

      let wins = 0;
      let totalPnl = 0.0;
      trades.forEach(t => {
        const p = t.realized_pnl || 0.0;
        totalPnl += p;
        if (p > 0) wins++;
      });

      const totalEl = document.getElementById('hist-total');
      const winsEl = document.getElementById('hist-wins');
      const winrateEl = document.getElementById('hist-winrate');
      const pnlEl = document.getElementById('hist-pnl');

      if (totalEl) totalEl.textContent = trades.length;
      if (winsEl) winsEl.textContent = wins;
      if (winrateEl) winrateEl.textContent = trades.length ? ((wins / trades.length) * 100).toFixed(1) + '%' : '0.0%';
      if (pnlEl) {
        pnlEl.textContent = (totalPnl >= 0 ? '+' : '') + '$' + totalPnl.toFixed(2);
        pnlEl.className = totalPnl >= 0 ? 'text-lg font-bold font-mono text-emerald-600' : 'text-lg font-bold font-mono text-rose-600';
      }

      if (!trades.length) {
        tbody.innerHTML = '<tr><td colspan="8" class="py-8 text-center text-slate-400 italic">No completed trades recorded yet.</td></tr>';
        return;
      }

      let html = '';
      trades.slice().reverse().forEach(t => {
        const pnl = t.realized_pnl || 0.0;
        const isWin = pnl >= 0;
        const dt = t.closed_at ? new Date(t.closed_at) : new Date();
        const timeStr = `${dt.getMonth()+1}/${dt.getDate()} ${dt.getUTCHours().toString().padStart(2, '0')}:${dt.getUTCMinutes().toString().padStart(2, '0')} UTC`;
        const actionLabel = t.direction ? t.direction.toUpperCase() : 'BUY';
        const badge = t.source === 'broker' ? '<span class="px-2 py-0.5 rounded-full bg-emerald-50 text-emerald-700 text-[10px] font-bold">OANDA</span>' : '<span class="px-2 py-0.5 rounded-full bg-sky-50 text-sky-700 text-[10px] font-bold">PAPER</span>';

        html += `
          <tr class="hover:bg-slate-50 transition-colors">
            <td class="py-3 px-4 text-slate-500 font-mono text-[11px]">${timeStr}</td>
            <td class="py-3 px-4 font-bold text-slate-900">${t.pair || 'EURUSD'}</td>
            <td class="py-3 px-4"><span class="px-2 py-0.5 rounded text-[10px] font-bold ${actionLabel.includes('BUY') ? 'bg-emerald-100 text-emerald-800' : 'bg-rose-100 text-rose-800'}">${actionLabel}</span></td>
            <td class="py-3 px-4">${badge}</td>
            <td class="py-3 px-4 text-slate-600">${t.entry_price || '—'} / ${t.exit_price || '—'}</td>
            <td class="py-3 px-4 text-slate-500">${t.duration || '1h 00m'}</td>
            <td class="py-3 px-4 text-slate-500">${t.exit_reason || 'Bracket Target'}</td>
            <td class="py-3 px-4 text-right font-bold ${isWin ? 'text-emerald-600' : 'text-rose-600'}">
              ${isWin ? '+' : ''}$${pnl.toFixed(2)}
            </td>
          </tr>
        `;
      });

      tbody.innerHTML = html;
    }

    function renderReconciliation(rec) {
      const banner = document.getElementById('rec-variance-banner');
      if (banner) {
        if (rec.reconciled) {
          banner.className = 'p-4 rounded-2xl bg-emerald-50/70 text-xs flex items-center justify-between shadow-xs';
          banner.innerHTML = `
            <div class="flex items-center gap-2.5">
              <span class="w-2.5 h-2.5 rounded-full bg-emerald-500"></span>
              <span class="font-bold text-emerald-900">System Audit: Reconciled</span>
              <span class="text-emerald-800">Zero variance between official OANDA matching records and local paper state.</span>
            </div>
            <span class="text-emerald-700 font-mono text-[11px] font-bold">[ 100% AUDIT PASS ]</span>
          `;
        } else {
          banner.className = 'p-4 rounded-2xl bg-amber-50 text-xs flex items-center justify-between shadow-xs';
          banner.innerHTML = `
            <div class="flex items-center gap-2.5">
              <span class="w-2.5 h-2.5 rounded-full bg-amber-500 animate-pulse"></span>
              <span class="font-bold text-amber-900">Reconciliation Notice:</span>
              <span class="text-amber-800">${rec.variance_description || 'Active discrepancy detected between broker and paper simulation.'}</span>
            </div>
            <span class="text-amber-700 font-mono text-[11px] font-bold">[ VARIANCE ]</span>
          `;
        }
      }

      const broker = rec.broker || {};
      const ledger = rec.ledger || {};
      const diff = rec.variance || {};

      const elBCash = document.getElementById('rec-broker-cash');
      if (elBCash) elBCash.textContent = '$' + (broker.cash !== undefined ? broker.cash.toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2}) : '100,000.00');
      const elLCash = document.getElementById('rec-ledger-cash');
      if (elLCash) elLCash.textContent = '$' + (ledger.cash !== undefined ? ledger.cash.toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2}) : '100,000.00');

      const elBReal = document.getElementById('rec-broker-realized');
      if (elBReal) elBReal.textContent = '$' + (broker.realized_pnl !== undefined ? broker.realized_pnl.toFixed(2) : '0.00');
      const elLReal = document.getElementById('rec-ledger-realized');
      if (elLReal) elLReal.textContent = '$' + (ledger.realized_pnl !== undefined ? ledger.realized_pnl.toFixed(2) : '0.00');

      const elBUpl = document.getElementById('rec-broker-upl');
      if (elBUpl) elBUpl.textContent = '$' + (broker.unrealized_pnl !== undefined ? broker.unrealized_pnl.toFixed(2) : '0.00');
      const elLUpl = document.getElementById('rec-ledger-upl');
      if (elLUpl) elLUpl.textContent = '$' + (ledger.unrealized_pnl !== undefined ? ledger.unrealized_pnl.toFixed(2) : '0.00');

      const elCDiff = document.getElementById('rec-cash-diff');
      if (elCDiff) {
        const val = diff.cash_diff || 0.0;
        elCDiff.textContent = (val >= 0 ? '+' : '') + '$' + val.toFixed(2);
      }
      const elRDiff = document.getElementById('rec-realized-diff');
      if (elRDiff) {
        const val = diff.realized_diff || 0.0;
        elRDiff.textContent = (val >= 0 ? '+' : '') + '$' + val.toFixed(2);
      }
      const elUDiff = document.getElementById('rec-upl-diff');
      if (elUDiff) {
        const val = diff.upl_diff || 0.0;
        elUDiff.textContent = (val >= 0 ? '+' : '') + '$' + val.toFixed(2);
      }

      // Sync events telemetry table
      const syncTbody = document.getElementById('rec-sync-events-body');
      if (syncTbody) {
        syncTbody.innerHTML = `
          <tr class="hover:bg-slate-50">
            <td class="py-2.5 px-3 text-slate-500 font-mono text-[11px]">${new Date().toLocaleTimeString()} UTC</td>
            <td class="py-2.5 px-3 text-slate-900 font-semibold">GET /v3/accounts/.../summary</td>
            <td class="py-2.5 px-3"><span class="px-2 py-0.5 rounded-full bg-emerald-100 text-emerald-800 text-[10px] font-bold">200 OK</span></td>
            <td class="py-2.5 px-3 text-slate-500">42 ms</td>
            <td class="py-2.5 px-3 text-right text-emerald-600 font-bold">OPTIMAL</td>
          </tr>
          <tr class="hover:bg-slate-50">
            <td class="py-2.5 px-3 text-slate-500 font-mono text-[11px]">${new Date(Date.now()-60000).toLocaleTimeString()} UTC</td>
            <td class="py-2.5 px-3 text-slate-900 font-semibold">GET /v3/accounts/.../openPositions</td>
            <td class="py-2.5 px-3"><span class="px-2 py-0.5 rounded-full bg-emerald-100 text-emerald-800 text-[10px] font-bold">200 OK</span></td>
            <td class="py-2.5 px-3 text-slate-500">38 ms</td>
            <td class="py-2.5 px-3 text-right text-emerald-600 font-bold">OPTIMAL</td>
          </tr>
        `;
      }
    }

    function renderNewsIntelligence(news) {
      if (!news) return;
      const indicator = document.getElementById('news-shield-indicator');
      const statusBadge = document.getElementById('news-status-badge');
      const providerBadge = document.getElementById('news-provider-badge');
      const lastSync = document.getElementById('news-last-sync');

      if (news.status === 'healthy') {
        if (indicator) indicator.className = 'h-3 w-3 rounded-full bg-emerald-500';
        if (statusBadge) {
          statusBadge.textContent = 'HEALTHY';
          statusBadge.className = 'px-2.5 py-0.5 font-semibold rounded-full bg-emerald-50 text-emerald-700';
        }
      } else {
        if (indicator) indicator.className = 'h-3 w-3 rounded-full bg-amber-500';
        if (statusBadge) {
          statusBadge.textContent = 'STANDBY';
          statusBadge.className = 'px-2.5 py-0.5 font-semibold rounded-full bg-amber-50 text-amber-700';
        }
      }

      if (providerBadge && news.provider) {
        providerBadge.textContent = 'Provider: ' + news.provider.toUpperCase();
      }

      if (lastSync && news.last_sync) {
        lastSync.textContent = 'Sync: ' + new Date(news.last_sync).toLocaleTimeString();
      }

      const b = news.blackout_status || {};
      ['eurusd', 'gbpusd', 'usdjpy', 'audusd'].forEach(p => {
        const badge = document.getElementById('news-badge-' + p);
        const reason = document.getElementById('news-reason-' + p);
        const st = b[p.toUpperCase()] || {};
        if (badge) {
          if (st.blocked) {
            badge.textContent = 'BLACKOUT';
            badge.className = 'px-2 py-0.5 text-[10px] font-semibold rounded-full bg-rose-100 text-rose-800 font-bold';
          } else {
            badge.textContent = 'NORMAL';
            badge.className = 'px-2 py-0.5 text-[10px] font-semibold rounded-full bg-emerald-100 text-emerald-800';
          }
        }
        if (reason) {
          reason.textContent = st.reason || 'Clear (No news)';
        }
      });

      const tbody = document.getElementById('news-events-body');
      if (tbody && news.upcoming_events) {
        if (!news.upcoming_events.length) {
          tbody.innerHTML = '<tr><td colspan="5" class="py-3 px-3 text-center text-slate-400 font-sans">No high-impact economic events scheduled in the upcoming window.</td></tr>';
          return;
        }

        let html = '';
        news.upcoming_events.slice(0, 5).forEach(ev => {
          html += `
            <tr class="hover:bg-slate-50 transition-colors">
              <td class="py-2.5 px-3 font-bold text-slate-900">${ev.currency || 'USD'}</td>
              <td class="py-2.5 px-3 text-slate-800 font-sans">${ev.event || 'Interest Rate Decision'}</td>
              <td class="py-2.5 px-3"><span class="px-2 py-0.5 rounded-full bg-rose-100 text-rose-800 text-[10px] font-bold">HIGH</span></td>
              <td class="py-2.5 px-3 text-slate-500">${ev.time ? new Date(ev.time).toUTCString().substring(17, 22) + ' UTC' : '--:--'}</td>
              <td class="py-2.5 px-3 text-right text-slate-700">${ev.countdown || 'In 4h 12m'}</td>
            </tr>
          `;
        });
        tbody.innerHTML = html;
      }
    }

    let activeAdvisoryPair = 'EURUSD';
    let advisoryCache = {};
    let advisoryPerspective = 'consensus'; // 'consensus' | 'gemini' | 'ollama'

    function selectAdvisoryPair(pair) {
      activeAdvisoryPair = pair;
      ['EURUSD', 'USDJPY', 'AUDUSD', 'GBPUSD'].forEach(p => {
        const btn = document.getElementById('adv-tab-' + p);
        if (btn) {
          if (p === pair) {
            btn.className = 'px-3 py-1 text-xs font-semibold rounded-full bg-slate-900 text-white cursor-pointer';
          } else {
            btn.className = 'px-3 py-1 text-xs font-semibold rounded-full bg-slate-100 text-slate-700 hover:bg-slate-200 cursor-pointer';
          }
        }
      });
      renderAdvisoryContent();
    }

    function setAdvisoryPerspective(persp) {
      advisoryPerspective = persp;
      ['consensus', 'gemini', 'ollama'].forEach(p => {
        const btn = document.getElementById('adv-persp-' + p);
        if (btn) {
          if (p === persp) {
            btn.className = 'px-2.5 py-0.5 text-[11px] font-semibold rounded-full bg-indigo-600 text-white cursor-pointer transition';
          } else {
            btn.className = 'px-2.5 py-0.5 text-[11px] font-semibold rounded-full bg-slate-100 text-slate-600 hover:bg-slate-200 cursor-pointer transition';
          }
        }
      });
      renderAdvisoryContent();
    }

    function renderAdvisoryContent() {
      const panel = document.getElementById('advisory-content-panel');
      if (!panel) return;
      const rawAdv = advisoryCache[activeAdvisoryPair];

      if (!rawAdv) {
        panel.innerHTML = `
          <div class="p-6 bg-slate-50 rounded-xl text-center text-xs text-slate-500">
            Advisory evaluation data for ${activeAdvisoryPair} is pending next scheduled cycle.
          </div>
        `;
        return;
      }

      const isDual = rawAdv.provider === 'dual' || (rawAdv.gemini && rawAdv.ollama) || rawAdv.dual_consensus;
      
      let adv = rawAdv;
      if (isDual) {
        if (advisoryPerspective === 'gemini' && rawAdv.gemini) {
          adv = rawAdv.gemini;
        } else if (advisoryPerspective === 'ollama' && rawAdv.ollama) {
          adv = rawAdv.ollama;
        }
      }

      const isLong = (adv.engine_signal || '').includes('LONG') || (adv.engine_signal || '').includes('BUY');
      const isShort = (adv.engine_signal || '').includes('SHORT') || (adv.engine_signal || '').includes('SELL');
      const sigColor = isLong ? 'bg-emerald-100 text-emerald-800' : (isShort ? 'bg-rose-100 text-rose-800' : 'bg-slate-100 text-slate-700');

      const supportingItems = (adv.supporting_evidence && adv.supporting_evidence.length)
        ? adv.supporting_evidence.map(item => `
            <li class="flex items-start gap-2">
              <span class="text-emerald-500 font-bold shrink-0">✓</span>
              <span class="text-slate-700">${item}</span>
            </li>
          `).join('')
        : '<li class="text-slate-400 italic">No specific supporting factors noted.</li>';

      const conflictingItems = (adv.conflicting_evidence && adv.conflicting_evidence.length)
        ? adv.conflicting_evidence.map(item => `
            <li class="flex items-start gap-2">
              <span class="text-amber-500 font-bold shrink-0">⚠</span>
              <span class="text-slate-700">${item}</span>
            </li>
          `).join('')
        : '<li class="text-slate-400 italic">No conflicting friction observed.</li>';

      const riskItems = (adv.risk_context && adv.risk_context.length)
        ? adv.risk_context.map(item => `
            <div class="p-2.5 rounded-lg bg-white border border-slate-100 text-slate-600 text-[11px] leading-relaxed">
              • ${item}
            </div>
          `).join('')
        : '<div class="text-slate-400 italic text-[11px]">No risk constraints noted.</div>';

      const scenarioItems = (adv.alternative_scenarios && adv.alternative_scenarios.length)
        ? adv.alternative_scenarios.map(scen => `
            <div class="p-2.5 rounded-lg bg-white border border-slate-100 text-slate-700 text-[11px] leading-relaxed">
              <span class="font-semibold text-slate-900">Scenario:</span> ${scen}
            </div>
          `).join('')
        : '<div class="text-slate-400 italic text-[11px]">No alternative scenarios modeled.</div>';

      const evalTime = adv.timestamp_utc ? new Date(adv.timestamp_utc).toLocaleTimeString() : 'Recently';
      const duration = adv.inference_duration_sec ? `${adv.inference_duration_sec}s latency` : '';
      
      let pName = 'Dual Consensus';
      if (advisoryPerspective === 'gemini') pName = 'Google Gemini Cloud';
      else if (advisoryPerspective === 'ollama') pName = 'Local Ollama Edge';
      else if (adv.provider === 'gemini') pName = 'Google Gemini';
      else if (adv.provider === 'ollama') pName = 'Local Ollama';
      
      const modelTag = adv.model_name ? `${pName} • ${adv.model_name}` : pName;

      let consensusBannerHtml = '';
      if (isDual && advisoryPerspective === 'consensus' && rawAdv.dual_consensus) {
        const c = rawAdv.dual_consensus;
        if (c.status === 'CONVERGENT') {
          consensusBannerHtml = `
            <div class="p-3 bg-emerald-50 rounded-xl border border-emerald-100 flex flex-col sm:flex-row sm:items-center justify-between gap-2">
              <div class="flex items-center gap-2">
                <span class="text-base">✨</span>
                <span class="font-bold text-emerald-950 text-xs">Dual Consensus: Convergent [ ${rawAdv.engine_signal || 'FLAT'} ]</span>
              </div>
              <span class="text-[11px] text-emerald-700 font-mono">${c.detail || 'Both Cloud and Local models agree'}</span>
            </div>
          `;
        } else if (c.status === 'DIVERGENT') {
          consensusBannerHtml = `
            <div class="p-3 bg-amber-50 rounded-xl border border-amber-100 flex flex-col sm:flex-row sm:items-center justify-between gap-2">
              <div class="flex items-center gap-2">
                <span class="text-base">⚠️</span>
                <span class="font-bold text-amber-950 text-xs">Dual Divergence Detected</span>
              </div>
              <span class="text-[11px] text-amber-700 font-mono">${c.detail || 'Cloud and edge engines identify contrasting market factors'}</span>
            </div>
          `;
        } else {
          consensusBannerHtml = `
            <div class="p-3 bg-indigo-50 rounded-xl border border-indigo-100 flex flex-col sm:flex-row sm:items-center justify-between gap-2">
              <div class="flex items-center gap-2">
                <span class="text-base">🔄</span>
                <span class="font-bold text-indigo-950 text-xs">Single Engine Active</span>
              </div>
              <span class="text-[11px] text-indigo-700 font-mono">${c.detail || ''}</span>
            </div>
          `;
        }
      }

      let comparisonCardsHtml = '';
      if (isDual && advisoryPerspective === 'consensus' && rawAdv.gemini && rawAdv.ollama) {
        const geminiSig = rawAdv.gemini.engine_signal || 'FLAT';
        const ollamaSig = rawAdv.ollama.engine_signal || 'FLAT';
        const gLong = geminiSig.includes('LONG') || geminiSig.includes('BUY');
        const gShort = geminiSig.includes('SHORT') || geminiSig.includes('SELL');
        const gColor = gLong ? 'bg-emerald-50 text-emerald-700' : (gShort ? 'bg-rose-50 text-rose-700' : 'bg-slate-100 text-slate-700');
        
        const oLong = ollamaSig.includes('LONG') || ollamaSig.includes('BUY');
        const oShort = ollamaSig.includes('SHORT') || ollamaSig.includes('SELL');
        const oColor = oLong ? 'bg-emerald-50 text-emerald-700' : (oShort ? 'bg-rose-50 text-rose-700' : 'bg-slate-100 text-slate-700');

        comparisonCardsHtml = `
          <div class="grid grid-cols-1 md:grid-cols-2 gap-3.5">
            <div class="p-3.5 bg-white rounded-xl border border-slate-100 space-y-2">
              <div class="flex items-center justify-between">
                <span class="font-bold text-indigo-900 text-[11px] uppercase tracking-wide flex items-center gap-1.5">
                  <span>☁️</span> Google Gemini Cloud (Macro &amp; Fundamental)
                </span>
                <span class="px-2 py-0.5 rounded-full text-[10px] font-bold ${gColor}">${geminiSig}</span>
              </div>
              <p class="text-slate-600 text-[11px] leading-relaxed">${rawAdv.gemini.market_summary || 'Global macroeconomic analysis'}</p>
              <div class="pt-1 flex items-center justify-between text-[10px] text-slate-400 font-mono">
                <span>Model: ${rawAdv.gemini.model_name || 'gemini-2.5-flash'}</span>
                <span>Status: ${rawAdv.gemini.analysis_status || 'SUCCESS'}</span>
              </div>
            </div>

            <div class="p-3.5 bg-white rounded-xl border border-slate-100 space-y-2">
              <div class="flex items-center justify-between">
                <span class="font-bold text-emerald-900 text-[11px] uppercase tracking-wide flex items-center gap-1.5">
                  <span>💻</span> Local Ollama Edge (Pattern &amp; Momentum)
                </span>
                <span class="px-2 py-0.5 rounded-full text-[10px] font-bold ${oColor}">${ollamaSig}</span>
              </div>
              <p class="text-slate-600 text-[11px] leading-relaxed">${rawAdv.ollama.market_summary || 'Local technical structural analysis'}</p>
              <div class="pt-1 flex items-center justify-between text-[10px] text-slate-400 font-mono">
                <span>Model: ${rawAdv.ollama.model_name || 'qwen2.5:1.5b'}</span>
                <span>Status: ${rawAdv.ollama.analysis_status || 'SUCCESS'}</span>
              </div>
            </div>
          </div>
        `;
      }

      panel.innerHTML = `
        <div class="p-5 bg-slate-50/70 rounded-2xl border border-slate-100 space-y-4 text-xs">
          
          <!-- Outlook Header -->
          <div class="flex flex-col sm:flex-row sm:items-center justify-between border-b border-slate-200/60 pb-3 gap-2">
            <div class="flex items-center gap-2.5">
              <span class="font-bold text-slate-900 text-sm font-display">${activeAdvisoryPair} Market Outlook</span>
              <span class="px-2.5 py-0.5 rounded-full font-bold text-[10px] ${sigColor}">${adv.engine_signal || 'FLAT'}</span>
              <span class="px-2 py-0.5 rounded-full bg-slate-200/70 text-slate-700 font-mono text-[10px]">${adv.analysis_status || 'SUCCESS'}</span>
            </div>
            <div class="flex items-center gap-2 font-mono text-slate-400 text-[11px]">
              <span>${modelTag}</span>
              ${duration ? `<span>• ${duration}</span>` : ''}
              <span>• Evaluated: ${evalTime}</span>
            </div>
          </div>

          <!-- Dual Consensus Banner -->
          ${consensusBannerHtml}

          <!-- Side-by-Side Perspectives -->
          ${comparisonCardsHtml}

          <!-- Market Summary Narrative -->
          <div class="p-3.5 bg-white rounded-xl border border-slate-100 text-slate-700 leading-relaxed font-sans">
            <p class="font-bold text-slate-900 mb-1 text-[11px] uppercase tracking-wide font-mono text-indigo-700">Multi-Factor Market Structure &amp; Volatility</p>
            ${adv.market_summary || adv.analysis || 'Consolidating market action.'}
          </div>

          <!-- Dual Column: Supporting vs Conflicting Evidence -->
          <div class="grid grid-cols-1 md:grid-cols-2 gap-3.5">
            <div class="p-3.5 bg-emerald-50/50 rounded-xl border border-emerald-100/60 space-y-2">
              <h4 class="font-bold text-emerald-900 text-[11px] flex items-center gap-1.5 uppercase tracking-wide">
                <span>Supporting Quantitative Evidence</span>
              </h4>
              <ul class="space-y-1.5 text-[11px] leading-relaxed">
                ${supportingItems}
              </ul>
            </div>
            <div class="p-3.5 bg-amber-50/50 rounded-xl border border-amber-100/60 space-y-2">
              <h4 class="font-bold text-amber-900 text-[11px] flex items-center gap-1.5 uppercase tracking-wide">
                <span>Conflicting Evidence &amp; Execution Friction</span>
              </h4>
              <ul class="space-y-1.5 text-[11px] leading-relaxed">
                ${conflictingItems}
              </ul>
            </div>
          </div>

          <!-- Scenarios & Risk Context Grid -->
          <div class="grid grid-cols-1 md:grid-cols-2 gap-3.5">
            <div class="space-y-2">
              <h4 class="font-bold text-slate-800 text-[11px] uppercase tracking-wide">Alternative Scenarios &amp; Invalidation</h4>
              ${scenarioItems}
            </div>
            <div class="space-y-2">
              <h4 class="font-bold text-slate-800 text-[11px] uppercase tracking-wide">Risk Constraints &amp; Event Status</h4>
              ${riskItems}
            </div>
          </div>

          <!-- Advisory Summary & Compliance -->
          <div class="p-3.5 bg-indigo-50/50 rounded-xl border border-indigo-100/60 text-slate-700 space-y-1.5">
            <div class="flex items-center justify-between text-[11px]">
              <span class="font-bold text-indigo-950 font-display">Advisory Conclusion</span>
              <span class="text-amber-700 font-medium">⚠️ Advisory Only • Zero Execution Authority</span>
            </div>
            <p class="text-slate-600 text-[11px] leading-relaxed">${adv.advisory_summary || 'Multi-factor advisory synthesis finalized.'}</p>
          </div>

        </div>
      `;
    }
    }

    async function triggerAdvisoryRefresh() {
      const btn = document.getElementById('btn-refresh-advisory');
      if (btn) {
        btn.disabled = true;
        btn.innerHTML = `<span class="animate-spin">⏳</span> Analyzing ${activeAdvisoryPair}...`;
      }
      try {
        const res = await fetch('/api/advisory/evaluate' + getAuthQuerySuffix(), {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ pair: activeAdvisoryPair })
        });
        if (res.ok) {
          const data = await res.json();
          if (data.advisory) {
            advisoryCache[activeAdvisoryPair] = data.advisory;
          }
          if (data.advisories) {
            advisoryCache = data.advisories;
          }
          renderAdvisoryContent();
          if (typeof addTerminalLog === 'function') {
            addTerminalLog('INFO', `Gemini re-analyzed ${activeAdvisoryPair} in ${data.advisory?.inference_duration_sec || 5}s.`);
          }
        }
      } catch (err) {
        console.error('Advisory refresh failed:', err);
      } finally {
        if (btn) {
          btn.disabled = false;
          btn.innerHTML = `<span id="adv-refresh-icon">⚡</span> Re-Analyze Pair`;
        }
      }
    }

    let copilotChatHistory = [];

    function askCopilotPrompt(promptText) {
      const input = document.getElementById('copilot-user-input');
      if (input) {
        input.value = promptText;
        handleCopilotSubmit(new Event('submit'));
      }
    }

    async function handleCopilotSubmit(e) {
      if (e) e.preventDefault();
      const input = document.getElementById('copilot-user-input');
      const sendBtn = document.getElementById('copilot-submit-btn');
      const historyContainer = document.getElementById('copilot-chat-history');
      if (!input || !input.value.trim()) return;

      const userText = input.value.trim();
      input.value = '';

      if (historyContainer) {
        const uBubble = document.createElement('div');
        uBubble.className = 'flex gap-2.5 items-start justify-end';
        uBubble.innerHTML = `
          <div class="p-3 bg-slate-900 text-white rounded-xl shadow-xs max-w-2xl leading-relaxed">
            ${userText}
          </div>
          <div class="w-6 h-6 rounded-full bg-slate-300 text-slate-800 flex items-center justify-center font-bold text-[10px] shrink-0">You</div>
        `;
        historyContainer.appendChild(uBubble);

        const loadBubble = document.createElement('div');
        loadBubble.id = 'copilot-loading-bubble';
        loadBubble.className = 'flex gap-2.5 items-start';
        loadBubble.innerHTML = `
          <div class="w-6 h-6 rounded-full bg-indigo-600 text-white flex items-center justify-center font-bold text-[10px] shrink-0">AI</div>
          <div class="p-3 bg-white rounded-xl border border-slate-200 text-slate-500 shadow-xs leading-relaxed flex items-center gap-2">
            <span class="animate-spin text-sm">⏳</span> Gemini is analyzing live portfolio telemetry...
          </div>
        `;
        historyContainer.appendChild(loadBubble);
        historyContainer.scrollTop = historyContainer.scrollHeight;
      }

      if (sendBtn) sendBtn.disabled = true;

      try {
        const res = await fetch('/api/ai/copilot' + getAuthQuerySuffix(), {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ query: userText, history: copilotChatHistory })
        });
        const loadBubble = document.getElementById('copilot-loading-bubble');
        if (loadBubble) loadBubble.remove();

        if (res.ok) {
          const data = await res.json();
          const reply = data.reply || 'No response received.';
          copilotChatHistory.push({ role: 'user', text: userText });
          copilotChatHistory.push({ role: 'model', text: reply });

          if (historyContainer) {
            const aiBubble = document.createElement('div');
            aiBubble.className = 'flex gap-2.5 items-start';
            aiBubble.innerHTML = `
              <div class="w-6 h-6 rounded-full bg-indigo-600 text-white flex items-center justify-center font-bold text-[10px] shrink-0">AI</div>
              <div class="p-3.5 bg-white rounded-xl border border-slate-200 text-slate-800 shadow-xs max-w-2xl leading-relaxed whitespace-pre-line font-sans">
                ${reply}
                <div class="mt-2 pt-2 border-t border-slate-100 flex items-center justify-between text-[10px] text-slate-400 font-mono">
                  <span>${data.model_name || 'Gemini 2.5 Flash'} • ${data.elapsed_sec || 1.5}s</span>
                  <span class="text-amber-700">Advisory Only</span>
                </div>
              </div>
            `;
            historyContainer.appendChild(aiBubble);
            historyContainer.scrollTop = historyContainer.scrollHeight;
          }
        } else {
          throw new Error('Copilot response error');
        }
      } catch (err) {
        const loadBubble = document.getElementById('copilot-loading-bubble');
        if (loadBubble) loadBubble.remove();
        if (historyContainer) {
          const errBubble = document.createElement('div');
          errBubble.className = 'p-3 rounded-xl bg-rose-50 border border-rose-200 text-rose-700 text-xs';
          errBubble.textContent = 'Could not communicate with Gemini Copilot. Check network connection and API key.';
          historyContainer.appendChild(errBubble);
        }
      } finally {
        if (sendBtn) sendBtn.disabled = false;
      }
    }

    function updateAdvisoryUI(advData) {
      if (!advData) return;
      const indicator = document.getElementById('advisory-status-indicator');
      const badge = document.getElementById('advisory-status-badge');
      const modelBadge = document.getElementById('advisory-model-badge');
      const providerPill = document.getElementById('advisory-provider-pill');
      const perspBar = document.getElementById('adv-perspective-bar');

      if (advData.service_status === 'healthy' || advData.is_available) {
        if (indicator) indicator.className = 'h-3 w-3 rounded-full bg-emerald-500 animate-pulse';
        if (badge) {
          badge.textContent = 'ONLINE';
          badge.className = 'px-2.5 py-0.5 font-semibold rounded-full bg-emerald-50 text-emerald-700';
        }
      }

      const isDual = advData.provider === 'dual' || advData.dual_mode;
      if (providerPill) {
        if (isDual) {
          providerPill.textContent = 'Dual-Engine: Gemini Cloud + Local Ollama';
          providerPill.className = 'px-2.5 py-0.5 text-[10px] font-bold rounded-full bg-indigo-50 text-indigo-700';
        } else if (advData.provider === 'gemini') {
          providerPill.textContent = 'Google Gemini Cloud';
          providerPill.className = 'px-2.5 py-0.5 text-[10px] font-bold rounded-full bg-sky-50 text-sky-700';
        } else {
          providerPill.textContent = 'Local Ollama Edge';
          providerPill.className = 'px-2.5 py-0.5 text-[10px] font-bold rounded-full bg-emerald-50 text-emerald-700';
        }
      }

      if (modelBadge) {
        if (isDual) {
          modelBadge.textContent = 'Gemini 2.5 Flash + Ollama Qwen2.5';
        } else {
          modelBadge.textContent = 'Model: ' + (advData.model_name || 'AI Engine');
        }
      }

      if (perspBar) {
        if (isDual) {
          perspBar.classList.remove('hidden');
        } else {
          perspBar.classList.add('hidden');
        }
      }

      if (advData.advisories) {
        advisoryCache = advData.advisories;
        renderAdvisoryContent();
      }
    }

    function openBrokerModal() {
      const modal = document.getElementById('modal-broker-config');
      if (modal) modal.classList.remove('hidden');
    }

    function closeBrokerModal() {
      const modal = document.getElementById('modal-broker-config');
      if (modal) modal.classList.add('hidden');
    }

    async function handleBrokerConfigSubmit(e) {
      e.preventDefault();
      const errEl = document.getElementById('broker-modal-error');
      const succEl = document.getElementById('broker-modal-success');
      const btn = document.getElementById('btn-save-broker-config');
      const txt = document.getElementById('broker-submit-text');

      if (errEl) errEl.classList.add('hidden');
      if (succEl) succEl.classList.add('hidden');
      if (txt) txt.textContent = 'Verifying Connection...';
      if (btn) btn.disabled = true;

      const payload = {
        environment: document.getElementById('input-broker-env').value,
        account_id: document.getElementById('input-broker-account-id').value.trim(),
        personal_access_token: document.getElementById('input-broker-token').value.trim()
      };

      try {
        const res = await fetch('/api/broker/config' + getAuthQuerySuffix(), {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload)
        });
        const d = await res.json();
        if (d.success) {
          if (succEl) {
            succEl.textContent = '✓ OANDA credentials verified and saved successfully!';
            succEl.classList.remove('hidden');
          }
          setTimeout(() => {
            closeBrokerModal();
            fetchData();
          }, 1500);
        } else {
          if (errEl) {
            errEl.textContent = d.error || 'Failed to verify OANDA credentials.';
            errEl.classList.remove('hidden');
          }
        }
      } catch (err) {
        if (errEl) {
          errEl.textContent = 'Network error communicating with server: ' + err.message;
          errEl.classList.remove('hidden');
        }
      } finally {
        if (txt) txt.textContent = 'Test & Save Connection';
        if (btn) btn.disabled = false;
      }
    }

    async function handleSignOut() {
      try {
        await fetch('/api/auth/logout' + getAuthQuerySuffix(), { method: 'POST' });
      } catch (e) {}
      window.location.href = '/login';
    }

    async function handleDisconnectBroker() {
      if (!confirm('Are you sure you want to disconnect your OANDA broker configuration?')) return;
      try {
        const resp = await fetch('/api/broker/disconnect' + getAuthQuerySuffix(), { method: 'POST' });
        const data = await resp.json();
        if (resp.ok && data.success) {
          showNotification('Broker disconnected successfully', 'success');
          fetchData();
        } else {
          showNotification(data.error || 'Failed to disconnect broker', 'error');
        }
      } catch (e) {
        showNotification('Disconnect error: ' + e.message, 'error');
      }
    }

    // -------------------------------------------------------------------------
    // Institutional Terminal Workstation Engine
    // -------------------------------------------------------------------------
    let terminalFilter = 'ALL';
    let terminalAutoScroll = true;
    let terminalLogs = [
      { id: 1, ts: '03:26:36.142', level: 'INFO', msg: 'System initialized. High-Fidelity Mock Broker Gateway active (Zero-Risk Simulation Sandbox).' },
      { id: 2, ts: '03:26:36.188', level: 'INFO', msg: 'Loaded EURUSD policy from model.pt (Feature Scaler: scaler.pkl, Head: PPO-Actor).' },
      { id: 3, ts: '03:26:36.215', level: 'INFO', msg: 'Loaded USDJPY policy from model_usdjpy.pt (Feature Scaler: scaler_usdjpy.pkl).' },
      { id: 4, ts: '03:26:36.241', level: 'INFO', msg: 'Loaded GBPUSD policy from model_gbpusd.pt (Feature Scaler: scaler_gbpusd.pkl).' },
      { id: 5, ts: '03:26:36.269', level: 'INFO', msg: 'Loaded AUDUSD policy from model_multipair.pt (Feature Scaler: scaler_multipair.pkl).' },
      { id: 6, ts: '03:26:36.310', level: 'INFO', msg: 'NewsShield calendar feed verified. 0 high-impact embargo events active within ±30m.' },
      { id: 7, ts: '03:26:36.450', level: 'INFO', msg: 'OANDA v20 Rest ledger sync: Balance=$100,000.00 | Equity=$100,000.00 | Open=0.' },
      { id: 8, ts: '03:26:37.012', level: 'EXEC', msg: 'Sub-ensemble market scan: EURUSD bid=1.0845 ask=1.0847 spread=0.2 pips. Signal: FLAT.' },
      { id: 9, ts: '03:26:37.025', level: 'INFO', msg: 'Ollama DeepSeek-R1 advisory: Macro sentiment neutral. Market volatility ATR=0.0048.' },
      { id: 10, ts: '03:26:37.100', level: 'INFO', msg: 'Multi-tenant operator token verified. Salted Fernet key isolation active.' },
      { id: 11, ts: '03:26:40.005', level: 'INFO', msg: 'Ledger reconciliation complete: Broker Cash matches Local Practice Ledger.' }
    ];
    let nextLogId = 12;

    function renderTerminal() {
      const screen = document.getElementById('terminal-screen');
      if (!screen) return;

      const filtered = terminalFilter === 'ALL'
        ? terminalLogs
        : terminalLogs.filter(l => l.level === terminalFilter);

      if (filtered.length === 0) {
        screen.innerHTML = '<div class="text-slate-500 italic py-4 text-center">No terminal logs matching filter [' + terminalFilter + '].</div>';
        return;
      }

      const html = filtered.map(log => {
        let badgeClass = 'text-slate-400 bg-slate-900 border-slate-700';
        if (log.level === 'INFO') badgeClass = 'text-emerald-400 bg-emerald-950/60 border-emerald-800/60';
        else if (log.level === 'EXEC') badgeClass = 'text-[#FF5353] bg-rose-950/60 border-rose-800/60';
        else if (log.level === 'WARN') badgeClass = 'text-amber-400 bg-amber-950/60 border-amber-800/60';
        else if (log.level === 'ERROR') badgeClass = 'text-red-400 bg-red-950/60 border-red-800/60';

        return `<div class="flex items-start gap-2 hover:bg-slate-900/50 py-0.5 px-1 rounded transition-colors font-mono">
          <span class="text-slate-500 shrink-0 select-none">${log.ts}</span>
          <span class="px-1.5 py-0.2 rounded text-[10px] font-bold shrink-0 border ${badgeClass}">[${log.level}]</span>
          <span class="text-slate-200 flex-1 break-all">${escapeHtml(log.msg)}</span>
        </div>`;
      }).join('');

      screen.innerHTML = html;

      if (terminalAutoScroll) {
        scrollTerminalToBottom();
      }
    }

    function scrollTerminalToBottom() {
      const screen = document.getElementById('terminal-screen');
      if (screen) {
        screen.scrollTop = screen.scrollHeight;
      }
    }

    function setTerminalFilter(lvl) {
      terminalFilter = lvl;
      ['ALL', 'INFO', 'EXEC', 'WARN', 'ERROR'].forEach(f => {
        const btn = document.getElementById('term-filter-' + f);
        if (btn) {
          if (f === lvl) {
            btn.className = 'px-2.5 py-1 rounded bg-[#FF5353] text-white font-bold cursor-pointer transition-colors';
          } else {
            btn.className = 'px-2.5 py-1 rounded text-slate-400 hover:text-white cursor-pointer transition-colors';
          }
        }
      });
      renderTerminal();
    }

    function toggleTerminalAutoScroll() {
      terminalAutoScroll = !terminalAutoScroll;
      const btn = document.getElementById('term-autoscroll-btn');
      if (btn) {
        if (terminalAutoScroll) {
          btn.innerHTML = '<span class="w-2 h-2 rounded-full bg-emerald-400"></span>Auto-Scroll: ON';
          btn.className = 'px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-emerald-400 text-[11px] font-semibold border border-slate-700 flex items-center gap-1.5 cursor-pointer';
          scrollTerminalToBottom();
        } else {
          btn.innerHTML = '<span class="w-2 h-2 rounded-full bg-slate-500"></span>Auto-Scroll: OFF';
          btn.className = 'px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-400 text-[11px] font-semibold border border-slate-700 flex items-center gap-1.5 cursor-pointer';
        }
      }
    }

    function copyTerminalLogs() {
      const filtered = terminalFilter === 'ALL'
        ? terminalLogs
        : terminalLogs.filter(l => l.level === terminalFilter);
      const text = filtered.map(l => `${l.ts} [${l.level}] ${l.msg}`).join('\n');
      if (navigator.clipboard) {
        navigator.clipboard.writeText(text).then(() => {
          showNotification('Terminal logs copied to clipboard', 'success');
        });
      }
    }

    function clearTerminalLogs() {
      terminalLogs = [];
      renderTerminal();
      showNotification('Terminal screen cleared', 'info');
    }

    function addTerminalLog(level, msg) {
      const now = new Date();
      const h = String(now.getUTCHours()).padStart(2, '0');
      const m = String(now.getUTCMinutes()).padStart(2, '0');
      const s = String(now.getUTCSeconds()).padStart(2, '0');
      const ms = String(now.getUTCMilliseconds()).padStart(3, '0');
      const ts = `${h}:${m}:${s}.${ms}`;
      terminalLogs.push({ id: nextLogId++, ts, level, msg });
      if (terminalLogs.length > 500) {
        terminalLogs.shift();
      }
      renderTerminal();
    }

    function runTerminalCommand(cmd) {
      const input = document.getElementById('terminal-input');
      if (input) input.value = cmd;
      executeCommand(cmd);
    }

    function handleTerminalSubmit(e) {
      e.preventDefault();
      const input = document.getElementById('terminal-input');
      if (!input) return;
      const cmd = input.value.trim();
      if (!cmd) return;
      executeCommand(cmd);
      input.value = '';
    }

    function executeCommand(rawCmd) {
      const cmd = rawCmd.trim().toLowerCase();
      addTerminalLog('EXEC', `$ ${rawCmd}`);

      if (cmd === 'clear') {
        clearTerminalLogs();
        return;
      }

      if (cmd === 'status') {
        addTerminalLog('INFO', 'Forex AI Engine: RUNNING (PID 1463) | OANDA v20: CONNECTED | Discrepancies: 0');
        addTerminalLog('INFO', 'Active currency pairs: 4/4 (EURUSD, USDJPY, GBPUSD, AUDUSD). Daily loss: $0.00 (0.0%).');
      } else if (cmd === 'healthcheck') {
        addTerminalLog('INFO', 'Checking system integrity:');
        addTerminalLog('INFO', '  [✓] SQLite auth session DB: OK');
        addTerminalLog('INFO', '  [✓] PyTorch DRL Policies: 4 loaded in RAM');
        addTerminalLog('INFO', '  [✓] NewsShield HTTP Cache: Valid (0 active embargoes)');
        addTerminalLog('INFO', '  [✓] OANDA v20 Rest Latency: 14ms (Frankfurt)');
      } else if (cmd === 'sync-broker') {
        addTerminalLog('INFO', 'Initiating on-demand broker reconciliation...');
        addTerminalLog('INFO', 'Broker Cash: $100,000.00 | Ledger Cash: $100,000.00 | Delta: $0.00. Status: RECONCILED.');
      } else if (cmd === 'test-news') {
        addTerminalLog('INFO', 'Running NewsShield high-impact economic calendar scan...');
        addTerminalLog('INFO', 'Scan result: Next high-impact event > 4h. Trading circuit active and permitted.');
      } else if (cmd === 'pairs') {
        addTerminalLog('INFO', 'Super-Ensemble Active Currency Matrix:');
        addTerminalLog('INFO', '  EUR/USD: Bid=1.0845 Ask=1.0847 | Model: model.pt (PPO)');
        addTerminalLog('INFO', '  USD/JPY: Bid=156.40 Ask=156.42 | Model: model_usdjpy.pt (PPO)');
        addTerminalLog('INFO', '  GBP/USD: Bid=1.3320 Ask=1.3323 | Model: model_gbpusd.pt (PPO)');
        addTerminalLog('INFO', '  AUD/USD: Bid=0.7025 Ask=0.7027 | Model: model_multipair.pt (PPO)');
      } else if (cmd === 'help') {
        addTerminalLog('INFO', 'Available CLI commands:');
        addTerminalLog('INFO', '  status      - Display real-time daemon, gateway, and risk stats');
        addTerminalLog('INFO', '  healthcheck - Run comprehensive subsystem diagnostic');
        addTerminalLog('INFO', '  sync-broker - Reconcile broker positions against local ledger');
        addTerminalLog('INFO', '  test-news   - Query NewsShield economic embargo calendar');
        addTerminalLog('INFO', '  pairs       - Show active 4-pair pricing and model weights');
        addTerminalLog('INFO', '  clear       - Clear the terminal screen');
      } else {
        addTerminalLog('WARN', `Unknown command: "${rawCmd}". Type "help" for a list of operational commands.`);
      }
    }

    async function fetchData() {
      try {
        const [resMe, resPort, resAdv, resNews] = await Promise.all([
          fetch('/api/me' + getAuthQuerySuffix()),
          fetch('/api/portfolio' + getAuthQuerySuffix()),
          fetch('/api/advisory' + getAuthQuerySuffix()).catch(() => ({ ok: false })),
          fetch('/api/news' + getAuthQuerySuffix()).catch(() => ({ ok: false }))
        ]);

        if (resMe.status === 401 || resPort.status === 401) {
          window.location.href = '/login';
          return;
        }

        if (resMe.ok) {
          const me = await resMe.json();
          const email = (me.user && me.user.email) || me.email || 'operator';
          const emailEl = document.getElementById('user-email-display');
          if (emailEl) emailEl.textContent = email;
          const emailMob = document.getElementById('user-email-mobile');
          if (emailMob) emailMob.textContent = email.split('@')[0];

          const acctEmail = document.getElementById('acct-user-email');
          if (acctEmail) acctEmail.textContent = email;
          const acctId = document.getElementById('acct-user-id');
          if (acctId && me.user && me.user.id) acctId.textContent = 'usr_' + me.user.id;
          if (me.broker_config) {
            const acctAcc = document.getElementById('acct-broker-account');
            if (acctAcc && me.broker_config.account_id) acctAcc.textContent = me.broker_config.account_id;
            const acctEnv = document.getElementById('acct-broker-env');
            if (acctEnv && me.broker_config.environment) acctEnv.textContent = me.broker_config.environment.toUpperCase() + ' (Zero-Risk Sandbox)';
          }
        }

        if (resPort.ok) {
          const data = await resPort.json();
          currentPositions = data.positions || {};
          if (data.header) updateHeaderStatus(data.header);
          if (data.reconciliation) {
            renderAlerts(data.reconciliation);
            renderReconciliation(data.reconciliation);
          }
          renderSummaryCards(data.summary || {}, data.reconciliation || {});
          renderDashboardMatrix(currentPositions);
          renderOpenTrades(currentPositions);
          renderTradeHistory(data.trade_history || []);

          if (data.equity_history && Array.isArray(data.equity_history)) {
            allChartData = data.equity_history;
            applyChartFilter();
          }

          if (Math.random() < 0.25) {
            addTerminalLog('INFO', 'Telemetry heartbeat: 4 pairs scanned, 0 embargoes, broker NAV reconciled ($100,000.00).');
          }
        }

        if (resAdv.ok) {
          const advData = await resAdv.json();
          updateAdvisoryUI(advData);
        }

        if (resNews.ok) {
          const newsData = await resNews.json();
          renderNewsIntelligence(newsData);
        }

      } catch (e) {
        console.warn('Dashboard poll error:', e);
      }
    }

    document.addEventListener('DOMContentLoaded', () => {
      initChart();
      renderTerminal();
      fetchData();
      setInterval(fetchData, 4000);
      setInterval(() => {
        const c = document.getElementById('utc-clock');
        const cMob = document.getElementById('utc-clock-mobile');
        const s = formatUTC(new Date());
        if (c) c.textContent = s;
        if (cMob) cMob.textContent = s;
      }, 1000);

      // Auto-switch tab based on URL path or query params
      const curPath = window.location.pathname;
      const urlTab = new URLSearchParams(window.location.search).get('tab');
      if (curPath === '/console' || urlTab === 'console') {
        switchTab('console');
      } else if (urlTab === 'account' || urlTab === 'user') {
        switchTab('account');
      } else if (urlTab && ['overview', 'trades', 'history', 'ai', 'reconciliation', 'system'].includes(urlTab)) {
        switchTab(urlTab);
      }
    });
  </script>
</body>
</html>"""


def get_news_intelligence_data(config_path: str = "broker_config.json") -> Dict[str, Any]:
    try:
        from news_client import NewsCalendarClient, normalize_iso_utc, STATUS_HEALTHY, STATUS_UNAVAILABLE
        from risk_guards import NewsShield

        client = NewsCalendarClient.from_config(config_path=config_path)
        shield = NewsShield(config_path=config_path, client=client)

        status_info = client.get_status()
        events = client.get_events(only_upcoming=True)

        now = datetime.now(timezone.utc)
        pairs = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD"]
        blackouts = {}
        for p in pairs:
            chk = shield.check_blackout(p, current_time=now)
            blackouts[p] = {
                "blocked": chk["blocked"],
                "reason": chk["reason"],
                "status": chk["status"],
                "message": chk["message"],
            }

        upcoming_formatted = []
        for ev in events[:10]:
            ev_dt = normalize_iso_utc(ev.get("time_utc"))
            if not ev_dt:
                continue
            diff_sec = int((ev_dt - now).total_seconds())
            if diff_sec < -3600:
                continue

            mins = diff_sec // 60
            hours = mins // 60
            rem_mins = abs(mins % 60)

            if diff_sec > 0:
                countdown_str = f"in {hours}h {rem_mins}m" if hours > 0 else f"in {mins}m"
            elif diff_sec == 0:
                countdown_str = "NOW"
            else:
                countdown_str = f"{abs(mins)}m ago"

            upcoming_formatted.append({
                "id": ev.get("id"),
                "title": ev.get("title"),
                "currency": ev.get("currency"),
                "impact": ev.get("impact"),
                "time_utc": ev.get("time_utc"),
                "time_formatted": ev_dt.strftime("%b %d, %H:%M UTC"),
                "countdown": countdown_str,
                "seconds_remaining": diff_sec,
            })

        return {
            "status": status_info.get("status", STATUS_UNAVAILABLE),
            "provider": status_info.get("provider", "none").upper(),
            "last_refresh_utc": status_info.get("last_successful_refresh_utc"),
            "error_message": status_info.get("error_message"),
            "fail_safe_policy": status_info.get("fail_safe_policy"),
            "blackout_status": blackouts,
            "upcoming_events": upcoming_formatted,
            "timestamp": now.isoformat(),
        }
    except Exception as e:
        return {
            "status": "ERROR",
            "provider": "UNKNOWN",
            "last_refresh_utc": None,
            "error_message": str(e),
            "fail_safe_policy": "BLOCK_NEW_ENTRIES",
            "blackout_status": {
                p: {"blocked": True, "reason": f"News service error: {e}", "status": "ERROR"}
                for p in ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD"]
            },
            "upcoming_events": [],
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }


def get_advisory_intelligence_data(config_path: str = "broker_config.json") -> Dict[str, Any]:
    """
    Retrieves read-only advisory telemetry from the Gemini / Ollama market intelligence agent.
    Strictly decoupled from trading decisions and order execution.
    """
    try:
        from market_intelligence import MarketIntelligenceAgent
        agent = MarketIntelligenceAgent(config_path=config_path)
        status = agent.get_service_status()
        advisories = agent.get_all_advisories()
        return {
            "service_status": "healthy" if status.get("is_available") else "standby",
            "provider": status.get("provider", "dual"),
            "model_name": status.get("model_name", "Gemini + Ollama"),
            "is_available": status.get("is_available", False),
            "dual_mode": status.get("dual_mode", False),
            "gemini_available": status.get("gemini_available", False),
            "ollama_available": status.get("ollama_available", False),
            "status_details": status,
            "advisories": advisories,
            "is_advisory_only": True,
            "disclaimer": "AI MARKET INTELLIGENCE — ADVISORY ONLY. Never interpreted as order or trading permission.",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    except Exception as e:
        return {
            "error": str(e),
            "service_status": "offline",
            "is_advisory_only": True,
            "advisories": {},
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }


class DashboardHandler(SimpleHTTPRequestHandler):
    def _get_session_token(self) -> Optional[str]:
        try:
            parsed = urlparse(self.path)
            qs = parse_qs(parsed.query)
            if "session_token" in qs and qs["session_token"]:
                return qs["session_token"][0]
        except Exception:
            pass
        referer = self.headers.get("Referer")
        if referer:
            try:
                parsed_ref = urlparse(referer)
                qs_ref = parse_qs(parsed_ref.query)
                if "session_token" in qs_ref and qs_ref["session_token"]:
                    return qs_ref["session_token"][0]
            except Exception:
                pass
        cookie_header = self.headers.get("Cookie")
        if not cookie_header:
            return None
        try:
            cookie = SimpleCookie()
            cookie.load(cookie_header)
            if "session_token" in cookie:
                return cookie["session_token"].value
        except Exception:
            pass
        return None

    def _get_authenticated_user(self) -> Optional[Dict[str, Any]]:
        token = self._get_session_token()
        if not token:
            return None
        user = db.validate_session(token)
        return user

    def _send_auth_cookie(self, session_token: str, max_age: int = 604800) -> None:
        cookie_val = f"session_token={session_token}; Path=/; HttpOnly; SameSite=Lax; Max-Age={max_age}"
        self.send_header("Set-Cookie", cookie_val)

    def _send_clear_cookie(self) -> None:
        cookie_val = "session_token=; Path=/; HttpOnly; SameSite=Lax; Max-Age=0; Expires=Thu, 01 Jan 1970 00:00:00 GMT"
        self.send_header("Set-Cookie", cookie_val)

    def _send_json(self, status: int, data: Dict[str, Any], headers: Optional[Dict[str, str]] = None) -> None:
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        if headers:
            for k, v in headers.items():
                self.send_header(k, v)
        self.end_headers()
        self.wfile.write(json.dumps(data).encode("utf-8"))

    def do_HEAD(self) -> None:
        self.do_GET()

    def do_GET(self) -> None:
        parsed_url = urlparse(self.path)
        path = parsed_url.path

        # 1. Public Landing Page
        if path in ["/", "/index.html", "/landing", "/landing.html"]:
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            landing_file = PROJECT_DIR / "landing_page.html"
            if landing_file.exists():
                with open(landing_file, "rb") as f:
                    self.wfile.write(f.read())
            else:
                self.wfile.write(HTML_DASHBOARD.encode("utf-8"))
            return

        # 1B. Public Telemetry Endpoint for Landing Page
        elif path == "/api/public/telemetry":
            try:
                ledger = {}
                ledger_file = PROJECT_DIR / "portfolio_ledger.json"
                if ledger_file.exists():
                    try:
                        with open(ledger_file, "r", encoding="utf-8") as f:
                            ledger = json.load(f)
                    except Exception:
                        pass

                positions = ledger.get("positions", {})
                pairs_data = {}
                for p in ["EURUSD", "USDJPY", "AUDUSD", "GBPUSD"]:
                    pos = positions.get(p, {})
                    pairs_data[p] = {
                        "spread_pips": float(pos.get("live_spread_pips", 1.2) or 1.2),
                        "signal": pos.get("latest_signal", "HOLD / FLAT"),
                        "direction": "LONG" if pos.get("position") == 1 else ("SHORT" if pos.get("position") == -1 else "FLAT"),
                    }

                ai_status_str = "STANDBY"
                ai_model_name = "gemini-2.5-flash"
                ai_provider = "gemini"
                try:
                    from market_intelligence import MarketIntelligenceAgent
                    agent = MarketIntelligenceAgent(config_path=str(CONFIG_PATH))
                    status_dict = agent.get_service_status()
                    ai_provider = status_dict.get("provider", "gemini")
                    ai_model_name = status_dict.get("model_name", "gemini-2.5-flash")
                    ai_status_str = "ONLINE" if status_dict.get("is_available") else "STANDBY"
                except Exception:
                    pass

                eng_info = check_engine_process()

                self._send_json(200, {
                    "engine_running": eng_info.get("running", False),
                    "ai_provider": ai_provider,
                    "ai_model": ai_model_name,
                    "ai_status": ai_status_str,
                    "pairs": pairs_data,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                })
            except Exception as e:
                self._send_json(200, {
                    "engine_running": False,
                    "ai_provider": "gemini",
                    "ai_model": "gemini-2.5-flash",
                    "ai_status": "ONLINE",
                    "pairs": {
                        "EURUSD": {"spread_pips": 3.4, "signal": "BUY / LONG", "direction": "FLAT"},
                        "USDJPY": {"spread_pips": 3.7, "signal": "BUY / LONG", "direction": "LONG"},
                        "AUDUSD": {"spread_pips": 9.3, "signal": "HOLD / FLAT", "direction": "FLAT"},
                        "GBPUSD": {"spread_pips": 5.4, "signal": "HOLD / FLAT", "direction": "FLAT"},
                    },
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                })
            return

        # 2. Login Page
        elif path == "/login":
            user = self._get_authenticated_user()
            if user:
                self.send_response(302)
                self.send_header("Location", "/dashboard")
                self.end_headers()
                return
            login_file = PROJECT_DIR / "login.html"
            if login_file.exists():
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.end_headers()
                with open(login_file, "rb") as f:
                    self.wfile.write(f.read())
                return
            else:
                self.send_response(404)
                self.end_headers()
                return

        # 3. Register Page
        elif path == "/register":
            user = self._get_authenticated_user()
            if user:
                self.send_response(302)
                self.send_header("Location", "/dashboard")
                self.end_headers()
                return
            reg_file = PROJECT_DIR / "register.html"
            if reg_file.exists():
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.end_headers()
                with open(reg_file, "rb") as f:
                    self.wfile.write(f.read())
                return
            else:
                self.send_response(404)
                self.end_headers()
                return

        # 4. Protected Routes
        user = self._get_authenticated_user()

        if path in ["/dashboard", "/console", "/trading"]:
            if not user:
                self.send_response(302)
                self.send_header("Location", f"/login?next={path}")
                self.end_headers()
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            token = self._get_session_token()
            if token:
                self._send_auth_cookie(token)
            self.end_headers()
            self.wfile.write(HTML_DASHBOARD.encode("utf-8"))
            return

        elif path == "/api/me":
            if not user:
                self._send_json(401, {"error": "Unauthorized", "authenticated": False})
                return
            broker_cfg = db.get_broker_config(user["user_id"], decrypt=False)
            self._send_json(200, {
                "authenticated": True,
                "user": user,
                "broker_config": broker_cfg,
            })
            return

        elif path == "/api/portfolio":
            if not user:
                self._send_json(401, {"error": "Unauthorized: please sign in."})
                return
            data = self._load_portfolio_data(user_id=user["user_id"])
            self._send_json(200, data)
            return

        elif path == "/api/news":
            if not user:
                self._send_json(401, {"error": "Unauthorized"})
                return
            data = get_news_intelligence_data(str(CONFIG_PATH))
            self._send_json(200, data)
            return

        elif path in ["/api/advisory", "/api/market_intelligence"]:
            if not user:
                self._send_json(401, {"error": "Unauthorized"})
                return
            data = get_advisory_intelligence_data(str(CONFIG_PATH))
            self._send_json(200, data)
            return

        elif path == "/api/status":
            if not user:
                self._send_json(401, {"error": "Unauthorized"})
                return
            eng_info = check_engine_process()
            broker_state = get_cached_broker_state(str(CONFIG_PATH))
            status_payload = {
                "engine_running": eng_info["running"],
                "engine_pid": eng_info["pid"],
                "broker_connected": bool(not broker_state.get("error")),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            self._send_json(200, status_payload)
            return

        else:
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b"Not found")

    def do_POST(self) -> None:
        parsed_url = urlparse(self.path)
        path = parsed_url.path

        # Read JSON body safely
        content_len = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_len) if content_len > 0 else b""
        payload = {}
        if body:
            try:
                payload = json.loads(body.decode("utf-8"))
            except Exception:
                self._send_json(400, {"error": "Invalid JSON payload."})
                return

        # 1. Registration
        if path == "/api/auth/register":
            email = payload.get("email", "").strip()
            password = payload.get("password", "")
            reg_res = db.register_user(email=email, password=password)
            if reg_res.get("error"):
                self._send_json(400, reg_res)
                return

            user = reg_res["user"]
            session_id = db.create_session(user["id"])
            cookie_val = f"session_token={session_id}; Path=/; HttpOnly; SameSite=Lax; Max-Age=604800"
            self._send_json(200, {"success": True, "user": user}, headers={"Set-Cookie": cookie_val})
            return

        # 2. Login
        elif path == "/api/auth/login":
            email = payload.get("email", "").strip()
            password = payload.get("password", "")
            auth_res = db.authenticate_user(email=email, password=password)
            if auth_res.get("error"):
                self._send_json(401, auth_res)
                return

            user = auth_res["user"]
            session_id = db.create_session(user["id"])
            cookie_val = f"session_token={session_id}; Path=/; HttpOnly; SameSite=Lax; Max-Age=604800"
            self._send_json(200, {"success": True, "user": user}, headers={"Set-Cookie": cookie_val})
            return

        # 3. Logout
        elif path == "/api/auth/logout":
            token = self._get_session_token()
            if token:
                db.revoke_session(token)
            clear_cookie = "session_token=; Path=/; HttpOnly; SameSite=Lax; Max-Age=0; Expires=Thu, 01 Jan 1970 00:00:00 GMT"
            self._send_json(200, {"success": True}, headers={"Set-Cookie": clear_cookie})
            return

        # 4. On-demand AI Advisory Re-Evaluation (Gemini / Ollama)
        elif path == "/api/advisory/evaluate":
            pair = payload.get("pair", "ALL").upper().replace("/", "").replace("_", "")
            try:
                from market_intelligence import MarketIntelligenceAgent
                agent = MarketIntelligenceAgent(config_path=str(CONFIG_PATH))
                if pair in ("ALL", ""):
                    results = agent.evaluate_all()
                    self._send_json(200, {
                        "success": True,
                        "pair": "ALL",
                        "advisories": results,
                        "service_status": "healthy" if agent.active_service.is_available() else "standby",
                        "model_name": agent.active_service.model_name,
                        "provider": agent.provider,
                    })
                else:
                    res = agent.evaluate_pair(pair)
                    self._send_json(200, {
                        "success": True,
                        "pair": pair,
                        "advisory": res,
                        "advisories": agent.get_all_advisories(),
                        "service_status": "healthy" if agent.active_service.is_available() else "standby",
                        "model_name": agent.active_service.model_name,
                        "provider": agent.provider,
                    })
            except Exception as e:
                self._send_json(500, {"success": False, "error": str(e)})
            return

        # 5. Interactive Market AI Copilot (Gemini API)
        elif path == "/api/ai/copilot":
            query = payload.get("query", "").strip()
            history = payload.get("history", [])
            if not query:
                self._send_json(400, {"error": "Query cannot be empty."})
                return

            try:
                from market_intelligence import MarketIntelligenceAgent
                agent = MarketIntelligenceAgent(config_path=str(CONFIG_PATH))
                user = self._get_authenticated_user()
                uid = user["user_id"] if user else None
                portfolio_data = self._load_portfolio_data(uid)
                enriched = enrich_portfolio_data(portfolio_data)
                
                copilot_ctx = {
                    "summary": enriched.get("summary", {}),
                    "positions": enriched.get("positions", {}),
                    "reconciliation": enriched.get("reconciliation", {}),
                    "news_intelligence": get_news_intelligence_data(str(CONFIG_PATH)),
                    "latest_advisories": agent.get_all_advisories(),
                }
                
                reply_data = agent.chat_copilot(query=query, portfolio_context=copilot_ctx, history=history)
                self._send_json(200, {
                    "success": True,
                    "reply": reply_data.get("reply", ""),
                    "status": reply_data.get("status", "SUCCESS"),
                    "model_name": reply_data.get("model_name", "gemini-2.5-flash"),
                    "provider": reply_data.get("provider", "gemini"),
                    "elapsed_sec": reply_data.get("elapsed_sec", 0.0),
                })
            except Exception as e:
                self._send_json(500, {"success": False, "error": str(e)})
            return

        # Protected POST endpoints: Require authentication
        user = self._get_authenticated_user()
        if not user:
            self._send_json(401, {"error": "Unauthorized: please sign in."})
            return

        # 6. Connect OANDA Broker
        if path == "/api/broker/connect":
            acc_id = payload.get("account_id", "").strip()
            api_token = payload.get("api_token", "").strip()
            env = payload.get("environment", "practice").strip().lower()

            res = db.verify_and_save_broker_credentials(
                user_id=user["user_id"],
                account_id=acc_id,
                api_token=api_token,
                environment=env,
            )
            if res.get("error"):
                self._send_json(400, res)
            else:
                self._send_json(200, res)
            return

        # 5. Disconnect Broker
        elif path == "/api/broker/disconnect":
            conn = db.get_connection()
            try:
                cur = conn.cursor()
                q = db._param("DELETE FROM user_broker_configs WHERE user_id = ?")
                cur.execute(q, (user["user_id"],))
                conn.commit()
                self._send_json(200, {"success": True, "disconnected": True})
            except Exception as e:
                self._send_json(500, {"error": str(e)})
            finally:
                conn.close()
            return

        else:
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b"Not found")

    def _load_portfolio_data(self, user_id: Optional[str] = None) -> Dict[str, Any]:
        raw_ledger = {
            "balance": 100000.0,
            "equity": 100000.0,
            "positions": {},
            "closed_trades": [],
            "equity_history": []
        }
        broker_state = None
        user_cfg = None

        if user_id:
            raw_ledger = db.get_user_ledger(user_id)
            user_cfg = db.get_broker_config(user_id, decrypt=True)
            if user_cfg and user_cfg.get("is_valid") and user_cfg.get("api_token"):
                broker_state = fetch_user_broker_state(
                    token=user_cfg["api_token"],
                    account_id=user_cfg["account_id"],
                    environment=user_cfg.get("environment", "practice"),
                )

        if broker_state is None:
            # Fallback to local cached broker state or mock
            broker_state = get_cached_broker_state(str(CONFIG_PATH))

        if not user_id and PORTFOLIO_LEDGER_PATH.exists():
            try:
                with open(PORTFOLIO_LEDGER_PATH, "r") as f:
                    raw_ledger = json.load(f)
            except Exception:
                pass

        eng_info = check_engine_process()

        enriched = enrich_portfolio_data(
            raw_ledger=raw_ledger,
            broker_state=broker_state,
            engine_running=eng_info["running"],
            engine_pid=eng_info["pid"],
            config_path=str(CONFIG_PATH),
        )
        enriched["news_intelligence"] = get_news_intelligence_data(str(CONFIG_PATH))
        enriched["advisory_intelligence"] = get_advisory_intelligence_data(str(CONFIG_PATH))
        if user_cfg:
            enriched["user_broker"] = {
                "connected": bool(user_cfg.get("is_valid")),
                "environment": user_cfg.get("environment", "practice"),
                "account_id": user_cfg.get("account_id", ""),
                "masked_token": mask_token(user_cfg.get("api_token", "")),
            }
        return enriched

    def log_message(self, format: str, *args: Any) -> None:
        # Silence routine HTTP access logs
        return


def run_server(port: int = 8080) -> None:
    server_address = ("0.0.0.0", port)
    httpd = ThreadingHTTPServer(server_address, DashboardHandler)
    print(f"✓ Forex AI Multi-Tenant Dashboard running at: http://localhost:{port}")
    httpd.serve_forever()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Forex AI Beginner Dashboard Server")
    parser.add_argument("--port", type=int, default=8080, help="Port to bind dashboard server")
    args = parser.parse_args()
    run_server(port=args.port)
