"""
Operations Helper for Forex AI Trading System Control Script (run_bot.sh).
Provides safe, read-only utilities for configuration validation, secret masking,
ledger inspection, and non-destructive broker health checks without external CLI tools.
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, Optional

# Pre-compiled regex for masking sensitive credentials in streams
TOKEN_PATTERN = re.compile(r"([0-9a-fA-F]{32,64}-[0-9a-fA-F]{32,64}|Bearer\s+[^\s\"']+)")


def mask_string(text: str) -> str:
    """Mask tokens and sensitive hex strings in text."""
    return TOKEN_PATTERN.sub("[REDACTED_TOKEN]", text)


def cmd_mask_stdin() -> None:
    """Stream filter that masks sensitive tokens from stdin to stdout."""
    for line in sys.stdin:
        sys.stdout.write(mask_string(line))


def get_config_info(config_path: str = "broker_config.json") -> Dict[str, Any]:
    """Safely load and validate broker configuration without exposing secrets."""
    cpath = Path(config_path)
    if not cpath.exists():
        return {"error": f"Configuration file not found: {config_path}"}

    try:
        with open(cpath, "r") as f:
            cfg = json.load(f)
    except Exception as e:
        return {"error": f"Invalid JSON syntax in {config_path}: {e}"}

    broker = str(cfg.get("broker", "mock")).lower().strip()
    environment = str(cfg.get("environment", "practice")).lower().strip()

    # Determine credentials presence (boolean only, never output value)
    has_token = False
    has_account = False
    masked_account = "None"

    if broker == "oanda":
        oanda_cfg = cfg.get("oanda", {})
        token = str(oanda_cfg.get("api_token", "")).strip()
        account_id = str(oanda_cfg.get("account_id", "")).strip()
        has_token = bool(token and "YOUR_" not in token)
        has_account = bool(account_id and "YOUR_" not in account_id)
        if has_account:
            # Mask middle digits for security display: 101-001-40816673-001 -> 101-001-***-001
            parts = account_id.split("-")
            if len(parts) >= 4:
                masked_account = f"{parts[0]}-{parts[1]}-***-{parts[-1]}"
            else:
                masked_account = f"{account_id[:4]}***{account_id[-3:]}"

    return {
        "broker": broker,
        "environment": environment,
        "has_token": has_token,
        "has_account": has_account,
        "masked_account": masked_account,
        "guards": cfg.get("guards", {}),
    }


def cmd_get_mode(config_path: str = "broker_config.json") -> None:
    """Output machine-readable configuration mode: broker|environment|is_valid|error_message"""
    info = get_config_info(config_path)
    if "error" in info:
        print(f"UNKNOWN|UNKNOWN|INVALID|{info['error']}")
        return

    broker = info["broker"]
    env = info["environment"]

    if broker not in ["oanda", "mock"]:
        print(f"{broker}|{env}|INVALID|Unsupported broker '{broker}'. Must be 'oanda' or 'mock'.")
        return

    if env not in ["practice", "live"]:
        print(f"{broker}|{env}|INVALID|Unsupported environment '{env}'. Must be 'practice' or 'live'.")
        return

    if broker == "oanda" and (not info["has_token"] or not info["has_account"]):
        print(f"{broker}|{env}|INVALID|OANDA broker selected but API token or Account ID is missing.")
        return

    print(f"{broker}|{env}|VALID|OK")


def normalize_symbol(sym: str) -> str:
    """Normalize currency pair symbol strings (e.g. AUD_USD, AUDUSD=X -> AUDUSD)."""
    return sym.upper().replace("_", "").replace("=X", "")


def fetch_broker_state(config_path: str = "broker_config.json") -> Dict[str, Any]:
    """Safe, read-only query to fetch broker account stats and open trades."""
    info = get_config_info(config_path)
    if "error" in info:
        return {"error": True, "message": info["error"]}

    broker = info["broker"]
    if broker == "mock":
        return {
            "broker": "mock",
            "environment": "practice",
            "balance": 10000.0,
            "equity": 10000.0,
            "unrealized_pnl": 0.0,
            "margin_used": 0.0,
            "margin_available": 10000.0,
            "open_position_count": 0,
            "trades": [],
        }

    try:
        with open(config_path, "r") as f:
            cfg = json.load(f)
        o_cfg = cfg.get("oanda", {})
        token = o_cfg.get("api_token")
        acc_id = o_cfg.get("account_id")
        env = cfg.get("environment", "practice")
    except Exception as e:
        return {"error": True, "message": f"Failed reading config: {e}"}

    base_url = "https://api-fxtrade.oanda.com/v3" if env == "live" else "https://api-fxpractice.oanda.com/v3"
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/json"}

    try:
        import requests
        # 1. Summary
        s_resp = requests.get(f"{base_url}/accounts/{acc_id}/summary", headers=headers, timeout=8)
        if s_resp.status_code != 200:
            return {"error": True, "message": f"Broker returned HTTP {s_resp.status_code}"}
        acc = s_resp.json().get("account", {})

        # 2. Open Trades
        t_resp = requests.get(f"{base_url}/accounts/{acc_id}/openTrades", headers=headers, timeout=8)
        trades_list = []
        if t_resp.status_code == 200:
            raw_trades = t_resp.json().get("trades", [])
            for t in raw_trades:
                sym = normalize_symbol(t.get("instrument", ""))
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
            "open_position_count": int(acc.get("openPositionCount", len(trades_list))),
            "trades": trades_list,
        }
    except Exception as e:
        return {"error": True, "message": f"Broker communication failure: {e}"}


def reconcile_positions(
    config_path: str = "broker_config.json",
    ledger_path: str = "portfolio_ledger.json",
) -> Dict[str, Any]:
    """
    Compare broker-side trading state against the local ledger.
    Categorizes every position without placing orders or modifying broker state.
    """
    b_state = fetch_broker_state(config_path)
    if b_state.get("error"):
        return {
            "overall_status": "BROKER_UNAVAILABLE",
            "error": b_state["message"],
            "positions": {},
        }

    lpath = Path(ledger_path)
    if not lpath.exists():
        return {
            "overall_status": "RECONCILIATION_REQUIRED",
            "error": "Local ledger file missing",
            "positions": {},
        }

    try:
        with open(lpath, "r") as f:
            ledger = json.load(f)
    except Exception as e:
        return {
            "overall_status": "RECONCILIATION_REQUIRED",
            "error": f"Corrupted local ledger: {e}",
            "positions": {},
        }

    b_trades_by_sym = {t["symbol"]: t for t in b_state.get("trades", [])}
    l_positions = ledger.get("positions", {})

    all_pairs = sorted(list(set(list(l_positions.keys()) + list(b_trades_by_sym.keys()))))

    positions_report = {}
    confirmed_count = 0
    simulated_count = 0
    unmatched_broker_count = 0
    closed_at_broker_count = 0
    mismatch_count = 0

    simulated_upl_sum = 0.0

    for pair in all_pairs:
        lp = l_positions.get(pair, {})
        bt = b_trades_by_sym.get(pair)
        pos = lp.get("position", 0)
        ticket = str(lp.get("broker_ticket", "")).strip()
        local_upl = float(lp.get("unrealized_pnl", 0.0))

        if pos != 0:
            if bt:
                # Both exist: check ticket and direction match
                b_ticket = bt["trade_id"] if "trade_id" in bt else bt.get("id", "")
                ticket_matches = (ticket == b_ticket) or (ticket in ["", "LOCAL", "N/A"])
                dir_matches = (pos == 1 and bt["direction"] == "LONG") or (pos == -1 and bt["direction"] == "SHORT")

                if ticket_matches and dir_matches:
                    cat = "BROKER_CONFIRMED"
                    confirmed_count += 1
                else:
                    cat = "RECONCILIATION_MISMATCH"
                    mismatch_count += 1
            else:
                if ticket and ticket not in ["N/A", "LOCAL", "None", ""]:
                    cat = "CLOSED_AT_BROKER"
                    closed_at_broker_count += 1
                else:
                    cat = "SIMULATED"
                    simulated_count += 1
                    simulated_upl_sum += local_upl
        else:
            if bt:
                cat = "UNMATCHED_BROKER"
                unmatched_broker_count += 1
            else:
                cat = "FLAT"

        positions_report[pair] = {
            "symbol": pair,
            "classification": cat,
            "local_pos": pos,
            "local_ticket": ticket if ticket else "N/A",
            "broker_trade_id": bt["id"] if bt else "N/A",
            "broker_units": bt["units"] if bt else 0,
            "local_size": float(lp.get("position_size", 0.0)),
            "local_upl": local_upl,
            "broker_upl": bt["unrealized_pl"] if bt else 0.0,
            "entry_price": float(lp.get("entry_price", 0.0)),
            "broker_price": bt["price"] if bt else 0.0,
        }

    # Determine overall status
    if unmatched_broker_count > 0 or closed_at_broker_count > 0 or mismatch_count > 0:
        overall_status = "RECONCILIATION_REQUIRED"
    elif simulated_count > 0:
        overall_status = "RECONCILED_WITH_SIMULATED_POSITIONS"
    else:
        overall_status = "RECONCILED"

    b_bal = float(b_state.get("balance", 0.0))
    b_eq = float(b_state.get("equity", 0.0))
    b_upl = float(b_state.get("unrealized_pnl", 0.0))

    l_bal = float(ledger.get("balance", 0.0))
    l_eq = float(ledger.get("equity", 0.0))
    l_upl = l_eq - l_bal

    discrepancy = l_eq - b_eq

    explanation = []
    if abs(l_bal - b_bal) > 0.01:
        explanation.append(f"Cash balance diff (${l_bal - b_bal:+.2f}) from local trade history")
    if simulated_count > 0:
        explanation.append(f"Includes ${simulated_upl_sum:+.2f} unrealized PnL from {simulated_count} simulated positions")
    if not explanation:
        explanation.append("Local strategy equity aligns with broker equity")

    return {
        "overall_status": overall_status,
        "broker": b_state,
        "ledger_summary": {
            "balance": l_bal,
            "equity": l_eq,
            "unrealized_pnl": l_upl,
            "closed_trades_count": len(ledger.get("closed_trades", [])),
        },
        "discrepancy_amount": round(discrepancy, 2),
        "discrepancy_explanation": "; ".join(explanation),
        "positions": positions_report,
        "counts": {
            "confirmed": confirmed_count,
            "simulated": simulated_count,
            "unmatched_broker": unmatched_broker_count,
            "closed_at_broker": closed_at_broker_count,
            "mismatch": mismatch_count,
        },
    }


def cmd_reconcile(
    config_path: str = "broker_config.json",
    ledger_path: str = "portfolio_ledger.json",
) -> None:
    """CLI output for position and equity reconciliation."""
    rep = reconcile_positions(config_path, ledger_path)
    status = rep.get("overall_status", "UNKNOWN")

    print(f"RECONCILIATION_STATUS: {status}")
    if "error" in rep:
        print(f"ERROR: {rep['error']}")
        return

    b = rep.get("broker", {})
    l = rep.get("ledger_summary", {})
    counts = rep.get("counts", {})

    print(f"BROKER_BALANCE: {b.get('balance', 0.0):.2f}")
    print(f"BROKER_EQUITY: {b.get('equity', 0.0):.2f}")
    print(f"BROKER_UNREALIZED_PNL: {b.get('unrealized_pnl', 0.0):+.2f}")
    print(f"BROKER_OPEN_TRADES: {len(b.get('trades', []))}")
    print(f"LOCAL_BALANCE: {l.get('balance', 0.0):.2f}")
    print(f"LOCAL_EQUITY: {l.get('equity', 0.0):.2f}")
    print(f"LOCAL_UNREALIZED_PNL: {l.get('unrealized_pnl', 0.0):+.2f}")
    print(f"EQUITY_DISCREPANCY: ${rep.get('discrepancy_amount', 0.0):+.2f}")
    print(f"DISCREPANCY_EXPLANATION: {rep.get('discrepancy_explanation', 'None')}")
    print(f"SUMMARY_COUNTS: Confirmed={counts.get('confirmed', 0)} | Simulated={counts.get('simulated', 0)} | Unmatched={counts.get('unmatched_broker', 0)} | ClosedAtBroker={counts.get('closed_at_broker', 0)}")

    for pair, p in rep.get("positions", {}).items():
        cat = p["classification"]
        lt = p["local_ticket"]
        bt = p["broker_trade_id"]
        lupl = p["local_upl"]
        bupl = p["broker_upl"]
        print(f"RECON_POS_{pair}: {cat}|LocalTicket:{lt}|BrokerID:{bt}|LocalPnL:${lupl:+.2f}|BrokerPnL:${bupl:+.2f}")


def cmd_broker_summary(config_path: str = "broker_config.json") -> None:
    """Safe, read-only query to broker API for live account stats."""
    info = get_config_info(config_path)
    if "error" in info:
        print(f"ERROR: {info['error']}")
        sys.exit(1)

    broker = info["broker"]
    if broker == "mock":
        print("BROKER: Mock Simulator (Local Sandbox)")
        print("ENVIRONMENT: practice")
        print("ACCOUNT_ID: MOCK-DEMO-001")
        print("CONNECTION: OK (Local Simulation)")
        return

    # OANDA read-only query
    try:
        with open(config_path, "r") as f:
            cfg = json.load(f)
        o_cfg = cfg.get("oanda", {})
        token = o_cfg.get("api_token")
        acc_id = o_cfg.get("account_id")
        env = cfg.get("environment", "practice")
    except Exception as e:
        print(f"ERROR: Failed reading config: {e}")
        sys.exit(1)

    base_url = "https://api-fxtrade.oanda.com/v3" if env == "live" else "https://api-fxpractice.oanda.com/v3"
    url = f"{base_url}/accounts/{acc_id}/summary"
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/json"}

    try:
        import requests
        resp = requests.get(url, headers=headers, timeout=8)
        if resp.status_code == 200:
            acc = resp.json().get("account", {})
            print(f"BROKER: OANDA v20 REST API")
            print(f"ENVIRONMENT: {env.upper()}")
            print(f"ACCOUNT_ID: {info['masked_account']}")
            print(f"CONNECTION: CONNECTED (HTTP 200)")
            print(f"CURRENCY: {acc.get('currency', 'USD')}")
            print(f"BALANCE: {float(acc.get('balance', 0.0)):.2f}")
            print(f"EQUITY: {float(acc.get('NAV', 0.0)):.2f}")
            print(f"UNREALIZED_PNL: {float(acc.get('unrealizedPL', 0.0)):+.2f}")
            print(f"MARGIN_USED: {float(acc.get('marginUsed', 0.0)):.2f}")
            print(f"MARGIN_AVAILABLE: {float(acc.get('marginAvailable', 0.0)):.2f}")
            print(f"OPEN_POSITIONS_COUNT: {int(acc.get('openPositionCount', 0))}")
        else:
            print(f"BROKER: OANDA v20 REST API")
            print(f"ENVIRONMENT: {env.upper()}")
            print(f"ACCOUNT_ID: {info['masked_account']}")
            print(f"CONNECTION: ERROR (HTTP {resp.status_code})")
            print(f"ERROR_MESSAGE: {mask_string(resp.text[:120])}")
    except Exception as e:
        print(f"BROKER: OANDA v20 REST API")
        print(f"ENVIRONMENT: {env.upper()}")
        print(f"ACCOUNT_ID: {info['masked_account']}")
        print(f"CONNECTION: UNREACHABLE")
        print(f"ERROR_MESSAGE: {e}")


def cmd_ledger_summary(ledger_path: str = "portfolio_ledger.json") -> None:
    """Read local portfolio ledger without modifying it."""
    lpath = Path(ledger_path)
    if not lpath.exists():
        print("LEDGER_STATUS: MISSING")
        return

    try:
        with open(lpath, "r") as f:
            data = json.load(f)
    except Exception as e:
        print(f"LEDGER_STATUS: CORRUPTED ({e})")
        return

    print("LEDGER_STATUS: VALID")
    print(f"LOCAL_BALANCE: {float(data.get('balance', 0.0)):.2f}")
    print(f"LOCAL_EQUITY: {float(data.get('equity', 0.0)):.2f}")
    unrealized = float(data.get("equity", 0.0)) - float(data.get("balance", 0.0))
    print(f"LOCAL_UNREALIZED_PNL: {unrealized:+.2f}")
    print(f"CLOSED_TRADES_COUNT: {len(data.get('closed_trades', []))}")
    print(f"LAST_UPDATED: {data.get('created_at', 'Unknown')}")

    # Output open positions
    positions = data.get("positions", {})
    active_count = sum(1 for p in positions.values() if p.get("position", 0) != 0)
    print(f"LOCAL_ACTIVE_POSITIONS_COUNT: {active_count}")

    for pair, p in positions.items():
        pos = p.get("position", 0)
        direction = "LONG" if pos == 1 else ("SHORT" if pos == -1 else "FLAT")
        ticket = p.get("broker_ticket", "N/A")
        if pos != 0:
            category = "BROKER_LINKED" if (ticket and ticket not in ["N/A", "LOCAL", "None", ""]) else "SIMULATED"
        else:
            category = "FLAT"
        entry = p.get("entry_price", 0.0)
        sl = p.get("stop_loss", 0.0)
        tp = p.get("take_profit", 0.0)
        pnl = p.get("unrealized_pnl", 0.0)
        hours = p.get("time_in_trade", 0)
        spread = p.get("live_spread", 0.0)
        print(f"POS_{pair}: {direction}|{entry:.5f}|{sl:.5f}|{tp:.5f}|{pnl:+.2f}|{hours}h|Ticket:{ticket}|Category:{category}|Spread:{spread:.1f}p")


def cmd_verify_deps() -> None:
    """Verify that all required Python packages are installed (fast spec check)."""
    import importlib.util
    required = [
        ("torch", "PyTorch"),
        ("numpy", "NumPy"),
        ("pandas", "Pandas"),
        ("sklearn", "Scikit-Learn"),
        ("ta", "Technical Analysis (ta)"),
        ("requests", "Requests"),
        ("yfinance", "Yahoo Finance (yfinance)"),
    ]
    missing = []
    for mod_name, label in required:
        try:
            if importlib.util.find_spec(mod_name) is None:
                missing.append(label)
        except Exception:
            missing.append(label)

    if missing:
        print(f"MISSING: {', '.join(missing)}")
        sys.exit(1)
    else:
        print("ALL_DEPENDENCIES_AVAILABLE")
        sys.exit(0)


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: ops_helper.py <get_mode|broker_summary|ledger_summary|reconcile|verify_deps|mask_stdin> [args...]")
        sys.exit(1)

    cmd = sys.argv[1]
    if cmd == "get_mode":
        cfg = sys.argv[2] if len(sys.argv) > 2 else "broker_config.json"
        cmd_get_mode(cfg)
    elif cmd == "broker_summary":
        cfg = sys.argv[2] if len(sys.argv) > 2 else "broker_config.json"
        cmd_broker_summary(cfg)
    elif cmd == "ledger_summary":
        led = sys.argv[2] if len(sys.argv) > 2 else "portfolio_ledger.json"
        cmd_ledger_summary(led)
    elif cmd == "reconcile":
        cfg = sys.argv[2] if len(sys.argv) > 2 else "broker_config.json"
        led = sys.argv[3] if len(sys.argv) > 3 else "portfolio_ledger.json"
        cmd_reconcile(cfg, led)
    elif cmd == "verify_deps":
        cmd_verify_deps()
    elif cmd == "mask_stdin":
        cmd_mask_stdin()
    else:
        print(f"Unknown command: {cmd}")
        sys.exit(1)


if __name__ == "__main__":
    main()
