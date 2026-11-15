"""
Broker Gateway Integration for Forex AI Trading System.
Provides a unified interface for order routing, account monitoring, and 
server-side bracket execution with OANDA v20 REST API and high-fidelity Mock simulation.
"""

from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

DEFAULT_CONFIG_PATH = "broker_config.json"

SYMBOL_OANDA_MAP = {
    "EURUSD": "EUR_USD",
    "GBPUSD": "GBP_USD",
    "USDJPY": "USD_JPY",
    "AUDUSD": "AUD_USD",
}

OANDA_SYMBOL_MAP = {v: k for k, v in SYMBOL_OANDA_MAP.items()}


class BaseBrokerGateway(ABC):
    """Abstract base class for all broker execution gateways."""

    @staticmethod
    def validate_order_parameters(
        symbol: str,
        units: int,
        stop_loss: float,
        take_profit: float,
        reference_price: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Rigorous pre-flight validation of order units, brackets, and pair-specific sizing constraints.
        Protects against zero, NaN, infinite, inverted brackets, and unnormalized point-sensitivity scalars.
        """
        import math
        # 1. Finite, integer, nonzero units validation
        if units is None or not isinstance(units, (int, float)):
            return {"valid": False, "error": f"Invalid units type: {type(units)}"}
        if math.isnan(units) or math.isinf(units):
            return {"valid": False, "error": f"Units cannot be NaN or Infinite (got {units})"}
        int_units = int(round(units))
        if int_units == 0:
            return {"valid": False, "error": "Order units cannot be zero"}
        if abs(int_units) > 10_000_000:
            return {"valid": False, "error": f"Units exceed maximum broker trade limit (got {int_units:,})"}

        # 2. Stop loss & Take profit validity
        if stop_loss is None or take_profit is None or stop_loss <= 0 or take_profit <= 0:
            return {"valid": False, "error": f"Stop-loss ({stop_loss}) and Take-profit ({take_profit}) must be positive numbers"}
        if math.isnan(stop_loss) or math.isinf(stop_loss) or math.isnan(take_profit) or math.isinf(take_profit):
            return {"valid": False, "error": "Stop-loss and Take-profit must be finite numbers"}

        # 3. Direction and bracket consistency
        clean_sym = symbol.upper().replace("_", "").replace("=X", "")
        if int_units > 0:  # BUY / LONG
            if stop_loss >= take_profit:
                return {"valid": False, "error": f"BUY order invalid brackets: Stop-loss ({stop_loss}) must be below Take-profit ({take_profit})"}
            if reference_price and (stop_loss >= reference_price or take_profit <= reference_price):
                return {"valid": False, "error": f"BUY order invalid brackets relative to price {reference_price}: SL must be < price < TP"}
        else:  # SELL / SHORT
            if stop_loss <= take_profit:
                return {"valid": False, "error": f"SELL order invalid brackets: Stop-loss ({stop_loss}) must be above Take-profit ({take_profit})"}
            if reference_price and (stop_loss <= reference_price or take_profit >= reference_price):
                return {"valid": False, "error": f"SELL order invalid brackets relative to price {reference_price}: TP must be < price < SL"}

        # 4. Pair-Specific Safeguards: USD/JPY Dollar Sensitivity Defense
        if "JPY" in clean_sym or clean_sym.startswith("USDJPY"):
            if abs(int_units) < 1000:
                return {
                    "valid": False,
                    "error": (
                        f"CRITICAL REJECTION: USD/JPY order units ({int_units:,}) is undersized (< 1,000 units). "
                        "Unnormalized dollar-sensitivity scalars must never be passed directly as broker base units."
                    )
                }
            if reference_price:
                sl_dist = abs(reference_price - stop_loss)
                expected_sl_risk_usd = (abs(int_units) * sl_dist) / stop_loss
                if expected_sl_risk_usd < 1.0:
                    return {
                        "valid": False,
                        "error": f"CRITICAL REJECTION: Calculated stop-loss risk (${expected_sl_risk_usd:.2f}) is negligible (< $1.00)."
                    }

        return {"valid": True, "units": int_units}

    @abstractmethod
    def get_account_summary(self) -> Dict[str, Any]:
        """Fetch real-time account balance, equity, and margin stats."""
        pass

    @abstractmethod
    def get_open_positions(self) -> Dict[str, Dict[str, Any]]:
        """Fetch all open positions on the broker server."""
        pass

    @abstractmethod
    def get_open_trades(self) -> List[Dict[str, Any]]:
        """Fetch all individual open trades with tickets, bracket orders, and live unrealized PnL."""
        pass

    @abstractmethod
    def place_market_bracket_order(
        self,
        symbol: str,
        units: int,
        stop_loss: float,
        take_profit: float,
    ) -> Dict[str, Any]:
        """
        Place an atomic market order with server-side stop-loss and take-profit brackets.
        Positive units = Buy/Long, Negative units = Sell/Short.
        """
        pass

    @abstractmethod
    def close_position(self, symbol: str) -> Dict[str, Any]:
        """Close open position for a given currency pair."""
        pass

    @abstractmethod
    def get_realtime_spread(self, symbol: str) -> float:
        """Fetch real-time broker spread in pips."""
        pass


class MockBrokerGateway(BaseBrokerGateway):
    """
    High-fidelity simulated broker sandbox.
    Mirrors real broker API contracts, order tickets, and atomic brackets without credentials.
    """

    def __init__(self, initial_balance: float = 10_000.0) -> None:
        self.balance = initial_balance
        self.equity = initial_balance
        self.positions: Dict[str, Dict[str, Any]] = {}
        self.order_history: List[Dict[str, Any]] = []
        self._ticket_counter = 1000

    def get_account_summary(self) -> Dict[str, Any]:
        unrealized = sum(p.get("unrealized_pnl", 0.0) for p in self.positions.values())
        self.equity = self.balance + unrealized
        return {
            "broker": "MOCK_SIMULATOR",
            "account_id": "MOCK-DEMO-001",
            "currency": "USD",
            "balance": round(self.balance, 2),
            "equity": round(self.equity, 2),
            "unrealized_pnl": round(unrealized, 2),
            "open_position_count": len([p for p in self.positions.values() if p["units"] != 0]),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    def get_open_positions(self) -> Dict[str, Dict[str, Any]]:
        return self.positions

    def get_open_trades(self) -> List[Dict[str, Any]]:
        trades: List[Dict[str, Any]] = []
        for sym, pos in self.positions.items():
            if pos.get("units", 0) != 0:
                trades.append({
                    "id": str(pos.get("ticket", "MOCK-1")),
                    "instrument": sym,
                    "symbol": sym,
                    "units": pos.get("signed_units", pos.get("units", 0)),
                    "direction": pos.get("direction", "LONG"),
                    "price": float(pos.get("entry_price", 0.0)),
                    "unrealized_pl": float(pos.get("unrealized_pnl", 0.0)),
                    "stop_loss": float(pos.get("stop_loss", 0.0)),
                    "take_profit": float(pos.get("take_profit", 0.0)),
                })
        return trades

    def get_realtime_spread(self, symbol: str) -> float:
        return 1.2

    def place_market_bracket_order(
        self,
        symbol: str,
        units: int,
        stop_loss: float,
        take_profit: float,
    ) -> Dict[str, Any]:
        val = self.validate_order_parameters(symbol, units, stop_loss, take_profit)
        if not val["valid"]:
            return {
                "status": "REJECTED",
                "symbol": symbol.upper(),
                "error": "ORDER_VALIDATION_FAILED",
                "message": val["error"],
            }
        units = val["units"]

        self._ticket_counter += 1
        ticket = f"TICK-{self._ticket_counter}"
        direction = "BUY" if units > 0 else "SELL"

        order_record = {
            "ticket": ticket,
            "symbol": symbol.upper(),
            "direction": direction,
            "units": abs(units),
            "signed_units": units,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "status": "FILLED",
            "fill_time": datetime.now(timezone.utc).isoformat(),
            "unrealized_pnl": 0.0,
        }

        self.positions[symbol.upper()] = order_record
        self.order_history.append(order_record)

        return {
            "status": "SUCCESS",
            "ticket": ticket,
            "symbol": symbol.upper(),
            "units": units,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "message": f"Simulated {direction} order filled with server-side brackets (SL: {stop_loss}, TP: {take_profit})",
        }

    def close_position(self, symbol: str) -> Dict[str, Any]:
        sym = symbol.upper()
        if sym in self.positions and self.positions[sym]["units"] != 0:
            closed = self.positions.pop(sym)
            pnl = closed.get("unrealized_pnl", 0.0)
            self.balance += pnl
            self.equity = self.balance
            return {
                "status": "CLOSED",
                "symbol": sym,
                "ticket": closed["ticket"],
                "pnl": pnl,
                "message": f"Closed position on {sym} with P&L: ${pnl:+.2f}",
            }
        return {"status": "NO_POSITION", "symbol": sym, "message": "No active position to close"}


class OandaGateway(BaseBrokerGateway):
    """
    OANDA v20 REST API Broker Gateway.
    Executes trades with native server-side Take-Profit & Stop-Loss brackets on OANDA.
    """

    def __init__(
        self,
        api_token: str,
        account_id: str,
        environment: str = "practice",
    ) -> None:
        self.api_token = api_token.strip()
        self.account_id = account_id.strip()
        self.environment = environment.lower()

        if self.environment == "live":
            self.base_url = "https://api-fxtrade.oanda.com/v3"
        else:
            self.base_url = "https://api-fxpractice.oanda.com/v3"

        self.headers = {
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    def _format_symbol(self, symbol: str) -> str:
        clean = symbol.upper().replace("_", "").replace("=X", "")
        return SYMBOL_OANDA_MAP.get(clean, f"{clean[:3]}_{clean[3:]}")

    def _format_price(self, symbol: str, price: float) -> str:
        clean = symbol.upper().replace("_", "")
        # JPY pairs use 3 decimal places on OANDA, all others use 5
        precision = 3 if "JPY" in clean else 5
        return f"{price:.{precision}f}"

    def get_account_summary(self) -> Dict[str, Any]:
        url = f"{self.base_url}/accounts/{self.account_id}/summary"
        try:
            resp = requests.get(url, headers=self.headers, timeout=10)
            if resp.status_code == 200:
                acc = resp.json().get("account", {})
                return {
                    "broker": "OANDA_V20",
                    "environment": self.environment,
                    "account_id": self.account_id,
                    "currency": acc.get("currency", "USD"),
                    "balance": float(acc.get("balance", 0.0)),
                    "equity": float(acc.get("NAV", 0.0)),
                    "unrealized_pnl": float(acc.get("unrealizedPL", 0.0)),
                    "margin_used": float(acc.get("marginUsed", 0.0)),
                    "margin_available": float(acc.get("marginAvailable", 0.0)),
                    "open_position_count": int(acc.get("openPositionCount", 0)),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
            else:
                return {
                    "error": True,
                    "status_code": resp.status_code,
                    "message": resp.text,
                }
        except Exception as e:
            return {"error": True, "message": str(e)}

    def get_open_positions(self) -> Dict[str, Dict[str, Any]]:
        url = f"{self.base_url}/accounts/{self.account_id}/openPositions"
        positions: Dict[str, Dict[str, Any]] = {}
        try:
            resp = requests.get(url, headers=self.headers, timeout=10)
            if resp.status_code == 200:
                raw_positions = resp.json().get("positions", [])
                for p in raw_positions:
                    instrument = p.get("instrument", "")
                    clean_sym = instrument.replace("_", "")
                    long_p = p.get("long", {})
                    short_p = p.get("short", {})
                    long_units = int(long_p.get("units", 0))
                    short_units = int(short_p.get("units", 0))

                    if long_units > 0:
                        trade_ids = long_p.get("tradeIDs", [])
                        positions[clean_sym] = {
                            "symbol": clean_sym,
                            "direction": "LONG",
                            "units": long_units,
                            "entry_price": float(long_p.get("averagePrice", 0.0)),
                            "unrealized_pnl": float(long_p.get("unrealizedPL", 0.0)),
                            "trade_ids": trade_ids,
                            "broker_ticket": str(trade_ids[0]) if trade_ids else "N/A",
                        }
                    elif short_units < 0:
                        trade_ids = short_p.get("tradeIDs", [])
                        positions[clean_sym] = {
                            "symbol": clean_sym,
                            "direction": "SHORT",
                            "units": abs(short_units),
                            "entry_price": float(short_p.get("averagePrice", 0.0)),
                            "unrealized_pnl": float(short_p.get("unrealizedPL", 0.0)),
                            "trade_ids": trade_ids,
                            "broker_ticket": str(trade_ids[0]) if trade_ids else "N/A",
                        }
        except Exception as e:
            print(f"Error fetching OANDA open positions: {e}")
        return positions

    def get_open_trades(self) -> List[Dict[str, Any]]:
        url = f"{self.base_url}/accounts/{self.account_id}/openTrades"
        trades: List[Dict[str, Any]] = []
        try:
            resp = requests.get(url, headers=self.headers, timeout=10)
            if resp.status_code == 200:
                raw_trades = resp.json().get("trades", [])
                for t in raw_trades:
                    instrument = t.get("instrument", "")
                    clean_sym = instrument.replace("_", "")
                    current_units = int(t.get("currentUnits", 0))
                    trades.append({
                        "id": str(t.get("id")),
                        "symbol": clean_sym,
                        "instrument": instrument,
                        "units": current_units,
                        "direction": "LONG" if current_units > 0 else "SHORT",
                        "price": float(t.get("price", 0.0)),
                        "unrealized_pl": float(t.get("unrealizedPL", 0.0)),
                        "stop_loss": float(t.get("stopLossOrder", {}).get("price", 0.0)) if "stopLossOrder" in t else 0.0,
                        "take_profit": float(t.get("takeProfitOrder", {}).get("price", 0.0)) if "takeProfitOrder" in t else 0.0,
                        "open_time": t.get("openTime"),
                    })
        except Exception as e:
            print(f"Error fetching OANDA open trades: {e}")
        return trades

    def get_realtime_spread(self, symbol: str) -> float:
        instrument = self._format_symbol(symbol)
        url = f"{self.base_url}/accounts/{self.account_id}/pricing?instruments={instrument}"
        try:
            resp = requests.get(url, headers=self.headers, timeout=6)
            if resp.status_code == 200:
                prices = resp.json().get("prices", [])
                if prices:
                    p = prices[0]
                    if p.get("bids") and p.get("asks"):
                        bid = float(p["bids"][0].get("price", 0.0))
                        ask = float(p["asks"][0].get("price", 0.0))
                    else:
                        bid = float(p.get("closeoutBid", 0.0))
                        ask = float(p.get("closeoutAsk", 0.0))
                    mult = 100.0 if "JPY" in instrument else 10000.0
                    return round((ask - bid) * mult, 2)
        except Exception as e:
            print(f"Error fetching real-time spread for {symbol}: {e}")
        return 0.0

    def place_market_bracket_order(
        self,
        symbol: str,
        units: int,
        stop_loss: float,
        take_profit: float,
    ) -> Dict[str, Any]:
        val = self.validate_order_parameters(symbol, units, stop_loss, take_profit)
        if not val["valid"]:
            return {
                "status": "REJECTED",
                "symbol": symbol.upper(),
                "error": "ORDER_VALIDATION_FAILED",
                "message": val["error"],
            }
        units = val["units"]

        instrument = self._format_symbol(symbol)
        sl_formatted = self._format_price(symbol, stop_loss)
        tp_formatted = self._format_price(symbol, take_profit)

        url = f"{self.base_url}/accounts/{self.account_id}/orders"
        order_payload = {
            "order": {
                "type": "MARKET",
                "instrument": instrument,
                "units": str(units),
                "timeInForce": "FOK",
                "positionFill": "DEFAULT",
                "stopLossOnFill": {
                    "price": sl_formatted,
                    "timeInForce": "GTC",
                },
                "takeProfitOnFill": {
                    "price": tp_formatted,
                    "timeInForce": "GTC",
                },
            }
        }

        try:
            resp = requests.post(url, headers=self.headers, json=order_payload, timeout=12)
            data = resp.json()
            if resp.status_code in [200, 201]:
                fill = data.get("orderFillTransaction", {})
                ticket = fill.get("id", data.get("orderCreateTransaction", {}).get("id", "UNKNOWN"))
                fill_price = float(fill.get("price", 0.0)) if fill else None
                return {
                    "status": "SUCCESS",
                    "ticket": str(ticket),
                    "symbol": symbol.upper(),
                    "units": units,
                    "fill_price": fill_price,
                    "stop_loss": stop_loss,
                    "take_profit": take_profit,
                    "message": f"OANDA Order #{ticket} FILLED @ {fill_price} with server-side brackets",
                    "raw": data,
                }
            else:
                return {
                    "status": "REJECTED",
                    "status_code": resp.status_code,
                    "symbol": symbol.upper(),
                    "message": data.get("errorMessage", resp.text),
                    "raw": data,
                }
        except Exception as e:
            return {
                "status": "ERROR",
                "symbol": symbol.upper(),
                "message": f"Network exception communicating with OANDA: {e}",
            }

    def close_position(self, symbol: str) -> Dict[str, Any]:
        instrument = self._format_symbol(symbol)
        url = f"{self.base_url}/accounts/{self.account_id}/positions/{instrument}/close"
        
        pos_map = self.get_open_positions()
        clean_sym = symbol.upper().replace("_", "").replace("=X", "")
        pos_entry = pos_map.get(clean_sym)

        if not pos_entry:
            return {
                "status": "NO_POSITION",
                "symbol": symbol.upper(),
                "message": f"No active broker position on {instrument} to close",
            }

        payload = {}
        if pos_entry.get("direction") == "LONG":
            payload = {"longUnits": "ALL"}
        elif pos_entry.get("direction") == "SHORT":
            payload = {"shortUnits": "ALL"}

        try:
            resp = requests.put(url, headers=self.headers, json=payload, timeout=12)
            if resp.status_code in [200, 201]:
                return {"status": "CLOSED", "symbol": symbol.upper(), "message": f"Closed position on {instrument}"}
            else:
                return {"status": "ERROR", "message": resp.text}
        except Exception as e:
            return {"status": "ERROR", "message": str(e)}


def create_broker_gateway(config_path: str = DEFAULT_CONFIG_PATH) -> BaseBrokerGateway:
    """Factory function to instantiate the configured broker gateway."""
    cpath = Path(config_path)
    if cpath.exists():
        try:
            with open(cpath, "r") as f:
                cfg = json.load(f)

            broker_type = cfg.get("broker", "mock").lower()
            if broker_type == "oanda":
                oanda_cfg = cfg.get("oanda", {})
                token = oanda_cfg.get("api_token", "")
                acc_id = oanda_cfg.get("account_id", "")
                env = cfg.get("environment", "practice")
                if token and acc_id and "YOUR_" not in token:
                    print(f"✓ Initialized OANDA Gateway ({env.upper()} mode, Account: {acc_id})")
                    return OandaGateway(api_token=token, account_id=acc_id, environment=env)
                else:
                    print("⚠️ OANDA credentials not set in broker_config.json. Defaulting to Mock Sandbox Gateway.")
        except Exception as e:
            print(f"Warning: Could not parse {config_path} ({e}). Defaulting to Mock Sandbox.")

    print("✓ Initialized High-Fidelity Mock Broker Gateway (Zero-Risk Simulation Sandbox)")
    return MockBrokerGateway()
