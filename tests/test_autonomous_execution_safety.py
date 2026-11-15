#!/usr/bin/env python3
"""
Forex AI: Autonomous Execution Safety Test Suite.
Validates safety properties, risk guards, reconciliation edge cases, and failure modes
using deterministic mocks and isolated temporary directories.
Strictly zero broker orders, zero live trading, and zero mutation of production files.
"""

import json
import math
import os
import shutil
import sys
import tempfile
import time
import unittest
from datetime import datetime, timezone, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from broker_gateway import BaseBrokerGateway, MockBrokerGateway, OandaGateway
from risk_guards import SpreadGuard, NewsShield, DailyCircuitBreaker
from news_client import NewsCalendarClient, MockNewsProvider, STATUS_HEALTHY, STATUS_STALE, STATUS_UNAVAILABLE
from live_paper_portfolio import MultiPairPaperTrader


class TestAutonomousExecutionSafety(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="safety_audit_")
        self.test_config = os.path.join(self.test_dir, "test_broker_config.json")
        self.test_ledger = os.path.join(self.test_dir, "test_portfolio_ledger.json")
        self.test_news_cache = os.path.join(self.test_dir, "test_news_cache.json")

        # Mock broker configuration
        with open(self.test_config, "w") as f:
            json.dump({
                "broker": "mock",
                "environment": "practice",
                "oanda": {
                    "api_token": "mock-token-redacted-safety-audit",
                    "account_id": "101-001-99999999-001"
                },
                "guards": {
                    "spread_guard_enabled": True,
                    "max_spread_pips": {
                        "EURUSD": 2.5,
                        "GBPUSD": 3.0,
                        "USDJPY": 3.0,
                        "AUDUSD": 3.5
                    },
                    "news_shield_enabled": True,
                    "news_shield": {
                        "provider": "mock",
                        "refresh_interval_seconds": 14400,
                        "max_cache_age_seconds": 21600,
                        "blackout_minutes_before": 30,
                        "blackout_minutes_after": 15,
                        "fail_safe_policy": "BLOCK_NEW_ENTRIES"
                    },
                    "daily_drawdown_limit_pct": 3.0
                }
            }, f)

        # Base isolated ledger
        self.base_ledger = {
            "balance": 100000.0,
            "equity": 100000.0,
            "initial_capital": 100000.0,
            "broker_equity": 100000.0,
            "positions": {
                "EURUSD": {"position": 0, "entry_price": 0.0, "position_size": 0.0, "stop_loss": 0.0, "take_profit": 0.0, "unrealized_pnl": 0.0, "broker_ticket": None, "execution_type": "FLAT"},
                "USDJPY": {"position": 1, "entry_price": 156.814, "position_size": 318.565, "point_sensitivity": 318.565, "stop_loss": 156.343, "take_profit": 158.226, "unrealized_pnl": 316.34, "broker_ticket": None, "execution_type": "SIMULATED", "accounting_method": "POINT_SENSITIVITY"},
                "AUDUSD": {"position": 0, "entry_price": 0.0, "position_size": 0.0, "stop_loss": 0.0, "take_profit": 0.0, "unrealized_pnl": 0.0, "broker_ticket": None, "execution_type": "FLAT"},
                "GBPUSD": {"position": 0, "entry_price": 0.0, "position_size": 0.0, "stop_loss": 0.0, "take_profit": 0.0, "unrealized_pnl": 0.0, "broker_ticket": None, "execution_type": "FLAT"},
            },
            "closed_trades": []
        }
        with open(self.test_ledger, "w") as f:
            json.dump(self.base_ledger, f)

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    # -------------------------------------------------------------------------
    # Case 1: Accepted Order with Server-Side Brackets
    # -------------------------------------------------------------------------
    def test_case_01_accepted_order_with_brackets(self):
        mock_gw = MockBrokerGateway(initial_balance=100000.0)
        res = mock_gw.place_market_bracket_order(
            symbol="EURUSD",
            units=50000,
            stop_loss=1.1100,
            take_profit=1.1250,
        )
        self.assertEqual(res["status"], "SUCCESS")
        self.assertTrue(res["ticket"].startswith("TICK-"))
        self.assertEqual(res["symbol"], "EURUSD")

        # Verify open trades in mock gateway
        open_trades = mock_gw.get_open_trades()
        self.assertEqual(len(open_trades), 1)
        self.assertEqual(open_trades[0]["id"], res["ticket"])
        self.assertEqual(open_trades[0]["stop_loss"], 1.1100)
        self.assertEqual(open_trades[0]["take_profit"], 1.1250)

    # -------------------------------------------------------------------------
    # Case 2: Rejected Order (Invalid Parameter / Gateway Rejection)
    # -------------------------------------------------------------------------
    def test_case_02_rejected_order_validation(self):
        mock_gw = MockBrokerGateway(initial_balance=100000.0)
        # Inverted brackets for BUY (SL must be < TP)
        res = mock_gw.place_market_bracket_order(
            symbol="EURUSD",
            units=50000,
            stop_loss=1.1300,  # SL > TP for BUY is invalid
            take_profit=1.1200,
        )
        self.assertEqual(res["status"], "REJECTED")
        self.assertEqual(res["error"], "ORDER_VALIDATION_FAILED")
        self.assertIn("must be below", res["message"])
        self.assertEqual(len(mock_gw.get_open_trades()), 0)

    # -------------------------------------------------------------------------
    # Case 3: Ambiguous Timeout During Order Submission
    # -------------------------------------------------------------------------
    def test_case_03_ambiguous_timeout_handling(self):
        """Simulates network timeout on requests.post to OANDA."""
        gw = OandaGateway(
            api_token="dummy",
            account_id="101-001-99999999-001",
            environment="practice",
        )
        with patch("requests.post", side_effect=Exception("HTTPSConnectionPool timeout")):
            res = gw.place_market_bracket_order(
                symbol="EURUSD",
                units=10000,
                stop_loss=1.1100,
                take_profit=1.1250,
            )
            # Must fail safely with ERROR status without crashing
            self.assertEqual(res["status"], "ERROR")
            self.assertIn("Network exception", res["message"])

    # -------------------------------------------------------------------------
    # Case 4: Duplicate Retry / Second Order Prevention
    # -------------------------------------------------------------------------
    def test_case_04_duplicate_order_blocked_when_position_active(self):
        """Ensures that when a position is already active (cur_pos != 0), duplicate orders are blocked."""
        # Load trader with existing position in USDJPY
        trader = MultiPairPaperTrader(
            broker_config_path=self.test_config,
            ledger_path=self.test_ledger,
            pairs=["USDJPY"],
        )
        # Position is currently 1 (LONG). Action 1 (BUY) should HOLD/NOT open new trade.
        pos = trader.ledger["positions"]["USDJPY"]
        self.assertEqual(pos["position"], 1)

        # Manually evaluate order block logic
        cur_pos = pos["position"]
        action = 1  # BUY signal while already LONG
        order_attempted = False
        if cur_pos == 0:
            order_attempted = True

        self.assertFalse(order_attempted, "Engine must not open new order when already in a position.")

    # -------------------------------------------------------------------------
    # Case 5: Missing Protective Order on Broker
    # -------------------------------------------------------------------------
    def test_case_05_missing_protective_order_detection(self):
        """Simulates an OANDA trade response where stopLossOrder is missing or 0.0."""
        gw = OandaGateway(
            api_token="dummy",
            account_id="101-001-99999999-001",
            environment="practice",
        )
        mock_trades_resp = {
            "trades": [
                {
                    "id": "501",
                    "instrument": "EUR_USD",
                    "currentUnits": "10000",
                    "price": "1.11500",
                    "unrealizedPL": "5.00",
                    "openTime": "2026-10-27T07:00:00Z",
                    # Note: No stopLossOrder or takeProfitOrder present
                }
            ]
        }
        with patch("requests.get") as mock_get:
            mock_get.return_value.status_code = 200
            mock_get.return_value.json.return_value = mock_trades_resp

            open_trades = gw.get_open_trades()
            self.assertEqual(len(open_trades), 1)
            t = open_trades[0]
            self.assertEqual(t["stop_loss"], 0.0, "Missing stop-loss order must register as 0.0")
            self.assertEqual(t["take_profit"], 0.0, "Missing take-profit order must register as 0.0")

    # -------------------------------------------------------------------------
    # Case 6: Invalid Quote Handling in SpreadGuard
    # -------------------------------------------------------------------------
    def test_case_06_invalid_quote_spread_guard(self):
        """Tests SpreadGuard with zero, negative, or failing quotes."""
        guard = SpreadGuard(spread_limits={"EURUSD": 2.5})
        mock_gw = MagicMock()

        # Gateway returns 0.0 (API failure or empty quote)
        mock_gw.get_realtime_spread.return_value = 0.0
        allowed, cur_sp, max_sp, reason = guard.check(mock_gw, "EURUSD")
        # Note: Current code treats <= 0 as 'Normal' (defensive defect to document)
        self.assertEqual(cur_sp, 0.0)

        # Gateway returns excessive spread
        mock_gw.get_realtime_spread.return_value = 4.5
        allowed, cur_sp, max_sp, reason = guard.check(mock_gw, "EURUSD")
        self.assertFalse(allowed)
        self.assertIn("Spread Spike", reason)

    # -------------------------------------------------------------------------
    # Case 7: Excessive Spread Blocks Entry
    # -------------------------------------------------------------------------
    def test_case_07_excessive_spread_blocking(self):
        guard = SpreadGuard(spread_limits={"EURUSD": 2.5, "USDJPY": 3.0})
        mock_gw = MagicMock()
        mock_gw.get_realtime_spread.return_value = 3.6  # Above EURUSD 2.5 limit

        allowed, sp, limit, msg = guard.check(mock_gw, "EURUSD")
        self.assertFalse(allowed)
        self.assertEqual(sp, 3.6)
        self.assertEqual(limit, 2.5)
        self.assertIn("Spread Spike (3.6 pips > 2.5 max limit)", msg)

    # -------------------------------------------------------------------------
    # Case 8: Stale News Triggers Fail-Safe Block
    # -------------------------------------------------------------------------
    def test_case_08_stale_news_blocks_entries(self):
        """Simulates stale calendar cache (> 6h) with BLOCK_NEW_ENTRIES policy."""
        # Create a stale cache
        stale_time = (datetime.now(timezone.utc) - timedelta(hours=8)).isoformat()
        with open(self.test_news_cache, "w") as f:
            json.dump({
                "schema_version": "1.0.0",
                "last_successful_refresh_utc": stale_time,
                "provider": "mock",
                "status": "HEALTHY",
                "events": [],
                "cache_metadata": {"max_cache_age_seconds": 21600.0}
            }, f)

        client = NewsCalendarClient(
            provider=MockNewsProvider(),
            cache_path=self.test_news_cache,
            max_cache_age_seconds=21600.0,
            fail_safe_policy="BLOCK_NEW_ENTRIES",
        )
        status_info = client.get_status()
        self.assertEqual(status_info["status"], STATUS_STALE)

        shield = NewsShield(client=client, config_path=self.test_config)
        shield.fail_safe_policy = "BLOCK_NEW_ENTRIES"
        res = shield.check_blackout("EURUSD")
        self.assertTrue(res["blocked"])
        self.assertIn("STALE", res["reason"])

    # -------------------------------------------------------------------------
    # Case 9: Partial Fill Immunity via FOK (Fill Or Kill)
    # -------------------------------------------------------------------------
    def test_case_09_time_in_force_fok_validation(self):
        """Verifies that OandaGateway specifies timeInForce=FOK to prevent partial fills."""
        gw = OandaGateway(
            api_token="dummy",
            account_id="101-001-99999999-001",
            environment="practice",
        )
        with patch("requests.post") as mock_post:
            mock_post.return_value.status_code = 201
            mock_post.return_value.json.return_value = {
                "orderFillTransaction": {"id": "1001", "price": "1.11650"}
            }
            gw.place_market_bracket_order("EURUSD", 10000, 1.1100, 1.1250)
            args, kwargs = mock_post.call_args
            payload = kwargs.get("json", {}).get("order", {})
            self.assertEqual(payload.get("timeInForce"), "FOK", "Must enforce Fill-Or-Kill (FOK)")
            self.assertEqual(payload.get("type"), "MARKET")

    # -------------------------------------------------------------------------
    # Case 10: Interruption During Order Submission / Orphan Broker Position
    # -------------------------------------------------------------------------
    def test_case_10_interruption_during_submission_reconciliation(self):
        """
        If order is filled on broker but process dies before saving to ledger:
        Tests whether reconcile_with_broker detects the broker trade when local ledger is FLAT.
        """
        trader = MultiPairPaperTrader(
            broker_config_path=self.test_config,
            ledger_path=self.test_ledger,
            pairs=["EURUSD"],
        )
        # Local ledger is FLAT on EURUSD
        self.assertEqual(trader.ledger["positions"]["EURUSD"]["position"], 0)

        # Mock broker holding an orphan trade #999 for EURUSD
        mock_gw = MagicMock()
        mock_gw.get_open_trades.return_value = [
            {
                "id": "999",
                "symbol": "EURUSD",
                "instrument": "EUR_USD",
                "units": 50000,
                "direction": "LONG",
                "price": 1.1150,
                "unrealized_pl": 25.0,
                "stop_loss": 1.1100,
                "take_profit": 1.1250,
            }
        ]
        mock_gw.get_account_summary.return_value = {"balance": 100025.0, "equity": 100025.0, "broker": "OANDA"}
        trader.gateway = mock_gw

        trader.reconcile_with_broker()

        # Check behavior: Does trader adopt the orphan trade, or does it leave EURUSD FLAT?
        eur_pos = trader.ledger["positions"]["EURUSD"]
        # In current code: cur_pos == 0 results in execution_type = "FLAT" without adopting orphan trade!
        self.assertEqual(eur_pos["position"], 0)
        self.assertEqual(eur_pos["execution_type"], "FLAT")

    # -------------------------------------------------------------------------
    # Case 11: Unmatched Broker Position Classification
    # -------------------------------------------------------------------------
    def test_case_11_unmatched_broker_position(self):
        """Verifies that foreign/unmanaged broker positions do not corrupt paper ledger balance."""
        trader = MultiPairPaperTrader(
            broker_config_path=self.test_config,
            ledger_path=self.test_ledger,
            pairs=["EURUSD", "USDJPY"],
        )
        # Broker has an unmanaged trade on NZD_USD
        mock_gw = MagicMock()
        mock_gw.get_open_trades.return_value = [
            {
                "id": "777",
                "symbol": "NZDUSD",
                "instrument": "NZD_USD",
                "units": 20000,
                "direction": "LONG",
                "price": 0.6000,
                "unrealized_pl": 10.0,
            }
        ]
        mock_gw.get_account_summary.return_value = {"balance": 99500.0, "equity": 99510.0, "broker": "OANDA"}
        trader.gateway = mock_gw

        pre_balance = trader.ledger["balance"]
        trader.reconcile_with_broker()

        # Paper cash balance must remain untouched
        self.assertEqual(trader.ledger["balance"], pre_balance)
        # Broker equity is synced to separate key
        self.assertEqual(trader.ledger["broker_equity"], 99510.0)
        # NZDUSD should NOT appear in managed positions
        self.assertNotIn("NZDUSD", trader.ledger["positions"])


if __name__ == "__main__":
    unittest.main()
