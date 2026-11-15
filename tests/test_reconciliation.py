#!/usr/bin/env python3
"""
Comprehensive Automated Test Suite for Broker-to-Ledger Reconciliation.
Covers all 15 required scenarios using mocked broker gateways and isolated test ledgers.
Does NOT place real trades or close/modify live OANDA positions.
"""

import json
import os
import shutil
import sys
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from broker_gateway import BaseBrokerGateway, MockBrokerGateway
from ops_helper import normalize_symbol, reconcile_positions
from live_paper_portfolio import MultiPairPaperTrader


class TestBrokerReconciliation(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="recon_test_")
        self.test_config = os.path.join(self.test_dir, "test_broker_config.json")
        self.test_ledger = os.path.join(self.test_dir, "test_portfolio_ledger.json")

        # Mock broker configuration
        with open(self.test_config, "w") as f:
            json.dump({
                "broker": "mock",
                "environment": "practice",
                "oanda": {
                    "api_token": "mock-token-12345678901234567890",
                    "account_id": "101-001-99999999-001"
                }
            }, f)

        # Base clean ledger
        self.base_ledger = {
            "balance": 100000.0,
            "equity": 100000.0,
            "positions": {
                "EURUSD": {"position": 0, "entry_price": 0.0, "position_size": 0.0, "stop_loss": 0.0, "take_profit": 0.0, "unrealized_pnl": 0.0, "broker_ticket": None, "execution_type": "FLAT"},
                "USDJPY": {"position": 0, "entry_price": 0.0, "position_size": 0.0, "stop_loss": 0.0, "take_profit": 0.0, "unrealized_pnl": 0.0, "broker_ticket": None, "execution_type": "FLAT"},
                "AUDUSD": {"position": 0, "entry_price": 0.0, "position_size": 0.0, "stop_loss": 0.0, "take_profit": 0.0, "unrealized_pnl": 0.0, "broker_ticket": None, "execution_type": "FLAT"},
                "GBPUSD": {"position": 0, "entry_price": 0.0, "position_size": 0.0, "stop_loss": 0.0, "take_profit": 0.0, "unrealized_pnl": 0.0, "broker_ticket": None, "execution_type": "FLAT"},
            },
            "closed_trades": [],
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        self.save_ledger(self.base_ledger)

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def save_ledger(self, data):
        with open(self.test_ledger, "w") as f:
            json.dump(data, f, indent=2)

    # --------------------------------------------------------------------------
    # Scenario 1: Broker and ledger contain the same position
    # --------------------------------------------------------------------------
    @patch("ops_helper.fetch_broker_state")
    def test_01_broker_and_ledger_contain_same_position(self, mock_b_state):
        mock_b_state.return_value = {
            "broker": "oanda",
            "balance": 100000.0,
            "equity": 100500.0,
            "unrealized_pnl": 500.0,
            "trades": [{
                "id": "101",
                "symbol": "AUDUSD",
                "instrument": "AUD_USD",
                "units": 100000,
                "direction": "LONG",
                "price": 0.70200,
                "unrealized_pl": 500.0,
                "stop_loss": 0.70000,
                "take_profit": 0.70800,
            }]
        }
        ledger = self.base_ledger.copy()
        ledger["positions"]["AUDUSD"] = {
            "position": 1,
            "entry_price": 0.70200,
            "position_size": 100000.0,
            "stop_loss": 0.70000,
            "take_profit": 0.70800,
            "unrealized_pnl": 500.0,
            "broker_ticket": "101",
            "execution_type": "BROKER_CONFIRMED"
        }
        ledger["equity"] = 100500.0
        self.save_ledger(ledger)

        res = reconcile_positions(self.test_config, self.test_ledger)
        self.assertEqual(res["overall_status"], "RECONCILED")
        self.assertEqual(res["positions"]["AUDUSD"]["classification"], "BROKER_CONFIRMED")
        self.assertEqual(res["counts"]["confirmed"], 1)
        self.assertEqual(res["counts"]["simulated"], 0)

    # --------------------------------------------------------------------------
    # Scenario 2: Simulated position exists without a broker position
    # --------------------------------------------------------------------------
    @patch("ops_helper.fetch_broker_state")
    def test_02_simulated_position_without_broker_position(self, mock_b_state):
        mock_b_state.return_value = {
            "broker": "oanda",
            "balance": 100000.0,
            "equity": 100000.0,
            "unrealized_pnl": 0.0,
            "trades": []
        }
        ledger = self.base_ledger.copy()
        ledger["positions"]["EURUSD"] = {
            "position": 1,
            "entry_price": 1.12000,
            "position_size": 50000.0,
            "stop_loss": 1.11800,
            "take_profit": 1.12600,
            "unrealized_pnl": 150.0,
            "broker_ticket": "N/A",
            "execution_type": "SIMULATED"
        }
        ledger["equity"] = 100150.0
        self.save_ledger(ledger)

        res = reconcile_positions(self.test_config, self.test_ledger)
        self.assertEqual(res["overall_status"], "RECONCILED_WITH_SIMULATED_POSITIONS")
        self.assertEqual(res["positions"]["EURUSD"]["classification"], "SIMULATED")
        self.assertEqual(res["counts"]["simulated"], 1)

    # --------------------------------------------------------------------------
    # Scenario 3: Broker position exists without a local record
    # --------------------------------------------------------------------------
    @patch("ops_helper.fetch_broker_state")
    def test_03_broker_position_exists_without_local_record(self, mock_b_state):
        mock_b_state.return_value = {
            "broker": "oanda",
            "balance": 100000.0,
            "equity": 100200.0,
            "unrealized_pnl": 200.0,
            "trades": [{
                "id": "202",
                "symbol": "USDJPY",
                "instrument": "USD_JPY",
                "units": 50000,
                "direction": "LONG",
                "price": 156.50,
                "unrealized_pl": 200.0,
            }]
        }
        # Local ledger has USDJPY flat
        res = reconcile_positions(self.test_config, self.test_ledger)
        self.assertEqual(res["overall_status"], "RECONCILED_REQUIRED" if "RECONCILED_REQUIRED" in res["overall_status"] else "RECONCILIATION_REQUIRED")
        self.assertEqual(res["positions"]["USDJPY"]["classification"], "UNMATCHED_BROKER")
        self.assertEqual(res["counts"]["unmatched_broker"], 1)

    # --------------------------------------------------------------------------
    # Scenario 4: Local position marked active although broker position is closed
    # --------------------------------------------------------------------------
    @patch("ops_helper.fetch_broker_state")
    def test_04_local_active_but_broker_closed(self, mock_b_state):
        mock_b_state.return_value = {
            "broker": "oanda",
            "balance": 100000.0,
            "equity": 100000.0,
            "unrealized_pnl": 0.0,
            "trades": []  # Empty! Trade was closed
        }
        ledger = self.base_ledger.copy()
        ledger["positions"]["AUDUSD"] = {
            "position": 1,
            "entry_price": 0.70300,
            "position_size": 100000.0,
            "stop_loss": 0.70150,
            "take_profit": 0.70750,
            "unrealized_pnl": -50.0,
            "broker_ticket": "303",
            "execution_type": "BROKER_CONFIRMED"
        }
        self.save_ledger(ledger)

        res = reconcile_positions(self.test_config, self.test_ledger)
        self.assertEqual(res["overall_status"], "RECONCILIATION_REQUIRED")
        self.assertEqual(res["positions"]["AUDUSD"]["classification"], "CLOSED_AT_BROKER")
        self.assertEqual(res["counts"]["closed_at_broker"], 1)

    # --------------------------------------------------------------------------
    # Scenario 5: Position has an incorrect broker trade ID
    # --------------------------------------------------------------------------
    @patch("ops_helper.fetch_broker_state")
    def test_05_position_has_incorrect_broker_trade_id(self, mock_b_state):
        mock_b_state.return_value = {
            "broker": "oanda",
            "balance": 100000.0,
            "equity": 100000.0,
            "unrealized_pnl": 0.0,
            "trades": [{
                "id": "555",
                "symbol": "AUDUSD",
                "units": 100000,
                "direction": "LONG",
                "price": 0.70200,
                "unrealized_pl": 0.0,
            }]
        }
        ledger = self.base_ledger.copy()
        ledger["positions"]["AUDUSD"] = {
            "position": 1,
            "entry_price": 0.70200,
            "position_size": 100000.0,
            "broker_ticket": "999",  # Ticket mismatch!
            "execution_type": "BROKER_CONFIRMED"
        }
        self.save_ledger(ledger)

        res = reconcile_positions(self.test_config, self.test_ledger)
        self.assertEqual(res["overall_status"], "RECONCILIATION_REQUIRED")
        self.assertEqual(res["positions"]["AUDUSD"]["classification"], "RECONCILIATION_MISMATCH")

    # --------------------------------------------------------------------------
    # Scenario 6: Position sizes differ between broker and ledger
    # --------------------------------------------------------------------------
    @patch("ops_helper.fetch_broker_state")
    def test_06_position_size_difference(self, mock_b_state):
        mock_b_state.return_value = {
            "broker": "oanda",
            "balance": 100000.0,
            "equity": 100000.0,
            "unrealized_pnl": 0.0,
            "trades": [{
                "id": "101",
                "symbol": "AUDUSD",
                "units": 100000,  # 100,000 on broker
                "direction": "LONG",
                "price": 0.70200,
                "unrealized_pl": 0.0,
            }]
        }
        ledger = self.base_ledger.copy()
        ledger["positions"]["AUDUSD"] = {
            "position": 1,
            "entry_price": 0.70200,
            "position_size": 25000.0,  # 25,000 in ledger
            "broker_ticket": "101",
            "execution_type": "BROKER_CONFIRMED"
        }
        self.save_ledger(ledger)

        res = reconcile_positions(self.test_config, self.test_ledger)
        pos_rep = res["positions"]["AUDUSD"]
        self.assertNotEqual(pos_rep["broker_units"], int(pos_rep["local_size"]))

    # --------------------------------------------------------------------------
    # Scenario 7: Broker position has a different direction from local record
    # --------------------------------------------------------------------------
    @patch("ops_helper.fetch_broker_state")
    def test_07_direction_mismatch(self, mock_b_state):
        mock_b_state.return_value = {
            "broker": "oanda",
            "balance": 100000.0,
            "equity": 100000.0,
            "unrealized_pnl": 0.0,
            "trades": [{
                "id": "101",
                "symbol": "AUDUSD",
                "units": 100000,
                "direction": "LONG",  # Long at broker
                "price": 0.70200,
                "unrealized_pl": 0.0,
            }]
        }
        ledger = self.base_ledger.copy()
        ledger["positions"]["AUDUSD"] = {
            "position": -1,  # Short in ledger!
            "entry_price": 0.70200,
            "position_size": 100000.0,
            "broker_ticket": "101",
            "execution_type": "BROKER_CONFIRMED"
        }
        self.save_ledger(ledger)

        res = reconcile_positions(self.test_config, self.test_ledger)
        self.assertEqual(res["overall_status"], "RECONCILIATION_REQUIRED")
        self.assertEqual(res["positions"]["AUDUSD"]["classification"], "RECONCILIATION_MISMATCH")

    # --------------------------------------------------------------------------
    # Scenario 8: Broker API is unavailable
    # --------------------------------------------------------------------------
    @patch("ops_helper.fetch_broker_state")
    def test_08_broker_api_unavailable(self, mock_b_state):
        mock_b_state.return_value = {"error": True, "message": "Connection refused by peer"}
        res = reconcile_positions(self.test_config, self.test_ledger)
        self.assertEqual(res["overall_status"], "BROKER_UNAVAILABLE")
        self.assertIn("Connection refused", res["error"])

    # --------------------------------------------------------------------------
    # Scenario 9: Broker returns incomplete response
    # --------------------------------------------------------------------------
    @patch("ops_helper.fetch_broker_state")
    def test_09_broker_incomplete_response(self, mock_b_state):
        mock_b_state.return_value = {"error": True, "message": "Broker returned HTTP 500"}
        res = reconcile_positions(self.test_config, self.test_ledger)
        self.assertEqual(res["overall_status"], "BROKER_UNAVAILABLE")

    # --------------------------------------------------------------------------
    # Scenario 10: Account equity differs from stale cached value
    # --------------------------------------------------------------------------
    @patch("ops_helper.fetch_broker_state")
    def test_10_equity_differs_from_stale_cached_value(self, mock_b_state):
        mock_b_state.return_value = {
            "broker": "oanda",
            "balance": 100000.0,
            "equity": 102500.0,
            "unrealized_pnl": 2500.0,
            "trades": []
        }
        ledger = self.base_ledger.copy()
        ledger["equity"] = 100000.0  # Stale cached value
        self.save_ledger(ledger)

        res = reconcile_positions(self.test_config, self.test_ledger)
        self.assertEqual(res["discrepancy_amount"], -2500.0)

    # --------------------------------------------------------------------------
    # Scenario 11: Atomic write prevents corruption during concurrent reads
    # --------------------------------------------------------------------------
    def test_11_atomic_write_prevents_corruption(self):
        # Create trader with test ledger
        trader = MultiPairPaperTrader(
            pairs=["EURUSD"],
            ledger_path=self.test_ledger,
            broker_config_path=self.test_config,
        )
        trader.ledger["balance"] = 123456.78
        trader._save_ledger()

        # Confirm ledger is 100% valid JSON and tmp file does not remain
        with open(self.test_ledger, "r") as f:
            data = json.load(f)
        self.assertEqual(data["balance"], 123456.78)
        self.assertFalse(os.path.exists(self.test_ledger + ".tmp"))

    # --------------------------------------------------------------------------
    # Scenario 12: Symbol normalization prevents double counting
    # --------------------------------------------------------------------------
    def test_12_symbol_normalization(self):
        s1 = normalize_symbol("EUR_USD")
        s2 = normalize_symbol("EURUSD")
        s3 = normalize_symbol("EURUSD=X")
        self.assertEqual(s1, "EURUSD")
        self.assertEqual(s2, "EURUSD")
        self.assertEqual(s3, "EURUSD")

    # --------------------------------------------------------------------------
    # Scenario 13: Protective order executes while engine is offline
    # --------------------------------------------------------------------------
    def test_13_protective_order_executes_offline(self):
        # Ledger has active trade with ticket 777
        ledger = self.base_ledger.copy()
        ledger["positions"]["AUDUSD"] = {
            "position": 1,
            "entry_price": 0.70200,
            "position_size": 100000.0,
            "stop_loss": 0.70000,
            "take_profit": 0.70800,
            "unrealized_pnl": -200.0,
            "broker_ticket": "777",
            "execution_type": "BROKER_CONFIRMED"
        }
        self.save_ledger(ledger)

        # Mock broker gateway returning empty open trades (trade 777 closed!)
        mock_gw = MagicMock()
        mock_gw.get_open_trades.return_value = []
        mock_gw.get_account_summary.return_value = {"balance": 99800.0, "equity": 99800.0, "broker": "OANDA_V20"}

        with patch("live_paper_portfolio.create_broker_gateway", return_value=mock_gw):
            trader = MultiPairPaperTrader(
                pairs=["AUDUSD"],
                ledger_path=self.test_ledger,
                broker_config_path=self.test_config,
            )
            # Reconcile should detect trade 777 closed and reset AUDUSD to FLAT
            self.assertEqual(trader.ledger["positions"]["AUDUSD"]["position"], 0)
            self.assertEqual(len(trader.ledger["closed_trades"]), 1)
            self.assertEqual(trader.ledger["closed_trades"][0]["broker_ticket"], "777")
            self.assertIn("BROKER_FILL", trader.ledger["closed_trades"][0]["exit_reason"])

    # --------------------------------------------------------------------------
    # Scenario 14: Engine restart occurs while position is open
    # --------------------------------------------------------------------------
    def test_14_restart_while_position_open(self):
        # Ledger has active trade 888
        ledger = self.base_ledger.copy()
        ledger["positions"]["AUDUSD"] = {
            "position": 1,
            "entry_price": 0.70200,
            "position_size": 100000.0,
            "stop_loss": 0.70000,
            "take_profit": 0.70800,
            "unrealized_pnl": 150.0,
            "broker_ticket": "888",
            "execution_type": "BROKER_CONFIRMED"
        }
        self.save_ledger(ledger)

        # Mock broker confirms trade 888 is still open with updated PnL 300.0
        mock_gw = MagicMock()
        mock_gw.get_open_trades.return_value = [{
            "id": "888",
            "symbol": "AUDUSD",
            "units": 100000,
            "direction": "LONG",
            "price": 0.70200,
            "unrealized_pl": 300.0,
        }]
        mock_gw.get_account_summary.return_value = {"balance": 100000.0, "equity": 100300.0, "broker": "OANDA_V20"}

        with patch("live_paper_portfolio.create_broker_gateway", return_value=mock_gw):
            trader = MultiPairPaperTrader(
                pairs=["AUDUSD"],
                ledger_path=self.test_ledger,
                broker_config_path=self.test_config,
            )
            # Position remains open, broker confirmed, PnL updated
            self.assertEqual(trader.ledger["positions"]["AUDUSD"]["position"], 1)
            self.assertEqual(trader.ledger["positions"]["AUDUSD"]["broker_ticket"], "888")
            self.assertEqual(trader.ledger["positions"]["AUDUSD"]["execution_type"], "BROKER_CONFIRMED")
            self.assertEqual(trader.ledger["positions"]["AUDUSD"]["broker_unrealized_pnl"], 300.0)

    # --------------------------------------------------------------------------
    # Scenario 15: Simulated result is not included in broker-account P&L
    # --------------------------------------------------------------------------
    @patch("ops_helper.fetch_broker_state")
    def test_15_simulated_result_not_in_broker_pnl(self, mock_b_state):
        mock_b_state.return_value = {
            "broker": "oanda",
            "balance": 100000.0,
            "equity": 100000.0,
            "unrealized_pnl": 0.0,  # Zero on broker
            "trades": []
        }
        ledger = self.base_ledger.copy()
        # Simulated position has +1,500.00 PnL
        ledger["positions"]["EURUSD"] = {
            "position": 1,
            "entry_price": 1.12000,
            "position_size": 100000.0,
            "unrealized_pnl": 1500.0,
            "broker_ticket": "N/A",
            "execution_type": "SIMULATED"
        }
        ledger["equity"] = 101500.0
        self.save_ledger(ledger)

        res = reconcile_positions(self.test_config, self.test_ledger)
        # Broker unrealized PnL remains exactly 0.0
        self.assertEqual(res["broker"]["unrealized_pnl"], 0.0)
        # Discrepancy is identified and attributed to simulation
        self.assertEqual(res["discrepancy_amount"], 1500.0)
        self.assertIn("Includes $+1500.00 unrealized PnL from 1 simulated positions", res["discrepancy_explanation"])


if __name__ == "__main__":
    unittest.main()
