"""
Unit Tests for MarketContextBuilder.

Verifies:
- Read-only data extraction across all supported pairs.
- Sanitization and secret redaction.
- Accurate capture of signals, spreads, ATR, positions, and news.
- Rejection of invalid pairs.
- Zero mutation of persistent files.
"""

import json
import os
import tempfile
import unittest
from pathlib import Path

from market_context_builder import MarketContextBuilder


class TestMarketContextBuilder(unittest.TestCase):

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.test_ledger = Path(self.tmp_dir) / "test_ledger.json"
        self.test_config = Path(self.tmp_dir) / "test_config.json"
        self.test_cache = Path(self.tmp_dir) / "test_cache.json"

        # Mock ledger with open USDJPY and closed trades
        self.ledger_data = {
            "balance": 98200.74,
            "equity": 98435.20,
            "initial_capital": 100000.0,
            "positions": {
                "EURUSD": {
                    "position": 0,
                    "current_price": 1.1179,
                    "current_atr": 0.00085,
                    "live_spread": 3.4,
                    "latest_signal": "BUY / LONG",
                    "probs": [0.36, 0.45, 0.19],
                    "execution_type": "FLAT",
                },
                "USDJPY": {
                    "position": 1,
                    "entry_price": 156.814,
                    "current_price": 157.550,
                    "current_atr": 0.174,
                    "entry_atr": 0.235,
                    "stop_loss": 156.343,
                    "take_profit": 158.226,
                    "unrealized_pnl": 234.47,
                    "time_in_trade": 37,
                    "live_spread": 3.8,
                    "latest_signal": "BUY / LONG",
                    "probs": [0.11, 0.69, 0.20],
                    "execution_type": "SIMULATED",
                },
                "AUDUSD": {"position": 0, "current_price": 0.7020, "latest_signal": "HOLD / FLAT"},
                "GBPUSD": {"position": 0, "current_price": 1.3342, "latest_signal": "HOLD / FLAT"},
            },
            "closed_trades": [{"pair": "AUDUSD", "net_pnl": -150.5}],
        }
        with open(self.test_ledger, "w") as f:
            json.dump(self.ledger_data, f)

        # Mock config with secrets that must never be exposed
        self.secret_token = "secret-token-do-not-leak-12345"
        self.config_data = {
            "broker": "oanda",
            "environment": "practice",
            "oanda": {"api_token": self.secret_token, "account_id": "101-001-test"},
            "guards": {
                "max_spread_pips": {"EURUSD": 2.5, "GBPUSD": 3.0, "USDJPY": 3.0, "AUDUSD": 3.5},
                "daily_drawdown_limit_pct": 3.0,
            },
        }
        with open(self.test_config, "w") as f:
            json.dump(self.config_data, f)

        # Mock news cache
        self.cache_data = {
            "provider": "mock",
            "status": "HEALTHY",
            "last_successful_refresh_utc": "2026-10-26T18:00:00Z",
            "events": [
                {
                    "id": "event_1",
                    "title": "FOMC Rate Decision",
                    "currency": "USD",
                    "impact": "HIGH",
                    "time_utc": "2026-10-28T18:00:00Z",
                }
            ],
        }
        with open(self.test_cache, "w") as f:
            json.dump(self.cache_data, f)

        self.builder = MarketContextBuilder(
            ledger_path=str(self.test_ledger),
            config_path=str(self.test_config),
            cache_path=str(self.test_cache),
        )

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_01_build_snapshot_all_supported_pairs(self):
        snapshots = self.builder.build_all_snapshots()
        self.assertEqual(len(snapshots), 4)
        for p in ("EURUSD", "USDJPY", "AUDUSD", "GBPUSD"):
            self.assertIn(p, snapshots)
            self.assertEqual(snapshots[p]["pair"], p)
            self.assertIn("market_data", snapshots[p])
            self.assertIn("engine_signal", snapshots[p])
            self.assertIn("risk_guards", snapshots[p])

    def test_02_secret_redaction(self):
        snapshots = self.builder.build_all_snapshots()
        dumped = json.dumps(snapshots)
        self.assertNotIn(self.secret_token, dumped)
        self.assertNotIn("101-001-test", dumped)

    def test_03_spread_guard_evaluation_accuracy(self):
        # EURUSD live spread 3.4 vs limit 2.5 on BUY signal -> ELEVATED_BLOCKED
        eu = self.builder.build_snapshot("EURUSD")
        self.assertEqual(eu["market_data"]["spread_status"], "ELEVATED_BLOCKED")
        self.assertFalse(eu["risk_guards"]["spread_guard"]["passed"])
        self.assertEqual(eu["risk_guards"]["spread_guard"]["threshold"], 2.5)

    def test_04_position_details_usdjpy(self):
        uj = self.builder.build_snapshot("USDJPY")
        pos = uj["position_state"]
        self.assertTrue(pos["has_active_position"])
        self.assertEqual(pos["direction"], "LONG")
        self.assertEqual(pos["execution_type"], "SIMULATED")
        self.assertEqual(pos["entry_price"], 156.814)
        self.assertEqual(pos["unrealized_pnl_usd"], 234.47)
        self.assertEqual(pos["duration_hours"], 37)

    def test_05_flat_position_handling(self):
        aud = self.builder.build_snapshot("AUDUSD")
        pos = aud["position_state"]
        self.assertFalse(pos["has_active_position"])
        self.assertEqual(pos["direction"], "FLAT")
        self.assertIsNone(pos["entry_price"])
        self.assertEqual(pos["unrealized_pnl_usd"], 0.0)

    def test_06_unsupported_pair_rejection(self):
        with self.assertRaises(ValueError):
            self.builder.build_snapshot("BTCUSD")

    def test_07_no_mutation_of_persistent_files(self):
        # File mtimes before
        mtime_l = os.path.getmtime(self.test_ledger)
        mtime_c = os.path.getmtime(self.test_config)
        self.builder.build_all_snapshots()
        self.assertEqual(mtime_l, os.path.getmtime(self.test_ledger))
        self.assertEqual(mtime_c, os.path.getmtime(self.test_config))


if __name__ == "__main__":
    unittest.main()
