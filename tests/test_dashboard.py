#!/usr/bin/env python3
"""
Comprehensive Test Suite for Beginner-Friendly Forex AI Dashboard.
Tests all 12 scenarios specified in Section 10 of the redesign task:
 1. Three actual open positions and one flat currency pair.
 2. No open positions (all flat).
 3. Negative unrealized profit or loss.
 4. Positive unrealized profit or loss.
 5. Missing or zero entry prices.
 6. Missing stop-loss or take-profit data.
 7. Broker API unavailable.
 8. Stale account data.
 9. Missing historical equity data.
 10. Discrepancy between broker positions and local ledger.
 11. HTML/JS structure supporting responsive desktop/mobile layouts.
 12. Correct display of PRACTICE environment without changing config.
"""

import json
import os
import sys
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from dashboard_server import enrich_portfolio_data, HTML_DASHBOARD


class TestDashboardScenarios(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="dash_scenarios_")
        self.config_path = os.path.join(self.test_dir, "test_config.json")
        with open(self.config_path, "w") as f:
            json.dump({
                "broker": "oanda",
                "environment": "practice",
                "oanda": {
                    "api_token": "mock-token-12345",
                    "account_id": "101-001-40816673-001"
                }
            }, f)

        self.sample_ledger = {
            "balance": 100000.0,
            "equity": 100000.0,
            "initial_capital": 100000.0,
            "positions": {
                "AUDUSD": {"position": 1, "entry_price": 0.70300, "current_price": 0.70400, "stop_loss": 0.70100, "take_profit": 0.70700, "unrealized_pnl": 100.0, "time_in_trade": 5, "broker_ticket": "16"},
                "EURUSD": {"position": 1, "entry_price": 1.12000, "current_price": 1.12200, "stop_loss": 1.11800, "take_profit": 1.12600, "unrealized_pnl": 200.0, "time_in_trade": 10, "broker_ticket": "N/A"},
                "USDJPY": {"position": 1, "entry_price": 156.000, "current_price": 156.500, "stop_loss": 155.000, "take_profit": 158.000, "unrealized_pnl": 50.0, "time_in_trade": 8, "broker_ticket": "N/A"},
                "GBPUSD": {"position": 0, "entry_price": 0.00000, "current_price": 1.33500, "stop_loss": 0.00000, "take_profit": 0.00000, "unrealized_pnl": 0.0, "time_in_trade": 0, "broker_ticket": None},
            },
            "closed_trades": [],
            "equity_history": [
                {"timestamp": "2026-10-26T10:00:00Z", "equity": 100000.0, "balance": 100000.0, "unrealized_pnl": 0.0},
                {"timestamp": "2026-10-26T11:00:00Z", "equity": 100350.0, "balance": 100000.0, "unrealized_pnl": 350.0},
            ]
        }

    def tearDown(self):
        import shutil
        shutil.rmtree(self.test_dir, ignore_errors=True)

    # 1. Three actual open positions and one flat currency pair
    def test_01_three_open_and_one_flat(self):
        data = enrich_portfolio_data(self.sample_ledger, broker_state=None, engine_running=True, config_path=self.config_path)
        self.assertEqual(data["account_summary"]["active_trades_count"], 3)
        self.assertTrue(data["positions"]["AUDUSD"]["has_active_trade"])
        self.assertTrue(data["positions"]["EURUSD"]["has_active_trade"])
        self.assertTrue(data["positions"]["USDJPY"]["has_active_trade"])
        self.assertFalse(data["positions"]["GBPUSD"]["has_active_trade"])
        self.assertIsNone(data["positions"]["GBPUSD"]["pips"])
        self.assertEqual(data["positions"]["GBPUSD"]["direction_label"], "No active trade")

    # 2. No open positions (all flat)
    def test_02_all_flat(self):
        flat_ledger = self.sample_ledger.copy()
        flat_ledger["positions"] = {p: {"position": 0, "entry_price": 0.0, "current_price": 1.0, "unrealized_pnl": 0.0} for p in ["AUDUSD", "EURUSD", "USDJPY", "GBPUSD"]}
        data = enrich_portfolio_data(flat_ledger, broker_state=None, engine_running=True, config_path=self.config_path)
        self.assertEqual(data["account_summary"]["active_trades_count"], 0)
        for p in data["positions"].values():
            self.assertFalse(p["has_active_trade"])
            self.assertEqual(p["direction_label"], "No active trade")
            self.assertIsNone(p["pips"])

    # 3. Negative unrealized profit or loss
    def test_03_negative_unrealized_pnl(self):
        loss_ledger = self.sample_ledger.copy()
        loss_ledger["positions"]["AUDUSD"]["unrealized_pnl"] = -450.0
        loss_ledger["positions"]["EURUSD"]["unrealized_pnl"] = -150.0
        loss_ledger["positions"]["USDJPY"]["unrealized_pnl"] = 0.0
        loss_ledger["equity"] = 99400.0
        data = enrich_portfolio_data(loss_ledger, broker_state=None, engine_running=True, config_path=self.config_path)
        self.assertEqual(data["account_summary"]["open_trades_result"], -600.0)
        self.assertLess(data["positions"]["AUDUSD"]["unrealized_pnl"], 0)

    # 4. Positive unrealized profit or loss
    def test_04_positive_unrealized_pnl(self):
        gain_ledger = self.sample_ledger.copy()
        gain_ledger["equity"] = 101250.0
        data = enrich_portfolio_data(gain_ledger, broker_state=None, engine_running=True, config_path=self.config_path)
        self.assertEqual(data["account_summary"]["open_trades_result"], 1250.0)
        self.assertGreater(data["account_summary"]["open_trades_result"], 0)

    # 5. Missing or zero entry prices
    def test_05_missing_or_zero_entry_prices(self):
        zero_ledger = self.sample_ledger.copy()
        zero_ledger["positions"]["AUDUSD"]["entry_price"] = 0.0
        data = enrich_portfolio_data(zero_ledger, broker_state=None, engine_running=True, config_path=self.config_path)
        # Even with zero entry price, does not crash or calculate infinite pips
        self.assertEqual(data["positions"]["AUDUSD"]["pips"], 0.0)

    # 6. Missing stop-loss or take-profit data
    def test_06_missing_sl_tp(self):
        no_bracket = self.sample_ledger.copy()
        no_bracket["positions"]["AUDUSD"]["stop_loss"] = 0.0
        no_bracket["positions"]["AUDUSD"]["take_profit"] = 0.0
        data = enrich_portfolio_data(no_bracket, broker_state=None, engine_running=True, config_path=self.config_path)
        self.assertEqual(data["positions"]["AUDUSD"]["stop_loss_status"], "Not set")
        self.assertEqual(data["positions"]["AUDUSD"]["take_profit_status"], "Not set")

    # 7. Broker API unavailable
    def test_07_broker_api_unavailable(self):
        broker_err = {"error": True, "message": "Connection refused"}
        data = enrich_portfolio_data(self.sample_ledger, broker_state=broker_err, engine_running=True, config_path=self.config_path)
        self.assertFalse(data["header"]["broker_connected"])
        self.assertIn("Connecting", data["header"]["broker_status"])

    # 8. Stale account data handled gracefully
    def test_08_stale_account_data(self):
        data = enrich_portfolio_data(self.sample_ledger, broker_state=None, engine_running=False, config_path=self.config_path)
        self.assertFalse(data["header"]["engine_running"])
        self.assertEqual(data["system_status"]["engine_status"], "Stopped")

    # 9. Missing historical equity data
    def test_09_missing_historical_equity(self):
        no_hist = self.sample_ledger.copy()
        no_hist["equity_history"] = []
        data = enrich_portfolio_data(no_hist, broker_state=None, engine_running=True, config_path=self.config_path)
        self.assertEqual(len(data["equity_history"]), 0)

    # 10. Discrepancy between broker positions and local ledger
    def test_10_broker_ledger_discrepancy(self):
        broker_state = {
            "broker": "oanda",
            "balance": 99999.98,
            "equity": 100050.0,
            "unrealized_pnl": 50.0,
            "trades": [{
                "id": "16",
                "symbol": "AUDUSD",
                "units": 1312729,
                "price": 0.70291,
                "unrealized_pl": 50.0,
                "stop_loss": 0.70195,
                "take_profit": 0.70651,
            }]
        }
        data = enrich_portfolio_data(self.sample_ledger, broker_state=broker_state, engine_running=True, config_path=self.config_path)
        # Broker values are used for verified total account value
        self.assertEqual(data["account_summary"]["total_account_value"], 100050.0)
        self.assertEqual(data["positions"]["AUDUSD"]["execution_category"], "BROKER_CONFIRMED")
        self.assertEqual(data["positions"]["EURUSD"]["execution_category"], "SIMULATED")

    # 11. Responsive layout and HTML validation
    def test_11_responsive_html_elements(self):
        self.assertIn("<!DOCTYPE html>", HTML_DASHBOARD)
        self.assertIn("Forex AI", HTML_DASHBOARD)
        self.assertIn("tab-overview", HTML_DASHBOARD)
        self.assertIn("tab-trades", HTML_DASHBOARD)
        self.assertIn("tab-history", HTML_DASHBOARD)
        self.assertIn("tab-ai", HTML_DASHBOARD)
        self.assertIn("tab-system", HTML_DASHBOARD)
        self.assertIn("equityChart", HTML_DASHBOARD)

    # 12. Correct display of PRACTICE environment
    def test_12_practice_mode_display(self):
        data = enrich_portfolio_data(self.sample_ledger, broker_state=None, engine_running=True, config_path=self.config_path)
        self.assertEqual(data["header"]["environment"], "PRACTICE")

    # 13. Separate broker account equity from local strategy paper equity
    def test_13_separate_broker_and_strategy_paper_equity(self):
        broker_state = {
            "broker": "oanda",
            "balance": 99999.98,
            "equity": 99737.43,
            "unrealized_pnl": -262.55,
            "margin_used": 27679.15,
            "margin_available": 72058.28,
            "trades": [{
                "id": "16",
                "symbol": "AUDUSD",
                "units": 1312729,
                "direction": "LONG",
                "price": 0.70291,
                "unrealized_pl": -262.55,
                "stop_loss": 0.70195,
                "take_profit": 0.70651,
            }]
        }
        data = enrich_portfolio_data(self.sample_ledger, broker_state=broker_state, engine_running=True, config_path=self.config_path)
        summary = data["account_summary"]
        self.assertIn("broker_account", summary)
        self.assertIn("strategy_paper", summary)
        self.assertEqual(summary["broker_account"]["equity"], 99737.43)
        self.assertEqual(summary["broker_account"]["balance"], 99999.98)
        self.assertEqual(summary["strategy_paper"]["equity"], 100000.0)
        self.assertEqual(summary["strategy_paper"]["balance"], 100000.0)
        # Never blend simulated paper trades into broker NAV
        self.assertNotEqual(summary["broker_account"]["equity"], summary["strategy_paper"]["equity"])

    # 14. Clean flat positions without fake prices or calculated pips
    def test_14_clean_flat_positions_no_fake_pips_or_prices(self):
        data = enrich_portfolio_data(self.sample_ledger, broker_state=None, engine_running=True, config_path=self.config_path)
        gbp = data["positions"]["GBPUSD"]
        self.assertEqual(gbp["execution_category"], "FLAT")
        self.assertIsNone(gbp["entry_price"])
        self.assertIsNone(gbp["pips"])
        self.assertIsNone(gbp["stop_loss_price"])
        self.assertIsNone(gbp["take_profit_price"])
        self.assertEqual(gbp["stop_loss_status"], "—")
        self.assertEqual(gbp["take_profit_status"], "—")
        self.assertEqual(gbp["direction_label"], "No active trade")

    # 15. Operational warnings for offline, stopped, unmatched, or discrepancies
    def test_15_operational_warnings_generation(self):
        # Broker disconnected & engine stopped
        data = enrich_portfolio_data(self.sample_ledger, broker_state=None, engine_running=False, config_path=self.config_path)
        rec = data["reconciliation"]
        codes = [w["code"] for w in rec["warnings"]]
        self.assertIn("BROKER_OFFLINE", codes)
        self.assertIn("ENGINE_STOPPED", codes)

        # Unmatched broker position warning
        unmatched_broker = {
            "broker": "oanda",
            "balance": 100000.0,
            "equity": 100000.0,
            "unrealized_pnl": 0.0,
            "trades": [
                {"id": "99", "symbol": "GBPUSD", "units": 50000, "direction": "LONG", "price": 1.3300, "unrealized_pl": 0.0}
            ]
        }
        data2 = enrich_portfolio_data(self.sample_ledger, broker_state=unmatched_broker, engine_running=True, config_path=self.config_path)
        codes2 = [w["code"] for w in data2["reconciliation"]["warnings"]]
        self.assertIn("UNMATCHED_BROKER", codes2)

    # 16. Reconciliation tab and UI elements present
    def test_16_reconciliation_tab_elements(self):
        self.assertIn("tab-reconciliation", HTML_DASHBOARD)
        self.assertIn("Broker &amp; Reconciliation", HTML_DASHBOARD.replace("Broker & Reconciliation", "Broker &amp; Reconciliation"))
        self.assertIn("alerts-container", HTML_DASHBOARD)
        self.assertIn("rec-broker-cash", HTML_DASHBOARD)
        self.assertIn("rec-ledger-cash", HTML_DASHBOARD)

    # 17. Separate realized P&L and timestamps for broker vs strategy paper
    def test_17_realized_pnl_and_timestamps(self):
        broker_state = {
            "broker": "oanda",
            "balance": 98726.63,
            "equity": 98726.63,
            "unrealized_pnl": 0.0,
            "realized_pnl": -1273.37,
            "margin_used": 0.0,
            "margin_available": 98726.63,
            "trades": []
        }
        ledger_with_history = self.sample_ledger.copy()
        ledger_with_history["balance"] = 98200.74
        ledger_with_history["equity"] = 98410.67
        ledger_with_history["created_at"] = "2026-10-26T14:01:00Z"
        ledger_with_history["closed_trades"] = [
            {"pair": "AUDUSD", "net_pnl": -150.50, "exit_time": "2026-10-26T11:00:00Z"},
            {"pair": "GBPUSD", "net_pnl": -150.50, "exit_time": "2026-10-26T11:00:00Z"},
            {"pair": "EURUSD", "net_pnl": -150.50, "exit_time": "2026-10-26T11:00:00Z"},
            {"pair": "AUDUSD", "net_pnl": -1498.24, "exit_time": "2026-10-26T14:01:00Z"},
        ]
        data = enrich_portfolio_data(ledger_with_history, broker_state=broker_state, engine_running=True, config_path=self.config_path)
        summary = data["account_summary"]
        
        # Verify broker realized P&L is independent of paper trades
        self.assertEqual(summary["broker_account"]["realized_pnl"], -1273.37)
        self.assertEqual(summary["broker_account"]["balance"], 98726.63)
        self.assertEqual(summary["broker_account"]["unrealized_pnl"], 0.0)
        self.assertIsNotNone(summary["broker_account"]["timestamp"])
        self.assertFalse(summary["broker_account"]["is_stale"])

        # Verify strategy paper realized P&L reflects simulated closed trades
        self.assertEqual(summary["strategy_paper"]["simulated_realized_pnl"], -1949.74)
        self.assertEqual(summary["strategy_paper"]["balance"], 98200.74)
        self.assertEqual(summary["strategy_paper"]["timestamp"], "2026-10-26T14:01:00Z")
        self.assertFalse(summary["strategy_paper"]["is_stale"])

        # Verify HTML contains cards and badges for both accounts
        self.assertIn("Verified OANDA Practice Account", HTML_DASHBOARD)
        self.assertIn("Local Strategy Paper Sandbox", HTML_DASHBOARD)
        self.assertIn("metric-broker-realized", HTML_DASHBOARD)
        self.assertIn("metric-strategy-realized", HTML_DASHBOARD)
        self.assertIn("rec-broker-realized", HTML_DASHBOARD)
        self.assertIn("rec-ledger-realized", HTML_DASHBOARD)

    # 18. Reconciliation differences and formulas (Broker minus Paper convention)
    def test_18_reconciliation_differences_formulas(self):
        broker_state = {
            "broker": "oanda",
            "balance": 98726.63,
            "equity": 98726.63,
            "unrealized_pnl": 0.0,
            "realized_pnl": -1273.37,
            "trades": []
        }
        import copy
        ledger = copy.deepcopy(self.sample_ledger)
        ledger["balance"] = 98200.74
        ledger["equity"] = 98410.67
        ledger["positions"] = {
            "AUDUSD": {"position": 0, "entry_price": 0.0, "current_price": 0.70175, "unrealized_pnl": 0.0, "broker_ticket": None},
            "EURUSD": {"position": 0, "entry_price": 0.0, "current_price": 1.11794, "unrealized_pnl": 0.0, "broker_ticket": None},
            "USDJPY": {"position": 1, "entry_price": 156.814, "current_price": 157.473, "unrealized_pnl": 209.94, "broker_ticket": None},
            "GBPUSD": {"position": 0, "entry_price": 0.0, "current_price": 1.33317, "unrealized_pnl": 0.0, "broker_ticket": None},
        }
        ledger["closed_trades"] = [
            {"pair": "AUDUSD", "net_pnl": -150.50, "exit_time": "2026-10-26T11:00:00Z"},
            {"pair": "GBPUSD", "net_pnl": -150.50, "exit_time": "2026-10-26T11:00:00Z"},
            {"pair": "EURUSD", "net_pnl": -150.50, "exit_time": "2026-10-26T11:00:00Z"},
            {"pair": "AUDUSD", "net_pnl": -1498.24, "exit_time": "2026-10-26T14:01:00Z"},
        ]
        data = enrich_portfolio_data(ledger, broker_state=broker_state, engine_running=True, config_path=self.config_path)
        rec = data["reconciliation"]

        # 1. Cash Difference = broker cash minus paper cash (+525.89)
        self.assertEqual(rec["cash_difference"], 525.89)
        self.assertEqual(rec["cash_variance"], 525.89)

        # 2. Equity Difference = broker equity minus paper equity (+315.96)
        self.assertEqual(rec["equity_difference"], 315.96)
        self.assertEqual(rec["discrepancy_amount"], 315.96)

        # 3. Realized Difference = broker realized minus paper realized (-1273.37 - (-1949.74) = +676.37)
        self.assertEqual(rec["realized_difference"], 676.37)

        # 4. Unrealized Difference = broker unrealized minus paper unrealized (0.0 - 209.94 = -209.94)
        self.assertEqual(rec["unrealized_difference"], -209.94)

        # 5. Proof of mathematical identity (Cash Diff + Unrealized Diff == Equity Diff within rounding)
        sum_components = round(rec["cash_difference"] + rec["unrealized_difference"], 2)
        self.assertAlmostEqual(sum_components, rec["equity_difference"], delta=0.01)

    # 19. Full-precision ledger accounting vs display rounding verification
    def test_19_full_precision_ledger_accounting_vs_display_rounding(self):
        # Actual full-precision values stored in live portfolio_ledger.json
        balance_full = 98200.73583000005
        upl_full = 209.93818264000097
        equity_full = 98410.67401264005

        # 1. Verify underlying accounting is exact (difference is 0.0)
        self.assertEqual(balance_full + upl_full, equity_full)

        # 2. Verify display rounding causes the apparent $0.01 divergence
        rounded_bal = round(balance_full, 2)  # 98200.74 (rounds up)
        rounded_upl = round(upl_full, 2)      # 209.94 (rounds up)
        rounded_eq = round(equity_full, 2)    # 98410.67 (rounds down)

        self.assertEqual(rounded_bal, 98200.74)
        self.assertEqual(rounded_upl, 209.94)
        self.assertEqual(rounded_eq, 98410.67)
        from decimal import Decimal
        d_bal = Decimal(str(rounded_bal))
        d_upl = Decimal(str(rounded_upl))
        d_eq = Decimal(str(rounded_eq))

        self.assertEqual(d_bal + d_upl, Decimal("98410.68"))
        self.assertEqual((d_bal + d_upl) - d_eq, Decimal("0.01"))

    # 20. Today's settled (UTC) vs cumulative realized P&L separation
    def test_20_todays_settled_vs_cumulative(self):
        ledger = self.sample_ledger.copy()
        # 1 trade closed yesterday, 1 trade closed today
        today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        ledger["closed_trades"] = [
            {"pair": "EURUSD", "net_pnl": 500.0, "exit_time": "2026-10-20T10:00:00Z"},  # Prior date
            {"pair": "USDJPY", "net_pnl": -150.0, "exit_time": f"{today_str}T12:00:00Z"},  # Today
        ]
        data = enrich_portfolio_data(ledger, broker_state=None, engine_running=True, config_path=self.config_path)
        summary = data["account_summary"]

        # Today's settled must only include today's trade (-150.0), NOT cumulative (350.0)
        self.assertEqual(summary["today_realized_pnl"], -150.0)
        self.assertEqual(summary["strategy_paper"]["simulated_realized_pnl"], 350.0)

        # Verify UI clearly denotes "Today's Settled (UTC Day)"
        self.assertIn("Today's Settled (UTC Day)", HTML_DASHBOARD)

    # 21. USD/JPY P&L calculation audit & accounting convention verification
    def test_21_usdjpy_pnl_accounting_convention(self):
        # 1. Stored live parameters from portfolio_ledger.json
        entry_price = 156.81399536132812
        entry_atr = 0.23543067506818544
        current_price = 157.47300720214844
        risk_budget = 150.0  # 1.5% of initial $10,000 capital
        
        # 2. Engine Position Sizing Logic (live_paper_portfolio.py line 428-429)
        sl_dist = 2.0 * entry_atr  # 0.47086135 JPY
        expected_pos_size = risk_budget / sl_dist  # 318.565114 USD/JPY sensitivity multiplier
        
        # 3. Engine P&L Logic (live_paper_portfolio.py line 489)
        price_gain_jpy = current_price - entry_price  # +0.65901184 JPY (+65.9 pips)
        calculated_upl_usd = price_gain_jpy * expected_pos_size  # +209.93818 USD
        
        # 4. Stop-Loss Risk Consistency Check
        # If price drops to SL (entry_price - sl_dist), loss must equal -$150.00 exactly
        sl_exit_price = entry_price - sl_dist
        sl_loss_usd = (sl_exit_price - entry_price) * expected_pos_size
        self.assertAlmostEqual(sl_loss_usd, -risk_budget, places=5)
        
        # 5. Verify consistency with stored ledger values
        self.assertAlmostEqual(calculated_upl_usd, 209.93818, places=4)
        self.assertEqual(round(calculated_upl_usd, 2), 209.94)
        
        # 6. Verify dashboard enrichment handles USD/JPY with 100.0 pip multiplier
        ledger = self.sample_ledger.copy()
        ledger["positions"]["USDJPY"] = {
            "position": 1,
            "entry_price": entry_price,
            "current_price": current_price,
            "entry_atr": entry_atr,
            "position_size": expected_pos_size,
            "unrealized_pnl": calculated_upl_usd,
            "stop_loss": sl_exit_price,
            "take_profit": entry_price + (6.0 * entry_atr),
            "broker_ticket": None
        }
        data = enrich_portfolio_data(ledger, broker_state=None, engine_running=True, config_path=self.config_path)
        uj = data["positions"]["USDJPY"]
        
        self.assertEqual(uj["pips"], 65.9)
        self.assertEqual(uj["execution_category"], "SIMULATED")
        self.assertEqual(round(uj["unrealized_pnl"], 2), 209.94)

    def test_22_news_intelligence_endpoint_and_html_elements(self):
        from dashboard_server import HTML_DASHBOARD, get_news_intelligence_data

        # 1. Verify HTML dashboard contains News Shield elements
        self.assertIn('id="news-shield-indicator"', HTML_DASHBOARD)
        self.assertIn('id="news-status-badge"', HTML_DASHBOARD)
        self.assertIn('id="news-provider-badge"', HTML_DASHBOARD)
        self.assertIn('id="news-badge-eurusd"', HTML_DASHBOARD)
        self.assertIn('id="news-badge-gbpusd"', HTML_DASHBOARD)
        self.assertIn('id="news-badge-usdjpy"', HTML_DASHBOARD)
        self.assertIn('id="news-badge-audusd"', HTML_DASHBOARD)
        self.assertIn('id="news-events-body"', HTML_DASHBOARD)

        # 2. Verify get_news_intelligence_data endpoint payload
        news_data = get_news_intelligence_data(config_path=self.config_path)
        self.assertIn("status", news_data)
        self.assertIn("provider", news_data)
        self.assertIn("blackout_status", news_data)
        self.assertIn("upcoming_events", news_data)

        # Check pair keys
        for pair in ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD"]:
            self.assertIn(pair, news_data["blackout_status"])
            self.assertIn("blocked", news_data["blackout_status"][pair])
            self.assertIn("reason", news_data["blackout_status"][pair])

        # 3. Verify no API secrets or tokens leaked
        payload_str = json.dumps(news_data)
        self.assertNotIn("api_token", payload_str)
        self.assertNotIn("token", payload_str.lower().replace("token", "", 0))  # Ensure no auth secrets
        self.assertNotIn("582354e0279fb5fa953279c59d441bb2", payload_str)

    def test_23_advisory_intelligence_endpoint_and_html_elements(self):
        from dashboard_server import HTML_DASHBOARD, get_advisory_intelligence_data

        # 1. Verify HTML dashboard contains Advisory Intelligence elements
        self.assertIn("LOCAL AI MARKET INTELLIGENCE — ADVISORY ONLY", HTML_DASHBOARD)
        self.assertIn('id="advisory-status-indicator"', HTML_DASHBOARD)
        self.assertIn('id="advisory-status-badge"', HTML_DASHBOARD)
        self.assertIn('id="advisory-model-badge"', HTML_DASHBOARD)
        self.assertIn('id="adv-tab-EURUSD"', HTML_DASHBOARD)
        self.assertIn('id="adv-tab-USDJPY"', HTML_DASHBOARD)
        self.assertIn('id="advisory-content-panel"', HTML_DASHBOARD)

        # 2. Verify get_advisory_intelligence_data payload
        adv_data = get_advisory_intelligence_data(config_path=self.config_path)
        self.assertIn("service_status", adv_data)
        self.assertIn("advisories", adv_data)
        self.assertTrue(adv_data["is_advisory_only"])
        self.assertIn("disclaimer", adv_data)

        # 3. Verify no secret leaks
        adv_str = json.dumps(adv_data)
        self.assertNotIn("api_token", adv_str)
        self.assertNotIn("582354e0279fb5fa953279c59d441bb2", adv_str)


if __name__ == "__main__":
    unittest.main()
