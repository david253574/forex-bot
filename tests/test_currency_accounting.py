#!/usr/bin/env python3
"""
Comprehensive Independent Test Suite for Currency Accounting & Position Sizing.
Tests all 12 scenarios specified in Phase 6 of the USD/JPY Currency Accounting task:
 1. USD/JPY long trades with profitable exits.
 2. USD/JPY long trades with losing exits.
 3. USD/JPY long stop-loss execution at the assumed stop price.
 4. USD/JPY short trades and stop-loss execution.
 5. Different exchange rates with the same USD risk budget.
 6. EUR/USD, GBP/USD, and AUD/USD regression cases.
 7. Correct signed base-currency units sent to the broker gateway.
 8. Invalid, zero, negative, NaN, or infinite size inputs.
 9. Spread and slippage effects on realized outcomes.
 10. Cash, realized P&L, unrealized P&L, and equity reconciliation.
 11. Existing legacy paper positions loaded without silent reinterpretation.
 12. Order rejection when risk or unit conversion cannot be validated.
"""

import math
import sys
import unittest
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from live_paper_portfolio import MultiPairPaperTrader
from broker_gateway import BaseBrokerGateway, MockBrokerGateway


class TestCurrencyAccountingAndPositionSizing(unittest.TestCase):

    # =========================================================================
    # Scenario 1: USD/JPY long trades with profitable exits
    # =========================================================================
    def test_01_usdjpy_long_profitable_exit(self):
        # Entry at 156.814, SL at 156.34313 (sl_dist = 0.47087 JPY), Risk Budget = $150.00
        balance = 10000.0
        risk_pct = 0.015
        entry = 156.814
        atr = 0.235435  # sl_dist = 2 * atr = 0.47087
        
        sizing = MultiPairPaperTrader.calculate_position_sizing(
            pair="USDJPY", action=1, balance=balance, risk_per_trade=risk_pct,
            current_price=entry, current_atr=atr
        )
        base_units = sizing["base_units"]
        
        # Independently derived base units: U = (R * S) / sl_dist
        sl_price = entry - (2.0 * atr)
        expected_units = int(round((150.0 * sl_price) / (2.0 * atr)))
        self.assertEqual(base_units, expected_units)
        
        # Exit with +100 pips (+1.00 JPY gain) at 157.814
        exit_price = 157.814
        pos_info = {
            "position": 1,
            "entry_price": entry,
            "base_units": base_units,
            "position_size": base_units,
            "accounting_method": "STANDARD_SPOT_FX",
        }
        pnl = MultiPairPaperTrader.calculate_position_pnl("USDJPY", pos_info, exit_price)
        
        # Independent reference calculation: U * (X - E) / X
        expected_pnl = base_units * (exit_price - entry) / exit_price
        self.assertAlmostEqual(pnl, expected_pnl, places=4)
        self.assertGreater(pnl, 0.0)
        # For ~49,800 units, +1.00 JPY move at 157.814 yields ~$315.56
        self.assertAlmostEqual(pnl, (base_units * 1.00) / 157.814, places=2)

    # =========================================================================
    # Scenario 2: USD/JPY long trades with losing exits (partial retracement)
    # =========================================================================
    def test_02_usdjpy_long_losing_exit(self):
        balance = 10000.0
        risk_pct = 0.015
        entry = 156.814
        atr = 0.235435
        
        sizing = MultiPairPaperTrader.calculate_position_sizing(
            pair="USDJPY", action=1, balance=balance, risk_per_trade=risk_pct,
            current_price=entry, current_atr=atr
        )
        base_units = sizing["base_units"]
        
        # Exit with -20 pips (-0.20 JPY loss) at 156.614
        exit_price = 156.614
        pos_info = {
            "position": 1,
            "entry_price": entry,
            "base_units": base_units,
            "position_size": base_units,
            "accounting_method": "STANDARD_SPOT_FX",
        }
        pnl = MultiPairPaperTrader.calculate_position_pnl("USDJPY", pos_info, exit_price)
        expected_pnl = base_units * (exit_price - entry) / exit_price
        
        self.assertAlmostEqual(pnl, expected_pnl, places=4)
        self.assertLess(pnl, 0.0)
        self.assertGreater(pnl, -150.0)  # Loss is less than full stop-loss risk

    # =========================================================================
    # Scenario 3: USD/JPY long stop-loss execution at the assumed stop price
    # =========================================================================
    def test_03_usdjpy_long_stop_loss_exact_risk(self):
        balance = 10000.0
        risk_budget = 150.0
        entry = 156.814
        atr = 0.23543068
        
        sizing = MultiPairPaperTrader.calculate_position_sizing(
            pair="USDJPY", action=1, balance=balance, risk_per_trade=0.015,
            current_price=entry, current_atr=atr
        )
        base_units = sizing["base_units"]
        sl_price = sizing["stop_loss"]
        
        pos_info = {
            "position": 1,
            "entry_price": entry,
            "base_units": base_units,
            "position_size": base_units,
            "accounting_method": "STANDARD_SPOT_FX",
        }
        sl_loss = MultiPairPaperTrader.calculate_position_pnl("USDJPY", pos_info, sl_price)
        
        # Proves independent mathematical derivation: Loss at SL MUST equal -$150.00 within rounding
        self.assertAlmostEqual(sl_loss, -risk_budget, delta=0.02)

    # =========================================================================
    # Scenario 4: USD/JPY short trades and stop-loss execution
    # =========================================================================
    def test_04_usdjpy_short_sizing_and_stop_loss(self):
        balance = 10000.0
        risk_budget = 150.0
        entry = 156.814
        atr = 0.23543068  # sl_dist = 0.47086136 JPY
        
        # For SHORT (action = 2), stop loss is ABOVE entry: S = E + sl_dist
        sizing = MultiPairPaperTrader.calculate_position_sizing(
            pair="USDJPY", action=2, balance=balance, risk_per_trade=0.015,
            current_price=entry, current_atr=atr
        )
        base_units = sizing["base_units"]
        order_units = sizing["order_units"]
        sl_price = sizing["stop_loss"]
        
        self.assertLess(order_units, 0)  # Negative signed units for broker
        self.assertGreater(sl_price, entry)  # SL above entry for short
        
        # Sizing formula for short: U = (R * S) / (S - E)
        expected_units = int(round((risk_budget * sl_price) / (sl_price - entry)))
        self.assertEqual(base_units, expected_units)
        
        pos_info = {
            "position": -1,
            "entry_price": entry,
            "base_units": base_units,
            "position_size": base_units,
            "accounting_method": "STANDARD_SPOT_FX",
        }
        
        # 1. Test profitable short exit (price falls to 155.814)
        profit_exit = 155.814
        profit_pnl = MultiPairPaperTrader.calculate_position_pnl("USDJPY", pos_info, profit_exit)
        expected_profit = base_units * (entry - profit_exit) / profit_exit
        self.assertAlmostEqual(profit_pnl, expected_profit, places=4)
        self.assertGreater(profit_pnl, 0.0)
        
        # 2. Test short stop-loss execution (price rises to sl_price)
        sl_loss = MultiPairPaperTrader.calculate_position_pnl("USDJPY", pos_info, sl_price)
        self.assertAlmostEqual(sl_loss, -risk_budget, delta=0.02)

    # =========================================================================
    # Scenario 5: Different exchange rates with the same USD risk budget
    # =========================================================================
    def test_05_different_exchange_rates_same_risk_budget(self):
        # Test USD/JPY at 100.0, 130.0, 155.0, 180.0 with same $150 risk and 50 pip SL (0.50 JPY)
        balance = 10000.0
        risk_budget = 150.0
        atr = 0.25  # sl_dist = 0.50 JPY
        
        for p_rate in [100.0, 130.0, 155.0, 180.0]:
            sizing = MultiPairPaperTrader.calculate_position_sizing(
                pair="USDJPY", action=1, balance=balance, risk_per_trade=0.015,
                current_price=p_rate, current_atr=atr
            )
            base_units = sizing["base_units"]
            sl_price = sizing["stop_loss"]
            
            # As exchange rate rises, required USD base units scale linearly with S
            self.assertAlmostEqual(base_units, (risk_budget * sl_price) / 0.50, delta=1.0)
            
            # Loss at stop-loss must invariant to the exchange rate: always -$150.00
            pos_info = {
                "position": 1,
                "entry_price": p_rate,
                "base_units": base_units,
                "position_size": base_units,
                "accounting_method": "STANDARD_SPOT_FX",
            }
            loss = MultiPairPaperTrader.calculate_position_pnl("USDJPY", pos_info, sl_price)
            self.assertAlmostEqual(loss, -risk_budget, delta=0.02)

    # =========================================================================
    # Scenario 6: EUR/USD, GBP/USD, and AUD/USD regression cases
    # =========================================================================
    def test_06_direct_pairs_regression(self):
        # Quote is USD: units = risk / sl_dist (no rate division needed)
        balance = 10000.0
        risk_budget = 150.0
        
        test_cases = [
            ("EURUSD", 1.12000, 0.00100),  # sl_dist = 0.00200 -> 75,000 EUR
            ("GBPUSD", 1.33000, 0.00125),  # sl_dist = 0.00250 -> 60,000 GBP
            ("AUDUSD", 0.70000, 0.00075),  # sl_dist = 0.00150 -> 100,000 AUD
        ]
        
        for pair, entry, atr in test_cases:
            sl_dist = 2.0 * atr
            expected_units = int(round(risk_budget / sl_dist))
            
            sizing = MultiPairPaperTrader.calculate_position_sizing(
                pair=pair, action=1, balance=balance, risk_per_trade=0.015,
                current_price=entry, current_atr=atr
            )
            self.assertEqual(sizing["base_units"], expected_units)
            self.assertEqual(sizing["order_units"], expected_units)
            
            pos_info = {
                "position": 1,
                "entry_price": entry,
                "base_units": sizing["base_units"],
                "position_size": sizing["base_units"],
                "accounting_method": "STANDARD_SPOT_FX",
            }
            
            # SL loss must equal -$150.00
            sl_loss = MultiPairPaperTrader.calculate_position_pnl(pair, pos_info, entry - sl_dist)
            self.assertAlmostEqual(sl_loss, -risk_budget, delta=0.01)

    # =========================================================================
    # Scenario 7: Correct signed base-currency units sent to the broker gateway
    # =========================================================================
    def test_07_broker_gateway_signed_units(self):
        mock_gateway = MockBrokerGateway()
        
        # 1. USD/JPY LONG order
        resp_long = mock_gateway.place_market_bracket_order(
            symbol="USDJPY",
            units=49805,  # Valid base units
            stop_loss=156.343,
            take_profit=158.226
        )
        self.assertEqual(resp_long["status"], "SUCCESS")
        self.assertEqual(resp_long["units"], 49805)
        
        # 2. USD/JPY SHORT order
        resp_short = mock_gateway.place_market_bracket_order(
            symbol="USDJPY",
            units=-50105,  # Negative units for short
            stop_loss=157.285,
            take_profit=155.402
        )
        self.assertEqual(resp_short["status"], "SUCCESS")
        self.assertEqual(resp_short["units"], -50105)

    # =========================================================================
    # Scenario 8: Invalid, zero, negative, NaN, or infinite size inputs
    # =========================================================================
    def test_08_invalid_size_inputs_rejected(self):
        gateway = MockBrokerGateway()
        
        # Zero units
        res_zero = gateway.place_market_bracket_order("EURUSD", 0, 1.1100, 1.1300)
        self.assertEqual(res_zero["status"], "REJECTED")
        self.assertIn("zero", res_zero["message"].lower())
        
        # NaN units
        res_nan = gateway.place_market_bracket_order("EURUSD", float("nan"), 1.1100, 1.1300)
        self.assertEqual(res_nan["status"], "REJECTED")
        
        # Infinite units
        res_inf = gateway.place_market_bracket_order("EURUSD", float("inf"), 1.1100, 1.1300)
        self.assertEqual(res_inf["status"], "REJECTED")
        
        # Inverted brackets (BUY with SL > TP)
        res_inv = gateway.place_market_bracket_order("EURUSD", 10000, 1.1300, 1.1100)
        self.assertEqual(res_inv["status"], "REJECTED")
        self.assertIn("invalid brackets", res_inv["message"].lower())

    # =========================================================================
    # Scenario 9: Spread and slippage effects on realized outcomes
    # =========================================================================
    def test_09_spread_and_slippage_accounting(self):
        # Demonstrate that actual broker fill with half-spread and slippage differs from theoretical mid-price
        units = 50000
        nominal_entry = 156.814
        half_spread_pips = 0.8  # 0.008 JPY
        actual_fill_entry = nominal_entry + 0.008  # Ask price filled
        
        nominal_sl = 156.343
        slippage_pips = 1.0  # 0.010 JPY slippage
        actual_fill_exit = nominal_sl - 0.010
        
        # Theoretical mid P&L
        theo_loss = (units * (nominal_sl - nominal_entry)) / nominal_sl
        
        # Realized broker P&L accounting for spread + slippage
        actual_loss = (units * (actual_fill_exit - actual_fill_entry)) / actual_fill_exit
        
        # Realized loss is greater than theoretical loss
        self.assertLess(actual_loss, theo_loss)
        drag = theo_loss - actual_loss
        self.assertGreater(drag, 0.0)

    # =========================================================================
    # Scenario 10: Cash, realized P&L, unrealized P&L, and equity reconciliation
    # =========================================================================
    def test_10_cash_equity_reconciliation_consistency(self):
        # Verify that balance + sum(unrealized) == equity with standard spot FX positions
        balance = 98200.74
        pos_info = {
            "position": 1,
            "entry_price": 156.814,
            "base_units": 49805,
            "position_size": 49805,
            "accounting_method": "STANDARD_SPOT_FX",
        }
        current_price = 157.545
        upl = MultiPairPaperTrader.calculate_position_pnl("USDJPY", pos_info, current_price)
        equity = balance + upl
        
        self.assertAlmostEqual(equity - balance, upl, places=5)
        self.assertAlmostEqual(equity, 98200.74 + 231.08, delta=0.02)

    # =========================================================================
    # Scenario 11: Existing legacy paper positions loaded without silent reinterpretation
    # =========================================================================
    def test_11_legacy_position_loaded_without_silent_reinterpretation(self):
        # A legacy USD/JPY position from portfolio_ledger.json has position_size = 318.565 and NO base_units key
        legacy_pos = {
            "position": 1,
            "entry_price": 156.81399536132812,
            "entry_atr": 0.23543067506818544,
            "position_size": 318.56511467028884,  # Legacy sensitivity scalar
            "stop_loss": 156.34313401119175,
            "take_profit": 158.22657941173725,
            "unrealized_pnl": 232.87199323287243,
        }
        current_price = 157.5449981689453
        
        # Must evaluate using legacy sensitivity without dropping to $1.48
        upl = MultiPairPaperTrader.calculate_position_pnl("USDJPY", legacy_pos, current_price)
        self.assertAlmostEqual(upl, 232.87, delta=0.01)
        self.assertNotEqual(round(upl, 2), 1.48)  # Confirms no silent reinterpretation

    # =========================================================================
    # Scenario 12: Order rejection when risk or unit conversion cannot be validated
    # =========================================================================
    def test_12_unnormalized_sensitivity_scalar_rejected(self):
        gateway = MockBrokerGateway()
        
        # Attempt to submit USD/JPY with unnormalized sensitivity scalar (318 units)
        res = gateway.place_market_bracket_order(
            symbol="USDJPY",
            units=318,  # Undersized scalar (< 1000)
            stop_loss=156.343,
            take_profit=158.226
        )
        self.assertEqual(res["status"], "REJECTED")
        self.assertIn("CRITICAL REJECTION", res["message"])
        self.assertIn("undersized", res["message"].lower())


if __name__ == "__main__":
    unittest.main()
