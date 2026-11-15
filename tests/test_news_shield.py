#!/usr/bin/env python3
"""
Comprehensive Automated Test Suite for NewsShield & NewsCalendarClient.
Covers all 20 required verification scenarios specified in Phase 9:
 1. Event 45 minutes away: allowed if calendar is healthy.
 2. Event 15 minutes away: blocked.
 3. Event currently in progress: blocked.
 4. Event 10 minutes after scheduled time: blocked.
 5. Event 30 minutes after scheduled time: allowed (with 15-minute post-event window).
 6. USD event blocks all four supported pairs (EUR/USD, GBP/USD, AUD/USD, USD/JPY).
 7. EUR-only event does not directly block USD/JPY.
 8. JPY-only event does not directly block EUR/USD.
 9. Events outside configured impact policy do not trigger blackouts.
 10. Missing calendar cache handled safely per fail-safe policy.
 11. Expired/stale calendar cache handled safely per fail-safe policy.
 12. Corrupted cache file handled safely without crashing.
 13. HTTP timeout and provider rate-limit response (HTTP 429) caught and handled.
 14. Malformed event data rejected; valid records preserved.
 15. Duplicate events deduplicated using composite key.
 16. UTC and timezone-boundary handling (daylight saving and offsets normalized).
 17. Cache updates are atomic and preserve valid data on failure.
 18. Execution loop makes zero synchronous news-provider HTTP requests.
 19. News-data failure blocks new entries without crashing existing-position management.
 20. Existing spread guard and daily circuit-breaker guards remain fully operational.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import unittest
from datetime import datetime, timezone, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from news_client import (
    BaseNewsProvider,
    FinnhubNewsProvider,
    MockNewsProvider,
    NewsCalendarClient,
    normalize_iso_utc,
    STATUS_HEALTHY,
    STATUS_STALE,
    STATUS_UNAVAILABLE,
    STATUS_ERROR,
    POLICY_BLOCK_NEW_ENTRIES,
    POLICY_ALLOW_WITH_SPREAD_GUARD,
    IMPACT_HIGH,
    IMPACT_MEDIUM,
    IMPACT_LOW,
)
from risk_guards import NewsShield, SpreadGuard, DailyCircuitBreaker
from live_paper_portfolio import MultiPairPaperTrader


class TestNewsShieldAndCalendarClient(unittest.TestCase):

    def setUp(self) -> None:
        self.temp_dir = tempfile.mkdtemp()
        self.cache_file = os.path.join(self.temp_dir, "test_news_calendar_cache.json")
        self.config_file = os.path.join(self.temp_dir, "test_broker_config.json")

        # Create valid test configuration
        cfg = {
            "guards": {
                "news_shield_enabled": True,
                "news_shield": {
                    "provider": "mock",
                    "api_key_env_var": "FINNHUB_API_KEY",
                    "refresh_interval_seconds": 14400,
                    "max_cache_age_seconds": 21600,
                    "blackout_minutes_before": 30,
                    "blackout_minutes_after": 15,
                    "fail_safe_policy": "BLOCK_NEW_ENTRIES",
                },
                "max_spread_pips": {
                    "EURUSD": 2.5,
                    "GBPUSD": 3.0,
                    "USDJPY": 3.0,
                    "AUDUSD": 3.5,
                },
                "daily_drawdown_limit_pct": 3.0,
            }
        }
        with open(self.config_file, "w") as f:
            json.dump(cfg, f)

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    # -------------------------------------------------------------------------
    # Helper to construct a synthetic healthy client and shield
    # -------------------------------------------------------------------------
    def _create_shield_with_events(self, events: list, now_dt: datetime) -> Tuple[NewsShield, NewsCalendarClient]:
        mock_provider = MockNewsProvider(events)
        client = NewsCalendarClient(
            provider=mock_provider,
            cache_path=self.cache_file,
            refresh_interval_seconds=14400,
            max_cache_age_seconds=21600,
            fail_safe_policy=POLICY_BLOCK_NEW_ENTRIES,
            auto_load_cache=False,
        )
        client.refresh_events()
        shield = NewsShield(
            blackout_minutes_before=30,
            blackout_minutes_after=15,
            config_path=self.config_file,
            cache_path=self.cache_file,
            client=client,
        )
        return shield, client

    # =========================================================================
    # Scenario 1: Event 45 minutes away: allowed if calendar is healthy
    # =========================================================================
    def test_01_event_45_min_away_allowed(self):
        now = datetime(2026, 10, 27, 12, 0, 0, tzinfo=timezone.utc)
        event_time = now + timedelta(minutes=45)
        events = [{
            "title": "US Non-Farm Payrolls",
            "currency": "USD",
            "impact": "HIGH",
            "time_utc": event_time.isoformat(),
        }]
        shield, _ = self._create_shield_with_events(events, now)
        res = shield.check_blackout("EURUSD", current_time=now)
        self.assertFalse(res["blocked"])
        self.assertIn("Clear", res["reason"])

    # =========================================================================
    # Scenario 2: Event 15 minutes away: blocked
    # =========================================================================
    def test_02_event_15_min_away_blocked(self):
        now = datetime(2026, 10, 27, 12, 0, 0, tzinfo=timezone.utc)
        event_time = now + timedelta(minutes=15)
        events = [{
            "title": "US CPI Inflation",
            "currency": "USD",
            "impact": "HIGH",
            "time_utc": event_time.isoformat(),
        }]
        shield, _ = self._create_shield_with_events(events, now)
        res = shield.check_blackout("EURUSD", current_time=now)
        self.assertTrue(res["blocked"])
        self.assertIn("High-Impact News", res["reason"])
        self.assertIn("US CPI Inflation", res["reason"])

    # =========================================================================
    # Scenario 3: Event currently in progress: blocked
    # =========================================================================
    def test_03_event_in_progress_blocked(self):
        now = datetime(2026, 10, 27, 12, 0, 0, tzinfo=timezone.utc)
        events = [{
            "title": "FOMC Rate Decision",
            "currency": "USD",
            "impact": "HIGH",
            "time_utc": now.isoformat(),
        }]
        shield, _ = self._create_shield_with_events(events, now)
        res = shield.check_blackout("USDJPY", current_time=now)
        self.assertTrue(res["blocked"])
        self.assertIn("now", res["reason"].lower())

    # =========================================================================
    # Scenario 4: Event 10 minutes after scheduled time: blocked (in 15m post-window)
    # =========================================================================
    def test_04_event_10_min_after_blocked(self):
        now = datetime(2026, 10, 27, 12, 10, 0, tzinfo=timezone.utc)
        event_time = now - timedelta(minutes=10)
        events = [{
            "title": "ECB Interest Rate Decision",
            "currency": "EUR",
            "impact": "HIGH",
            "time_utc": event_time.isoformat(),
        }]
        shield, _ = self._create_shield_with_events(events, now)
        res = shield.check_blackout("EURUSD", current_time=now)
        self.assertTrue(res["blocked"])
        self.assertIn("10m ago", res["reason"])

    # =========================================================================
    # Scenario 5: Event 30 minutes after scheduled time: allowed (post-window expired)
    # =========================================================================
    def test_05_event_30_min_after_allowed(self):
        now = datetime(2026, 10, 27, 12, 30, 0, tzinfo=timezone.utc)
        event_time = now - timedelta(minutes=30)
        events = [{
            "title": "ECB Interest Rate Decision",
            "currency": "EUR",
            "impact": "HIGH",
            "time_utc": event_time.isoformat(),
        }]
        shield, _ = self._create_shield_with_events(events, now)
        res = shield.check_blackout("EURUSD", current_time=now)
        self.assertFalse(res["blocked"])
        self.assertIn("Clear", res["reason"])

    # =========================================================================
    # Scenario 6: USD event blocks all four supported pairs
    # =========================================================================
    def test_06_usd_event_blocks_all_four_pairs(self):
        now = datetime(2026, 10, 27, 12, 0, 0, tzinfo=timezone.utc)
        event_time = now + timedelta(minutes=10)
        events = [{
            "title": "Federal Reserve Rate Decision",
            "currency": "USD",
            "impact": "HIGH",
            "time_utc": event_time.isoformat(),
        }]
        shield, _ = self._create_shield_with_events(events, now)
        for pair in ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD"]:
            res = shield.check_blackout(pair, current_time=now)
            self.assertTrue(res["blocked"], f"Expected {pair} to be blocked by USD news")

    # =========================================================================
    # Scenario 7: EUR-only event does not directly block USD/JPY
    # =========================================================================
    def test_07_eur_only_event_does_not_block_usdjpy(self):
        now = datetime(2026, 10, 27, 12, 0, 0, tzinfo=timezone.utc)
        event_time = now + timedelta(minutes=10)
        events = [{
            "title": "German GDP Growth",
            "currency": "EUR",
            "impact": "HIGH",
            "time_utc": event_time.isoformat(),
        }]
        shield, _ = self._create_shield_with_events(events, now)
        # Blocked for EUR/USD
        res_eur = shield.check_blackout("EURUSD", current_time=now)
        self.assertTrue(res_eur["blocked"])

        # Allowed for USD/JPY
        res_jpy = shield.check_blackout("USDJPY", current_time=now)
        self.assertFalse(res_jpy["blocked"])

    # =========================================================================
    # Scenario 8: JPY-only event does not directly block EUR/USD
    # =========================================================================
    def test_08_jpy_only_event_does_not_block_eurusd(self):
        now = datetime(2026, 10, 27, 12, 0, 0, tzinfo=timezone.utc)
        event_time = now + timedelta(minutes=10)
        events = [{
            "title": "Bank of Japan Monetary Policy Statement",
            "currency": "JPY",
            "impact": "HIGH",
            "time_utc": event_time.isoformat(),
        }]
        shield, _ = self._create_shield_with_events(events, now)
        # Blocked for USD/JPY
        res_jpy = shield.check_blackout("USDJPY", current_time=now)
        self.assertTrue(res_jpy["blocked"])

        # Allowed for EUR/USD
        res_eur = shield.check_blackout("EURUSD", current_time=now)
        self.assertFalse(res_eur["blocked"])

    # =========================================================================
    # Scenario 9: Events outside configured impact policy do not trigger blackouts
    # =========================================================================
    def test_09_low_and_medium_impact_do_not_block(self):
        now = datetime(2026, 10, 27, 12, 0, 0, tzinfo=timezone.utc)
        event_time = now + timedelta(minutes=5)
        events = [
            {"title": "US Weekly EIA Petroleum Status", "currency": "USD", "impact": "LOW", "time_utc": event_time.isoformat()},
            {"title": "German Wholesale Price Index", "currency": "EUR", "impact": "MEDIUM", "time_utc": event_time.isoformat()},
        ]
        shield, _ = self._create_shield_with_events(events, now)
        res = shield.check_blackout("EURUSD", current_time=now)
        self.assertFalse(res["blocked"])

    # =========================================================================
    # Scenario 10: Missing calendar cache handled safely per fail-safe policy
    # =========================================================================
    def test_10_missing_cache_blocks_under_default_policy(self):
        non_existent_cache = os.path.join(self.temp_dir, "missing_cache.json")
        client = NewsCalendarClient(
            provider=MockNewsProvider([]),
            cache_path=non_existent_cache,
            fail_safe_policy=POLICY_BLOCK_NEW_ENTRIES,
            auto_load_cache=True,
        )
        shield = NewsShield(client=client, config_path=self.config_file)
        res = shield.check_blackout("EURUSD")
        self.assertTrue(res["blocked"])
        self.assertEqual(res["status"], STATUS_UNAVAILABLE)
        self.assertIn("BLOCK_NEW_ENTRIES", res["reason"])

    # =========================================================================
    # Scenario 11: Expired/stale calendar cache handled safely per fail-safe policy
    # =========================================================================
    def test_11_expired_cache_blocks_under_default_policy(self):
        stale_time = (datetime.now(timezone.utc) - timedelta(hours=10)).isoformat()
        stale_data = {
            "schema_version": "1.0.0",
            "last_successful_refresh_utc": stale_time,
            "provider": "mock",
            "status": STATUS_HEALTHY,
            "events": [],
            "cache_metadata": {"max_cache_age_seconds": 21600},
        }
        with open(self.cache_file, "w") as f:
            json.dump(stale_data, f)

        client = NewsCalendarClient(
            cache_path=self.cache_file,
            max_cache_age_seconds=21600,
            fail_safe_policy=POLICY_BLOCK_NEW_ENTRIES,
            auto_load_cache=True,
        )
        shield = NewsShield(client=client, config_path=self.config_file)
        res = shield.check_blackout("EURUSD")
        self.assertTrue(res["blocked"])
        self.assertEqual(res["status"], STATUS_STALE)

    # =========================================================================
    # Scenario 12: Corrupted cache file handled safely without crashing
    # =========================================================================
    def test_12_corrupted_cache_handled_safely(self):
        with open(self.cache_file, "w") as f:
            f.write("{ incomplete json: syntax error ...")

        client = NewsCalendarClient(
            cache_path=self.cache_file,
            fail_safe_policy=POLICY_BLOCK_NEW_ENTRIES,
            auto_load_cache=True,
        )
        self.assertEqual(client.get_status()["status"], STATUS_ERROR)

        shield = NewsShield(client=client, config_path=self.config_file)
        blocked, reason = shield.is_blackout("EURUSD")
        self.assertTrue(blocked)
        self.assertIn("ERROR", reason)

    # =========================================================================
    # Scenario 13: HTTP timeout and provider rate-limit response (HTTP 429) caught
    # =========================================================================
    def test_13_http_timeout_and_rate_limit_handled(self):
        finnhub = FinnhubNewsProvider(api_token="test_token")

        with patch("requests.get") as mock_get:
            # Test HTTP 429 Rate Limit
            mock_resp_429 = MagicMock()
            mock_resp_429.status_code = 429
            mock_get.return_value = mock_resp_429
            with self.assertRaises(PermissionError):
                finnhub.fetch_events(datetime.now(timezone.utc), datetime.now(timezone.utc))

            # Test HTTP Timeout
            mock_get.side_effect = TimeoutError("Request timed out")
            with self.assertRaises(TimeoutError):
                finnhub.fetch_events(datetime.now(timezone.utc), datetime.now(timezone.utc))

    # =========================================================================
    # Scenario 14: Malformed event data rejected; valid records preserved
    # =========================================================================
    def test_14_malformed_event_data_rejected(self):
        valid_ev = {
            "title": "US CPI",
            "currency": "USD",
            "impact": "HIGH",
            "time_utc": "2026-10-27T12:00:00Z",
        }
        malformed_cases = [
            None,
            "not a dict",
            {"currency": "USD"},  # Missing title and time
            {"title": "No time", "currency": "USD"},  # Missing time
            {"title": "Bad ccy", "currency": "XYZ", "time_utc": "2026-10-27T12:00:00Z"},  # Unsupported currency
            {"title": "Bad date", "currency": "USD", "time_utc": "invalid-date-string"},  # Bad date
        ]
        self.assertIsNotNone(NewsCalendarClient.validate_and_normalize_event(valid_ev))
        for bad in malformed_cases:
            self.assertIsNone(NewsCalendarClient.validate_and_normalize_event(bad))

    # =========================================================================
    # Scenario 15: Duplicate events deduplicated using composite key
    # =========================================================================
    def test_15_duplicate_events_deduplicated(self):
        duplicate_events = [
            {"title": "US Non-Farm Payrolls", "currency": "USD", "impact": "HIGH", "time_utc": "2026-11-06T13:30:00Z"},
            {"title": "US Non-Farm Payrolls", "currency": "USD", "impact": "HIGH", "time_utc": "2026-11-06T13:30:00Z"},
            {"title": "US Non-Farm Payrolls", "currency": "USD", "impact": "HIGH", "time_utc": "2026-11-06T13:30:00Z"},
        ]
        client = NewsCalendarClient(
            provider=MockNewsProvider(duplicate_events),
            cache_path=self.cache_file,
            auto_load_cache=False,
        )
        ok, msg = client.refresh_events()
        self.assertTrue(ok)
        events = client.get_events(only_upcoming=False)
        self.assertEqual(len(events), 1)

    # =========================================================================
    # Scenario 16: UTC and timezone-boundary handling
    # =========================================================================
    def test_16_utc_timezone_boundary_normalization(self):
        # Test offset string +02:00 and daylight saving representation
        offset_str = "2026-10-27T14:30:00+02:00"  # Equivalent to 12:30:00 UTC
        norm = normalize_iso_utc(offset_str)
        self.assertIsNotNone(norm)
        self.assertEqual(norm.tzinfo, timezone.utc)
        self.assertEqual(norm.hour, 12)
        self.assertEqual(norm.minute, 30)

        # UNIX timestamp normalization
        epoch_norm = normalize_iso_utc(1700000000)
        self.assertEqual(epoch_norm.tzinfo, timezone.utc)

    # =========================================================================
    # Scenario 17: Cache updates are atomic and preserve valid data on failure
    # =========================================================================
    def test_17_atomic_cache_preserves_valid_data_on_failure(self):
        # 1. Establish valid cache
        initial_events = [{"title": "Initial Event", "currency": "USD", "impact": "HIGH", "time_utc": "2026-10-27T12:00:00Z"}]
        client = NewsCalendarClient(
            provider=MockNewsProvider(initial_events),
            cache_path=self.cache_file,
            auto_load_cache=False,
        )
        client.refresh_events()
        self.assertEqual(len(client.get_events(only_upcoming=False)), 1)

        # 2. Simulate subsequent provider crash
        failing_provider = MagicMock()
        failing_provider.provider_name = "failing_mock"
        failing_provider.fetch_events.side_effect = ConnectionError("Network down")
        client.provider = failing_provider

        ok, err = client.refresh_events()
        self.assertFalse(ok)

        # 3. Verify existing cached events are STILL PRESERVED on disk and in memory
        reloaded_client = NewsCalendarClient(cache_path=self.cache_file, auto_load_cache=True)
        reloaded_events = reloaded_client.get_events(only_upcoming=False)
        self.assertEqual(len(reloaded_events), 1)
        self.assertEqual(reloaded_events[0]["title"], "Initial Event")

    # =========================================================================
    # Scenario 18: Execution loop makes zero synchronous news-provider HTTP requests
    # =========================================================================
    def test_18_zero_synchronous_http_calls_during_execution(self):
        shield, _ = self._create_shield_with_events([], datetime.now(timezone.utc))
        with patch("requests.get") as mock_get:
            shield.is_blackout("EURUSD")
            shield.check_blackout("EURUSD")
            mock_get.assert_not_called()

    # =========================================================================
    # Scenario 19: News failure blocks new entries without crashing existing positions
    # =========================================================================
    def test_19_news_failure_blocks_entries_without_crashing_portfolio(self):
        # Initialize client in UNAVAILABLE state
        empty_cache = os.path.join(self.temp_dir, "unavailable_cache.json")
        client = NewsCalendarClient(
            provider=MockNewsProvider([]),
            cache_path=empty_cache,
            fail_safe_policy=POLICY_BLOCK_NEW_ENTRIES,
            auto_load_cache=True,
        )
        shield = NewsShield(client=client, config_path=self.config_file)

        # 1. New entry is blocked
        blocked, reason = shield.is_blackout("EURUSD")
        self.assertTrue(blocked)
        self.assertIn("BLOCK_NEW_ENTRIES", reason)

        # 2. Existing position calculations proceed normally without crashing
        open_pos = {
            "position": 1,
            "entry_price": 1.1200,
            "base_units": 50000,
            "position_size": 50000,
            "accounting_method": "STANDARD_SPOT_FX",
        }
        pnl = MultiPairPaperTrader.calculate_position_pnl("EURUSD", open_pos, 1.1220)
        self.assertAlmostEqual(pnl, 100.0, places=2)

    # =========================================================================
    # Scenario 20: Existing spread and daily circuit-breaker guards remain operational
    # =========================================================================
    def test_20_spread_and_circuit_breaker_remain_operational(self):
        # Test SpreadGuard
        spread_guard = SpreadGuard()
        mock_gateway = MagicMock()
        mock_gateway.get_realtime_spread.return_value = 4.5  # Exceeds EUR/USD limit (2.5 pips)
        sp_ok, cur_sp, max_sp, sp_msg = spread_guard.check(mock_gateway, "EURUSD")
        self.assertFalse(sp_ok)
        self.assertIn("Spread Spike", sp_msg)

        # Test DailyCircuitBreaker
        circuit_breaker = DailyCircuitBreaker(max_daily_drawdown_pct=3.0)
        circuit_breaker.check(current_equity=100000.0)  # Establish starting equity
        cb_ok, dd_pct, cb_msg = circuit_breaker.check(current_equity=96000.0)  # 4% drawdown
        self.assertFalse(cb_ok)
        self.assertIn("Circuit Breaker Tripped", cb_msg)


if __name__ == "__main__":
    unittest.main()
