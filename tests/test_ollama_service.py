"""
Unit Tests for OllamaService.

Verifies:
- Hardware profile inspection and RAM budgeting.
- Strict JSON schema validation.
- Rejection of malformed JSON, missing fields, and pair mismatches.
- Non-crashing fallback handling when Ollama is offline or times out.
- Text sanitization and length bounding.
- Mocked inference responses.
"""

import unittest
from unittest.mock import MagicMock, patch

from ollama_service import OllamaService, sanitize_text


class TestOllamaService(unittest.TestCase):

    def setUp(self):
        self.service = OllamaService(
            base_url="http://localhost:11434",
            model_name="qwen2.5:1.5b",
            timeout_seconds=5.0,
        )
        self.sample_snapshot = {
            "snapshot_timestamp_utc": "2026-10-27T00:00:00Z",
            "pair": "EURUSD",
            "market_data": {
                "current_price": 1.1179,
                "current_atr": 0.00085,
                "live_spread_pips": 3.4,
                "max_spread_limit_pips": 2.5,
                "spread_status": "ELEVATED_BLOCKED",
            },
            "engine_signal": {
                "signal": "BUY / LONG",
                "probabilities": {"flat": 0.36, "long": 0.45, "short": 0.19},
            },
            "position_state": {"direction": "FLAT", "has_active_position": False},
            "risk_guards": {
                "spread_guard": {"passed": False},
                "circuit_breaker": {"tripped": False},
                "news_shield": {"cache_status": "HEALTHY", "upcoming_events": []},
            },
            "data_quality": {"status": "VALID", "limitations": ["Test limitation"]},
        }

    def test_01_hardware_inspection(self):
        hw = OllamaService.inspect_hardware()
        self.assertIn("cpu_model", hw)
        self.assertIn("cpu_cores", hw)
        self.assertIn("available_ram_mb", hw)
        self.assertIn("recommended_max_params", hw)
        self.assertIsInstance(hw["cpu_cores"], int)
        self.assertGreater(hw["cpu_cores"], 0)
        self.assertIn(hw["recommended_max_params"], ("0.5B", "1.5B", "3B"))

    def test_02_sanitize_text(self):
        dirty = "Hello\x00\x08World! " + ("A" * 1500)
        cleaned = sanitize_text(dirty, max_length=100)
        self.assertEqual(len(cleaned), 100)
        self.assertNotIn("\x00", cleaned)
        self.assertTrue(cleaned.startswith("HelloWorld!"))

    def test_03_valid_schema_parsing(self):
        valid_mock = {
            "timestamp_utc": "2026-10-27T00:00:00Z",
            "pair": "EURUSD",
            "engine_signal": "BUY / LONG",
            "analysis_status": "SUCCESS",
            "market_summary": "EUR/USD consolidating near 1.1180 with moderate ATR.",
            "supporting_evidence": ["Policy model long probability at 45%", "Short probability low at 19%"],
            "conflicting_evidence": ["Spread is elevated at 3.4 pips exceeding 2.5 pip ceiling"],
            "alternative_scenarios": [
                {"scenario": "Bullish breakout above 1.1185", "key_condition": "Spread tightens below 2.5p"},
                {"scenario": "Range-bound drift", "key_condition": "Persistent Asian session consolidation"}
            ],
            "risk_context": ["Execution blocked by spread guard"],
            "invalidation_conditions": ["Break below 1.1160"],
            "news_data_status": "HEALTHY",
            "data_quality": {"status": "VALID", "limitations": ["Mock schedule"]},
            "uncertainties": ["Thin liquidity during late night hours"],
            "advisory_summary": "Advisory context: Favorable directional bias but entry blocked by spread guard.",
        }

        result = self.service.generate_advisory(self.sample_snapshot, mock_response=valid_mock)
        self.assertEqual(result["analysis_status"], "SUCCESS")
        self.assertEqual(result["pair"], "EURUSD")
        self.assertTrue(result["is_advisory_only"])
        self.assertEqual(len(result["supporting_evidence"]), 2)
        self.assertEqual(len(result["conflicting_evidence"]), 1)
        self.assertEqual(len(result["alternative_scenarios"]), 2)

    def test_04_missing_schema_fields_fallback(self):
        invalid_mock = {
            "pair": "EURUSD",
            "market_summary": "Incomplete output",
        }
        result = self.service.generate_advisory(self.sample_snapshot, mock_response=invalid_mock)
        self.assertEqual(result["analysis_status"], "SCHEMA_MISSING_FIELDS")
        self.assertTrue(result["is_advisory_only"])
        self.assertIn("Missing required schema fields", result["error_detail"])

    def test_05_pair_mismatch_rejection(self):
        mismatch_mock = {
            "timestamp_utc": "2026-10-27T00:00:00Z",
            "pair": "USDJPY",  # Mismatched from EURUSD!
            "engine_signal": "BUY / LONG",
            "analysis_status": "SUCCESS",
            "market_summary": "Wrong pair output",
            "supporting_evidence": [],
            "conflicting_evidence": [],
            "alternative_scenarios": [],
            "risk_context": [],
            "invalidation_conditions": [],
            "news_data_status": "HEALTHY",
            "data_quality": {"status": "VALID", "limitations": []},
            "uncertainties": [],
            "advisory_summary": "Mismatched pair.",
        }
        result = self.service.generate_advisory(self.sample_snapshot, mock_response=mismatch_mock)
        self.assertEqual(result["analysis_status"], "PAIR_MISMATCH")
        self.assertTrue(result["is_advisory_only"])

    def test_06_ollama_unavailable_fallback(self):
        with patch.object(self.service, "is_available", return_value=False):
            result = self.service.generate_advisory(self.sample_snapshot)
            self.assertEqual(result["analysis_status"], "OLLAMA_UNAVAILABLE")
            self.assertTrue(result["is_advisory_only"])
            self.assertIn("not reachable", result["error_detail"])

    def test_07_inference_timeout_fallback(self):
        with patch.object(self.service, "is_available", return_value=True):
            import requests
            with patch("requests.post", side_effect=requests.exceptions.Timeout("Timed out")):
                result = self.service.generate_advisory(self.sample_snapshot)
                self.assertEqual(result["analysis_status"], "TIMEOUT")
                self.assertTrue(result["is_advisory_only"])
                self.assertIn("timed out", result["error_detail"])


if __name__ == "__main__":
    unittest.main()
