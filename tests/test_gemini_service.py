"""
Unit Tests for GeminiService.

Verifies:
- Service initialization and API key loading.
- Advisory prompt rendering adhering to advisory-only guidelines.
- Strict schema validation against ADVISORY_SCHEMA_FIELDS.
- Sanitization of text, lists, and scenarios.
- Graceful fallback when API key is missing or calls fail.
- Copilot chat handling with live telemetry context.
"""

import json
import unittest
from unittest.mock import MagicMock, patch

from gemini_service import GeminiService, sanitize_text


class TestGeminiService(unittest.TestCase):

    def setUp(self):
        self.service = GeminiService(
            api_key="test_fake_api_key",
            model_name="gemini-2.5-flash",
            timeout_seconds=5.0,
            temperature=0.2,
        )
        self.sample_snapshot = {
            "snapshot_timestamp_utc": "2026-11-15T05:00:00Z",
            "pair": "EURUSD",
            "market_data": {
                "current_price": 1.1138,
                "current_atr": 0.00076,
                "live_spread_pips": 3.4,
                "max_spread_limit_pips": 2.5,
                "spread_status": "ELEVATED_BLOCKED",
            },
            "engine_signal": {
                "signal": "BUY / LONG",
                "probabilities": {"flat": 0.36, "long": 0.454, "short": 0.186},
                "confidence_margin": 0.268,
            },
            "position_state": {"direction": "FLAT", "has_active_position": False},
            "risk_guards": {
                "spread_guard": {"passed": False},
                "circuit_breaker": {"tripped": False},
                "news_shield": {"cache_status": "HEALTHY", "upcoming_events": []},
            },
            "data_quality": {"status": "VALID", "limitations": []},
        }

    def test_01_sanitize_text(self):
        dirty = "Test\x00\x07Text! " + ("Z" * 2000)
        clean = sanitize_text(dirty, max_length=150)
        self.assertEqual(len(clean), 150)
        self.assertNotIn("\x00", clean)
        self.assertTrue(clean.startswith("TestText!"))

    def test_02_build_advisory_prompt(self):
        sys_p, usr_p = self.service.build_advisory_prompt(self.sample_snapshot)
        self.assertIn("STRICTLY ADVISORY", sys_p)
        self.assertIn("DO NOT issue trade commands", sys_p)
        self.assertIn("EURUSD", usr_p)
        self.assertIn("advisory_summary", usr_p)

    def test_03_generate_advisory_with_valid_mock(self):
        valid_mock = {
            "timestamp_utc": "2026-11-15T05:00:00Z",
            "pair": "EURUSD",
            "engine_signal": "BUY / LONG",
            "analysis_status": "SUCCESS",
            "market_summary": "EURUSD in low-volatility consolidation.",
            "supporting_evidence": ["Model long probability at 45.4%"],
            "conflicting_evidence": ["Spread elevated at 3.4 pips"],
            "alternative_scenarios": [
                {"scenario": "Range consolidation", "key_condition": "Spread above 2.5 pips"}
            ],
            "risk_context": ["Spread guard blocking entry"],
            "invalidation_conditions": ["Price breaks below 1.1120"],
            "news_data_status": "HEALTHY",
            "data_quality": {"status": "VALID", "limitations": []},
            "uncertainties": ["Weekend gap risk"],
            "advisory_summary": "Advisory only; no trade execution permitted.",
        }

        result = self.service.generate_advisory(self.sample_snapshot, mock_response=valid_mock)
        self.assertEqual(result["analysis_status"], "SUCCESS")
        self.assertEqual(result["pair"], "EURUSD")
        self.assertEqual(result["provider"], "gemini")
        self.assertEqual(result["model_name"], "gemini-2.5-flash")
        self.assertTrue(result["is_advisory_only"])
        self.assertEqual(len(result["supporting_evidence"]), 1)

    def test_04_schema_validation_rejection(self):
        invalid_mock = {
            "pair": "EURUSD",
            "market_summary": "Missing fields",
        }
        res = self.service.generate_advisory(self.sample_snapshot, mock_response=invalid_mock)
        self.assertEqual(res["analysis_status"], "SCHEMA_MISSING_FIELDS")
        self.assertTrue(res["is_advisory_only"])

    def test_05_pair_mismatch_fallback(self):
        mismatch_mock = {
            "timestamp_utc": "2026-11-15T05:00:00Z",
            "pair": "USDJPY",
            "engine_signal": "BUY / LONG",
            "analysis_status": "SUCCESS",
            "market_summary": "Wrong pair returned",
            "supporting_evidence": [],
            "conflicting_evidence": [],
            "alternative_scenarios": [],
            "risk_context": [],
            "invalidation_conditions": [],
            "news_data_status": "HEALTHY",
            "data_quality": {"status": "VALID", "limitations": []},
            "uncertainties": [],
            "advisory_summary": "Wrong pair",
        }
        res = self.service.generate_advisory(self.sample_snapshot, mock_response=mismatch_mock)
        self.assertEqual(res["analysis_status"], "PAIR_MISMATCH")

    def test_06_missing_api_key_fallback(self):
        no_key_service = GeminiService(api_key="")
        res = no_key_service.generate_advisory(self.sample_snapshot)
        self.assertEqual(res["analysis_status"], "GEMINI_KEY_MISSING")
        self.assertTrue(res["is_advisory_only"])

    @patch("requests.post")
    def test_07_gemini_api_timeout(self, mock_post):
        import requests
        mock_post.side_effect = requests.exceptions.Timeout("Connection timed out")
        res = self.service.generate_advisory(self.sample_snapshot)
        self.assertEqual(res["analysis_status"], "TIMEOUT")
        self.assertIn("timed out", res["market_summary"])

    @patch("requests.post")
    def test_08_chat_copilot_success(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {"text": "Current portfolio exposure is balanced across 4 pairs."}
                        ]
                    }
                }
            ]
        }
        mock_post.return_value = mock_resp

        res = self.service.chat_copilot(
            query="What is my risk exposure?",
            portfolio_context={"balance": 100000.0, "drawdown_pct": 1.44},
        )
        self.assertEqual(res["status"], "SUCCESS")
        self.assertIn("exposure is balanced", res["reply"])
        self.assertEqual(res["provider"], "gemini")


if __name__ == "__main__":
    unittest.main()
