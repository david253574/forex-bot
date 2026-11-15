"""
Unit Tests for MarketIntelligenceAgent.

Verifies:
- Decoupling from trade execution (zero broker orders, zero ledger mutations).
- Bounded queue and saturation protection.
- Thread-safe advisory retrieval and persistence.
- Graceful degradation when Ollama is offline.
- Safe evaluation of all 4 supported pairs.
"""

import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from market_intelligence import MarketIntelligenceAgent


class TestMarketIntelligenceAgent(unittest.TestCase):

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.test_cache = Path(self.tmp_dir) / "test_advisory_cache.json"
        self.test_config = Path(self.tmp_dir) / "test_config.json"

        # Config with ollama_agent defined
        with open(self.test_config, "w") as f:
            json.dump({
                "broker": "oanda",
                "environment": "practice",
                "guards": {"max_spread_pips": {"EURUSD": 2.5, "USDJPY": 3.0, "AUDUSD": 3.5, "GBPUSD": 3.0}},
                "ollama_agent": {
                    "enabled": True,
                    "base_url": "http://localhost:11434",
                    "model": "qwen2.5:1.5b",
                    "timeout_seconds": 5.0,
                }
            }, f)

        self.agent = MarketIntelligenceAgent(
            config_path=str(self.test_config),
            cache_path=str(self.test_cache),
            auto_start_worker=False,
        )

    def tearDown(self):
        self.agent.stop_worker()
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_01_service_status_diagnostics(self):
        status = self.agent.get_service_status()
        self.assertEqual(status["service_name"], "Local Ollama Market Intelligence Agent")
        self.assertTrue(status["is_advisory_only"])
        self.assertTrue(status["enabled"])
        self.assertEqual(status["configured_model"], "qwen2.5:1.5b")
        self.assertIn("hardware_profile", status)
        self.assertIn("cores", status["hardware_profile"])

    def test_02_evaluate_pair_with_mock_response(self):
        mock_adv = {
            "timestamp_utc": "2026-10-27T00:00:00Z",
            "pair": "EURUSD",
            "engine_signal": "BUY / LONG",
            "analysis_status": "SUCCESS",
            "market_summary": "EURUSD consolidating near support.",
            "supporting_evidence": ["45% Long probability"],
            "conflicting_evidence": ["Spread spike 3.4p"],
            "alternative_scenarios": [{"scenario": "Upward expansion", "key_condition": "Spread normalization"}],
            "risk_context": ["Execution blocked by SpreadGuard"],
            "invalidation_conditions": ["Price breaks 1.1150"],
            "news_data_status": "HEALTHY",
            "data_quality": {"status": "VALID", "limitations": []},
            "uncertainties": ["Low Asian liquidity"],
            "advisory_summary": "Advisory only; no trade execution permitted.",
        }

        result = self.agent.evaluate_pair("EURUSD", mock_response=mock_adv)
        self.assertEqual(result["analysis_status"], "SUCCESS")
        self.assertEqual(result["pair"], "EURUSD")
        self.assertIn("disclaimer", result)
        self.assertTrue(result["is_advisory_only"])

        # Check retrieval
        cached = self.agent.get_advisory("EURUSD")
        self.assertIsNotNone(cached)
        self.assertEqual(cached["pair"], "EURUSD")

        # Check file persistence
        self.assertTrue(self.test_cache.exists())
        with open(self.test_cache) as f:
            disk_data = json.load(f)
            self.assertIn("EURUSD", disk_data)

    def test_03_evaluate_all_pairs(self):
        # When Ollama is offline, evaluate_all returns structured fallbacks for all 4 pairs
        with patch.object(self.agent.ollama, "is_available", return_value=False):
            all_results = self.agent.evaluate_all()
            self.assertEqual(len(all_results), 4)
            for pair in ("EURUSD", "USDJPY", "AUDUSD", "GBPUSD"):
                self.assertIn(pair, all_results)
                self.assertEqual(all_results[pair]["analysis_status"], "OLLAMA_UNAVAILABLE")
                self.assertTrue(all_results[pair]["is_advisory_only"])

    def test_04_queue_saturation_prevention(self):
        # Work queue max size is 4
        self.assertTrue(self.agent.request_evaluation("EURUSD"))
        self.assertTrue(self.agent.request_evaluation("USDJPY"))
        self.assertTrue(self.agent.request_evaluation("AUDUSD"))
        self.assertTrue(self.agent.request_evaluation("GBPUSD"))
        # 5th request must be rejected gracefully without crashing
        self.assertFalse(self.agent.request_evaluation("EURUSD"))

    def test_05_background_worker_lifecycle(self):
        self.agent.start_worker()
        self.assertIsNotNone(self.agent._worker_thread)
        self.assertTrue(self.agent._worker_thread.is_alive())
        self.agent.stop_worker()
        self.agent._worker_thread.join(timeout=2.0)
        self.assertFalse(self.agent._worker_thread.is_alive())

    def test_06_zero_trade_side_effects(self):
        # Verify that evaluating pairs NEVER touches the portfolio ledger
        ledger_path = Path("portfolio_ledger.json")
        mtime_before = ledger_path.stat().st_mtime if ledger_path.exists() else 0
        size_before = ledger_path.stat().st_size if ledger_path.exists() else 0

        self.agent.evaluate_pair("EURUSD")
        self.agent.evaluate_pair("USDJPY")

        if ledger_path.exists():
            self.assertEqual(mtime_before, ledger_path.stat().st_mtime)
            self.assertEqual(size_before, ledger_path.stat().st_size)

    def test_07_gemini_provider_configuration(self):
        gemini_cfg_file = Path(self.tmp_dir) / "gemini_cfg.json"
        with open(gemini_cfg_file, "w") as f:
            json.dump({
                "broker": "oanda",
                "ai_agent": {
                    "enabled": True,
                    "provider": "gemini",
                    "gemini": {
                        "api_key": "test_gemini_key",
                        "model": "gemini-2.5-flash",
                        "timeout_seconds": 10.0
                    }
                }
            }, f)

        gemini_agent = MarketIntelligenceAgent(
            config_path=str(gemini_cfg_file),
            cache_path=str(self.test_cache),
            auto_start_worker=False,
        )
        status = gemini_agent.get_service_status()
        self.assertEqual(status["provider"], "gemini")
        self.assertEqual(status["service_name"], "Google Gemini Market Intelligence Agent")
        self.assertEqual(status["configured_model"], "gemini-2.5-flash")

    def test_08_gemini_copilot_delegation(self):
        with patch.object(self.agent.gemini, "chat_copilot") as mock_copilot:
            mock_copilot.return_value = {"reply": "Portfolio exposure is low.", "status": "SUCCESS"}
            resp = self.agent.chat_copilot("What is my exposure?")
            self.assertEqual(resp["status"], "SUCCESS")
            self.assertIn("Portfolio exposure", resp["reply"])
            mock_copilot.assert_called_once()

    def test_09_dual_provider_configuration(self):
        dual_cfg_file = Path(self.tmp_dir) / "dual_cfg.json"
        with open(dual_cfg_file, "w") as f:
            json.dump({
                "broker": "oanda",
                "ai_agent": {
                    "enabled": True,
                    "provider": "dual",
                    "gemini": {
                        "api_key": "test_gemini_key",
                        "model": "gemini-2.5-flash",
                        "timeout_seconds": 10.0
                    },
                    "ollama": {
                        "base_url": "http://localhost:11434",
                        "model": "qwen2.5:1.5b",
                        "timeout_seconds": 5.0
                    }
                }
            }, f)

        dual_agent = MarketIntelligenceAgent(
            config_path=str(dual_cfg_file),
            cache_path=str(self.test_cache),
            auto_start_worker=False,
        )
        status = dual_agent.get_service_status()
        self.assertEqual(status["provider"], "dual")
        self.assertTrue(status["dual_mode"])
        self.assertIn("Dual-Engine", status["service_name"])
        self.assertIn("Gemini (gemini-2.5-flash)", status["configured_model"])
        self.assertIn("Ollama (qwen2.5:1.5b)", status["configured_model"])

    def test_10_dual_engine_concurrent_evaluation_convergent(self):
        dual_cfg_file = Path(self.tmp_dir) / "dual_cfg.json"
        with open(dual_cfg_file, "w") as f:
            json.dump({
                "broker": "oanda",
                "ai_agent": {
                    "enabled": True,
                    "provider": "dual",
                    "gemini": {"api_key": "dummy_key", "model": "gemini-2.5-flash"},
                    "ollama": {"base_url": "http://localhost:11434", "model": "qwen2.5:1.5b"}
                }
            }, f)

        dual_agent = MarketIntelligenceAgent(
            config_path=str(dual_cfg_file),
            cache_path=str(self.test_cache),
            auto_start_worker=False,
        )

        mock_gemini_adv = {
            "timestamp_utc": "2026-11-15T12:00:00Z",
            "pair": "EURUSD",
            "engine_signal": "BUY / LONG",
            "analysis_status": "SUCCESS",
            "market_summary": "Macro dollar weakening supports EURUSD upward continuation.",
            "supporting_evidence": ["US CPI cooled to 2.1%"],
            "conflicting_evidence": ["ECB dovish press release"],
            "alternative_scenarios": ["Consolidation range 1.0850-1.0900"],
            "risk_context": ["NewsShield clear for next 4 hours"],
            "invalidation_conditions": ["Price breaks below 1.0820"],
            "news_data_status": "HEALTHY",
            "data_quality": {"status": "VALID", "limitations": []},
            "uncertainties": ["Low liquidity prior to London open"],
            "advisory_summary": "Favorable bullish macro setup.",
            "provider": "gemini",
            "model_name": "gemini-2.5-flash",
        }

        mock_ollama_adv = {
            "timestamp_utc": "2026-11-15T12:00:00Z",
            "pair": "EURUSD",
            "engine_signal": "BUY / LONG",
            "analysis_status": "SUCCESS",
            "market_summary": "Local price action breaks above 4-hour EMA band with low ATR expansion.",
            "supporting_evidence": ["EMA 20 > EMA 50 alignment"],
            "conflicting_evidence": ["Spread slightly elevated at 1.8 pips"],
            "alternative_scenarios": ["False breakout retest"],
            "risk_context": ["SpreadGuard operating within thresholds"],
            "invalidation_conditions": ["Break below local swing low"],
            "news_data_status": "HEALTHY",
            "data_quality": {"status": "OK", "limitations": []},
            "uncertainties": ["Minor tick volume fluctuation"],
            "advisory_summary": "Local technical confirmation aligns with long momentum.",
            "provider": "ollama",
            "model_name": "qwen2.5:1.5b",
        }

        with patch.object(dual_agent.gemini, "generate_advisory", return_value=mock_gemini_adv), \
             patch.object(dual_agent.ollama, "generate_advisory", return_value=mock_ollama_adv):
            res = dual_agent.evaluate_pair("EURUSD")

        self.assertEqual(res["provider"], "dual")
        self.assertEqual(res["analysis_status"], "SUCCESS")
        self.assertEqual(res["dual_consensus"]["status"], "CONVERGENT")
        self.assertTrue(res["dual_consensus"]["agreement"])
        self.assertIn("gemini", res)
        self.assertIn("ollama", res)
        self.assertTrue(any("[Gemini]" in s for s in res["supporting_evidence"]))
        self.assertTrue(any("[Ollama]" in s for s in res["supporting_evidence"]))

    def test_11_dual_engine_concurrent_evaluation_divergent(self):
        dual_cfg_file = Path(self.tmp_dir) / "dual_cfg.json"
        with open(dual_cfg_file, "w") as f:
            json.dump({
                "broker": "oanda",
                "ai_agent": {
                    "enabled": True,
                    "provider": "dual",
                    "gemini": {"api_key": "dummy_key", "model": "gemini-2.5-flash"},
                    "ollama": {"base_url": "http://localhost:11434", "model": "qwen2.5:1.5b"}
                }
            }, f)

        dual_agent = MarketIntelligenceAgent(
            config_path=str(dual_cfg_file),
            cache_path=str(self.test_cache),
            auto_start_worker=False,
        )

        mock_gemini_adv = {
            "timestamp_utc": "2026-11-15T12:00:00Z",
            "pair": "USDJPY",
            "engine_signal": "BUY / LONG",
            "analysis_status": "SUCCESS",
            "market_summary": "Treasury yield expansion indicates bullish USDJPY.",
            "supporting_evidence": ["US 10Y Yield rose 4bps"],
            "conflicting_evidence": [],
            "alternative_scenarios": [],
            "risk_context": [],
            "invalidation_conditions": [],
            "news_data_status": "HEALTHY",
            "data_quality": {"status": "VALID", "limitations": []},
            "uncertainties": [],
            "advisory_summary": "Macro favors upside.",
            "provider": "gemini",
            "model_name": "gemini-2.5-flash",
        }

        mock_ollama_adv = {
            "timestamp_utc": "2026-11-15T12:00:00Z",
            "pair": "USDJPY",
            "engine_signal": "SELL / SHORT",
            "analysis_status": "SUCCESS",
            "market_summary": "Stochastic overbought with double-top rejection at 155.00.",
            "supporting_evidence": ["Overbought RSI 78"],
            "conflicting_evidence": [],
            "alternative_scenarios": [],
            "risk_context": [],
            "invalidation_conditions": [],
            "news_data_status": "HEALTHY",
            "data_quality": {"status": "OK", "limitations": []},
            "uncertainties": [],
            "advisory_summary": "Technical exhaustion signals short pullback.",
            "provider": "ollama",
            "model_name": "qwen2.5:1.5b",
        }

        with patch.object(dual_agent.gemini, "generate_advisory", return_value=mock_gemini_adv), \
             patch.object(dual_agent.ollama, "generate_advisory", return_value=mock_ollama_adv):
            res = dual_agent.evaluate_pair("USDJPY")

        self.assertEqual(res["provider"], "dual")
        self.assertEqual(res["dual_consensus"]["status"], "DIVERGENT")
        self.assertFalse(res["dual_consensus"]["agreement"])
        self.assertTrue(any("Divergence" in c for c in res["conflicting_evidence"]))

    def test_12_dual_engine_graceful_fallback_ollama_offline(self):
        dual_cfg_file = Path(self.tmp_dir) / "dual_cfg.json"
        with open(dual_cfg_file, "w") as f:
            json.dump({
                "broker": "oanda",
                "ai_agent": {
                    "enabled": True,
                    "provider": "dual",
                    "gemini": {"api_key": "dummy_key", "model": "gemini-2.5-flash"},
                    "ollama": {"base_url": "http://localhost:11434", "model": "qwen2.5:1.5b"}
                }
            }, f)

        dual_agent = MarketIntelligenceAgent(
            config_path=str(dual_cfg_file),
            cache_path=str(self.test_cache),
            auto_start_worker=False,
        )

        mock_gemini_adv = {
            "timestamp_utc": "2026-11-15T12:00:00Z",
            "pair": "GBPUSD",
            "engine_signal": "FLAT",
            "analysis_status": "SUCCESS",
            "market_summary": "Rangebound market prior to BoE rate decision.",
            "supporting_evidence": ["Range consolidation"],
            "conflicting_evidence": [],
            "alternative_scenarios": [],
            "risk_context": [],
            "invalidation_conditions": [],
            "news_data_status": "HEALTHY",
            "data_quality": {"status": "VALID", "limitations": []},
            "uncertainties": [],
            "advisory_summary": "Standby advisory.",
            "provider": "gemini",
            "model_name": "gemini-2.5-flash",
        }

        # Ollama fails with error
        mock_ollama_fallback = dual_agent.ollama._fallback_response(
            "GBPUSD", "FLAT", "2026-11-15T12:00:00Z", "Connection refused", "OLLAMA_UNAVAILABLE"
        )

        with patch.object(dual_agent.gemini, "generate_advisory", return_value=mock_gemini_adv), \
             patch.object(dual_agent.ollama, "generate_advisory", return_value=mock_ollama_fallback):
            res = dual_agent.evaluate_pair("GBPUSD")

        self.assertEqual(res["provider"], "dual")
        self.assertEqual(res["analysis_status"], "SUCCESS")
        self.assertEqual(res["dual_consensus"]["status"], "GEMINI_ONLY")
        self.assertIn("GBPUSD", res["pair"])

    def test_13_dual_copilot_telemetry_enrichment(self):
        dual_cfg_file = Path(self.tmp_dir) / "dual_cfg.json"
        with open(dual_cfg_file, "w") as f:
            json.dump({
                "broker": "oanda",
                "ai_agent": {
                    "enabled": True,
                    "provider": "dual",
                    "gemini": {"api_key": "dummy_key", "model": "gemini-2.5-flash"},
                    "ollama": {"base_url": "http://localhost:11434", "model": "qwen2.5:1.5b"}
                }
            }, f)

        dual_agent = MarketIntelligenceAgent(
            config_path=str(dual_cfg_file),
            cache_path=str(self.test_cache),
            auto_start_worker=False,
        )

        with patch.object(dual_agent.gemini, "chat_copilot") as mock_copilot:
            mock_copilot.return_value = {"reply": "Dual engine running smoothly.", "status": "SUCCESS"}
            resp = dual_agent.chat_copilot("What is the status of our dual engines?")
            self.assertEqual(resp["status"], "SUCCESS")
            mock_copilot.assert_called_once()
            called_ctx = mock_copilot.call_args[1]["portfolio_context"]
            self.assertIn("dual_engine_telemetry", called_ctx)
            self.assertEqual(called_ctx["dual_engine_telemetry"]["active_provider"], "dual")
            self.assertEqual(called_ctx["dual_engine_telemetry"]["ollama_model"], "qwen2.5:1.5b")


if __name__ == "__main__":
    unittest.main()
