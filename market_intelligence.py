"""
Market Intelligence Orchestrator & Advisory Manager.

Coordinates MarketContextBuilder with GeminiService (Google Gemini API)
and OllamaService (local fallback) via a non-blocking bounded worker.
Maintains atomic advisory cache for dashboard presentation.
Completely decoupled and isolated from trade execution.
"""

from __future__ import annotations

import concurrent.futures
import json
import logging
import os
import queue
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from gemini_service import GeminiService, load_env_gemini_key
from market_context_builder import MarketContextBuilder
from ollama_service import OllamaService

logger = logging.getLogger("ForexAI.MarketIntelligence")

DEFAULT_ADVISORY_CACHE = "advisory_cache.json"


class MarketIntelligenceAgent:
    """
    Manages AI market advisory evaluation across the 4 currency pairs.
    Operates as a background service with a bounded queue and strict timeouts.
    Supports both Google Gemini API (cloud, fast, rich) and Ollama (local offline).
    Never alters orders, risk settings, or ledger balances.
    """

    def __init__(
        self,
        config_path: str = "broker_config.json",
        cache_path: str = DEFAULT_ADVISORY_CACHE,
        auto_start_worker: bool = False,
    ) -> None:
        self.config_path = Path(config_path)
        self.cache_path = Path(cache_path)
        self._lock = threading.Lock()
        self._work_queue: queue.Queue = queue.Queue(maxsize=4)
        self._worker_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        # Load config
        cfg = self._load_config()

        # Check for modern ai_agent or legacy ollama_agent
        ai_cfg = cfg.get("ai_agent") or cfg.get("gemini_agent") or cfg.get("ollama_agent", {})
        has_explicit_ollama_only = "ollama_agent" in cfg and "ai_agent" not in cfg and "gemini_agent" not in cfg

        self.enabled = ai_cfg.get("enabled", True if ("ai_agent" in cfg or "gemini_agent" in cfg) else False)

        # Provider selection: "gemini", "ollama", "dual" / "hybrid", or "auto"
        env_provider = os.environ.get("AI_PROVIDER", "").strip().lower()
        cfg_provider = str(ai_cfg.get("provider", "")).strip().lower()

        # Setup Ollama configuration
        ollama_subcfg = ai_cfg.get("ollama", {}) if "ollama" in ai_cfg else cfg.get("ollama_agent", ai_cfg)
        ollama_base_url = os.environ.get("OLLAMA_BASE_URL") or ollama_subcfg.get("base_url", "http://localhost:11434")
        ollama_model = os.environ.get("OLLAMA_MODEL") or ollama_subcfg.get("model", "qwen2.5:1.5b")
        ollama_timeout = float(os.environ.get("OLLAMA_TIMEOUT") or ollama_subcfg.get("timeout_seconds", 35.0))
        ollama_temp = float(ollama_subcfg.get("temperature", 0.2))
        ollama_max_tokens = int(os.environ.get("OLLAMA_MAX_TOKENS") or ollama_subcfg.get("max_tokens", 350))

        self.ollama = OllamaService(
            base_url=ollama_base_url,
            model_name=ollama_model,
            timeout_seconds=ollama_timeout,
            temperature=ollama_temp,
            max_tokens=ollama_max_tokens,
        )

        # Setup Gemini configuration
        gemini_subcfg = ai_cfg.get("gemini", {}) if "gemini" in ai_cfg else ai_cfg
        gemini_key = gemini_subcfg.get("api_key") or load_env_gemini_key()
        gemini_model = os.environ.get("GEMINI_MODEL") or gemini_subcfg.get("model") or "gemini-2.5-flash"
        gemini_timeout = float(os.environ.get("GEMINI_TIMEOUT") or gemini_subcfg.get("timeout_seconds", 25.0))
        gemini_temp = float(gemini_subcfg.get("temperature", 0.2))

        self.gemini = GeminiService(
            api_key=gemini_key,
            model_name=gemini_model,
            timeout_seconds=gemini_timeout,
            temperature=gemini_temp,
        )

        # Resolve active provider
        if has_explicit_ollama_only:
            # Preserve exact backward compatibility with test suites configuring ollama_agent
            self.provider = "ollama"
        elif cfg_provider in ("dual", "hybrid", "both"):
            self.provider = "dual"
        elif cfg_provider in ("gemini", "ollama"):
            self.provider = cfg_provider
        elif env_provider in ("dual", "hybrid", "both"):
            self.provider = "dual"
        elif env_provider in ("gemini", "ollama"):
            self.provider = env_provider
        elif self.gemini.api_key and self.ollama.is_available():
            self.provider = "dual"
        elif self.gemini.api_key:
            self.provider = "gemini"
        elif self.ollama.is_available():
            self.provider = "ollama"
        else:
            self.provider = "gemini" if self.gemini.api_key else "ollama"

        self.context_builder = MarketContextBuilder(config_path=str(self.config_path))
        self._advisories: Dict[str, Dict[str, Any]] = self._load_advisory_cache()

        if auto_start_worker and self.enabled:
            self.start_worker()

    def _load_config(self) -> Dict[str, Any]:
        if not self.config_path.exists():
            return {}
        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}

    def _load_advisory_cache(self) -> Dict[str, Dict[str, Any]]:
        """Loads previously saved advisory cache safely."""
        if not self.cache_path.exists():
            return {}
        try:
            with open(self.cache_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    def _save_advisory_cache(self) -> None:
        """Atomically saves in-memory advisories to disk."""
        with self._lock:
            try:
                tmp = self.cache_path.with_suffix(".tmp")
                with open(tmp, "w", encoding="utf-8") as f:
                    json.dump(self._advisories, f, indent=2)
                tmp.replace(self.cache_path)
            except Exception as e:
                logger.warning(f"Could not save advisory cache to {self.cache_path}: {e}")

    def start_worker(self) -> None:
        """Starts background evaluation thread."""
        with self._lock:
            if self._worker_thread and self._worker_thread.is_alive():
                return
            self._stop_event.clear()
            self._worker_thread = threading.Thread(
                target=self._worker_loop,
                name="MarketIntelligenceWorker",
                daemon=True,
            )
            self._worker_thread.start()

    def stop_worker(self) -> None:
        """Stops background evaluation thread cleanly."""
        self._stop_event.set()
        try:
            self._work_queue.put_nowait(None)
        except Exception:
            pass

    def _worker_loop(self) -> None:
        """Bounded single-worker loop processing requested pairs."""
        while not self._stop_event.is_set():
            try:
                item = self._work_queue.get(timeout=1.0)
                if item is None or self._stop_event.is_set():
                    break
                pair = str(item)
                self.evaluate_pair(pair)
                self._work_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error in advisory worker loop: {e}")

    @property
    def active_service(self) -> Any:
        """Returns the primary active AI service instance."""
        if self.provider == "gemini":
            return self.gemini
        if self.provider == "dual":
            return self.gemini
        return self.ollama

    def evaluate_pair(self, pair: str, mock_response: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Executes advisory evaluation for a single pair without blocking trade loops.
        Thread-safe, read-only, and fully isolated.
        Supports single-engine (Gemini or Ollama) and concurrent Dual-Engine modes.
        """
        clean_pair = pair.upper().replace("/", "").replace("_", "")

        try:
            snapshot = self.context_builder.build_snapshot(clean_pair)
        except Exception as e:
            service = self.active_service
            fallback = service._fallback_response(
                clean_pair,
                "UNKNOWN",
                datetime.now(timezone.utc).isoformat(),
                f"Snapshot build failed: {e}",
                "SNAPSHOT_ERROR",
            )
            fallback["provider"] = self.provider
            with self._lock:
                self._advisories[clean_pair] = fallback
            self._save_advisory_cache()
            return fallback

        if self.provider == "dual":
            advisory = self._evaluate_pair_dual(clean_pair, snapshot, mock_response=mock_response)
        else:
            service = self.active_service
            advisory = service.generate_advisory(snapshot, mock_response=mock_response)

        advisory["disclaimer"] = "AI MARKET INTELLIGENCE — ADVISORY ONLY. Never interpreted as order or trading permission."

        with self._lock:
            self._advisories[clean_pair] = advisory
        self._save_advisory_cache()
        return advisory

    def _evaluate_pair_dual(
        self,
        clean_pair: str,
        snapshot: Dict[str, Any],
        mock_response: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Runs Gemini (cloud) and Ollama (local) concurrently via ThreadPoolExecutor.
        Synthesizes findings into a unified Dual-Engine advisory with consensus tracking.
        """
        mock_gemini = None
        mock_ollama = None
        if isinstance(mock_response, dict):
            if "gemini" in mock_response or "ollama" in mock_response:
                mock_gemini = mock_response.get("gemini")
                mock_ollama = mock_response.get("ollama")
            else:
                mock_gemini = mock_response
                mock_ollama = mock_response

        with concurrent.futures.ThreadPoolExecutor(max_workers=2, thread_name_prefix="DualAI") as executor:
            fut_gemini = executor.submit(self.gemini.generate_advisory, snapshot, mock_gemini)
            fut_ollama = executor.submit(self.ollama.generate_advisory, snapshot, mock_ollama)

            try:
                gemini_res = fut_gemini.result()
            except Exception as e:
                logger.warning(f"Dual Gemini evaluation error for {clean_pair}: {e}")
                gemini_res = self.gemini._fallback_response(
                    clean_pair,
                    snapshot.get("engine_signal", {}).get("signal", "FLAT"),
                    datetime.now(timezone.utc).isoformat(),
                    f"Gemini execution exception: {e}",
                    "GEMINI_ERROR",
                )

            try:
                ollama_res = fut_ollama.result()
            except Exception as e:
                logger.warning(f"Dual Ollama evaluation error for {clean_pair}: {e}")
                ollama_res = self.ollama._fallback_response(
                    clean_pair,
                    snapshot.get("engine_signal", {}).get("signal", "FLAT"),
                    datetime.now(timezone.utc).isoformat(),
                    f"Ollama execution exception: {e}",
                    "OLLAMA_ERROR",
                )

        return self._synthesize_dual_advisory(clean_pair, gemini_res, ollama_res, snapshot)

    def _synthesize_dual_advisory(
        self,
        clean_pair: str,
        gemini_res: Dict[str, Any],
        ollama_res: Dict[str, Any],
        snapshot: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Merges cloud macro insights from Gemini with local technical interpretations from Ollama.
        Determines consensus (CONVERGENT vs DIVERGENT) and provides graceful fallback continuity.
        """
        ts = datetime.now(timezone.utc).isoformat()
        g_ok = gemini_res.get("analysis_status") == "SUCCESS"
        o_ok = ollama_res.get("analysis_status") == "SUCCESS"

        g_sig = (gemini_res.get("engine_signal") or "FLAT").strip().upper()
        o_sig = (ollama_res.get("engine_signal") or "FLAT").strip().upper()

        def get_bias(sig: str) -> str:
            if "LONG" in sig or "BUY" in sig:
                return "BULLISH"
            if "SHORT" in sig or "SELL" in sig:
                return "BEARISH"
            return "NEUTRAL"

        g_bias = get_bias(g_sig)
        o_bias = get_bias(o_sig)

        if g_ok and o_ok:
            if g_bias == o_bias:
                consensus_status = "CONVERGENT"
                agreement = True
                consensus_detail = f"Cloud & Local models converge: Both affirm {g_bias} bias ({g_sig})."
            else:
                consensus_status = "DIVERGENT"
                agreement = False
                consensus_detail = f"Model divergence: Gemini sees {g_bias} ({g_sig}) vs Ollama {o_bias} ({o_sig})."
        elif g_ok:
            consensus_status = "GEMINI_ONLY"
            agreement = True
            consensus_detail = f"Cloud Gemini active ({self.gemini.model_name}); Local Ollama unreachable or timed out."
        elif o_ok:
            consensus_status = "OLLAMA_ONLY"
            agreement = True
            consensus_detail = f"Local Ollama active ({self.ollama.model_name}); Cloud Gemini unreachable."
        else:
            consensus_status = "DUAL_OFFLINE"
            agreement = False
            consensus_detail = "Both cloud Gemini and local Ollama engines are unavailable."

        # Select primary signal
        if g_ok:
            primary_signal = gemini_res.get("engine_signal", "FLAT")
        elif o_ok:
            primary_signal = ollama_res.get("engine_signal", "FLAT")
        else:
            primary_signal = snapshot.get("engine_signal", {}).get("signal", "FLAT")

        # Synthesize narrative summary
        if g_ok and o_ok:
            market_summary = (
                f"[Cloud Gemini]: {gemini_res.get('market_summary', '')} "
                f"[Local Ollama]: {ollama_res.get('market_summary', '')}"
            ).strip()
        elif g_ok:
            market_summary = gemini_res.get("market_summary", "")
        else:
            market_summary = ollama_res.get("market_summary", "")

        # Supporting evidence merging
        supp = []
        if g_ok:
            supp.extend([f"[Gemini] {s}" if not s.startswith("[") else s for s in gemini_res.get("supporting_evidence", [])])
        if o_ok:
            supp.extend([f"[Ollama] {s}" if not s.startswith("[") else s for s in ollama_res.get("supporting_evidence", [])])

        # Conflicting evidence merging
        conf = []
        if consensus_status == "DIVERGENT":
            conf.append(f"Model Divergence: Cloud Gemini ({g_sig}) conflicts with Local Ollama ({o_sig}).")
        if g_ok:
            conf.extend([f"[Gemini] {c}" if not c.startswith("[") else c for c in gemini_res.get("conflicting_evidence", [])])
        if o_ok:
            conf.extend([f"[Ollama] {c}" if not c.startswith("[") else c for c in ollama_res.get("conflicting_evidence", [])])

        # Alternative scenarios
        scenarios = []
        if g_ok:
            scenarios.extend(gemini_res.get("alternative_scenarios", []))
        if o_ok:
            for sc in ollama_res.get("alternative_scenarios", []):
                if sc not in scenarios:
                    scenarios.append(sc)

        # Risk context
        risks = []
        if g_ok:
            risks.extend(gemini_res.get("risk_context", []))
        if o_ok:
            for r in ollama_res.get("risk_context", []):
                if r not in risks:
                    risks.append(r)

        # Invalidation conditions
        invalidation = []
        if g_ok:
            invalidation.extend(gemini_res.get("invalidation_conditions", []))
        if o_ok:
            for inv in ollama_res.get("invalidation_conditions", []):
                if inv not in invalidation:
                    invalidation.append(inv)

        # Uncertainties
        uncertainties = []
        if g_ok:
            uncertainties.extend(gemini_res.get("uncertainties", []))
        if o_ok:
            for u in ollama_res.get("uncertainties", []):
                if u not in uncertainties:
                    uncertainties.append(u)

        # Advisory summary
        if g_ok and o_ok:
            adv_summary = (
                f"Dual-Engine Advisory [{consensus_status}]. "
                f"Gemini: {gemini_res.get('advisory_summary', '')} | "
                f"Ollama: {ollama_res.get('advisory_summary', '')}"
            ).strip()
        elif g_ok:
            adv_summary = gemini_res.get("advisory_summary", "")
        else:
            adv_summary = ollama_res.get("advisory_summary", "")

        analysis_status = "SUCCESS" if (g_ok or o_ok) else "DUAL_UNAVAILABLE"

        return {
            "timestamp_utc": ts,
            "pair": clean_pair,
            "engine_signal": primary_signal,
            "analysis_status": analysis_status,
            "provider": "dual",
            "model_name": f"Gemini ({self.gemini.model_name}) + Ollama ({self.ollama.model_name})",
            "dual_consensus": {
                "status": consensus_status,
                "detail": consensus_detail,
                "agreement": agreement,
                "gemini_signal": g_sig,
                "ollama_signal": o_sig,
                "gemini_status": gemini_res.get("analysis_status", "UNKNOWN"),
                "ollama_status": ollama_res.get("analysis_status", "UNKNOWN"),
            },
            "gemini": gemini_res,
            "ollama": ollama_res,
            "market_summary": market_summary,
            "supporting_evidence": supp[:8],
            "conflicting_evidence": conf[:8],
            "alternative_scenarios": scenarios[:6],
            "risk_context": risks[:6],
            "invalidation_conditions": invalidation[:6],
            "news_data_status": gemini_res.get("news_data_status") if g_ok else ollama_res.get("news_data_status", "UNKNOWN"),
            "data_quality": gemini_res.get("data_quality") if g_ok else ollama_res.get("data_quality", {"status": "OK", "limitations": []}),
            "uncertainties": uncertainties[:6],
            "advisory_summary": adv_summary,
            "is_advisory_only": True,
        }

    def evaluate_all(self, mock_responses: Optional[Dict[str, Dict[str, Any]]] = None) -> Dict[str, Dict[str, Any]]:
        """Evaluates all 4 supported currency pairs sequentially."""
        results = {}
        for pair in MarketContextBuilder.SUPPORTED_PAIRS:
            m_resp = mock_responses.get(pair) if mock_responses else None
            results[pair] = self.evaluate_pair(pair, mock_response=m_resp)
        return results

    def request_evaluation(self, pair: Optional[str] = None) -> bool:
        """
        Non-blocking request to queue evaluation for a pair or all pairs.
        Returns False if queue is saturated.
        """
        pairs = [pair.upper()] if pair else list(MarketContextBuilder.SUPPORTED_PAIRS)
        for p in pairs:
            try:
                self._work_queue.put_nowait(p)
            except queue.Full:
                logger.warning(f"Advisory evaluation queue saturated. Dropped request for {p}.")
                return False
        return True

    def get_advisory(self, pair: str) -> Optional[Dict[str, Any]]:
        """Thread-safe retrieval of cached advisory for a single pair."""
        clean_pair = pair.upper().replace("/", "").replace("_", "")
        with self._lock:
            return self._advisories.get(clean_pair)

    def get_all_advisories(self) -> Dict[str, Any]:
        """Thread-safe retrieval of all cached advisories."""
        with self._lock:
            return dict(self._advisories)

    def chat_copilot(
        self,
        query: str,
        portfolio_context: Optional[Dict[str, Any]] = None,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        """Provides interactive AI copilot analysis using the active Gemini engine, enriched with Ollama telemetry."""
        ctx = dict(portfolio_context or {})
        ctx["dual_engine_telemetry"] = {
            "active_provider": self.provider,
            "gemini_model": self.gemini.model_name,
            "gemini_available": self.gemini.is_available(),
            "ollama_model": self.ollama.model_name,
            "ollama_available": self.ollama.is_available(),
            "installed_local_models": self.ollama.list_installed_models() if self.ollama.is_available() else [],
            "hardware": self.ollama.inspect_hardware(),
        }
        return self.gemini.chat_copilot(query, portfolio_context=ctx, history=history)

    def get_service_status(self) -> Dict[str, Any]:
        """Returns diagnostic and health status of advisory subsystem."""
        hw = self.ollama.inspect_hardware()
        gemini_avail = self.gemini.is_available()
        ollama_avail = self.ollama.is_available()
        is_dual = self.provider == "dual"

        if is_dual:
            service_name = "Dual-Engine (Google Gemini + Local Ollama) Intelligence Agent"
            is_avail = gemini_avail or ollama_avail
            configured_model = f"Gemini ({self.gemini.model_name}) + Ollama ({self.ollama.model_name})"
            model_name = f"{self.gemini.model_name} + {self.ollama.model_name}"
            base_url = f"Cloud: {self.gemini.base_url} | Local: {self.ollama.base_url}"
        elif self.provider == "gemini":
            service_name = "Google Gemini Market Intelligence Agent"
            is_avail = gemini_avail
            configured_model = self.gemini.model_name
            model_name = self.gemini.model_name
            base_url = self.gemini.base_url
        else:
            service_name = "Local Ollama Market Intelligence Agent"
            is_avail = ollama_avail
            configured_model = self.ollama.model_name
            model_name = self.ollama.model_name
            base_url = self.ollama.base_url

        installed_ollama = self.ollama.list_installed_models() if ollama_avail else []

        with self._lock:
            advisory_count = len(self._advisories)
            last_eval = max(
                (adv.get("timestamp_utc", "") for adv in self._advisories.values()),
                default="None",
            )

        return {
            "service_name": service_name,
            "provider": self.provider,
            "is_advisory_only": True,
            "disclaimer": "AI MARKET INTELLIGENCE — ADVISORY ONLY",
            "enabled": self.enabled,
            "is_available": is_avail,
            "dual_mode": is_dual,
            "gemini_available": gemini_avail,
            "ollama_available": ollama_avail,
            "configured_model": configured_model,
            "active_provider": self.provider,
            "model_name": model_name,
            "base_url": base_url,
            "installed_models": installed_ollama,
            "queue_size": self._work_queue.qsize(),
            "cached_advisories_count": advisory_count,
            "last_evaluation_utc": last_eval,
            "hardware_profile": {
                "cpu": hw.get("cpu_model"),
                "cores": hw.get("cpu_cores"),
                "available_ram_mb": hw.get("available_ram_mb"),
                "free_swap_mb": hw.get("free_swap_mb"),
                "recommended_max_params": hw.get("recommended_max_params"),
            },
        }
