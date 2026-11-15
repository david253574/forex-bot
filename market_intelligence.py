"""
Market Intelligence Orchestrator & Advisory Manager.

Coordinates MarketContextBuilder with GeminiService (Google Gemini API)
and OllamaService (local fallback) via a non-blocking bounded worker.
Maintains atomic advisory cache for dashboard presentation.
Completely decoupled and isolated from trade execution.
"""

from __future__ import annotations

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

        # Provider selection: "gemini", "ollama", or "auto"
        env_provider = os.environ.get("AI_PROVIDER", "").strip().lower()
        cfg_provider = str(ai_cfg.get("provider", "")).strip().lower()

        # Setup Ollama configuration
        ollama_subcfg = ai_cfg.get("ollama", {}) if "ollama" in ai_cfg else cfg.get("ollama_agent", ai_cfg)
        ollama_base_url = ollama_subcfg.get("base_url", "http://localhost:11434")
        ollama_model = ollama_subcfg.get("model", "qwen2.5:1.5b")
        ollama_timeout = float(ollama_subcfg.get("timeout_seconds", 30.0))
        ollama_temp = float(ollama_subcfg.get("temperature", 0.2))

        self.ollama = OllamaService(
            base_url=ollama_base_url,
            model_name=ollama_model,
            timeout_seconds=ollama_timeout,
            temperature=ollama_temp,
        )

        # Setup Gemini configuration
        gemini_subcfg = ai_cfg.get("gemini", {}) if "gemini" in ai_cfg else ai_cfg
        gemini_key = gemini_subcfg.get("api_key") or load_env_gemini_key()
        gemini_model = gemini_subcfg.get("model") or os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
        gemini_timeout = float(gemini_subcfg.get("timeout_seconds", 25.0))
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
        elif env_provider in ("gemini", "ollama"):
            self.provider = env_provider
        elif cfg_provider in ("gemini", "ollama"):
            self.provider = cfg_provider
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
        """Returns the active AI service instance (GeminiService or OllamaService)."""
        if self.provider == "gemini":
            return self.gemini
        return self.ollama

    def evaluate_pair(self, pair: str, mock_response: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Executes advisory evaluation for a single pair without blocking trade loops.
        Thread-safe, read-only, and fully isolated.
        """
        clean_pair = pair.upper().replace("/", "").replace("_", "")
        service = self.active_service

        try:
            snapshot = self.context_builder.build_snapshot(clean_pair)
        except Exception as e:
            fallback = service._fallback_response(
                clean_pair,
                "UNKNOWN",
                datetime.now(timezone.utc).isoformat(),
                f"Snapshot build failed: {e}",
                "SNAPSHOT_ERROR",
            )
            with self._lock:
                self._advisories[clean_pair] = fallback
            self._save_advisory_cache()
            return fallback

        advisory = service.generate_advisory(snapshot, mock_response=mock_response)
        advisory["disclaimer"] = "AI MARKET INTELLIGENCE — ADVISORY ONLY. Never interpreted as order or trading permission."

        with self._lock:
            self._advisories[clean_pair] = advisory
        self._save_advisory_cache()
        return advisory

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
        """Provides interactive AI copilot analysis using the active Gemini engine."""
        return self.gemini.chat_copilot(query, portfolio_context=portfolio_context, history=history)

    def get_service_status(self) -> Dict[str, Any]:
        """Returns diagnostic and health status of advisory subsystem."""
        hw = self.ollama.inspect_hardware()
        is_gemini = self.provider == "gemini"
        active_service = self.active_service
        is_avail = active_service.is_available()

        installed_ollama = self.ollama.list_installed_models() if self.ollama.is_available() else []

        with self._lock:
            advisory_count = len(self._advisories)
            last_eval = max(
                (adv.get("timestamp_utc", "") for adv in self._advisories.values()),
                default="None",
            )

        service_name = (
            "Local Ollama Market Intelligence Agent"
            if self.provider == "ollama"
            else "Google Gemini Market Intelligence Agent"
        )

        return {
            "service_name": service_name,
            "provider": self.provider,
            "is_advisory_only": True,
            "disclaimer": "AI MARKET INTELLIGENCE — ADVISORY ONLY",
            "enabled": self.enabled,
            "is_available": is_avail,
            "gemini_available": self.gemini.is_available(),
            "ollama_available": self.ollama.is_available(),
            "configured_model": active_service.model_name,
            "active_provider": self.provider,
            "model_name": active_service.model_name,
            "base_url": getattr(active_service, "base_url", "https://generativelanguage.googleapis.com"),
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
