"""
Ollama Service Module for Forex AI Local Market Intelligence.

Provides hardware inspection, connection management, structured prompt construction,
and JSON schema validation for local Ollama-powered market analysis.
Completely isolated from trade execution.
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

logger = logging.getLogger("ForexAI.OllamaService")

# JSON schema specification for structured model responses
ADVISORY_SCHEMA_FIELDS = {
    "timestamp_utc": str,
    "pair": str,
    "engine_signal": str,
    "analysis_status": str,
    "market_summary": str,
    "supporting_evidence": list,
    "conflicting_evidence": list,
    "alternative_scenarios": list,
    "risk_context": list,
    "invalidation_conditions": list,
    "news_data_status": str,
    "data_quality": dict,
    "uncertainties": list,
    "advisory_summary": str,
}


def sanitize_text(text: str, max_length: int = 1000) -> str:
    """Escapes and bounds untrusted model output strings."""
    if not isinstance(text, str):
        return ""
    # Strip dangerous characters and bound length
    clean = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)
    return clean[:max_length].strip()


class OllamaService:
    """
    Manages local Ollama API interaction, hardware profiling,
    prompt rendering, and rigorous output validation.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model_name: str = "qwen2.5:1.5b",
        timeout_seconds: float = 30.0,
        temperature: float = 0.2,
        max_tokens: int = 1024,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model_name = os.environ.get("OLLAMA_MODEL") or model_name
        self.timeout_seconds = float(os.environ.get("OLLAMA_TIMEOUT") or timeout_seconds)
        self.temperature = temperature
        self.max_tokens = int(os.environ.get("OLLAMA_MAX_TOKENS") or max_tokens)

    @staticmethod
    def inspect_hardware() -> Dict[str, Any]:
        """
        Inspects host CPU, RAM, swap, and disk space to determine
        the maximum feasible local model size without risking OOM.
        """
        info: Dict[str, Any] = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "cpu_model": "Unknown",
            "cpu_cores": os.cpu_count() or 1,
            "total_ram_mb": 0,
            "available_ram_mb": 0,
            "free_swap_mb": 0,
            "available_disk_gb": 0,
            "gpu_present": False,
            "recommended_max_params": "1.5B",
        }

        # CPU info
        try:
            with open("/proc/cpuinfo", "r") as f:
                for line in f:
                    if line.startswith("model name"):
                        info["cpu_model"] = line.split(":", 1)[1].strip()
                        break
        except Exception:
            pass

        # Memory info
        try:
            with open("/proc/meminfo", "r") as f:
                mem_data = {}
                for line in f:
                    parts = line.split(":")
                    if len(parts) == 2:
                        k = parts[0].strip()
                        v = parts[1].strip().split()[0]
                        mem_data[k] = int(v)
                info["total_ram_mb"] = mem_data.get("MemTotal", 0) // 1024
                info["available_ram_mb"] = mem_data.get("MemAvailable", 0) // 1024
                info["free_swap_mb"] = mem_data.get("SwapFree", 0) // 1024
        except Exception:
            pass

        # Disk space
        try:
            st = os.statvfs("/home")
            info["available_disk_gb"] = round((st.f_bavail * st.f_frsize) / (1024**3), 1)
        except Exception:
            pass

        # GPU detection
        try:
            lspci_out = subprocess.check_output(["lspci"], stderr=subprocess.DEVNULL).decode()
            if "NVIDIA" in lspci_out or "GeForce" in lspci_out or "Radeon" in lspci_out:
                info["gpu_present"] = True
        except Exception:
            pass

        # Memory budget determination
        avail_mb = info["available_ram_mb"]
        if avail_mb < 1500:
            info["recommended_max_params"] = "0.5B"
        elif avail_mb < 3000:
            info["recommended_max_params"] = "1.5B"
        else:
            info["recommended_max_params"] = "3B"

        return info

    def is_available(self) -> bool:
        """Checks if local Ollama daemon is reachable and responding."""
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=2.0)
            return resp.status_code == 200
        except Exception:
            return False

    def list_installed_models(self) -> List[str]:
        """Returns list of models currently present in local Ollama storage."""
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=3.0)
            if resp.status_code == 200:
                data = resp.json()
                models = data.get("models", [])
                return [m.get("name", "") for m in models if m.get("name")]
        except Exception:
            pass
        return []

    def build_advisory_prompt(self, snapshot: Dict[str, Any]) -> Tuple[str, str]:
        """
        Renders system prompt and user prompt for multi-factor market analysis.
        Strictly forbids generating executable trade orders.
        """
        pair = snapshot.get("pair", "UNKNOWN")
        m_data = snapshot.get("market_data", {})
        sig_data = snapshot.get("engine_signal", {})
        pos_data = snapshot.get("position_state", {})
        guards = snapshot.get("risk_guards", {})
        quality = snapshot.get("data_quality", {})

        system_prompt = (
            "You are a neutral, highly disciplined quantitative Market Intelligence Advisory Assistant "
            "for an automated multi-pair Forex trading system.\n"
            "Your role is STRICTLY ADVISORY: analyze evidence, interpret the existing algorithmic signals, "
            "highlight contradictory indicators, identify risk constraints, and outline alternative scenarios.\n"
            "CRITICAL RULES:\n"
            "1. You DO NOT issue trade commands (never command BUY, SELL, or specify position sizes).\n"
            "2. Never fabricate prices, economic events, or indicators not provided in the snapshot.\n"
            "3. Do not invent numerical win probabilities or rank scenarios with arbitrary percentages.\n"
            "4. Compare supporting evidence vs conflicting evidence neutrally.\n"
            "5. If spreads are elevated or guards are blocked, highlight the execution risk.\n"
            "6. You MUST return ONLY a single valid JSON object strictly matching the required schema.\n"
            "No preamble, no markdown backticks, no text outside the JSON."
        )

        user_prompt = f"""Analyze the following market snapshot for currency pair {pair}:

MARKET SNAPSHOT:
{json.dumps(snapshot, indent=2)}

OUTPUT FORMAT REQUIREMENTS:
Return a JSON object with EXACTLY these keys:
{{
  "timestamp_utc": "{snapshot.get('snapshot_timestamp_utc')}",
  "pair": "{pair}",
  "engine_signal": "{sig_data.get('signal')}",
  "analysis_status": "SUCCESS",
  "market_summary": "1-2 sentence neutral summary of current price action and ATR.",
  "supporting_evidence": ["Evidence point 1 consistent with signal", "Evidence point 2"],
  "conflicting_evidence": ["Evidence point 1 disagreeing with signal or showing resistance", "Point 2"],
  "alternative_scenarios": [
    {{"scenario": "Primary thesis (trend continuation/reversal)", "key_condition": "condition to observe"}},
    {{"scenario": "Opposing thesis (reversal/breakdown)", "key_condition": "condition to observe"}}
  ],
  "risk_context": ["Spread condition note", "Drawdown / circuit breaker note", "News event note"],
  "invalidation_conditions": ["Specific level or event that invalidates the engine thesis"],
  "news_data_status": "{guards.get('news_shield', {}).get('cache_status', 'UNKNOWN')}",
  "data_quality": {{
    "status": "{quality.get('status', 'VALID')}",
    "limitations": {json.dumps(quality.get('limitations', []))}
  }},
  "uncertainties": ["Data limitation or ambiguous technical signal"],
  "advisory_summary": "Concise advisory conclusion (advisory only; no trade command)."
}}"""
        return system_prompt, user_prompt

    def generate_advisory(
        self,
        snapshot: Dict[str, Any],
        mock_response: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Executes inference against local Ollama instance with timeout and schema validation.
        If mock_response is provided (e.g. for testing), returns validated mock response.
        """
        pair = snapshot.get("pair", "UNKNOWN")
        sig = snapshot.get("engine_signal", {}).get("signal", "FLAT")
        ts = snapshot.get("snapshot_timestamp_utc", datetime.now(timezone.utc).isoformat())

        # If running in mock test mode
        if mock_response is not None:
            return self._validate_and_sanitize(mock_response, pair, sig, ts)

        if not self.is_available():
            return self._fallback_response(
                pair=pair,
                signal=sig,
                timestamp=ts,
                error="Local Ollama server is not reachable at " + self.base_url,
                status="OLLAMA_UNAVAILABLE",
            )

        system_prompt, user_prompt = self.build_advisory_prompt(snapshot)

        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
            "format": "json",
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
                "num_thread": 4,  # Cap CPU threads to leave headroom
            },
        }

        start_time = time.perf_counter()
        try:
            resp = requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=self.timeout_seconds,
            )
            elapsed_sec = round(time.perf_counter() - start_time, 2)
        except requests.exceptions.Timeout:
            return self._fallback_response(
                pair=pair,
                signal=sig,
                timestamp=ts,
                error=f"Ollama inference timed out after {self.timeout_seconds}s",
                status="TIMEOUT",
            )
        except Exception as e:
            return self._fallback_response(
                pair=pair,
                signal=sig,
                timestamp=ts,
                error=f"Inference connection failed: {e}",
                status="CONNECTION_ERROR",
            )

        if resp.status_code != 200:
            return self._fallback_response(
                pair=pair,
                signal=sig,
                timestamp=ts,
                error=f"Ollama returned HTTP status {resp.status_code}: {resp.text[:200]}",
                status="OLLAMA_ERROR",
            )

        try:
            result_json = resp.json()
            msg_content = result_json.get("message", {}).get("content", "").strip()
            parsed_data = json.loads(msg_content)
        except Exception as e:
            return self._fallback_response(
                pair=pair,
                signal=sig,
                timestamp=ts,
                error=f"Failed to parse model JSON: {e}",
                status="MALFORMED_JSON",
            )

        sanitized = self._validate_and_sanitize(parsed_data, pair, sig, ts)
        sanitized["inference_duration_sec"] = elapsed_sec
        sanitized["model_name"] = self.model_name
        return sanitized

    def _validate_and_sanitize(
        self,
        data: Dict[str, Any],
        expected_pair: str,
        expected_signal: str,
        expected_ts: str,
    ) -> Dict[str, Any]:
        """
        Validates model output against ADVISORY_SCHEMA_FIELDS.
        Ensures proper types, limits string sizes, and guarantees fallback safety.
        """
        if not isinstance(data, dict):
            return self._fallback_response(expected_pair, expected_signal, expected_ts, "Output is not a JSON object", "INVALID_SCHEMA")

        # Required fields check
        missing = [f for f in ADVISORY_SCHEMA_FIELDS if f not in data]
        if missing:
            return self._fallback_response(
                expected_pair,
                expected_signal,
                expected_ts,
                f"Missing required schema fields: {missing}",
                "SCHEMA_MISSING_FIELDS",
            )

        # Enforce consistency with input snapshot
        validated_pair = str(data.get("pair", expected_pair)).upper().replace("/", "")
        if validated_pair != expected_pair:
            return self._fallback_response(
                expected_pair,
                expected_signal,
                expected_ts,
                f"Pair identifier mismatch (got {validated_pair}, expected {expected_pair})",
                "PAIR_MISMATCH",
            )

        # Sanitize text arrays
        def clean_list(items: Any, max_items: int = 8) -> List[str]:
            if not isinstance(items, list):
                return []
            out = []
            for item in items[:max_items]:
                if isinstance(item, str):
                    out.append(sanitize_text(item, 250))
                elif isinstance(item, dict):
                    # Format scenario dicts cleanly
                    scen = sanitize_text(str(item.get("scenario", "")), 150)
                    cond = sanitize_text(str(item.get("key_condition", "")), 150)
                    out.append(f"{scen} (Watch: {cond})" if cond else scen)
            return [x for x in out if x]

        dq = data.get("data_quality", {})
        if not isinstance(dq, dict):
            dq = {"status": "OK", "limitations": []}

        return {
            "timestamp_utc": expected_ts,
            "pair": expected_pair,
            "engine_signal": expected_signal,
            "analysis_status": "SUCCESS",
            "market_summary": sanitize_text(str(data.get("market_summary", "")), 400),
            "supporting_evidence": clean_list(data.get("supporting_evidence")),
            "conflicting_evidence": clean_list(data.get("conflicting_evidence")),
            "alternative_scenarios": clean_list(data.get("alternative_scenarios")),
            "risk_context": clean_list(data.get("risk_context")),
            "invalidation_conditions": clean_list(data.get("invalidation_conditions")),
            "news_data_status": sanitize_text(str(data.get("news_data_status", "UNKNOWN")), 50),
            "data_quality": {
                "status": sanitize_text(str(dq.get("status", "OK")), 30),
                "limitations": clean_list(dq.get("limitations")),
            },
            "uncertainties": clean_list(data.get("uncertainties")),
            "advisory_summary": sanitize_text(str(data.get("advisory_summary", "")), 400),
            "is_advisory_only": True,
        }

    def _fallback_response(
        self,
        pair: str,
        signal: str,
        timestamp: str,
        error: str,
        status: str,
    ) -> Dict[str, Any]:
        """Provides a safe, non-crashing structured fallback when Ollama is unavailable or fails."""
        return {
            "timestamp_utc": timestamp,
            "pair": pair,
            "engine_signal": signal,
            "analysis_status": status,
            "market_summary": f"Advisory evaluation unavailable: {error}",
            "supporting_evidence": [],
            "conflicting_evidence": [],
            "alternative_scenarios": [],
            "risk_context": ["Local LLM advisory service is offline or timed out; trading engine operating independently."],
            "invalidation_conditions": [],
            "news_data_status": "UNAVAILABLE",
            "data_quality": {
                "status": "FALLBACK",
                "limitations": [f"Failure reason: {error}"],
            },
            "uncertainties": ["Advisory intelligence offline."],
            "advisory_summary": "No advisory available. Engine trading decisions remain strictly guarded by PyTorch policy and risk guards.",
            "is_advisory_only": True,
            "error_detail": error,
        }
