"""
Gemini Service Module for Forex AI Market Intelligence & Copilot.

Provides cloud-scale, high-fidelity market intelligence and interactive
copilot capabilities via Google's Gemini API (e.g. gemini-2.5-flash).
Adheres strictly to zero-execution advisory boundaries and schema compliance.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

logger = logging.getLogger("ForexAI.GeminiService")

# Schema specification for structured advisory outputs (must match system specification)
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


def sanitize_text(text: str, max_length: int = 1500) -> str:
    """Escapes and bounds untrusted model output strings."""
    if not isinstance(text, str):
        return ""
    clean = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)
    return clean[:max_length].strip()


def load_env_gemini_key() -> str:
    """Retrieves GEMINI_API_KEY from environment or .env file."""
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if key and key.strip():
        return key.strip()

    # Search project .env
    for candidate in [Path(".env"), Path(__file__).resolve().parent / ".env"]:
        if candidate.exists():
            try:
                with open(candidate, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith("#"):
                            continue
                        if "=" in line:
                            k, v = line.split("=", 1)
                            k = k.strip()
                            v = v.strip().strip("'\"")
                            if k in ("GEMINI_API_KEY", "GOOGLE_API_KEY") and v:
                                return v
            except Exception:
                pass
    return ""


class GeminiService:
    """
    Manages Google Gemini API interactions for quantitative forex advisory
    and real-time market copilot queries. Completely decoupled from trade execution.
    """

    DEFAULT_MODEL = "gemini-2.5-flash"

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        timeout_seconds: float = 25.0,
        temperature: float = 0.2,
    ) -> None:
        if api_key is None:
            api_key = load_env_gemini_key()
        self.api_key = (api_key or "").strip()
        self.model_name = (model_name or os.environ.get("GEMINI_MODEL") or self.DEFAULT_MODEL).strip()
        self.timeout_seconds = float(timeout_seconds)
        self.temperature = float(temperature)
        self.base_url = "https://generativelanguage.googleapis.com/v1beta"

    def is_available(self) -> bool:
        """Checks if Gemini API key is configured and can reach the service."""
        if not self.api_key:
            return False
        try:
            url = f"{self.base_url}/models/{self.model_name}?key={self.api_key}"
            resp = requests.get(url, timeout=5.0)
            return resp.status_code == 200
        except Exception as e:
            logger.debug(f"Gemini API check failed: {e}")
            return False

    def build_advisory_prompt(self, snapshot: Dict[str, Any]) -> Tuple[str, str]:
        """Renders system prompt and structured prompt for market snapshot analysis."""
        pair = snapshot.get("pair", "UNKNOWN")
        sig_data = snapshot.get("engine_signal", {})
        guards = snapshot.get("risk_guards", {})
        quality = snapshot.get("data_quality", {})

        system_prompt = (
            "You are a neutral, highly disciplined quantitative Market Intelligence Advisory Assistant "
            "for an automated multi-pair Forex trading system.\n"
            "Your role is STRICTLY ADVISORY: analyze evidence, interpret algorithmic signals, "
            "highlight contradictory indicators, identify risk constraints, and outline alternative scenarios.\n"
            "CRITICAL RULES:\n"
            "1. You DO NOT issue trade commands (never command BUY, SELL, or specify position sizes).\n"
            "2. Never fabricate prices, economic events, or indicators not provided in the snapshot.\n"
            "3. Do not invent numerical win probabilities or rank scenarios with arbitrary percentages.\n"
            "4. Compare supporting evidence vs conflicting evidence neutrally.\n"
            "5. If spreads are elevated or guards are blocked, highlight the execution risk clearly.\n"
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
  "market_summary": "1-2 sentence neutral summary of current price action, ATR, and immediate structure.",
  "supporting_evidence": ["Evidence point 1 consistent with signal or trend bias", "Evidence point 2"],
  "conflicting_evidence": ["Evidence point 1 disagreeing with signal, resistance, or spread friction", "Point 2"],
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
        Generates structured multi-factor market intelligence advisory using Gemini.
        Validates schema, bounds string lengths, and ensures non-execution disclaimers.
        """
        pair = snapshot.get("pair", "UNKNOWN")
        sig = snapshot.get("engine_signal", {}).get("signal", "FLAT")
        ts = snapshot.get("snapshot_timestamp_utc", datetime.now(timezone.utc).isoformat())

        if mock_response is not None:
            return self._validate_and_sanitize(mock_response, pair, sig, ts)

        if not self.api_key:
            return self._fallback_response(
                pair=pair,
                signal=sig,
                timestamp=ts,
                error="GEMINI_API_KEY is not configured.",
                status="GEMINI_KEY_MISSING",
            )

        system_prompt, user_prompt = self.build_advisory_prompt(snapshot)

        payload = {
            "system_instruction": {"parts": [{"text": system_prompt}]},
            "contents": [{"parts": [{"text": user_prompt}]}],
            "generationConfig": {
                "responseMimeType": "application/json",
                "temperature": self.temperature,
                "thinkingConfig": {"thinkingBudget": 0},
            },
        }

        url = f"{self.base_url}/models/{self.model_name}:generateContent?key={self.api_key}"
        start_time = time.perf_counter()

        try:
            resp = requests.post(url, json=payload, timeout=self.timeout_seconds)
            elapsed_sec = round(time.perf_counter() - start_time, 2)
        except requests.exceptions.Timeout:
            return self._fallback_response(
                pair=pair,
                signal=sig,
                timestamp=ts,
                error=f"Gemini API request timed out after {self.timeout_seconds}s",
                status="TIMEOUT",
            )
        except Exception as e:
            return self._fallback_response(
                pair=pair,
                signal=sig,
                timestamp=ts,
                error=f"Gemini API connection error: {e}",
                status="CONNECTION_ERROR",
            )

        if resp.status_code != 200:
            return self._fallback_response(
                pair=pair,
                signal=sig,
                timestamp=ts,
                error=f"Gemini API HTTP {resp.status_code}: {resp.text[:250]}",
                status="GEMINI_ERROR",
            )

        try:
            result_json = resp.json()
            candidates = result_json.get("candidates", [])
            if not candidates:
                return self._fallback_response(
                    pair=pair,
                    signal=sig,
                    timestamp=ts,
                    error="Gemini returned no candidates (possible content safety filter).",
                    status="EMPTY_RESPONSE",
                )
            msg_content = candidates[0].get("content", {}).get("parts", [{}])[0].get("text", "").strip()
            parsed_data = json.loads(msg_content)
        except Exception as e:
            return self._fallback_response(
                pair=pair,
                signal=sig,
                timestamp=ts,
                error=f"Failed to parse Gemini JSON output: {e}",
                status="MALFORMED_JSON",
            )

        sanitized = self._validate_and_sanitize(parsed_data, pair, sig, ts)
        sanitized["inference_duration_sec"] = elapsed_sec
        sanitized["provider"] = "gemini"
        sanitized["model_name"] = self.model_name
        return sanitized

    def chat_copilot(
        self,
        query: str,
        portfolio_context: Optional[Dict[str, Any]] = None,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        """
        Interactive Market Intelligence Copilot powered by Gemini.
        Answers user questions using real-time portfolio telemetry, open positions,
        risk parameters, and upcoming macro calendar events.
        """
        if not self.api_key:
            return {
                "reply": "⚠️ Gemini API key is not configured. Please set GEMINI_API_KEY in your environment or .env.",
                "status": "KEY_MISSING",
                "model_name": self.model_name,
            }

        sanitized_query = sanitize_text(query, max_length=1000)
        if not sanitized_query:
            return {"reply": "Please provide a valid question or market prompt.", "status": "EMPTY_QUERY"}

        system_instruction = (
            "You are the Velotrade Quantitative Market & Portfolio AI Copilot, an elite quantitative assistant "
            "embedded in a multi-pair algorithmic Forex trading terminal.\n"
            "You have direct visibility into the user's real-time portfolio telemetry, open positions, neural policy signals, "
            "spread guard status, drawdown limits, and economic calendar events.\n"
            "CRITICAL GUIDELINES:\n"
            "1. ADVISORY ONLY: You do NOT execute or modify orders directly. Never say 'I bought' or 'I sold'.\n"
            "2. Ground all answers strictly in the provided real-time telemetry.\n"
            "3. Be quantitative, precise, concise, and structured. Use bullet points or markdown tables where appropriate.\n"
            "4. If spreads are high or news blackout shields are active, proactively mention risk factors.\n"
            "5. Always maintain a professional, risk-conscious institutional tone."
        )

        context_str = json.dumps(portfolio_context, indent=2) if portfolio_context else "No telemetry context provided."

        user_content = f"""USER QUESTION:
{sanitized_query}

CURRENT REAL-TIME PORTFOLIO & MARKET TELEMETRY:
{context_str}

Respond helpfully and quantitatively to the user."""

        contents: List[Dict[str, Any]] = []
        if history:
            for turn in history[-6:]:  # Keep recent context window
                role = "user" if turn.get("role") == "user" else "model"
                contents.append({"role": role, "parts": [{"text": sanitize_text(turn.get("text", ""), 1000)}]})

        contents.append({"role": "user", "parts": [{"text": user_content}]})

        payload = {
            "system_instruction": {"parts": [{"text": system_instruction}]},
            "contents": contents,
            "generationConfig": {
                "temperature": 0.3,
                "thinkingConfig": {"thinkingBudget": 0},
            },
        }

        url = f"{self.base_url}/models/{self.model_name}:generateContent?key={self.api_key}"
        start_time = time.perf_counter()

        try:
            resp = requests.post(url, json=payload, timeout=self.timeout_seconds)
            elapsed = round(time.perf_counter() - start_time, 2)
            if resp.status_code == 200:
                candidates = resp.json().get("candidates", [])
                reply_text = candidates[0].get("content", {}).get("parts", [{}])[0].get("text", "") if candidates else ""
                return {
                    "reply": reply_text.strip(),
                    "status": "SUCCESS",
                    "elapsed_sec": elapsed,
                    "model_name": self.model_name,
                    "provider": "gemini",
                }
            else:
                return {
                    "reply": f"Gemini API returned HTTP {resp.status_code}: {resp.text[:200]}",
                    "status": "API_ERROR",
                    "elapsed_sec": elapsed,
                }
        except Exception as e:
            return {
                "reply": f"Could not reach Gemini API: {e}",
                "status": "CONNECTION_ERROR",
            }

    def _validate_and_sanitize(
        self,
        data: Dict[str, Any],
        expected_pair: str,
        expected_signal: str,
        expected_ts: str,
    ) -> Dict[str, Any]:
        """Validates structured model output against schema and bounds text."""
        if not isinstance(data, dict):
            return self._fallback_response(
                expected_pair, expected_signal, expected_ts, "Output is not a JSON object", "INVALID_SCHEMA"
            )

        missing = [f for f in ADVISORY_SCHEMA_FIELDS if f not in data]
        if missing:
            return self._fallback_response(
                expected_pair,
                expected_signal,
                expected_ts,
                f"Missing required schema fields: {missing}",
                "SCHEMA_MISSING_FIELDS",
            )

        validated_pair = str(data.get("pair", expected_pair)).upper().replace("/", "").replace("_", "")
        if validated_pair != expected_pair:
            return self._fallback_response(
                expected_pair,
                expected_signal,
                expected_ts,
                f"Pair mismatch (got {validated_pair}, expected {expected_pair})",
                "PAIR_MISMATCH",
            )

        def clean_list(items: Any, max_items: int = 8) -> List[str]:
            if not isinstance(items, list):
                return []
            out = []
            for item in items[:max_items]:
                if isinstance(item, str):
                    out.append(sanitize_text(item, 300))
                elif isinstance(item, dict):
                    scen = sanitize_text(str(item.get("scenario", "")), 200)
                    cond = sanitize_text(str(item.get("key_condition", "")), 200)
                    out.append(f"{scen} (Watch: {cond})" if cond else scen)
            return [x for x in out if x]

        dq = data.get("data_quality", {})
        if not isinstance(dq, dict):
            dq = {"status": "OK", "limitations": []}

        return {
            "timestamp_utc": sanitize_text(str(data.get("timestamp_utc", expected_ts)), 50),
            "pair": expected_pair,
            "engine_signal": sanitize_text(str(data.get("engine_signal", expected_signal)), 40),
            "analysis_status": sanitize_text(str(data.get("analysis_status", "SUCCESS")), 40),
            "market_summary": sanitize_text(str(data.get("market_summary", "")), 600),
            "supporting_evidence": clean_list(data.get("supporting_evidence")),
            "conflicting_evidence": clean_list(data.get("conflicting_evidence")),
            "alternative_scenarios": clean_list(data.get("alternative_scenarios")),
            "risk_context": clean_list(data.get("risk_context")),
            "invalidation_conditions": clean_list(data.get("invalidation_conditions")),
            "news_data_status": sanitize_text(str(data.get("news_data_status", "UNKNOWN")), 30),
            "data_quality": {
                "status": sanitize_text(str(dq.get("status", "VALID")), 30),
                "limitations": [sanitize_text(str(x), 150) for x in dq.get("limitations", [])[:5]],
            },
            "uncertainties": clean_list(data.get("uncertainties")),
            "advisory_summary": sanitize_text(str(data.get("advisory_summary", "")), 600),
            "is_advisory_only": True,
            "provider": "gemini",
            "model_name": self.model_name,
        }

    def _fallback_response(
        self,
        pair: str,
        signal: str,
        timestamp: str,
        error: str,
        status: str,
    ) -> Dict[str, Any]:
        """Provides safe fallback payload conforming strictly to schema when errors occur."""
        return {
            "timestamp_utc": timestamp,
            "pair": pair,
            "engine_signal": signal,
            "analysis_status": status,
            "market_summary": f"Advisory evaluation unavailable: {error}",
            "supporting_evidence": [],
            "conflicting_evidence": [],
            "alternative_scenarios": [],
            "risk_context": [
                "Gemini advisory service is unavailable; trading engine operating independently under local risk guards."
            ],
            "invalidation_conditions": [],
            "news_data_status": "UNAVAILABLE",
            "data_quality": {
                "status": "FALLBACK",
                "limitations": [f"Failure reason: {error}"],
            },
            "uncertainties": ["Advisory intelligence offline."],
            "advisory_summary": (
                "No advisory available. Engine trading decisions remain strictly guarded by PyTorch policy and risk guards."
            ),
            "is_advisory_only": True,
            "error_detail": error,
            "provider": "gemini",
            "model_name": self.model_name,
        }
