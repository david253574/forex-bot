"""
Forex AI Multi-Tenant Authentication & Encrypted Credential Engine.
Supports Supabase / PostgreSQL (production) with automatic SQLite fallback (local).
Provides AES/Fernet encryption at rest for OANDA API tokens and Bcrypt password hashing.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import sqlite3
import time
import uuid
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import bcrypt
from cryptography.fernet import Fernet, InvalidToken

# Project paths
PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
SQLITE_DB_PATH = DATA_DIR / "auth.db"

# Auto-load .env if present
def _load_env_file() -> None:
    env_file = PROJECT_DIR / ".env"
    if env_file.exists():
        try:
            with open(env_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, _, v = line.partition("=")
                        k = k.strip()
                        v = v.strip().strip("'\" ")
                        if k and k not in os.environ:
                            os.environ[k] = v
        except Exception:
            pass

_load_env_file()

# Master encryption key resolution
def _get_or_create_encryption_key() -> bytes:
    """Retrieve or generate 32-byte Fernet key."""
    env_key = os.environ.get("CREDENTIAL_ENCRYPTION_KEY", "").strip()
    if env_key:
        try:
            # Validate Fernet key format
            key_bytes = env_key.encode("utf-8")
            Fernet(key_bytes)
            return key_bytes
        except Exception:
            pass

    # Deterministic fallback tied to machine/project seed if not in env
    seed = (str(PROJECT_DIR) + "-forex-ai-auth-seed").encode("utf-8")
    digest = hashlib.sha256(seed).digest()
    return base64.urlsafe_b64encode(digest)


_FERNET_KEY = _get_or_create_encryption_key()
_CIPHER = Fernet(_FERNET_KEY)

# Session expiry (default: 7 days)
SESSION_DURATION = timedelta(days=7)


def encrypt_token(raw_token: str) -> str:
    """Encrypt a sensitive API token using AES-128-CBC via Fernet."""
    if not raw_token:
        return ""
    encrypted = _CIPHER.encrypt(raw_token.strip().encode("utf-8"))
    return encrypted.decode("utf-8")


def decrypt_token(cipher_token: str) -> str:
    """Decrypt a stored API token. Returns empty string on failure."""
    if not cipher_token:
        return ""
    try:
        decrypted = _CIPHER.decrypt(cipher_token.encode("utf-8"))
        return decrypted.decode("utf-8")
    except InvalidToken:
        return ""
    except Exception:
        return ""


def mask_token(raw_token: str) -> str:
    """Produce a safe masked preview of an API key (e.g. ••••••••••••3a8f)."""
    if not raw_token:
        return "None"
    token = raw_token.strip()
    if len(token) <= 6:
        return "••••" + token[-2:]
    return "••••••••••••" + token[-4:]


class DatabaseManager:
    """Unified DB layer for Supabase (PostgreSQL) and SQLite."""

    def __init__(self, db_url: Optional[str] = None):
        self.db_url = db_url if db_url is not None else os.environ.get("DATABASE_URL", "").strip()
        self.is_postgres = bool(self.db_url and (
            self.db_url.startswith("postgres://") or self.db_url.startswith("postgresql://")
        ))
        
        # Standardize postgres protocol for psycopg2 if needed
        if self.is_postgres and self.db_url.startswith("postgres://"):
            self.db_url = "postgresql://" + self.db_url[len("postgres://"):]

        self._ensure_schema()

    def get_connection(self):
        """Get an active database connection."""
        if self.is_postgres:
            import psycopg2
            from psycopg2.extras import RealDictCursor
            # Supabase requires SSL mode
            conn = psycopg2.connect(self.db_url, sslmode="require", cursor_factory=RealDictCursor)
            conn.autocommit = False
            return conn
        else:
            DATA_DIR.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(str(SQLITE_DB_PATH))
            conn.row_factory = sqlite3.Row
            return conn

    def _ensure_schema(self) -> None:
        """Create necessary tables if they do not exist."""
        conn = self.get_connection()
        try:
            cur = conn.cursor()
            id_type = "VARCHAR(36)"
            ts_type = "TIMESTAMPTZ" if self.is_postgres else "DATETIME"
            bool_type = "BOOLEAN"
            text_type = "TEXT"

            # 1. Users table
            cur.execute(f"""
            CREATE TABLE IF NOT EXISTS users (
                id {id_type} PRIMARY KEY,
                email VARCHAR(255) UNIQUE NOT NULL,
                password_hash VARCHAR(255) NOT NULL,
                role VARCHAR(50) DEFAULT 'user',
                created_at {ts_type} DEFAULT CURRENT_TIMESTAMP,
                last_login_at {ts_type},
                is_active {bool_type} DEFAULT TRUE
            );
            """)

            # 2. User broker configurations (encrypted API token)
            cur.execute(f"""
            CREATE TABLE IF NOT EXISTS user_broker_configs (
                user_id {id_type} PRIMARY KEY,
                broker VARCHAR(50) DEFAULT 'oanda',
                environment VARCHAR(20) DEFAULT 'practice',
                account_id VARCHAR(100) NOT NULL,
                encrypted_api_token {text_type} NOT NULL,
                is_valid {bool_type} DEFAULT FALSE,
                last_verified_at {ts_type},
                account_summary {text_type} DEFAULT '{{}}',
                updated_at {ts_type} DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            );
            """)

            # 3. User sessions
            cur.execute(f"""
            CREATE TABLE IF NOT EXISTS user_sessions (
                session_id VARCHAR(64) PRIMARY KEY,
                user_id {id_type} NOT NULL,
                created_at {ts_type} DEFAULT CURRENT_TIMESTAMP,
                expires_at {ts_type} NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            );
            """)

            # 4. User ledgers
            cur.execute(f"""
            CREATE TABLE IF NOT EXISTS user_ledgers (
                user_id {id_type} PRIMARY KEY,
                ledger_data {text_type} NOT NULL,
                updated_at {ts_type} DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            );
            """)

            conn.commit()
        finally:
            conn.close()

    def _param(self, query: str) -> str:
        """Translate ? to %s if using Postgres."""
        if self.is_postgres:
            return query.replace("?", "%s")
        return query

    # --- User Operations ---

    def register_user(self, email: str, password: str, role: str = "user") -> Dict[str, Any]:
        """Create new user profile with hashed password."""
        email = email.strip().lower()
        if not email or "@" not in email:
            return {"error": "Invalid email address."}
        if len(password) < 8:
            return {"error": "Password must be at least 8 characters long."}

        # Check existing
        existing = self.get_user_by_email(email)
        if existing:
            return {"error": "An account with this email already exists."}

        # Hash password with bcrypt
        salt = bcrypt.gensalt(rounds=12)
        pwd_hash = bcrypt.hashpw(password.encode("utf-8"), salt).decode("utf-8")
        user_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()

        conn = self.get_connection()
        try:
            cur = conn.cursor()
            q = self._param(
                "INSERT INTO users (id, email, password_hash, role, created_at, is_active) "
                "VALUES (?, ?, ?, ?, ?, ?)"
            )
            cur.execute(q, (user_id, email, pwd_hash, role, now, True))
            conn.commit()
            return {
                "success": True,
                "user": {
                    "id": user_id,
                    "email": email,
                    "role": role,
                    "created_at": now,
                }
            }
        except Exception as e:
            conn.rollback()
            return {"error": f"Registration failed: {str(e)}"}
        finally:
            conn.close()

    def authenticate_user(self, email: str, password: str) -> Dict[str, Any]:
        """Verify user credentials and update last login."""
        email = email.strip().lower()
        user = self.get_user_by_email(email)
        if not user:
            return {"error": "Invalid email or password."}

        if not user.get("is_active"):
            return {"error": "This account has been deactivated."}

        pwd_hash = user.get("password_hash", "")
        if not bcrypt.checkpw(password.encode("utf-8"), pwd_hash.encode("utf-8")):
            return {"error": "Invalid email or password."}

        # Update last login
        now = datetime.now(timezone.utc).isoformat()
        conn = self.get_connection()
        try:
            cur = conn.cursor()
            q = self._param("UPDATE users SET last_login_at = ? WHERE id = ?")
            cur.execute(q, (now, user["id"]))
            conn.commit()
        except Exception:
            pass
        finally:
            conn.close()

        # Sanitize return object
        user_clean = {k: v for k, v in user.items() if k != "password_hash"}
        user_clean["last_login_at"] = now
        return {"success": True, "user": user_clean}

    def get_user_by_id(self, user_id: str) -> Optional[Dict[str, Any]]:
        conn = self.get_connection()
        try:
            cur = conn.cursor()
            q = self._param("SELECT * FROM users WHERE id = ?")
            cur.execute(q, (user_id,))
            row = cur.fetchone()
            if not row:
                return None
            return dict(row)
        finally:
            conn.close()

    def get_user_by_email(self, email: str) -> Optional[Dict[str, Any]]:
        conn = self.get_connection()
        try:
            cur = conn.cursor()
            q = self._param("SELECT * FROM users WHERE email = ?")
            cur.execute(q, (email.strip().lower(),))
            row = cur.fetchone()
            if not row:
                return None
            return dict(row)
        finally:
            conn.close()

    # --- Session Management ---

    def create_session(self, user_id: str) -> str:
        """Create new active session token."""
        session_id = uuid.uuid4().hex + uuid.uuid4().hex
        now = datetime.now(timezone.utc)
        expires_at = (now + SESSION_DURATION).isoformat()
        created_at = now.isoformat()

        conn = self.get_connection()
        try:
            cur = conn.cursor()
            q = self._param("INSERT INTO user_sessions (session_id, user_id, created_at, expires_at) VALUES (?, ?, ?, ?)")
            cur.execute(q, (session_id, user_id, created_at, expires_at))
            conn.commit()
            return session_id
        finally:
            conn.close()

    def validate_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Validate session token, returning user profile if valid and active."""
        if not session_id or len(session_id) < 32:
            return None

        conn = self.get_connection()
        try:
            cur = conn.cursor()
            q = self._param("""
            SELECT s.session_id, s.expires_at, u.id, u.email, u.role, u.is_active
            FROM user_sessions s
            JOIN users u ON s.user_id = u.id
            WHERE s.session_id = ?
            """)
            cur.execute(q, (session_id,))
            row = cur.fetchone()
            if not row:
                return None

            data = dict(row)
            if not data.get("is_active"):
                return None

            expires_str = data.get("expires_at")
            if isinstance(expires_str, str):
                expires_at = datetime.fromisoformat(expires_str)
            else:
                expires_at = expires_str

            if expires_at.tzinfo is None:
                expires_at = expires_at.replace(tzinfo=timezone.utc)

            if datetime.now(timezone.utc) > expires_at:
                self.revoke_session(session_id)
                return None

            return {
                "session_id": session_id,
                "user_id": data["id"],
                "email": data["email"],
                "role": data["role"],
            }
        finally:
            conn.close()

    def revoke_session(self, session_id: str) -> bool:
        """Delete session token on logout."""
        conn = self.get_connection()
        try:
            cur = conn.cursor()
            q = self._param("DELETE FROM user_sessions WHERE session_id = ?")
            cur.execute(q, (session_id,))
            conn.commit()
            return True
        except Exception:
            return False
        finally:
            conn.close()

    # --- OANDA Broker Credentials Management ---

    def save_broker_config(
        self,
        user_id: str,
        account_id: str,
        api_token: str,
        environment: str = "practice",
    ) -> Dict[str, Any]:
        """Encrypt and store user's OANDA broker connection parameters."""
        account_id = account_id.strip()
        api_token = api_token.strip()
        env = environment.strip().lower()
        if env not in ["practice", "live"]:
            env = "practice"

        if not account_id:
            return {"error": "OANDA Account ID is required."}
        if not api_token:
            return {"error": "OANDA API Token is required."}

        enc_token = encrypt_token(api_token)
        now = datetime.now(timezone.utc).isoformat()

        conn = self.get_connection()
        try:
            cur = conn.cursor()
            # Upsert broker config
            if self.is_postgres:
                q = """
                INSERT INTO user_broker_configs 
                    (user_id, broker, environment, account_id, encrypted_api_token, is_valid, updated_at)
                VALUES (%s, 'oanda', %s, %s, %s, FALSE, %s)
                ON CONFLICT (user_id) DO UPDATE SET
                    environment = EXCLUDED.environment,
                    account_id = EXCLUDED.account_id,
                    encrypted_api_token = EXCLUDED.encrypted_api_token,
                    is_valid = FALSE,
                    updated_at = EXCLUDED.updated_at
                """
                cur.execute(q, (user_id, env, account_id, enc_token, now))
            else:
                q = """
                INSERT INTO user_broker_configs 
                    (user_id, broker, environment, account_id, encrypted_api_token, is_valid, updated_at)
                VALUES (?, 'oanda', ?, ?, ?, 0, ?)
                ON CONFLICT(user_id) DO UPDATE SET
                    environment = excluded.environment,
                    account_id = excluded.account_id,
                    encrypted_api_token = excluded.encrypted_api_token,
                    is_valid = 0,
                    updated_at = excluded.updated_at
                """
                cur.execute(q, (user_id, env, account_id, enc_token, now))
            conn.commit()
            return {
                "success": True,
                "broker": "oanda",
                "environment": env,
                "account_id": account_id,
                "masked_token": mask_token(api_token),
                "is_valid": False,
            }
        except Exception as e:
            conn.rollback()
            return {"error": f"Failed to save broker configuration: {str(e)}"}
        finally:
            conn.close()

    def get_broker_config(self, user_id: str, decrypt: bool = False) -> Optional[Dict[str, Any]]:
        """Retrieve broker configuration with decrypted or masked token."""
        conn = self.get_connection()
        try:
            cur = conn.cursor()
            q = self._param("SELECT * FROM user_broker_configs WHERE user_id = ?")
            cur.execute(q, (user_id,))
            row = cur.fetchone()
            if not row:
                return None
            data = dict(row)
            enc_token = data.get("encrypted_api_token", "")
            raw_token = decrypt_token(enc_token) if enc_token else ""

            summary = {}
            raw_summary = data.get("account_summary", "{}")
            if raw_summary:
                try:
                    summary = json.loads(raw_summary)
                except Exception:
                    pass

            result = {
                "user_id": user_id,
                "broker": data.get("broker", "oanda"),
                "environment": data.get("environment", "practice"),
                "account_id": data.get("account_id", ""),
                "is_valid": bool(data.get("is_valid")),
                "last_verified_at": data.get("last_verified_at"),
                "account_summary": summary,
            }

            if decrypt:
                result["api_token"] = raw_token
            else:
                result["masked_token"] = mask_token(raw_token)

            return result
        finally:
            conn.close()

    def update_broker_verification(
        self,
        user_id: str,
        is_valid: bool,
        account_summary: Dict[str, Any]
    ) -> bool:
        """Update connection test result after checking against OANDA API."""
        now = datetime.now(timezone.utc).isoformat()
        summary_str = json.dumps(account_summary)

        conn = self.get_connection()
        try:
            cur = conn.cursor()
            q = self._param("""
            UPDATE user_broker_configs 
            SET is_valid = ?, last_verified_at = ?, account_summary = ?, updated_at = ?
            WHERE user_id = ?
            """)
            cur.execute(q, (bool(is_valid), now, summary_str, now, user_id))
            conn.commit()
            return True
        except Exception:
            return False
        finally:
            conn.close()

    # --- Per-User Portfolio Ledgers ---

    def get_user_ledger(self, user_id: str) -> Dict[str, Any]:
        """Fetch user-specific portfolio ledger or default starting ledger."""
        default_ledger = {
            "balance": 100000.0,
            "equity": 100000.0,
            "positions": {},
            "closed_trades": [],
            "equity_history": []
        }
        conn = self.get_connection()
        try:
            cur = conn.cursor()
            q = self._param("SELECT ledger_data FROM user_ledgers WHERE user_id = ?")
            cur.execute(q, (user_id,))
            row = cur.fetchone()
            if not row:
                return default_ledger
            raw = dict(row).get("ledger_data", "{}")
            return json.loads(raw)
        except Exception:
            return default_ledger
        finally:
            conn.close()

    def save_user_ledger(self, user_id: str, ledger: Dict[str, Any]) -> bool:
        """Persist user-specific portfolio ledger."""
        raw_json = json.dumps(ledger)
        now = datetime.now(timezone.utc).isoformat()

        conn = self.get_connection()
        try:
            cur = conn.cursor()
            if self.is_postgres:
                q = """
                INSERT INTO user_ledgers (user_id, ledger_data, updated_at)
                VALUES (%s, %s, %s)
                ON CONFLICT (user_id) DO UPDATE SET
                    ledger_data = EXCLUDED.ledger_data,
                    updated_at = EXCLUDED.updated_at
                """
                cur.execute(q, (user_id, raw_json, now))
            else:
                q = """
                INSERT INTO user_ledgers (user_id, ledger_data, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(user_id) DO UPDATE SET
                    ledger_data = excluded.ledger_data,
                    updated_at = excluded.updated_at
                """
                cur.execute(q, (user_id, raw_json, now))
            conn.commit()
            return True
        except Exception:
            return False
        finally:
            conn.close()


    def verify_and_save_broker_credentials(
        self,
        user_id: str,
        account_id: str,
        api_token: str,
        environment: str = "practice",
    ) -> Dict[str, Any]:
        """
        Tests credentials directly against OANDA REST API.
        If valid, encrypts and saves them with verified account summary.
        """
        import requests

        account_id = account_id.strip()
        api_token = api_token.strip()
        env = environment.strip().lower()
        if env not in ["practice", "live"]:
            env = "practice"

        if not account_id:
            return {"error": "OANDA Account ID is required."}
        if not api_token:
            return {"error": "OANDA API Token is required."}

        base_url = "https://api-fxtrade.oanda.com/v3" if env == "live" else "https://api-fxpractice.oanda.com/v3"
        headers = {
            "Authorization": f"Bearer {api_token}",
            "Accept": "application/json",
            "User-Agent": "ForexAI-Ensemble/2.0"
        }

        try:
            resp = requests.get(f"{base_url}/accounts/{account_id}/summary", headers=headers, timeout=10)
            if resp.status_code == 200:
                acc_data = resp.json().get("account", {})
                summary = {
                    "currency": acc_data.get("currency", "USD"),
                    "balance": float(acc_data.get("balance", 0.0)),
                    "equity": float(acc_data.get("NAV", 0.0)),
                    "unrealized_pl": float(acc_data.get("unrealizedPL", 0.0)),
                    "margin_available": float(acc_data.get("marginAvailable", 0.0)),
                    "margin_used": float(acc_data.get("marginUsed", 0.0)),
                    "open_trades": int(acc_data.get("openTradeCount", 0)),
                }
                # Save and mark verified
                self.save_broker_config(user_id, account_id, api_token, env)
                self.update_broker_verification(user_id, is_valid=True, account_summary=summary)
                return {
                    "success": True,
                    "verified": True,
                    "account_summary": summary,
                    "masked_token": mask_token(api_token),
                    "environment": env,
                    "account_id": account_id,
                }
            elif resp.status_code == 401:
                return {"error": "Unauthorized: Invalid OANDA API Personal Access Token."}
            elif resp.status_code == 404:
                return {"error": f"OANDA Account '{account_id}' not found in '{env}' environment."}
            else:
                return {"error": f"OANDA returned HTTP {resp.status_code}: {resp.text[:120]}"}
        except requests.exceptions.Timeout:
            return {"error": "Connection timed out while verifying with OANDA."}
        except Exception as e:
            return {"error": f"OANDA verification failed: {str(e)}"}


# Singleton instance
db = DatabaseManager()
