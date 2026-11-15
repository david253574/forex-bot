"""
Unit tests for auth_db.py (Multi-tenant authentication & Fernet encrypted credentials).
"""

import os
import unittest
from pathlib import Path
from auth_db import (
    DatabaseManager,
    encrypt_token,
    decrypt_token,
    mask_token,
)


class TestAuthDb(unittest.TestCase):
    def setUp(self):
        # Use an isolated temporary sqlite DB for testing
        self.test_db_path = Path(__file__).resolve().parent / "test_auth_temp.db"
        if self.test_db_path.exists():
            self.test_db_path.unlink()
        
        # Override sqlite db path for testing
        import auth_db
        auth_db.SQLITE_DB_PATH = self.test_db_path
        self.db = DatabaseManager(db_url="")

    def tearDown(self):
        if self.test_db_path.exists():
            try:
                self.test_db_path.unlink()
            except Exception:
                pass

    def test_token_encryption_and_decryption(self):
        secret_token = "0123456789abcdef0123456789abcdef-0123456789abcdef0123456789abcdef"
        encrypted = encrypt_token(secret_token)
        self.assertNotEqual(secret_token, encrypted)
        
        decrypted = decrypt_token(encrypted)
        self.assertEqual(secret_token, decrypted)

    def test_token_masking(self):
        token = "9876543210abcdef"
        masked = mask_token(token)
        self.assertTrue(masked.startswith("••••"))
        self.assertTrue(masked.endswith("cdef"))

    def test_user_registration_and_authentication(self):
        email = "trader_test@example.com"
        password = "SecurePassword123!"

        # Register
        reg_res = self.db.register_user(email=email, password=password)
        self.assertTrue(reg_res.get("success"), f"Registration failed: {reg_res}")
        user_info = reg_res["user"]
        self.assertEqual(user_info["email"], email)
        user_id = user_info["id"]

        # Duplicate registration should fail
        dup_res = self.db.register_user(email=email, password=password)
        self.assertIn("error", dup_res)

        # Successful Auth
        auth_res = self.db.authenticate_user(email=email, password=password)
        self.assertTrue(auth_res.get("success"))
        self.assertEqual(auth_res["user"]["id"], user_id)

        # Failed Auth (bad password)
        bad_auth = self.db.authenticate_user(email=email, password="WrongPassword!")
        self.assertIn("error", bad_auth)

        # Failed Auth (nonexistent user)
        ghost_auth = self.db.authenticate_user(email="ghost@example.com", password=password)
        self.assertIn("error", ghost_auth)

    def test_session_lifecycle(self):
        reg = self.db.register_user(email="session_user@example.com", password="Password1234!")
        user_id = reg["user"]["id"]

        # Create session
        session_id = self.db.create_session(user_id)
        self.assertTrue(bool(session_id))

        # Validate session
        validated = self.db.validate_session(session_id)
        self.assertIsNotNone(validated)
        self.assertEqual(validated["user_id"], user_id)
        self.assertEqual(validated["email"], "session_user@example.com")

        # Revoke session (logout)
        revoked = self.db.revoke_session(session_id)
        self.assertTrue(revoked)
        self.assertIsNone(self.db.validate_session(session_id))

    def test_broker_credentials_encryption_and_persistence(self):
        reg = self.db.register_user(email="broker_trader@example.com", password="Password1234!")
        user_id = reg["user"]["id"]

        account_id = "101-004-9876543-001"
        api_token = "secret-oanda-live-key-abcdef123456"

        # Save config
        save_res = self.db.save_broker_config(
            user_id=user_id,
            account_id=account_id,
            api_token=api_token,
            environment="practice",
        )
        self.assertTrue(save_res.get("success"))
        self.assertTrue(save_res["masked_token"].endswith("3456"))

        # Retrieve masked (for UI display)
        config_masked = self.db.get_broker_config(user_id, decrypt=False)
        self.assertEqual(config_masked["account_id"], account_id)
        self.assertEqual(config_masked["environment"], "practice")
        self.assertTrue(config_masked["masked_token"].startswith("••••"))
        self.assertNotIn("api_token", config_masked)

        # Retrieve decrypted (for backend execution)
        config_decrypted = self.db.get_broker_config(user_id, decrypt=True)
        self.assertEqual(config_decrypted["api_token"], api_token)

        # Update verification status
        self.db.update_broker_verification(
            user_id=user_id,
            is_valid=True,
            account_summary={"currency": "USD", "balance": "10000.00"}
        )
        updated_config = self.db.get_broker_config(user_id)
        self.assertTrue(updated_config["is_valid"])
        self.assertEqual(updated_config["account_summary"]["currency"], "USD")

    def test_user_ledger_isolation(self):
        reg_a = self.db.register_user(email="user_a@example.com", password="Password1234!")
        reg_b = self.db.register_user(email="user_b@example.com", password="Password1234!")
        user_a = reg_a["user"]["id"]
        user_b = reg_b["user"]["id"]

        ledger_a = {"balance": 150000.0, "positions": {"EUR_USD": {"units": 10000}}}
        ledger_b = {"balance": 50000.0, "positions": {}}

        self.db.save_user_ledger(user_a, ledger_a)
        self.db.save_user_ledger(user_b, ledger_b)

        fetched_a = self.db.get_user_ledger(user_a)
        fetched_b = self.db.get_user_ledger(user_b)

        self.assertEqual(fetched_a["balance"], 150000.0)
        self.assertEqual(fetched_b["balance"], 50000.0)

    from unittest.mock import patch, MagicMock

    @patch("requests.get")
    def test_verify_and_save_broker_credentials_success(self, mock_get):
        reg = self.db.register_user(email="mock_oanda@example.com", password="Password1234!")
        user_id = reg["user"]["id"]

        mock_resp = self.MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "account": {
                "currency": "EUR",
                "balance": "25000.50",
                "NAV": "25100.75",
                "unrealizedPL": "100.25",
                "marginAvailable": "24000.00",
                "marginUsed": "1100.75",
                "openTradeCount": 2,
            }
        }
        mock_get.return_value = mock_resp

        res = self.db.verify_and_save_broker_credentials(
            user_id=user_id,
            account_id="001-001-1234567-001",
            api_token="valid_token_12345678",
            environment="practice",
        )
        self.assertTrue(res.get("success"))
        self.assertTrue(res.get("verified"))
        self.assertEqual(res["account_summary"]["currency"], "EUR")
        self.assertEqual(res["account_summary"]["balance"], 25000.50)

        # Check DB state
        saved = self.db.get_broker_config(user_id)
        self.assertTrue(saved["is_valid"])
        self.assertEqual(saved["account_summary"]["currency"], "EUR")

    @patch("requests.get")
    def test_verify_and_save_broker_credentials_invalid(self, mock_get):
        reg = self.db.register_user(email="mock_bad_oanda@example.com", password="Password1234!")
        user_id = reg["user"]["id"]

        mock_resp = self.MagicMock()
        mock_resp.status_code = 401
        mock_get.return_value = mock_resp

        res = self.db.verify_and_save_broker_credentials(
            user_id=user_id,
            account_id="001-001-1234567-001",
            api_token="bad_token",
            environment="practice",
        )
        self.assertIn("error", res)
        self.assertIn("Unauthorized", res["error"])


if __name__ == "__main__":
    unittest.main()
