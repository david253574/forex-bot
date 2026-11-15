"""
Integration tests for multi-tenant authenticated dashboard server.
Verifies public landing, login/register flows, session cookies, route protection,
and OANDA broker credential management.
"""

import json
import os
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

from auth_db import db
import dashboard_server


class TestMultiTenantServer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Configure test sqlite db path
        cls.test_db_path = Path(__file__).resolve().parent / "test_server_auth.db"
        if cls.test_db_path.exists():
            cls.test_db_path.unlink()
        
        import auth_db
        auth_db.SQLITE_DB_PATH = cls.test_db_path
        cls.db = auth_db.DatabaseManager(db_url="")
        dashboard_server.db = cls.db

    @classmethod
    def tearDownClass(cls):
        if cls.test_db_path.exists():
            try:
                cls.test_db_path.unlink()
            except Exception:
                pass

    def setUp(self):
        # Mock request handler helper
        self.handler = dashboard_server.DashboardHandler.__new__(dashboard_server.DashboardHandler)
        self.handler.headers = {}
        self.handler.wfile = MagicMock()
        self.handler.rfile = MagicMock()
        self.handler.send_response = MagicMock()
        self.handler.send_header = MagicMock()
        self.handler.end_headers = MagicMock()

    def test_public_routes_accessible(self):
        # 1. Landing page
        self.handler.path = "/"
        self.handler.do_GET()
        self.handler.send_response.assert_called_with(200)

        # 2. Login page
        self.handler.path = "/login"
        self.handler.do_GET()
        self.handler.send_response.assert_called_with(200)

        # 3. Register page
        self.handler.path = "/register"
        self.handler.do_GET()
        self.handler.send_response.assert_called_with(200)

        # 4. Public Telemetry endpoint
        self.handler.path = "/api/public/telemetry"
        self.handler.do_GET()
        self.handler.send_response.assert_called_with(200)

    def test_protected_routes_unauthenticated_redirection(self):
        # /dashboard without cookie -> 302 redirect to /login
        self.handler.path = "/dashboard"
        self.handler.headers = {}
        self.handler.do_GET()
        self.handler.send_response.assert_called_with(302)
        self.handler.send_header.assert_any_call("Location", "/login?next=/dashboard")

        # /api/portfolio without cookie -> 401
        self.handler.path = "/api/portfolio"
        self.handler.headers = {}
        self.handler.do_GET()
        self.handler.send_response.assert_called_with(401)

        # /api/me without cookie -> 401
        self.handler.path = "/api/me"
        self.handler.headers = {}
        self.handler.do_GET()
        self.handler.send_response.assert_called_with(401)

    def test_full_auth_and_broker_lifecycle(self):
        email = "full_cycle@example.com"
        password = "SecurePassword123!"

        # 1. Register
        reg_payload = json.dumps({"email": email, "password": password}).encode("utf-8")
        self.handler.path = "/api/auth/register"
        self.handler.headers = {"Content-Length": str(len(reg_payload))}
        self.handler.rfile.read.return_value = reg_payload
        self.handler.do_POST()

        self.handler.send_response.assert_called_with(200)
        # Check that Set-Cookie was sent
        cookie_header_call = [call for call in self.handler.send_header.call_args_list if call[0][0] == "Set-Cookie"]
        self.assertTrue(len(cookie_header_call) > 0)
        cookie_val = cookie_header_call[-1][0][1]
        session_token = cookie_val.split(";")[0].split("=")[1]
        self.assertTrue(len(session_token) > 20)

        # 2. Access /dashboard with session cookie
        self.handler.path = "/dashboard"
        self.handler.headers = {"Cookie": f"session_token={session_token}"}
        self.handler.do_GET()
        self.handler.send_response.assert_called_with(200)

        # 3. Access /api/me with session cookie
        self.handler.path = "/api/me"
        self.handler.headers = {"Cookie": f"session_token={session_token}"}
        self.handler.do_GET()
        self.handler.send_response.assert_called_with(200)

        # 4. Connect OANDA broker with valid token
        with patch("requests.get") as mock_get:
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = {
                "account": {
                    "currency": "USD",
                    "balance": "50000.00",
                    "NAV": "50120.00",
                    "unrealizedPL": "120.00",
                    "marginAvailable": "49000.00",
                    "marginUsed": "1120.00",
                    "openTradeCount": 1,
                }
            }
            mock_get.return_value = mock_resp

            connect_payload = json.dumps({
                "account_id": "101-004-5555555-001",
                "api_token": "valid_token_secret_12345",
                "environment": "practice",
            }).encode("utf-8")
            self.handler.path = "/api/broker/connect"
            self.handler.headers = {
                "Cookie": f"session_token={session_token}",
                "Content-Length": str(len(connect_payload)),
            }
            self.handler.rfile.read.return_value = connect_payload
            self.handler.do_POST()
            self.handler.send_response.assert_called_with(200)

        # 5. Access /api/me to verify broker info is attached
        self.handler.path = "/api/me"
        self.handler.headers = {"Cookie": f"session_token={session_token}"}
        self.handler.do_GET()
        self.handler.send_response.assert_called_with(200)

        # 6. Logout
        self.handler.path = "/api/auth/logout"
        self.handler.headers = {"Cookie": f"session_token={session_token}"}
        self.handler.do_POST()
        self.handler.send_response.assert_called_with(200)

        # 7. Post-logout /dashboard should now 302 redirect again
        self.handler.path = "/dashboard"
        self.handler.headers = {"Cookie": f"session_token={session_token}"}
        self.handler.do_GET()
        self.handler.send_response.assert_called_with(302)

    def test_advisory_evaluate_endpoint(self):
        with patch("market_intelligence.MarketIntelligenceAgent.evaluate_pair") as mock_eval:
            mock_eval.return_value = {
                "pair": "EURUSD",
                "analysis_status": "SUCCESS",
                "market_summary": "EURUSD consolidating.",
                "inference_duration_sec": 1.2,
            }
            payload = json.dumps({"pair": "EURUSD"}).encode("utf-8")
            self.handler.path = "/api/advisory/evaluate"
            self.handler.headers = {"Content-Length": str(len(payload))}
            self.handler.rfile.read.return_value = payload
            self.handler.do_POST()
            self.handler.send_response.assert_called_with(200)

    def test_ai_copilot_endpoint(self):
        with patch("market_intelligence.MarketIntelligenceAgent.chat_copilot") as mock_copilot:
            mock_copilot.return_value = {
                "reply": "Current drawdown is 1.44%, safely below limit.",
                "status": "SUCCESS",
                "elapsed_sec": 1.1,
            }
            payload = json.dumps({"query": "What is my drawdown?"}).encode("utf-8")
            self.handler.path = "/api/ai/copilot"
            self.handler.headers = {"Content-Length": str(len(payload))}
            self.handler.rfile.read.return_value = payload
            self.handler.do_POST()
            self.handler.send_response.assert_called_with(200)


if __name__ == "__main__":
    unittest.main()
