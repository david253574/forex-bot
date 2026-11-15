# forex-bot

Autonomous Multi-Pair Algorithmic Currency Trading Platform powered by PyTorch Deep Reinforcement Learning (PPO) and OANDA v20 REST API.

---

## Overview

**Forex AI** is an institutional-grade algorithmic currency execution platform coordinating four independent neural policy models across major liquid G10 currency pairs:
- **EUR/USD** — Intraday Momentum & Mean Reversion
- **GBP/USD** — High-Beta London Volatility Breakout
- **USD/JPY** — Macro Trend & Interest Rate Differential (100&times; JPY pip multiplier)
- **AUD/USD** — Commodity Cycle & Global Risk Proxy

The platform features:
- **Headless Linux Daemon**: 24/5 continuous process supervision under `systemd` or background runner.
- **Server-Side Order Brackets**: Native broker-enforced Stop-Loss (2.0&times; ATR) and Take-Profit (6.0&times; ATR) attached directly inside OANDA matching payloads.
- **Macro Blackout Shield**: Integrates Finnhub economic calendar polling to embargo trade openings during high-impact news releases (NFP, CPI, FOMC, BoE, ECB).
- **Multi-Tenant Web Terminal**: High-fidelity dashboard, real-time telemetry, 3D portfolio topology, and operational workstation.

---

## Repository Structure

```text
├── dashboard_server.py         # Multi-tenant HTTP dashboard & REST API server (:8080)
├── live_paper_portfolio.py     # Hourly continuous multi-pair trading loop
├── broker_gateway.py           # OANDA v20 REST client with Mock Sandbox fallback
├── risk_guards.py              # Spread guard, news blackout shield, drawdown limits
├── auth_db.py                  # User authentication & AES-128 token encryption at rest
├── gemini_service.py           # Market intelligence & AI Copilot integration
├── news_client.py              # Economic calendar fetcher & event blackout manager
├── landing_page.html           # Public landing page with 3D Portfolio Topology
├── login.html / register.html  # Authentication interfaces
├── model_*.pt & scaler_*.pkl   # Pre-trained PyTorch LSTM policies and MinMax scalers
├── tests/                      # Comprehensive unit and integration test suite (112 tests)
├── .env.example                # Template for environment configuration
├── broker_config.example.json  # Template for OANDA credentials & guard parameters
└── run_bot.sh                  # Process management CLI (start, stop, status, restart)
```

---

## Quick Start & Setup

### 1. Prerequisites
- Linux / macOS (Python 3.10+)
- OANDA v20 Practice or Live Account
- Finnhub API Key (optional, for macro news blackout shield)
- Google Gemini API Key (optional, for market intelligence copilot)

### 2. Environment Configuration
Copy the template files and fill in your credentials:

```bash
# 1. Configure environment variables
cp .env.example .env

# 2. Configure broker settings
cp broker_config.example.json broker_config.json
```

Edit `.env` to configure your settings:
```ini
TRADING_MODE=practice
ALLOW_LIVE_TRADING=false
DASHBOARD_PORT=8080
FINNHUB_API_KEY=your_finnhub_key_here
GEMINI_API_KEY=your_gemini_key_here
```

Edit `broker_config.json` with your OANDA v20 credentials:
```json
{
  "broker": "oanda",
  "environment": "practice",
  "oanda": {
    "api_token": "YOUR_OANDA_V20_API_TOKEN",
    "account_id": "YOUR_OANDA_ACCOUNT_ID"
  }
}
```

### 3. Install Dependencies
```bash
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
```

### 4. Run the Automated Test Suite
```bash
python -m unittest discover tests
```

### 5. Start the Services

You can manage the trading engine and web dashboard via `run_bot.sh`:

```bash
# Start both Trading Engine and Dashboard
./run_bot.sh start

# Check real-time process status
./run_bot.sh status

# Tail trading logs
./run_bot.sh logs

# Stop all processes gracefully
./run_bot.sh stop
```

Alternatively, start the dashboard standalone:
```bash
python dashboard_server.py
```
Open [http://localhost:8080](http://localhost:8080) in your browser to launch the Web Terminal.

---

## Security & Risk Isolation

- **Secret Protection**: `.env` and `broker_config.json` are excluded via `.gitignore` to prevent secret leakage.
- **Client Token Encryption**: User broker tokens entered via the web dashboard are encrypted with AES-128 Fernet at rest.
- **Circuit Breakers**: Daily portfolio drawdown limit (default: 3.0%) halts new orders if capital thresholds are breached.
- **Practice Sandbox**: System defaults to `practice` mode. Live trading requires explicit configuration (`ALLOW_LIVE_TRADING=true`).

---

## License & Disclaimer

Foreign exchange trading carries significant financial risk. Automated machine learning models generate probabilistic likelihoods, not profit guarantees. Always validate policies thoroughly in Practice / Demo sandbox environments prior to committing live capital.
