#!/usr/bin/env bash
# ==============================================================================
# Forex AI 4-Pair Super-Ensemble Trading System Control Script
# Provides production management: start, stop, status, logs, and health checks.
# ==============================================================================

set -o pipefail

# Determine script and project directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

# ------------------------------------------------------------------------------
# Configuration & Defaults
# ------------------------------------------------------------------------------
PROJECT_DIR="$SCRIPT_DIR"

# Source .env if present (without exposing credentials)
if [ -f "$PROJECT_DIR/.env" ]; then
    # shellcheck disable=SC1091
    source "$PROJECT_DIR/.env"
fi

PYTHON_BIN="${PYTHON_BIN:-$PROJECT_DIR/env/bin/python}"
if [ ! -x "$PYTHON_BIN" ]; then
    if command -v python3 >/dev/null 2>&1; then
        PYTHON_BIN="$(command -v python3)"
    else
        echo "ERROR: Python interpreter not found at $PYTHON_BIN or in PATH." >&2
        exit 1
    fi
fi

PID_DIR="${PID_DIR:-$PROJECT_DIR/.run}"
ENGINE_PID_FILE="$PID_DIR/engine.pid"
DASHBOARD_PID_FILE="$PID_DIR/dashboard.pid"

LOG_DIR="${LOG_DIR:-$PROJECT_DIR}"
ENGINE_LOG="${ENGINE_LOG:-$LOG_DIR/paper_trading.log}"
DASHBOARD_LOG="${DASHBOARD_LOG:-$LOG_DIR/dashboard.log}"

CONFIG_FILE="${CONFIG_FILE:-$PROJECT_DIR/broker_config.json}"
LEDGER_FILE="${LEDGER_FILE:-$PROJECT_DIR/portfolio_ledger.json}"
OPS_HELPER="$PROJECT_DIR/ops_helper.py"

DASHBOARD_PORT="${DASHBOARD_PORT:-8080}"
SHUTDOWN_TIMEOUT="${SHUTDOWN_TIMEOUT:-15}"
ALLOW_LIVE_TRADING="${ALLOW_LIVE_TRADING:-false}"

# Ensure required run directory exists
mkdir -p "$PID_DIR"

# ------------------------------------------------------------------------------
# Process Identification Utilities
# ------------------------------------------------------------------------------

# Check if a given PID is currently active in the OS
is_pid_alive() {
    local pid="$1"
    if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
        return 0
    fi
    return 1
}

# Verify that a PID actually belongs to the live trading engine
is_engine_process() {
    local pid="$1"
    if ! is_pid_alive "$pid"; then
        return 1
    fi
    local exe=""
    exe=$(readlink -f "/proc/$pid/exe" 2>/dev/null || true)
    if [[ "$exe" != *"python"* ]]; then
        return 1
    fi
    local cmdline=""
    if [ -f "/proc/$pid/cmdline" ]; then
        cmdline=$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null)
    else
        cmdline=$(ps -p "$pid" -o cmd= 2>/dev/null)
    fi
    if [[ "$cmdline" == *"live_paper_portfolio.py"* ]]; then
        return 0
    fi
    return 1
}

# Verify that a PID actually belongs to the dashboard server
is_dashboard_process() {
    local pid="$1"
    if ! is_pid_alive "$pid"; then
        return 1
    fi
    local exe=""
    exe=$(readlink -f "/proc/$pid/exe" 2>/dev/null || true)
    if [[ "$exe" != *"python"* ]]; then
        return 1
    fi
    local cmdline=""
    if [ -f "/proc/$pid/cmdline" ]; then
        cmdline=$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null)
    else
        cmdline=$(ps -p "$pid" -o cmd= 2>/dev/null)
    fi
    if [[ "$cmdline" == *"dashboard_server.py"* ]]; then
        return 0
    fi
    return 1
}

# Find the active trading engine PID safely (PID file with fallback to proc check)
get_engine_pid() {
    if [ -f "$ENGINE_PID_FILE" ]; then
        local pid
        pid=$(cat "$ENGINE_PID_FILE" 2>/dev/null | tr -d '[:space:]')
        if [ -n "$pid" ] && is_engine_process "$pid"; then
            echo "$pid"
            return 0
        else
            # Stale PID file
            rm -f "$ENGINE_PID_FILE"
        fi
    fi

    # Fallback: discover existing instance matching this project directory
    local candidate
    for candidate in $(pgrep -f "live_paper_portfolio.py" 2>/dev/null); do
        if is_engine_process "$candidate"; then
            local cwd
            cwd=$(readlink -f "/proc/$candidate/cwd" 2>/dev/null)
            if [ "$cwd" = "$PROJECT_DIR" ]; then
                echo "$candidate" > "$ENGINE_PID_FILE"
                echo "$candidate"
                return 0
            fi
        fi
    done
    return 1
}

# Find the active dashboard server PID safely
get_dashboard_pid() {
    if [ -f "$DASHBOARD_PID_FILE" ]; then
        local pid
        pid=$(cat "$DASHBOARD_PID_FILE" 2>/dev/null | tr -d '[:space:]')
        if [ -n "$pid" ] && is_dashboard_process "$pid"; then
            echo "$pid"
            return 0
        else
            rm -f "$DASHBOARD_PID_FILE"
        fi
    fi

    local candidate
    for candidate in $(pgrep -f "dashboard_server.py" 2>/dev/null); do
        if is_dashboard_process "$candidate"; then
            local cwd
            cwd=$(readlink -f "/proc/$candidate/cwd" 2>/dev/null)
            if [ "$cwd" = "$PROJECT_DIR" ]; then
                echo "$candidate" > "$DASHBOARD_PID_FILE"
                echo "$candidate"
                return 0
            fi
        fi
    done
    return 1
}

# ------------------------------------------------------------------------------
# Mode Validation & Safety Checks
# ------------------------------------------------------------------------------

validate_mode() {
    local mode_info
    mode_info=$("$PYTHON_BIN" "$OPS_HELPER" get_mode "$CONFIG_FILE" 2>/dev/null)
    local broker
    local env
    local status
    local message

    IFS='|' read -r broker env status message <<< "$mode_info"

    if [ "$status" != "VALID" ]; then
        echo "CONFIGURATION ERROR: $message" >&2
        return 1
    fi

    # Explicit protection against unintended live trading
    if [ "$env" = "live" ]; then
        if [ "$ALLOW_LIVE_TRADING" != "true" ] && [ "$CONFIRM_LIVE_TRADING" != "1" ]; then
            echo "SAFETY LOCKOUT: Configuration specifies 'live' trading, but ALLOW_LIVE_TRADING is not set to 'true'." >&2
            echo "To protect real capital, live trading cannot start without explicit authorization in .env." >&2
            return 2
        fi
    fi

    return 0
}

# ------------------------------------------------------------------------------
# Command: start
# ------------------------------------------------------------------------------
cmd_start() {
    echo "================================================================================"
    echo "Starting Forex AI Trading System"
    echo "================================================================================"

    # 1. Pre-flight verification of required files
    local required_files=("live_paper_portfolio.py" "dashboard_server.py" "broker_gateway.py" "risk_guards.py" "ops_helper.py")
    for f in "${required_files[@]}"; do
        if [ ! -f "$PROJECT_DIR/$f" ]; then
            echo "STARTUP FAILED: Missing required project file: $f" >&2
            return 1
        fi
    done

    # 2. Check Python environment and dependencies
    if ! "$PYTHON_BIN" "$OPS_HELPER" verify_deps >/dev/null 2>&1; then
        echo "STARTUP FAILED: Python dependencies verification failed." >&2
        "$PYTHON_BIN" "$OPS_HELPER" verify_deps >&2
        return 1
    fi

    # 3. Validate operating mode and safety locks
    local mode_rc=0
    validate_mode || mode_rc=$?
    if [ "$mode_rc" -ne 0 ]; then
        return "$mode_rc"
    fi

    # Extract mode for display
    local mode_info
    mode_info=$("$PYTHON_BIN" "$OPS_HELPER" get_mode "$CONFIG_FILE" 2>/dev/null)
    local broker env status msg
    IFS='|' read -r broker env status msg <<< "$mode_info"

    # 4. Check for duplicate engine process
    local existing_engine_pid
    if existing_engine_pid=$(get_engine_pid); then
        echo "DUPLICATE PREVENTED: Trading engine is already running (PID: $existing_engine_pid)."
        echo "Use './run_bot.sh status' to inspect, or './run_bot.sh stop' to shut it down."
        return 0
    fi

    # 5. Verify writability of log and run directories
    touch "$ENGINE_LOG" "$DASHBOARD_LOG" 2>/dev/null || {
        echo "STARTUP FAILED: Cannot write to log files in $LOG_DIR." >&2
        return 1
    }

    echo "Operating Mode:       ${broker^^} [${env^^}]"
    if [ "$env" = "live" ]; then
        echo "WARNING:              REAL CAPITAL AT RISK (Live Mode Authorized)"
    fi
    echo "Log Output:           $ENGINE_LOG"
    echo "Dashboard Port:       $DASHBOARD_PORT"

    # 6. Start Trading Engine
    echo -n "Starting trading engine... "
    nohup setsid "$PYTHON_BIN" -u "$PROJECT_DIR/live_paper_portfolio.py" --continuous >> "$ENGINE_LOG" 2>&1 &
    local new_engine_pid=$!
    echo "$new_engine_pid" > "$ENGINE_PID_FILE"

    # Wait for process initialization
    sleep 2

    if ! is_engine_process "$new_engine_pid"; then
        echo "[FAILED]"
        echo "ERROR: Trading engine process exited immediately after startup." >&2
        echo "Recent log output ($ENGINE_LOG):" >&2
        tail -n 10 "$ENGINE_LOG" | "$PYTHON_BIN" "$OPS_HELPER" mask_stdin >&2
        rm -f "$ENGINE_PID_FILE"
        return 1
    fi
    echo "[OK] (PID: $new_engine_pid)"

    # 7. Start Dashboard Server (if not already running)
    local new_dash_pid
    if new_dash_pid=$(get_dashboard_pid); then
        echo "Dashboard server:     Already running (PID: $new_dash_pid)"
    else
        echo -n "Starting dashboard server... "
        nohup setsid "$PYTHON_BIN" "$PROJECT_DIR/dashboard_server.py" --port "$DASHBOARD_PORT" >> "$DASHBOARD_LOG" 2>&1 &
        new_dash_pid=$!
        echo "$new_dash_pid" > "$DASHBOARD_PID_FILE"
        sleep 1

        if ! is_dashboard_process "$new_dash_pid"; then
            echo "[FAILED]"
            echo "WARNING: Dashboard failed to start. Trading engine remains active." >&2
            tail -n 5 "$DASHBOARD_LOG" | "$PYTHON_BIN" "$OPS_HELPER" mask_stdin >&2
            rm -f "$DASHBOARD_PID_FILE"
        else
            echo "[OK] (PID: $new_dash_pid)"
        fi
    fi

    echo "--------------------------------------------------------------------------------"
    echo "System startup completed successfully."
    echo "  - Trading Engine PID:  $new_engine_pid"
    echo "  - Dashboard URL:       http://localhost:$DASHBOARD_PORT"
    echo "  - Operating Mode:      ${broker^^} (${env^^})"
    echo "  - Active Currency:     EUR/USD, USD/JPY, AUD/USD, GBP/USD"
    echo "  - Note:                Starting the system does not create or modify existing positions."
    echo "================================================================================"
    return 0
}

# ------------------------------------------------------------------------------
# Command: stop
# ------------------------------------------------------------------------------
cmd_stop() {
    echo "================================================================================"
    echo "Stopping Forex AI Trading System"
    echo "================================================================================"
    echo "Trading Safety Note:"
    echo "  Stopping the software stops automated evaluations and new order management."
    echo "  Server-side Stop-Loss and Take-Profit brackets previously placed on OANDA"
    echo "  will continue to remain active directly on the broker's servers."
    echo "  This script does NOT close or liquidate open broker positions."
    echo "--------------------------------------------------------------------------------"

    local stopped_any=false

    # 1. Stop Trading Engine
    local engine_pid
    if engine_pid=$(get_engine_pid); then
        echo -n "Stopping trading engine (PID: $engine_pid)... "
        # Send SIGINT (Ctrl+C equivalent) for graceful shutdown loop exit
        kill -INT "$engine_pid" 2>/dev/null

        local waited=0
        while is_pid_alive "$engine_pid" && [ "$waited" -lt "$SHUTDOWN_TIMEOUT" ]; do
            sleep 1
            waited=$((waited + 1))
        done

        if is_pid_alive "$engine_pid"; then
            # Escalate to SIGTERM
            kill -TERM "$engine_pid" 2>/dev/null
            sleep 2
        fi

        if is_pid_alive "$engine_pid"; then
            echo "[TIMEOUT]"
            echo "WARNING: Process did not exit within $SHUTDOWN_TIMEOUT seconds. Sending SIGKILL..." >&2
            kill -KILL "$engine_pid" 2>/dev/null
            sleep 1
        fi

        if ! is_pid_alive "$engine_pid"; then
            echo "[STOPPED]"
            rm -f "$ENGINE_PID_FILE"
            stopped_any=true
        else
            echo "[FAILED]"
            echo "ERROR: Unable to terminate trading engine process $engine_pid." >&2
        fi
    else
        echo "Trading engine:       Already stopped"
    fi

    # 2. Stop Dashboard Server
    local dash_pid
    if dash_pid=$(get_dashboard_pid); then
        echo -n "Stopping dashboard server (PID: $dash_pid)... "
        kill -TERM "$dash_pid" 2>/dev/null
        local d_waited=0
        while is_pid_alive "$dash_pid" && [ "$d_waited" -lt 5 ]; do
            sleep 1
            d_waited=$((d_waited + 1))
        done
        if is_pid_alive "$dash_pid"; then
            kill -KILL "$dash_pid" 2>/dev/null
        fi
        echo "[STOPPED]"
        rm -f "$DASHBOARD_PID_FILE"
        stopped_any=true
    else
        echo "Dashboard server:     Already stopped"
    fi

    echo "================================================================================"
    if [ "$stopped_any" = true ]; then
        echo "System shutdown complete."
    else
        echo "No running components found."
    fi
    return 0
}

# ------------------------------------------------------------------------------
# Command: status
# ------------------------------------------------------------------------------
cmd_status() {
    echo "================================================================================"
    echo "Forex AI 4-Pair Super-Ensemble System Status"
    echo "================================================================================"

    # Process statuses
    local engine_pid dash_pid
    engine_pid=$(get_engine_pid || true)
    dash_pid=$(get_dashboard_pid || true)

    echo -n "Trading Engine:       "
    if [ -n "$engine_pid" ]; then
        echo "RUNNING (PID: $engine_pid)"
    else
        echo "STOPPED"
    fi

    echo -n "Dashboard Server:     "
    if [ -n "$dash_pid" ]; then
        echo "RUNNING (PID: $dash_pid, Port: $DASHBOARD_PORT)"
    else
        echo "STOPPED"
    fi

    # Mode from config
    local mode_info
    mode_info=$("$PYTHON_BIN" "$OPS_HELPER" get_mode "$CONFIG_FILE" 2>/dev/null)
    local broker env status msg
    IFS='|' read -r broker env status msg <<< "$mode_info"

    echo "Operating Mode:       ${broker^^} [${env^^}] (Config: $CONFIG_FILE)"
    if [ "$env" = "live" ]; then
        echo "Mode Security:        [LIVE] REAL MONEY TRADING"
    else
        echo "Mode Security:        [PRACTICE] Demo / Risk-Free Simulation"
    fi

    echo "--------------------------------------------------------------------------------"
    echo "BROKER CONNECTION (Read-Only Live Query):"
    "$PYTHON_BIN" "$OPS_HELPER" broker_summary "$CONFIG_FILE" 2>/dev/null | while IFS= read -r line; do
        echo "  $line"
    done

    echo "--------------------------------------------------------------------------------"
    echo "LOCAL PORTFOLIO STATE (Cached in $LEDGER_FILE):"
    "$PYTHON_BIN" "$OPS_HELPER" ledger_summary "$LEDGER_FILE" 2>/dev/null | while IFS= read -r line; do
        if [[ "$line" == POS_* ]]; then
            # Format position line cleanly
            local pair="${line%%:*}"
            pair="${pair#POS_}"
            local details="${line#*: }"
            echo "  [$pair] $details"
        else
            echo "  $line"
        fi
    done

    echo "--------------------------------------------------------------------------------"
    echo "BROKER-TO-LEDGER RECONCILIATION:"
    local recon_raw
    recon_raw=$("$PYTHON_BIN" "$OPS_HELPER" reconcile "$CONFIG_FILE" "$LEDGER_FILE" 2>/dev/null)
    local recon_status
    recon_status=$(echo "$recon_raw" | grep "^RECONCILIATION_STATUS:" | cut -d: -f2- | tr -d '[:space:]')
    local recon_expl
    recon_expl=$(echo "$recon_raw" | grep "^DISCREPANCY_EXPLANATION:" | cut -d: -f2- | sed 's/^ //')
    local recon_counts
    recon_counts=$(echo "$recon_raw" | grep "^SUMMARY_COUNTS:" | cut -d: -f2- | sed 's/^ //')
    local recon_disc
    recon_disc=$(echo "$recon_raw" | grep "^EQUITY_DISCREPANCY:" | cut -d: -f2- | tr -d '[:space:]')

    echo "  Status:               $recon_status"
    echo "  Position Summary:     $recon_counts"
    echo "  Equity Discrepancy:   $recon_disc (Strategy vs Broker)"
    echo "  Explanation:          $recon_expl"

    echo "--------------------------------------------------------------------------------"
    echo -n "System State:         "
    if [ -n "$engine_pid" ] && [ -n "$dash_pid" ]; then
        echo "OPERATIONAL"
    elif [ -n "$engine_pid" ] || [ -n "$dash_pid" ]; then
        echo "DEGRADED (Partial service running)"
    else
        echo "STOPPED"
    fi

    echo "Last Log Activity:"
    if [ -f "$ENGINE_LOG" ]; then
        echo "  $(tail -n 1 "$ENGINE_LOG" 2>/dev/null | "$PYTHON_BIN" "$OPS_HELPER" mask_stdin)"
    else
        echo "  (No log file found)"
    fi
    echo "================================================================================"
}

# ------------------------------------------------------------------------------
# Command: logs
# ------------------------------------------------------------------------------
cmd_logs() {
    local target_log="$ENGINE_LOG"
    local lines=30
    local follow=false

    while [[ $# -gt 0 ]]; do
        case "$1" in
            -f|--follow)
                follow=true
                shift
                ;;
            -n|--lines)
                lines="$2"
                shift 2
                ;;
            --dashboard)
                target_log="$DASHBOARD_LOG"
                shift
                ;;
            *)
                shift
                ;;
        esac
    done

    if [ ! -f "$target_log" ]; then
        echo "Log file not found: $target_log"
        echo "Logs will be generated once the trading system is started using './run_bot.sh start'."
        return 1
    fi

    echo "Streaming logs from: $target_log (Tokens and secrets automatically redacted)"
    if [ "$follow" = true ]; then
        tail -n "$lines" -f "$target_log" | "$PYTHON_BIN" "$OPS_HELPER" mask_stdin
    else
        tail -n "$lines" "$target_log" | "$PYTHON_BIN" "$OPS_HELPER" mask_stdin
    fi
}

# ------------------------------------------------------------------------------
# Command: health
# ------------------------------------------------------------------------------
cmd_health() {
    echo "================================================================================"
    echo "Forex AI System Non-Destructive Health Checks"
    echo "================================================================================"

    local failures=0
    local warnings=0

    # Check 1: Python environment
    echo -n "1. Python Virtual Environment:    "
    if [ -x "$PYTHON_BIN" ]; then
        local py_version
        py_version=$("$PYTHON_BIN" --version 2>&1)
        echo "[PASS] ($py_version)"
    else
        echo "[FAIL] (Interpreter not executable at $PYTHON_BIN)"
        failures=$((failures + 1))
    fi

    # Check 2: Dependencies
    echo -n "2. Required Dependencies:         "
    if "$PYTHON_BIN" "$OPS_HELPER" verify_deps >/dev/null 2>&1; then
        echo "[PASS] (All core packages installed)"
    else
        echo "[FAIL] ($("$PYTHON_BIN" "$OPS_HELPER" verify_deps 2>&1))"
        failures=$((failures + 1))
    fi

    # Check 3: Neural Model Checkpoints & Scalers
    echo -n "3. AI Models & Scalers:           "
    local models=("model.pt" "model_usdjpy.pt" "model_multipair.pt")
    local scalers=("scaler.pkl" "scaler_usdjpy.pkl" "scaler_multipair.pkl")
    local missing_assets=0
    for m in "${models[@]}" "${scalers[@]}"; do
        if [ ! -f "$PROJECT_DIR/$m" ]; then
            missing_assets=$((missing_assets + 1))
        fi
    done
    if [ "$missing_assets" -eq 0 ]; then
        echo "[PASS] (All 4-pair super-ensemble assets verified)"
    else
        echo "[FAIL] ($missing_assets checkpoints or scalers missing)"
        failures=$((failures + 1))
    fi

    # Check 4: Configuration & Practice/Live Safety
    echo -n "4. Configuration & Safety Mode:   "
    local mode_info
    mode_info=$("$PYTHON_BIN" "$OPS_HELPER" get_mode "$CONFIG_FILE" 2>/dev/null)
    local broker env status msg
    IFS='|' read -r broker env status msg <<< "$mode_info"

    if [ "$status" = "VALID" ]; then
        if [ "$env" = "live" ] && [ "$ALLOW_LIVE_TRADING" != "true" ] && [ "$CONFIRM_LIVE_TRADING" != "1" ]; then
            echo "[FAIL] (Live mode configured but safety lock ALLOW_LIVE_TRADING is false)"
            failures=$((failures + 1))
        else
            echo "[PASS] (${broker^^} in ${env^^} mode)"
        fi
    else
        echo "[FAIL] ($msg)"
        failures=$((failures + 1))
    fi

    # Check 5: Process Health & Duplicate Detection
    echo -n "5. Process State & Duplicate Check: "
    local engine_pid
    engine_pid=$(get_engine_pid || true)
    local running_engines=0
    local cand
    for cand in $(pgrep -f "live_paper_portfolio.py" 2>/dev/null); do
        if is_engine_process "$cand"; then
            local cand_cwd
            cand_cwd=$(readlink -f "/proc/$cand/cwd" 2>/dev/null || true)
            if [ "$cand_cwd" = "$PROJECT_DIR" ]; then
                running_engines=$((running_engines + 1))
            fi
        fi
    done

    if [ "$running_engines" -gt 1 ]; then
        echo "[FAIL] (Multiple duplicate engine processes detected: $running_engines)"
        failures=$((failures + 1))
    elif [ "$running_engines" -eq 1 ]; then
        echo "[PASS] (Engine running normally, PID: $engine_pid)"
    else
        echo "[INFO] (Engine is currently stopped)"
    fi

    # Check 6: Dashboard Availability
    echo -n "6. Dashboard Availability:        "
    local dash_pid
    dash_pid=$(get_dashboard_pid || true)
    if [ -n "$dash_pid" ]; then
        if curl -s -o /dev/null -m 2 "http://localhost:$DASHBOARD_PORT/"; then
            echo "[PASS] (Responsive at http://localhost:$DASHBOARD_PORT)"
        else
            echo "[WARN] (Process running PID $dash_pid, but HTTP port not responding)"
            warnings=$((warnings + 1))
        fi
    else
        echo "[INFO] (Dashboard server is stopped)"
    fi

    # Check 7: Read-Only Broker Connectivity
    echo -n "7. Broker API Connectivity:       "
    if [ "$broker" = "mock" ]; then
        echo "[PASS] (Mock Simulator - Local Mode)"
    elif [ "$broker" = "oanda" ]; then
        local b_status
        b_status=$("$PYTHON_BIN" "$OPS_HELPER" broker_summary "$CONFIG_FILE" 2>/dev/null | grep "CONNECTION:" | cut -d: -f2- | tr -d '[:space:]')
        if [[ "$b_status" == *"CONNECTED"* ]]; then
            echo "[PASS] (OANDA v20 API Connected)"
        else
            echo "[WARN] (OANDA API returned non-200 or timeout)"
            warnings=$((warnings + 1))
        fi
    else
        echo "[WARN] (Unknown broker type)"
        warnings=$((warnings + 1))
    fi

    # Check 8: Directory Writability
    echo -n "8. Log & PID Storage Writability: "
    if [ -w "$PROJECT_DIR" ] && [ -w "$PID_DIR" ]; then
        echo "[PASS] (Writable)"
    else
        echo "[FAIL] (Permission error writing to project or .run directory)"
        failures=$((failures + 1))
    fi

    # Check 9: Ledger State File
    echo -n "9. Portfolio Ledger File:         "
    if [ -f "$LEDGER_FILE" ]; then
        if "$PYTHON_BIN" -c "import json, sys; json.load(open(sys.argv[1]))" "$LEDGER_FILE" >/dev/null 2>&1; then
            echo "[PASS] (Valid JSON structure)"
        else
            echo "[FAIL] (Corrupted JSON in $LEDGER_FILE)"
            failures=$((failures + 1))
        fi
    else
        echo "[WARN] (No ledger file yet - will initialize on first startup)"
        warnings=$((warnings + 1))
    fi

    # Check 10: Broker-to-Ledger Reconciliation
    echo -n "10. Broker Reconciliation:        "
    local recon_out
    recon_out=$("$PYTHON_BIN" "$OPS_HELPER" reconcile "$CONFIG_FILE" "$LEDGER_FILE" 2>/dev/null)
    local r_stat
    r_stat=$(echo "$recon_out" | grep "^RECONCILIATION_STATUS:" | cut -d: -f2- | tr -d '[:space:]')
    if [ "$r_stat" = "RECONCILED" ]; then
        echo "[PASS] (All broker positions matched, zero discrepancy)"
    elif [ "$r_stat" = "RECONCILED_WITH_SIMULATED_POSITIONS" ]; then
        local r_counts
        r_counts=$(echo "$recon_out" | grep "^SUMMARY_COUNTS:" | cut -d: -f2- | sed 's/^ //')
        echo "[PASS] ($r_stat: $r_counts)"
    elif [ "$r_stat" = "RECONCILIATION_REQUIRED" ]; then
        echo "[WARN] (Unmatched position or ticket mismatch detected)"
        warnings=$((warnings + 1))
    else
        echo "[WARN] (Broker unavailable for reconciliation)"
        warnings=$((warnings + 1))
    fi

    echo "================================================================================"
    if [ "$failures" -eq 0 ]; then
        echo "Health Check Result: PASSED ($warnings warnings, 0 failures)"
        return 0
    else
        echo "Health Check Result: FAILED ($failures critical failures, $warnings warnings)"
        return 1
    fi
}

# ------------------------------------------------------------------------------
# Interactive Menu
# ------------------------------------------------------------------------------
interactive_menu() {
    while true; do
        echo ""
        echo "Forex AI Trading System Control"
        echo "-------------------------------"
        echo "1) Start trading system"
        echo "2) Stop trading system"
        echo "3) View system status"
        echo "4) View recent logs"
        echo "5) Run health checks"
        echo "6) Exit"
        echo ""
        read -r -p "Select an option [1-6]: " choice
        case "$choice" in
            1) cmd_start ;;
            2) cmd_stop ;;
            3) cmd_status ;;
            4) cmd_logs ;;
            5) cmd_health ;;
            6) echo "Exiting."; exit 0 ;;
            *) echo "Invalid selection. Please choose 1 to 6." ;;
        esac
    done
}

# ------------------------------------------------------------------------------
# Main Dispatcher
# ------------------------------------------------------------------------------
case "${1:-}" in
    start)
        cmd_start
        ;;
    stop)
        cmd_stop
        ;;
    status)
        cmd_status
        ;;
    logs)
        shift
        cmd_logs "$@"
        ;;
    health)
        cmd_health
        ;;
    help|--help|-h)
        echo "Usage: ./run_bot.sh [start|stop|status|logs|health|help]"
        echo ""
        echo "Commands:"
        echo "  start     Start the trading engine and monitoring web dashboard"
        echo "  stop      Gracefully stop running trading processes (leaves broker orders intact)"
        echo "  status    Display real-time process state, broker stats, and ledger positions"
        echo "  logs      View recent activity logs (use -f to follow, -n for line count)"
        echo "  health    Execute non-destructive diagnostic and connectivity checks"
        echo "  help      Display this help reference"
        echo ""
        echo "Running without arguments opens the interactive menu."
        exit 0
        ;;
    "")
        interactive_menu
        ;;
    *)
        echo "Unknown command: $1" >&2
        echo "Usage: ./run_bot.sh [start|stop|status|logs|health|help]" >&2
        exit 1
        ;;
esac
