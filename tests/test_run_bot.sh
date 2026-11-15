#!/usr/bin/env bash
# ==============================================================================
# Automated Test Suite for run_bot.sh and Operational Tools
# ==============================================================================
set -u

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

TOTAL_TESTS=0
PASSED_TESTS=0
FAILED_TESTS=0

run_test() {
    local test_name="$1"
    local command="$2"
    local expected_code="${3:-0}"

    TOTAL_TESTS=$((TOTAL_TESTS + 1))
    echo -n "Test $TOTAL_TESTS: $test_name... "

    local output
    local exit_code=0
    output=$(eval "$command" 2>&1) || exit_code=$?

    if [ "$exit_code" -eq "$expected_code" ]; then
        echo "[PASS]"
        PASSED_TESTS=$((PASSED_TESTS + 1))
        return 0
    else
        echo "[FAIL] (Expected exit code $expected_code, got $exit_code)"
        echo "Command output:"
        echo "$output" | sed 's/^/  /'
        FAILED_TESTS=$((FAILED_TESTS + 1))
        return 1
    fi
}

echo "================================================================================"
echo "Running Control Script Test Suite (tests/test_run_bot.sh)"
echo "Project Directory: $PROJECT_DIR"
echo "================================================================================"

# ------------------------------------------------------------------------------
# Test 1: Help flag and command returns 0
# ------------------------------------------------------------------------------
run_test "Help option '--help' displays usage and exits 0" \
    "./run_bot.sh --help | grep -q 'Usage: ./run_bot.sh'" 0

run_test "Help command 'help' displays usage and exits 0" \
    "./run_bot.sh help | grep -q 'Commands:'" 0

# ------------------------------------------------------------------------------
# Test 2: Invalid command exits with non-zero code
# ------------------------------------------------------------------------------
run_test "Unknown command returns exit code 1" \
    "./run_bot.sh invalid_command_xyz" 1

# ------------------------------------------------------------------------------
# Test 3: Health check executes and returns 0
# ------------------------------------------------------------------------------
run_test "Health check exits 0 with all checks passing" \
    "./run_bot.sh health | grep -q 'Health Check Result: PASSED'" 0

# ------------------------------------------------------------------------------
# Test 4: Status check executes and returns 0
# ------------------------------------------------------------------------------
run_test "Status command exits 0 and reports operating mode" \
    "./run_bot.sh status | grep -q 'Operating Mode:'" 0

# ------------------------------------------------------------------------------
# Test 5: Duplicate trading engine prevention
# ------------------------------------------------------------------------------
run_test "Duplicate start prevention reports already running and exits 0" \
    "./run_bot.sh start | grep -q 'DUPLICATE PREVENTED'" 0

# ------------------------------------------------------------------------------
# Test 6: Stale PID detection and cleanup
# ------------------------------------------------------------------------------
test_stale_pid() {
    local fake_pid=999999
    # Ensure fake pid is truly not alive
    if kill -0 "$fake_pid" 2>/dev/null; then
        return 1
    fi
    local test_pid_file="$PROJECT_DIR/.run/test_stale.pid"
    echo "$fake_pid" > "$test_pid_file"
    # Source run_bot helper functions to test is_engine_process on dead PID
    source <(grep -E '^(is_pid_alive|is_engine_process)\(\)' -A 20 ./run_bot.sh)
    if is_engine_process "$fake_pid"; then
        rm -f "$test_pid_file"
        return 1
    fi
    rm -f "$test_pid_file"
    return 0
}
run_test "Stale PID rejection correctly identifies dead processes" \
    "test_stale_pid" 0

# ------------------------------------------------------------------------------
# Test 7: Secret redaction in status and health
# ------------------------------------------------------------------------------
test_secret_masking() {
    # Extract raw OANDA token from broker_config.json
    local raw_token
    raw_token=$(grep -o '"api_token": *"[^"]*"' "$PROJECT_DIR/broker_config.json" | cut -d'"' -f4)
    if [ -z "$raw_token" ]; then
        return 0
    fi
    # Check if raw token is leaked in status output
    local status_out
    status_out=$(./run_bot.sh status 2>&1)
    if echo "$status_out" | grep -q "$raw_token"; then
        echo "Leak detected in status output!"
        return 1
    fi
    # Check if raw token is leaked in health output
    local health_out
    health_out=$(./run_bot.sh health 2>&1)
    if echo "$health_out" | grep -q "$raw_token"; then
        echo "Leak detected in health output!"
        return 1
    fi
    return 0
}
run_test "Secret redaction verifies OANDA API token is never exposed" \
    "test_secret_masking" 0

# ------------------------------------------------------------------------------
# Test 8: Live mode safety lock rejects unconfirmed live trading
# ------------------------------------------------------------------------------
test_live_lockout() {
    local tmp_cfg="$PROJECT_DIR/.run/test_live_cfg.json"
    mkdir -p "$PROJECT_DIR/.run"
    sed 's/"environment": "practice"/"environment": "live"/' "$PROJECT_DIR/broker_config.json" > "$tmp_cfg"
    
    # Test with CONFIG_FILE set to tmp_cfg and ALLOW_LIVE_TRADING=false
    local exit_code=0
    CONFIG_FILE="$tmp_cfg" ALLOW_LIVE_TRADING=false ./run_bot.sh start >/dev/null 2>&1 || exit_code=$?
    rm -f "$tmp_cfg"
    if [ "$exit_code" -eq 2 ]; then
        return 0
    else
        return 1
    fi
}
run_test "Practice vs Live safety lock halts unconfirmed live execution (exit 2)" \
    "test_live_lockout" 0

# ------------------------------------------------------------------------------
# Test 9: Verify run_bot.sh has zero trading or closeout side-effects
# ------------------------------------------------------------------------------
test_no_destructive_calls() {
    # Verify that run_bot.sh does not call close_position, cancel_order, or order endpoints
    if grep -E "(close_position|order_post|cancel_order|positions/close)" ./run_bot.sh >/dev/null; then
        echo "Dangerous trading execution found in run_bot.sh!"
        return 1
    fi
    return 0
}
run_test "Control script safety audit verifies zero trading or liquidation side-effects" \
    "test_no_destructive_calls" 0

# ------------------------------------------------------------------------------
# Test 11: Broker-to-ledger reconciliation check
# ------------------------------------------------------------------------------
run_test "Broker reconciliation check reports valid status" \
    "./env/bin/python ops_helper.py reconcile broker_config.json portfolio_ledger.json | grep -q 'RECONCILIATION_STATUS:'" 0

# ------------------------------------------------------------------------------
# Test Summary
# ------------------------------------------------------------------------------
echo "================================================================================"
echo "Test Results: $PASSED_TESTS / $TOTAL_TESTS tests passed ($FAILED_TESTS failures)"
echo "================================================================================"

if [ "$FAILED_TESTS" -eq 0 ]; then
    exit 0
else
    exit 1
fi
