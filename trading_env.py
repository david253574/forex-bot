"""Custom Gymnasium trading environment for PPO-based forex agents with Multi-Pair and Advanced Risk Management."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class TradingEnv(gym.Env):
    """
    Institutional forex trading environment supporting:
    - Multi-currency pair pool (EUR/USD, GBP/USD, USD/JPY, AUD/USD)
    - Dynamic ATR position sizing (strictly 2% account risk)
    - Chandelier ATR Trailing Stop (ratchets behind new highs/lows)
    - Partial profit taking scale-out (realizes 50% at 3.0 ATR, moves stop to Breakeven)
    - Real-time portfolio state awareness (position, unrealized PnL ratio, duration)
    - Realistic broker spread, commissions, and execution slippage

    Actions:
        0 = Hold / Stay Flat (or stay in current position)
        1 = Buy  (open Long if flat; close Short to flat)
        2 = Sell (open Short if flat; close Long to flat)
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        feature_data: Optional[np.ndarray] = None,
        price_data: Optional[np.ndarray] = None,
        high_data: Optional[np.ndarray] = None,
        low_data: Optional[np.ndarray] = None,
        raw_atr_data: Optional[np.ndarray] = None,
        pair_data: Optional[Dict[str, Dict[str, np.ndarray]]] = None,
        feature_map: Optional[Dict[str, int]] = None,
        window_size: int = 32,
        initial_balance: float = 10_000.0,
        trade_size: float = 100_000.0,
        holding_penalty: float = 0.0,
        max_hold_steps: int = 72,
        inactivity_threshold: int = 64,
        reward_scaling: float = 100.0,
        execution_penalty: float = 0.05,
        spread: float = 0.00008,
        commission: float = 0.5,
        slippage_std: float = 0.00003,
        risk_per_trade: float = 0.02,
        min_hold_steps: int = 4,
        include_portfolio_state: bool = True,
        random_start: bool = False,
        # Advanced Risk Management parameters (default False for proven asymmetric 1:3 payoff)
        enable_trailing_stop: bool = False,
        chandelier_atr_mult: float = 2.5,
        activation_atr_mult: float = 2.0,
        enable_partial_tp: bool = False,
        partial_tp_mult: float = 3.0,
        partial_tp_ratio: float = 0.5,
    ) -> None:
        super().__init__()

        self.pair_data = pair_data
        self.available_pairs: List[str] = list(pair_data.keys()) if pair_data else []
        self.current_pair: str = self.available_pairs[0] if self.available_pairs else "DEFAULT"

        if pair_data:
            # Extract initial pair dataset
            first_pair = self.available_pairs[0]
            feature_data = pair_data[first_pair]["feature_data"]
            price_data = pair_data[first_pair]["price_data"]
            high_data = pair_data[first_pair].get("high_data", price_data)
            low_data = pair_data[first_pair].get("low_data", price_data)
            raw_atr_data = pair_data[first_pair].get("raw_atr_data")

        if feature_data is None or price_data is None:
            raise ValueError("Either pair_data or (feature_data, price_data) must be provided")

        if len(feature_data) != len(price_data):
            raise ValueError("feature_data and price_data must have the same length")

        feature_map = feature_map or {}

        self.feature_data = feature_data.astype(np.float32)
        self.price_data = price_data.astype(np.float32)
        self.high_data = high_data.astype(np.float32) if high_data is not None else self.price_data
        self.low_data = low_data.astype(np.float32) if low_data is not None else self.price_data
        self.raw_atr_data = raw_atr_data.astype(np.float32) if raw_atr_data is not None else None
        self.atr_idx = int(feature_map.get("atr", 7))

        self.window_size = int(window_size)
        self.initial_balance = float(initial_balance)
        self.trade_size = float(trade_size)
        self.holding_penalty = float(holding_penalty)
        self.max_hold_steps = int(max_hold_steps)
        self.inactivity_threshold = int(inactivity_threshold)
        self.reward_scaling = float(reward_scaling)
        self.execution_penalty = float(execution_penalty)
        self.spread = float(spread)
        self.commission = float(commission)
        self.slippage_std = float(slippage_std)
        self.risk_per_trade = float(risk_per_trade)
        self.min_hold_steps = int(min_hold_steps)
        self.include_portfolio_state = bool(include_portfolio_state)
        self.random_start = bool(random_start)

        # Advanced Risk Management Settings
        self.enable_trailing_stop = bool(enable_trailing_stop)
        self.chandelier_atr_mult = float(chandelier_atr_mult)
        self.activation_atr_mult = float(activation_atr_mult)
        self.enable_partial_tp = bool(enable_partial_tp)
        self.partial_tp_mult = float(partial_tp_mult)
        self.partial_tp_ratio = float(partial_tp_ratio)

        # 4 portfolio features: position, unrealized_pnl_ratio, trade_duration_ratio, inactivity_ratio
        portfolio_features_dim = 4 if self.include_portfolio_state else 0
        total_feature_dim = self.feature_data.shape[1] + portfolio_features_dim

        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(
            low=-10.0,
            high=10.0,
            shape=(self.window_size, total_feature_dim),
            dtype=np.float32,
        )

        self.current_step = self.window_size
        self.position = 0  # -1 short, 0 flat, +1 long
        self.entry_price = 0.0
        self.entry_atr = 0.0010
        self.stop_loss_price = 0.0
        self.take_profit_price = 0.0
        self.position_size = 0.0

        # Trailing stop and partial TP tracking
        self.highest_price = 0.0
        self.lowest_price = float("inf")
        self.partial_tp_hit = False

        self.balance = self.initial_balance
        self.equity = self.initial_balance
        self.peak_equity = self.initial_balance
        self.max_drawdown = 0.0
        self.time_in_trade = 0
        self.consecutive_hold_steps = 0
        self.total_closed_trades = 0
        self.profitable_closed_trades = 0
        self.active_trade_open_cost = 0.0

    def update_frictions(
        self,
        current_iter: int,
        total_iters: int,
        max_spread: float,
        max_commission: float,
        max_slippage: float,
    ) -> None:
        """Ramp execution friction from zero to full cost over the first 80% of training."""
        if total_iters <= 1:
            multiplier = 1.0
        else:
            ramp_end = max(total_iters * 0.8, 1.0)
            ramp_denominator = max(ramp_end - 1.0, 1e-8)
            multiplier = min(1.0, max(0.0, (current_iter - 1.0) / ramp_denominator))

        self.spread = float(max_spread * multiplier)
        self.commission = float(max_commission * multiplier)
        self.slippage_std = float(max_slippage * multiplier)

    def _sample_slippage(self) -> float:
        if self.slippage_std <= 0.0:
            return 0.0
        return max(0.0, float(self.np_random.normal(0.0, self.slippage_std)))

    def _calculate_unrealized_pnl(self, price: float) -> float:
        if self.position == 1:
            return (price - self.entry_price) * self.position_size
        if self.position == -1:
            return (self.entry_price - price) * self.position_size
        return 0.0

    def _get_observation(self) -> np.ndarray:
        start = self.current_step - self.window_size
        end = self.current_step
        market_features = self.feature_data[start:end]

        if not self.include_portfolio_state:
            return market_features

        curr_price = float(self.price_data[self.current_step])
        unrealized = self._calculate_unrealized_pnl(curr_price)

        pos_feat = float(self.position)
        unrealized_ratio = float(np.clip(unrealized / max(self.balance, 1.0), -0.2, 0.2)) * 5.0
        duration_ratio = (
            min(1.0, float(self.time_in_trade) / max(float(self.max_hold_steps), 1.0))
            if self.position != 0
            else 0.0
        )
        inactivity_ratio = min(
            1.0,
            float(self.consecutive_hold_steps) / max(float(self.inactivity_threshold), 1.0),
        )

        port_features = np.array(
            [pos_feat, unrealized_ratio, duration_ratio, inactivity_ratio],
            dtype=np.float32,
        )
        port_matrix = np.tile(port_features, (self.window_size, 1))

        return np.concatenate([market_features, port_matrix], axis=1)

    def _open_position(self, action: int, price: float, current_atr: float) -> Tuple[float, float]:
        if current_atr <= 0.0 or np.isnan(current_atr):
            current_atr = price * 0.0010

        self.entry_price = float(price)
        self.entry_atr = float(current_atr)

        # 2% account risk
        risk_amount = self.balance * self.risk_per_trade
        sl_distance = 2.0 * self.entry_atr

        # Risk / (2 * ATR) guarantees hitting the SL loses exactly risk_amount
        raw_size = risk_amount / max(sl_distance, 1e-6)

        # Cap leverage at 20:1 for account safety
        max_leverage_size = (self.balance * 20.0) / max(self.entry_price, 1e-4)
        self.position_size = min(raw_size, max_leverage_size)

        self.position = 1 if action == 1 else -1

        # Anchor Stop-Loss (2 ATR) and Take-Profit (6.0 ATR -> 1:3 Risk/Reward)
        if self.position == 1:
            self.stop_loss_price = self.entry_price - sl_distance
            self.take_profit_price = self.entry_price + (6.0 * self.entry_atr)
            self.highest_price = self.entry_price
            self.lowest_price = self.entry_price
        else:
            self.stop_loss_price = self.entry_price + sl_distance
            self.take_profit_price = self.entry_price - (6.0 * self.entry_atr)
            self.highest_price = self.entry_price
            self.lowest_price = self.entry_price

        self.partial_tp_hit = False

        slippage = self._sample_slippage()
        slippage_cost = slippage * self.position_size
        spread_cost = self.spread * self.position_size
        entry_cost = spread_cost + slippage_cost

        self.balance -= entry_cost
        self.active_trade_open_cost = entry_cost
        self.time_in_trade = 0

        return spread_cost, slippage_cost

    def _close_position(self, exit_price: float) -> Tuple[float, float, float, float]:
        if self.position == 1:
            gross_pnl = (exit_price - self.entry_price) * self.position_size
        elif self.position == -1:
            gross_pnl = (self.entry_price - exit_price) * self.position_size
        else:
            return 0.0, 0.0, 0.0, 0.0

        close_slippage = self._sample_slippage()
        close_slippage_cost = close_slippage * self.position_size
        commission_cost = self.commission

        net_pnl = gross_pnl - close_slippage_cost - commission_cost - self.active_trade_open_cost

        self.balance += gross_pnl - close_slippage_cost - commission_cost
        if net_pnl > 0.0:
            self.profitable_closed_trades += 1
        self.total_closed_trades += 1

        self.position = 0
        self.entry_price = 0.0
        self.position_size = 0.0
        self.time_in_trade = 0
        self.active_trade_open_cost = 0.0
        self.partial_tp_hit = False
        self.equity = self.balance

        return gross_pnl, close_slippage_cost, commission_cost, net_pnl

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)

        # In multi-pair mode, optionally select a specific pair or randomly sample from the pool
        if self.pair_data:
            if options and "pair" in options and options["pair"] in self.pair_data:
                self.current_pair = options["pair"]
            else:
                self.current_pair = str(self.np_random.choice(self.available_pairs))

            pair_dict = self.pair_data[self.current_pair]
            self.feature_data = pair_dict["feature_data"]
            self.price_data = pair_dict["price_data"]
            self.high_data = pair_dict.get("high_data", self.price_data)
            self.low_data = pair_dict.get("low_data", self.price_data)
            self.raw_atr_data = pair_dict.get("raw_atr_data")

        # In training mode, pick a random valid starting point to ensure exposure to diverse regimes
        if self.random_start and len(self.price_data) > self.window_size + 1500:
            max_start = len(self.price_data) - 1000
            self.current_step = int(self.np_random.integers(self.window_size, max_start))
        else:
            self.current_step = self.window_size

        self.position = 0
        self.entry_price = 0.0
        self.entry_atr = 0.0010
        self.stop_loss_price = 0.0
        self.take_profit_price = 0.0
        self.position_size = 0.0
        self.highest_price = 0.0
        self.lowest_price = float("inf")
        self.partial_tp_hit = False

        self.balance = self.initial_balance
        self.equity = self.initial_balance
        self.peak_equity = self.initial_balance
        self.max_drawdown = 0.0
        self.time_in_trade = 0
        self.consecutive_hold_steps = 0
        self.total_closed_trades = 0
        self.profitable_closed_trades = 0
        self.active_trade_open_cost = 0.0

        return self._get_observation(), {"balance": self.balance, "equity": self.equity, "pair": self.current_pair}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action: {action}")

        old_equity = self.equity
        price = float(self.price_data[self.current_step])
        high = float(self.high_data[self.current_step])
        low = float(self.low_data[self.current_step])

        if self.raw_atr_data is not None:
            current_atr = float(self.raw_atr_data[self.current_step])
        else:
            current_atr = price * float(self.feature_data[self.current_step, self.atr_idx])

        trade_closed = False
        trade_opened = False
        realized_pnl = 0.0
        spread_paid = 0.0
        slippage_paid = 0.0
        commission_paid = 0.0
        closed_trade_net_pnl = 0.0

        # --- 1. ADVANCED RISK MANAGEMENT ---
        if self.position != 0:
            # Track intra-trade extreme prices
            self.highest_price = max(self.highest_price, high)
            self.lowest_price = min(self.lowest_price, low)

            # A. Chandelier ATR Trailing Stop (Ratchets stop-loss behind peak price)
            if self.enable_trailing_stop:
                if self.position == 1 and self.highest_price >= self.entry_price + (self.activation_atr_mult * self.entry_atr):
                    trailing_level = self.highest_price - (self.chandelier_atr_mult * self.entry_atr)
                    self.stop_loss_price = max(self.stop_loss_price, trailing_level)
                elif self.position == -1 and self.lowest_price <= self.entry_price - (self.activation_atr_mult * self.entry_atr):
                    trailing_level = self.lowest_price + (self.chandelier_atr_mult * self.entry_atr)
                    self.stop_loss_price = min(self.stop_loss_price, trailing_level)

            # B. Partial Profit Taking Scale-Out (takes 50% at partial_tp_mult * ATR)
            if self.enable_partial_tp and not self.partial_tp_hit:
                partial_target = (
                    self.entry_price + (self.partial_tp_mult * self.entry_atr)
                    if self.position == 1
                    else self.entry_price - (self.partial_tp_mult * self.entry_atr)
                )
                partial_hit = (self.position == 1 and high >= partial_target) or (
                    self.position == -1 and low <= partial_target
                )

                if partial_hit:
                    self.partial_tp_hit = True
                    scale_size = self.position_size * self.partial_tp_ratio
                    gpnl = (
                        (partial_target - self.entry_price) * scale_size
                        if self.position == 1
                        else (self.entry_price - partial_target) * scale_size
                    )
                    comm = self.commission * self.partial_tp_ratio
                    self.balance += gpnl - comm
                    realized_pnl += gpnl
                    commission_paid += comm

                    self.position_size -= scale_size
                    self.active_trade_open_cost *= (1.0 - self.partial_tp_ratio)

                    # Move Stop-Loss to Breakeven (+0.1 ATR) to guarantee remaining half is risk-free
                    if self.position == 1:
                        self.stop_loss_price = max(self.stop_loss_price, self.entry_price + (0.1 * self.entry_atr))
                    else:
                        self.stop_loss_price = min(self.stop_loss_price, self.entry_price - (0.1 * self.entry_atr))

            # C. Check full bracket exit (Stop-Loss or Final Take-Profit)
            sl_hit = False
            tp_hit = False
            exit_price = price

            if self.position == 1:  # Long
                if low <= self.stop_loss_price:
                    sl_hit = True
                    exit_price = self.stop_loss_price
                elif high >= self.take_profit_price:
                    tp_hit = True
                    exit_price = self.take_profit_price
            elif self.position == -1:  # Short
                if high >= self.stop_loss_price:
                    sl_hit = True
                    exit_price = self.stop_loss_price
                elif low <= self.take_profit_price:
                    tp_hit = True
                    exit_price = self.take_profit_price

            if sl_hit or tp_hit:
                gpnl, slip, comm, npnl = self._close_position(exit_price)
                realized_pnl += gpnl
                slippage_paid += slip
                commission_paid += comm
                closed_trade_net_pnl = npnl
                trade_closed = True

        # --- 2. Check Maximum Holding Duration Exit ---
        if self.position != 0 and self.time_in_trade >= self.max_hold_steps:
            gpnl, slip, comm, npnl = self._close_position(price)
            realized_pnl += gpnl
            slippage_paid += slip
            commission_paid += comm
            closed_trade_net_pnl = npnl
            trade_closed = True

        # --- 3. Process Agent Action ---
        if action == 1:  # Buy / Bullish Signal
            if self.position == -1 and self.time_in_trade >= self.min_hold_steps and not trade_closed:
                # Close short position to flat
                gpnl, slip, comm, npnl = self._close_position(price)
                realized_pnl += gpnl
                slippage_paid += slip
                commission_paid += comm
                closed_trade_net_pnl = npnl
                trade_closed = True
            elif self.position == 0 and not trade_closed:
                # Open Long
                sp, slip = self._open_position(action=1, price=price, current_atr=current_atr)
                spread_paid += sp
                slippage_paid += slip
                trade_opened = True

        elif action == 2:  # Sell / Bearish Signal
            if self.position == 1 and self.time_in_trade >= self.min_hold_steps and not trade_closed:
                # Close long position to flat
                gpnl, slip, comm, npnl = self._close_position(price)
                realized_pnl += gpnl
                slippage_paid += slip
                commission_paid += comm
                closed_trade_net_pnl = npnl
                trade_closed = True
            elif self.position == 0 and not trade_closed:
                # Open Short
                sp, slip = self._open_position(action=2, price=price, current_atr=current_atr)
                spread_paid += sp
                slippage_paid += slip
                trade_opened = True

        # Track consecutive hold actions
        if action == 0 and self.position == 0:
            self.consecutive_hold_steps += 1
        else:
            self.consecutive_hold_steps = 0

        if self.position != 0:
            self.time_in_trade += 1
        else:
            self.time_in_trade = 0

        # Update equity
        unrealized = self._calculate_unrealized_pnl(price)
        self.equity = self.balance + unrealized
        self.peak_equity = max(self.peak_equity, self.equity)
        current_dd = max(0.0, (self.peak_equity - self.equity) / max(self.peak_equity, 1e-8))
        self.max_drawdown = max(self.max_drawdown, current_dd)

        # Advance step
        self.current_step += 1
        terminated = self.current_step >= len(self.price_data) - 1
        truncated = False

        # End of episode: force close any open position
        if terminated and self.position != 0:
            final_price = float(self.price_data[self.current_step])
            gpnl, slip, comm, npnl = self._close_position(final_price)
            realized_pnl += gpnl
            slippage_paid += slip
            commission_paid += comm
            closed_trade_net_pnl = npnl
            trade_closed = True
            self.equity = self.balance

        # Liquidation condition (loss of 80% capital)
        if self.equity <= 0.20 * self.initial_balance:
            terminated = True

        # --- REWARD CALCULATION ---
        equity_delta = self.equity - old_equity
        pct_return = (equity_delta / max(old_equity, 1e-8)) * 100.0

        open_penalty = 0.0
        if trade_opened and self.execution_penalty > 0.0:
            open_penalty = (self.execution_penalty / self.initial_balance) * 100.0

        inactivity_penalty = 0.0
        if self.consecutive_hold_steps > self.inactivity_threshold:
            inactivity_penalty = 0.002

        hold_penalty = 0.0
        if self.position != 0 and self.holding_penalty > 0.0:
            hold_penalty = (self.holding_penalty / self.initial_balance) * 100.0

        step_reward = pct_return - open_penalty - inactivity_penalty - hold_penalty
        reward = float(np.clip(step_reward, -2.0, 2.0))

        info = {
            "balance": self.balance,
            "equity": self.equity,
            "position": self.position,
            "realized_pnl": realized_pnl,
            "max_drawdown": self.max_drawdown,
            "price": price,
            "spread_paid": spread_paid,
            "slippage_paid": slippage_paid,
            "commission_paid": commission_paid,
            "trade_closed": trade_closed,
            "closed_trade_net_pnl": closed_trade_net_pnl,
            "total_closed_trades": self.total_closed_trades,
            "profitable_closed_trades": self.profitable_closed_trades,
            "partial_tp_hit": self.partial_tp_hit,
            "pair": self.current_pair,
        }

        return self._get_observation(), reward, terminated, truncated, info

    def render(self) -> None:
        print(
            f"pair={self.current_pair} step={self.current_step} balance={self.balance:.2f} "
            f"equity={self.equity:.2f} position={self.position} max_dd={self.max_drawdown:.4f}"
        )
