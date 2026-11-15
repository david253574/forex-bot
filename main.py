"""Main training entrypoint for PPO-based forex trading agent."""

from __future__ import annotations

import argparse
import os
import pickle
import tempfile
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.optim import Adam

from data_loader import load_and_preprocess_data
from model import TradingLSTMPolicy
from trading_env import TradingEnv


@dataclass
class RolloutBuffer:
    states: List[np.ndarray] = field(default_factory=list)
    actions: List[int] = field(default_factory=list)
    log_probs: List[float] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)
    dones: List[float] = field(default_factory=list)
    values: List[float] = field(default_factory=list)

    def clear(self) -> None:
        self.states.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.dones.clear()
        self.values.clear()


class PPOTrainer:
    def __init__(
        self,
        env: TradingEnv,
        model: TradingLSTMPolicy,
        device: torch.device,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.02,
        learning_rate: float = 3e-4,
        rollout_steps: int = 512,
        ppo_epochs: int = 4,
        mini_batch_size: int = 64,
        max_spread: float = 0.0,
        max_commission: float = 0.0,
        max_slippage: float = 0.0,
    ) -> None:
        self.env = env
        self.model = model.to(device)
        self.device = device

        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.rollout_steps = rollout_steps
        self.ppo_epochs = ppo_epochs
        self.mini_batch_size = mini_batch_size
        self.max_spread = max_spread
        self.max_commission = max_commission
        self.max_slippage = max_slippage

        # Adam optimizer with modest weight decay for regularization
        self.optimizer = Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-4)
        self.buffer = RolloutBuffer()

        # Persistent state across rollouts to ensure continuous traversal across full dataset
        self._current_state = None

    def _select_action(self, state: np.ndarray) -> Tuple[int, float, float]:
        state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            action_probs, state_value, _ = self.model(state_tensor)

        dist = Categorical(probs=action_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        return int(action.item()), float(log_prob.item()), float(state_value.item())

    def _compute_gae(self, next_value: float) -> Tuple[np.ndarray, np.ndarray]:
        rewards = np.array(self.buffer.rewards, dtype=np.float32)
        dones = np.array(self.buffer.dones, dtype=np.float32)
        values = np.array(self.buffer.values + [next_value], dtype=np.float32)

        advantages = np.zeros_like(rewards, dtype=np.float32)
        gae = 0.0

        for step in reversed(range(len(rewards))):
            mask = 1.0 - dones[step]
            delta = rewards[step] + self.gamma * values[step + 1] * mask - values[step]
            gae = delta + self.gamma * self.gae_lambda * mask * gae
            advantages[step] = gae

        returns = advantages + values[:-1]
        return advantages, returns

    def _collect_rollout(self) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
        self.buffer.clear()

        # Initialize state if starting for the first time or after a completed run
        if self._current_state is None:
            self._current_state, _ = self.env.reset()

        state = self._current_state
        rollout_rewards = 0.0
        closed_trades_start = self.env.total_closed_trades
        profitable_start = self.env.profitable_closed_trades

        for _ in range(self.rollout_steps):
            action, log_prob, value = self._select_action(state)
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated

            self.buffer.states.append(state)
            self.buffer.actions.append(action)
            self.buffer.log_probs.append(log_prob)
            self.buffer.rewards.append(float(reward))
            self.buffer.dones.append(float(done))
            self.buffer.values.append(value)

            rollout_rewards += float(reward)
            state = next_state

            if done:
                state, _ = self.env.reset()

        self._current_state = state

        # Bootstrap value for the last step
        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            _, next_value, _ = self.model(state_tensor)

        advantages, returns = self._compute_gae(float(next_value.item()))

        new_closed_trades = self.env.total_closed_trades - closed_trades_start
        new_profitable = self.env.profitable_closed_trades - profitable_start
        win_rate = (new_profitable / new_closed_trades * 100.0) if new_closed_trades > 0 else 0.0

        stats = {
            "rollout_reward": rollout_rewards,
            "equity": self.env.equity,
            "closed_trades": new_closed_trades,
            "win_rate": win_rate,
            "max_dd": self.env.max_drawdown * 100.0,
        }

        return advantages, returns, stats

    def _update_policy(self, advantages: np.ndarray, returns: np.ndarray) -> Dict[str, float]:
        states = torch.tensor(np.array(self.buffer.states), dtype=torch.float32, device=self.device)
        actions = torch.tensor(np.array(self.buffer.actions), dtype=torch.long, device=self.device)
        old_log_probs = torch.tensor(np.array(self.buffer.log_probs), dtype=torch.float32, device=self.device)
        advantages_t = torch.tensor(advantages, dtype=torch.float32, device=self.device)
        returns_t = torch.tensor(returns, dtype=torch.float32, device=self.device)

        # Global advantage normalization across entire rollout buffer
        advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)

        total_loss_value = 0.0
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        num_updates = 0

        num_samples = states.size(0)

        for _ in range(self.ppo_epochs):
            indices = torch.randperm(num_samples, device=self.device)

            for start in range(0, num_samples, self.mini_batch_size):
                end = start + self.mini_batch_size
                mb_idx = indices[start:end]

                mb_states = states[mb_idx]
                mb_actions = actions[mb_idx]
                mb_old_log_probs = old_log_probs[mb_idx]
                mb_advantages = advantages_t[mb_idx]
                mb_returns = returns_t[mb_idx]

                action_probs, state_values, _ = self.model(mb_states)
                dist = Categorical(probs=action_probs)
                new_log_probs = dist.log_prob(mb_actions)
                entropy = dist.entropy().mean()

                ratio = torch.exp(new_log_probs - mb_old_log_probs)
                clipped_ratio = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon)

                surrogate_1 = ratio * mb_advantages
                surrogate_2 = clipped_ratio * mb_advantages
                policy_loss = -torch.min(surrogate_1, surrogate_2).mean()

                value_loss = F.mse_loss(state_values, mb_returns)
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
                self.optimizer.step()

                total_loss_value += float(loss.item())
                total_policy_loss += float(policy_loss.item())
                total_value_loss += float(value_loss.item())
                total_entropy += float(entropy.item())
                num_updates += 1

        return {
            "loss": total_loss_value / max(num_updates, 1),
            "policy_loss": total_policy_loss / max(num_updates, 1),
            "value_loss": total_value_loss / max(num_updates, 1),
            "entropy": total_entropy / max(num_updates, 1),
        }

    def train(self, iterations: int = 100) -> None:
        try:
            for iteration in range(1, iterations + 1):
                self.env.update_frictions(
                    current_iter=iteration,
                    total_iters=iterations,
                    max_spread=self.max_spread,
                    max_commission=self.max_commission,
                    max_slippage=self.max_slippage,
                )
                advantages, returns, stats = self._collect_rollout()
                metrics = self._update_policy(advantages, returns)

                if iteration % 5 == 0 or iteration == 1 or iteration == iterations:
                    print(
                        f"Iter {iteration:03d}/{iterations:03d} | "
                        f"Loss: {metrics['loss']:.4f} (Pol: {metrics['policy_loss']:.4f}, Val: {metrics['value_loss']:.4f}, Ent: {metrics['entropy']:.4f}) | "
                        f"Rollout Return: {stats['rollout_reward']:+.2f}% | "
                        f"Trades: {stats['closed_trades']} (Win: {stats['win_rate']:.1f}%) | "
                        f"Equity: ${stats['equity']:.2f} | "
                        f"Friction (Sprd: {self.env.spread:.5f})"
                    )
        except KeyboardInterrupt:
            print("\nTraining interrupted! Saving current model weights...")
            torch.save(self.model.state_dict(), "model.pt")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a PPO forex trading agent")
    parser.add_argument("--csv", type=str, default="forex_data.csv", help="Path to OHLCV CSV file")
    parser.add_argument("--iterations", type=int, default=100, help="PPO training iterations")
    parser.add_argument("--window-size", type=int, default=32, help="Observation sequence window")
    parser.add_argument("--rollout-steps", type=int, default=512, help="Steps collected per PPO rollout")
    parser.add_argument("--learning-rate", type=float, default=3e-4, help="Learning rate for Adam")
    parser.add_argument("--entropy-coef", type=float, default=0.02, help="Entropy coefficient for exploration")
    parser.add_argument("--spread", type=float, default=0.00008, help="Spread charged on trade open (0.8 pips default)")
    parser.add_argument(
        "--commission",
        type=float,
        default=0.5,
        help="Flat commission charged when closing trades",
    )
    parser.add_argument(
        "--slippage-std",
        type=float,
        default=0.00003,
        help="Stddev for slippage overhead per execution",
    )
    parser.add_argument("--save-path", type=str, default="model.pt", help="Model checkpoint output path")
    parser.add_argument("--scaler-path", type=str, default="scaler.pkl", help="Fitted scaler output path")
    parser.add_argument("--test-export-path", type=str, default="forex_data_test.csv", help="Test export path")
    parser.add_argument(
        "--reset-weights",
        action="store_true",
        help="Delete existing checkpoint to force random initialization",
    )
    parser.add_argument("--execution-penalty", type=float, default=0.05, help="Penalty per trade open")
    parser.add_argument("--max-hold-steps", type=int, default=72, help="Max steps before forced exit")
    parser.add_argument("--holding-penalty", type=float, default=0.0, help="Penalty per step in position")
    parser.add_argument(
        "--train-split",
        type=float,
        default=0.8,
        help="Fraction of data used for training (remainder is test set)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if getattr(args, "reset_weights", False) and os.path.exists(args.save_path):
        os.remove(args.save_path)
        print(f"Removed existing checkpoint: {args.save_path}")

    # Load and preprocess dataset
    _, _, feature_columns, enriched_df = load_and_preprocess_data(args.csv)
    print(f"Total dataset samples: {len(enriched_df)}")

    # Strict chronological split: train first 80%, test last 20%
    split_idx = int(len(enriched_df) * args.train_split)
    train_enriched = enriched_df.iloc[:split_idx].reset_index(drop=True)
    test_enriched = enriched_df.iloc[split_idx:].reset_index(drop=True)

    print(f"Train samples: {len(train_enriched)} | Test samples: {len(test_enriched)}")

    # Fit scaler strictly on training features to prevent lookahead bias
    from sklearn.preprocessing import MinMaxScaler
    train_scaler = MinMaxScaler()
    train_normalized = pd.DataFrame(
        train_scaler.fit_transform(train_enriched[feature_columns]),
        columns=feature_columns,
    )

    with open(args.scaler_path, "wb") as scaler_file:
        pickle.dump(train_scaler, scaler_file)
    print(f"Fitted training scaler saved to {args.scaler_path}")

    # Export quarantined out-of-sample test data
    test_export = test_enriched.copy()
    test_export[feature_columns] = train_scaler.transform(test_enriched[feature_columns])
    test_export.to_csv(args.test_export_path, index=False)
    print(f"Quarantined test set exported to {args.test_export_path} ({len(test_export)} bars)")

    raw_train_atr = train_enriched["atr_14"].values.astype(np.float32)

    env = TradingEnv(
        feature_data=train_normalized[feature_columns].values,
        price_data=train_enriched["close"].values,
        high_data=train_enriched["high"].values,
        low_data=train_enriched["low"].values,
        raw_atr_data=raw_train_atr,
        feature_map={"atr": 7},
        window_size=args.window_size,
        spread=args.spread,
        commission=args.commission,
        slippage_std=args.slippage_std,
        execution_penalty=args.execution_penalty,
        max_hold_steps=args.max_hold_steps,
        holding_penalty=args.holding_penalty,
        include_portfolio_state=True,
        random_start=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_size = env.observation_space.shape[1]
    print(f"Observation space: {env.observation_space.shape} (Features: {input_size}) on Device: {device}")

    model = TradingLSTMPolicy(input_size=input_size)
    if not getattr(args, "reset_weights", False) and os.path.exists(args.save_path):
        try:
            model.load_state_dict(torch.load(args.save_path, map_location=device))
            print(f"Resuming training from existing checkpoint: {args.save_path}")
        except Exception as e:
            print(f"Could not load existing checkpoint ({e}), starting with fresh weights.")

    trainer = PPOTrainer(
        env=env,
        model=model,
        device=device,
        learning_rate=args.learning_rate,
        entropy_coef=args.entropy_coef,
        rollout_steps=args.rollout_steps,
        max_spread=args.spread,
        max_commission=args.commission,
        max_slippage=args.slippage_std,
    )

    print("\nStarting PPO training across full historical dataset...")
    trainer.train(iterations=args.iterations)

    # Atomic checkpoint save
    dirpath = os.path.dirname(os.path.abspath(args.save_path)) or "."
    with tempfile.NamedTemporaryFile(dir=dirpath, delete=False) as tmpf:
        tmp_path = tmpf.name
    try:
        torch.save(model.state_dict(), tmp_path)
        os.replace(tmp_path, args.save_path)
        print(f"\nSuccessfully saved trained model weights to: {args.save_path}")
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


if __name__ == "__main__":
    main()
