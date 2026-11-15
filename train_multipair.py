"""
Multi-currency pair training script for PPO-based forex trading agent.
Trains a single generalized policy across EURUSD, GBPUSD, USDJPY, and AUDUSD.
"""

from __future__ import annotations

import argparse
import os
import pickle
import tempfile
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler

from data_loader import load_and_preprocess_data
from main import PPOTrainer
from model import TradingLSTMPolicy
from trading_env import TradingEnv

DEFAULT_PAIRS = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD"]


def load_multipair_data(
    pairs: List[str], data_dir: str = "data", train_split: float = 0.8
) -> Tuple[Dict[str, Dict[str, np.ndarray]], Dict[str, Dict[str, np.ndarray]], MinMaxScaler, List[str]]:
    data_path = Path(data_dir)
    raw_train_dfs: Dict[str, pd.DataFrame] = {}
    raw_test_dfs: Dict[str, pd.DataFrame] = {}
    feature_columns: List[str] = []

    print("Loading and preprocessing currency pair datasets...")
    for pair in pairs:
        csv_file = data_path / f"{pair}.csv"
        if not csv_file.exists():
            raise FileNotFoundError(f"Missing data file: {csv_file}. Please run download_pairs.py first.")

        _, _, cols, enriched_df = load_and_preprocess_data(str(csv_file))
        feature_columns = cols

        split_idx = int(len(enriched_df) * train_split)
        train_df = enriched_df.iloc[:split_idx].reset_index(drop=True)
        test_df = enriched_df.iloc[split_idx:].reset_index(drop=True)

        raw_train_dfs[pair] = train_df
        raw_test_dfs[pair] = test_df
        print(f"  {pair}: {len(train_df)} train bars, {len(test_df)} test bars")

    # Fit a single unified MinMaxScaler across all training pair features combined
    combined_train_features = pd.concat(
        [raw_train_dfs[pair][feature_columns] for pair in pairs], ignore_index=True
    )
    global_scaler = MinMaxScaler()
    global_scaler.fit(combined_train_features)
    print(f"✓ Fitted global feature scaler on {len(combined_train_features)} combined bars across {len(pairs)} pairs")

    with open("scaler_multipair.pkl", "wb") as f:
        pickle.dump(global_scaler, f)

    # Transform features for each pair
    train_pair_data: Dict[str, Dict[str, np.ndarray]] = {}
    test_pair_data: Dict[str, Dict[str, np.ndarray]] = {}

    for pair in pairs:
        train_df = raw_train_dfs[pair]
        test_df = raw_test_dfs[pair]

        train_scaled = global_scaler.transform(train_df[feature_columns])
        test_scaled = global_scaler.transform(test_df[feature_columns])

        train_pair_data[pair] = {
            "feature_data": train_scaled.astype(np.float32),
            "price_data": train_df["close"].values.astype(np.float32),
            "high_data": train_df["high"].values.astype(np.float32),
            "low_data": train_df["low"].values.astype(np.float32),
            "raw_atr_data": train_df["atr_14"].values.astype(np.float32),
        }

        test_pair_data[pair] = {
            "feature_data": test_scaled.astype(np.float32),
            "price_data": test_df["close"].values.astype(np.float32),
            "high_data": test_df["high"].values.astype(np.float32),
            "low_data": test_df["low"].values.astype(np.float32),
            "raw_atr_data": test_df["atr_14"].values.astype(np.float32),
        }

        # Export test set for evaluation
        test_export = test_df.copy()
        test_export[feature_columns] = test_scaled
        test_export.to_csv(data_path / f"{pair}_test.csv", index=False)

    return train_pair_data, test_pair_data, global_scaler, feature_columns


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a multi-pair PPO forex trading agent")
    parser.add_argument("--pairs", nargs="+", default=DEFAULT_PAIRS, help="Pairs to train on")
    parser.add_argument("--iterations", type=int, default=100, help="PPO training iterations")
    parser.add_argument("--window-size", type=int, default=32, help="Observation sequence window")
    parser.add_argument("--rollout-steps", type=int, default=512, help="Steps collected per PPO rollout")
    parser.add_argument("--learning-rate", type=float, default=3e-4, help="Learning rate for Adam")
    parser.add_argument("--entropy-coef", type=float, default=0.03, help="Entropy coefficient")
    parser.add_argument("--spread", type=float, default=0.00008, help="Spread charged on trade entry")
    parser.add_argument("--commission", type=float, default=0.5, help="Flat commission charged per trade")
    parser.add_argument("--slippage-std", type=float, default=0.00003, help="Slippage stddev")
    parser.add_argument("--save-path", type=str, default="model_multipair.pt", help="Model checkpoint path")
    parser.add_argument("--reset-weights", action="store_true", help="Start training from fresh weights")
    parser.add_argument("--enable-trailing-stop", action="store_true", help="Enable Chandelier ATR trailing stop")
    parser.add_argument("--enable-partial-tp", action="store_true", help="Enable partial profit taking scale-out")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    train_pair_data, _, _, feature_columns = load_multipair_data(pairs=args.pairs)

    env = TradingEnv(
        pair_data=train_pair_data,
        feature_map={"atr": 7},
        window_size=args.window_size,
        spread=args.spread,
        commission=args.commission,
        slippage_std=args.slippage_std,
        include_portfolio_state=True,
        random_start=True,
        enable_trailing_stop=args.enable_trailing_stop,
        chandelier_atr_mult=2.5,
        activation_atr_mult=2.0,
        enable_partial_tp=args.enable_partial_tp,
        partial_tp_mult=3.0,
        partial_tp_ratio=0.5,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_size = env.observation_space.shape[1]
    print(f"\nMulti-Pair Observation space: {env.observation_space.shape} (Features: {input_size}) on Device: {device}")

    model = TradingLSTMPolicy(input_size=input_size)
    if not args.reset_weights and os.path.exists(args.save_path):
        try:
            model.load_state_dict(torch.load(args.save_path, map_location=device))
            print(f"Resuming training from checkpoint: {args.save_path}")
        except Exception as e:
            print(f"Could not load existing checkpoint ({e}), initializing fresh weights.")

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

    print(f"\nStarting Multi-Pair PPO training across {args.pairs} ({args.iterations} iterations)...")
    trainer.train(iterations=args.iterations)

    dirpath = os.path.dirname(os.path.abspath(args.save_path)) or "."
    with tempfile.NamedTemporaryFile(dir=dirpath, delete=False) as tmpf:
        tmp_path = tmpf.name
    try:
        torch.save(model.state_dict(), tmp_path)
        os.replace(tmp_path, args.save_path)
        print(f"\n✓ Successfully saved multi-pair model weights to: {args.save_path}")
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


if __name__ == "__main__":
    main()
