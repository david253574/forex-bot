"""Data loading and preprocessing utilities for forex OHLCV data."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator, MACD
from ta.volatility import AverageTrueRange, BollingerBands


@dataclass
class ForexDataPreprocessor:
    """Loads raw OHLCV data, builds indicators, and normalizes model features."""

    scaler: MinMaxScaler | None = None

    def load_and_preprocess(
        self, csv_path: str
    ) -> Tuple[pd.DataFrame, MinMaxScaler, List[str], pd.DataFrame]:
        """
        Load OHLCV data and return a normalized feature dataframe for RL training.

        Returns:
            normalized_df: Scaled dataframe containing model-ready features.
            scaler: Fitted MinMaxScaler instance.
            feature_columns: Ordered list of feature names used by the model.
            enriched_df: Original numeric dataframe with indicators (unscaled).
        """
        df = pd.read_csv(csv_path)

        # Normalize column names to avoid schema mismatches from broker exports.
        df.columns = [col.strip().lower() for col in df.columns]

        required_cols = {"open", "high", "low", "close", "volume"}
        missing = required_cols.difference(df.columns)
        if missing:
            raise ValueError(f"CSV is missing required OHLCV columns: {sorted(missing)}")

        # Cast to numeric and drop malformed rows before indicator generation.
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # Stationary log returns and candle geometry are computed before the
        # global cleanup step below so the early NaNs from shifting are removed
        # naturally by the existing dropna().reset_index(drop=True) call.
        df["log_ret_close"] = np.log(df["close"] / df["close"].shift(1))
        df["log_ret_open"] = np.log(df["open"] / df["open"].shift(1))
        df["log_ret_high"] = np.log(df["high"] / df["high"].shift(1))
        df["log_ret_low"] = np.log(df["low"] / df["low"].shift(1))

        range_calc = df["high"] - df["low"] + 1e-8
        df["body_size"] = (df["close"] - df["open"]) / range_calc
        df["upper_wick"] = (df["high"] - df[["open", "close"]].max(axis=1)) / range_calc
        df["lower_wick"] = (df[["open", "close"]].min(axis=1) - df["low"]) / range_calc

        df = df.dropna(subset=["open", "high", "low", "close", "volume"]).reset_index(drop=True)

        # Datetime parsing (prefer 'date' column but accept common alternatives)
        if "date" in df.columns:
            datetime_col = pd.to_datetime(df["date"])
        elif "datetime" in df.columns:
            datetime_col = pd.to_datetime(df["datetime"])
        elif "time" in df.columns:
            datetime_col = pd.to_datetime(df["time"])
        else:
            # Fallback: try to infer from index if possible, otherwise set NaT
            try:
                datetime_col = pd.to_datetime(df.index)
            except Exception:
                datetime_col = pd.to_datetime(df["close"], errors="coerce")

        # Ensure datetime_col is a Series so `.dt` accessor is available
        datetime_col = pd.Series(pd.to_datetime(datetime_col))

        # Cyclical time features (keeps them at the end of feature_columns)
        df["hour_sin"] = np.sin(2 * np.pi * datetime_col.dt.hour / 24.0)
        df["hour_cos"] = np.cos(2 * np.pi * datetime_col.dt.hour / 24.0)
        # Normalize day_of_week to [0,1]
        df["day_of_week"] = datetime_col.dt.dayofweek / 6.0

        macd = MACD(close=df["close"])
        df["macd"] = macd.macd() / df["close"]
        df["macd_signal"] = macd.macd_signal() / df["close"]
        df["rsi_14"] = RSIIndicator(close=df["close"], window=14).rsi()
        df["ema_50"] = EMAIndicator(close=df["close"], window=50).ema_indicator()
        df["ema_200"] = EMAIndicator(close=df["close"], window=200).ema_indicator()
        df["atr_14"] = AverageTrueRange(
            high=df["high"], low=df["low"], close=df["close"], window=14
        ).average_true_range()
        df["atr_norm"] = df["atr_14"] / df["close"]

        bb = BollingerBands(close=df["close"], window=20)
        df["bb_width"] = (bb.bollinger_hband() - bb.bollinger_lband()) / df["close"]
        df["dist_ema_50"] = (df["close"] - df["ema_50"]) / df["ema_50"]
        df["dist_ema_200"] = (df["close"] - df["ema_200"]) / df["ema_200"]

        df = df.dropna().reset_index(drop=True)

        feature_columns = [
            "log_ret_open",
            "log_ret_high",
            "log_ret_low",
            "log_ret_close",
            "macd",
            "macd_signal",
            "rsi_14",
            "atr_14",
            "bb_width",
            "dist_ema_50",
            "dist_ema_200",
            # Temporal cyclical features appended last to avoid shifting index of existing indicators
            "hour_sin",
            "hour_cos",
            "day_of_week",
            "body_size",
            "upper_wick",
            "lower_wick",
        ]

        self.scaler = MinMaxScaler()
        normalized_values = self.scaler.fit_transform(df[feature_columns])

        normalized_df = pd.DataFrame(normalized_values, columns=feature_columns, index=df.index)
        return normalized_df, self.scaler, feature_columns, df


def load_and_preprocess_data(
    csv_path: str,
) -> Tuple[pd.DataFrame, MinMaxScaler, List[str], pd.DataFrame]:
    """Convenience function used by training scripts."""
    preprocessor = ForexDataPreprocessor()
    return preprocessor.load_and_preprocess(csv_path)
