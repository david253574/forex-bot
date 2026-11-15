"""
Concurrent downloader for multi-currency pair 1-Hour historical data from FXCM.
Downloads EURUSD, GBPUSD, USDJPY, AUDUSD (2020-2026).
"""

from __future__ import annotations

import concurrent.futures
import gzip
import io
import os
from pathlib import Path
from typing import List, Optional

import pandas as pd
import requests
from tqdm import tqdm

PAIRS = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD"]
YEARS = range(2020, 2027)  # 2020 to 2026
WEEKS = range(1, 53)
DATA_DIR = Path("data")


def fetch_week(session: requests.Session, pair: str, year: int, week: int) -> Optional[pd.DataFrame]:
    url = f"https://candledata.fxcorporate.com/H1/{pair}/{year}/{week}.csv.gz"
    try:
        resp = session.get(url, timeout=8)
        if resp.status_code == 200:
            with gzip.GzipFile(fileobj=io.BytesIO(resp.content)) as gz:
                df = pd.read_csv(gz)
                return df
    except Exception:
        pass
    return None


def clean_pair_df(dfs: List[pd.DataFrame], pair: str) -> pd.DataFrame:
    combined = pd.concat(dfs, ignore_index=True)
    combined.columns = [col.strip().lower() for col in combined.columns]

    if "bidopen" in combined.columns and "askopen" in combined.columns:
        combined["open"] = (pd.to_numeric(combined["bidopen"], errors="coerce") + pd.to_numeric(combined["askopen"], errors="coerce")) / 2
        combined["high"] = (pd.to_numeric(combined["bidhigh"], errors="coerce") + pd.to_numeric(combined["askhigh"], errors="coerce")) / 2
        combined["low"] = (pd.to_numeric(combined["bidlow"], errors="coerce") + pd.to_numeric(combined["asklow"], errors="coerce")) / 2
        combined["close"] = (pd.to_numeric(combined["bidclose"], errors="coerce") + pd.to_numeric(combined["askclose"], errors="coerce")) / 2
        combined["volume"] = 0
    else:
        for c in ["open", "high", "low", "close"]:
            combined[c] = pd.to_numeric(combined[c], errors="coerce")
        combined["volume"] = 0

    date_col = None
    for c in ["date", "datetime", "time"]:
        if c in combined.columns:
            date_col = c
            break
    if date_col and date_col != "date":
        combined["date"] = combined[date_col]
    elif not date_col:
        combined["date"] = combined.iloc[:, 0]

    combined = combined.dropna(subset=["date", "open", "high", "low", "close"]).reset_index(drop=True)
    combined["date"] = pd.to_datetime(combined["date"])
    combined = combined.sort_values("date").drop_duplicates(subset=["date"]).reset_index(drop=True)

    final_df = combined[["date", "open", "high", "low", "close", "volume"]]
    return final_df


def download_pair(pair: str) -> None:
    print(f"\n--- Downloading {pair} (2020-2026) ---")
    tasks = [(year, week) for year in YEARS for week in WEEKS]
    dfs: List[pd.DataFrame] = []

    with requests.Session() as session:
        # Use a thread pool per pair for fast parallel downloads
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(fetch_week, session, pair, y, w) for y, w in tasks]
            for f in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc=pair, unit="wk"):
                res = f.result()
                if res is not None:
                    dfs.append(res)

    if not dfs:
        print(f"Error: No data downloaded for {pair}")
        return

    clean_df = clean_pair_df(dfs, pair)
    output_path = DATA_DIR / f"{pair}.csv"
    clean_df.to_csv(output_path, index=False)
    print(f"✓ Saved {pair}: {len(clean_df)} bars ({clean_df['date'].min()} to {clean_df['date'].max()}) -> {output_path}")


def main() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    for pair in PAIRS:
        download_pair(pair)
    print("\nAll currency pairs successfully downloaded and processed!")


if __name__ == "__main__":
    main()
