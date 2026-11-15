"""
Download historical 1-Hour EUR/USD data from FXCM's free data repository.
Covers 2020 to present, with robust error handling and progress tracking.
"""

import os
import io
import gzip
import requests
import pandas as pd
import time
from tqdm import tqdm

# Configuration
BASE_URL = "https://candledata.fxcorporate.com/H1/EURUSD/{year}/{week}.csv.gz"
YEARS = range(2020, 2027)  # 2020 to 2026
WEEKS = range(1, 53)  # 1 to 52
OUTPUT_FILE = "forex_data.csv"
REQUEST_TIMEOUT = 10  # seconds
SLEEP_DURATION = 0.5  # seconds between requests

def download_week(year, week):
    """
    Download a single week's data from FXCM.
    
    Args:
        year (int): Year (2020-2026)
        week (int): Week number (1-52)
    
    Returns:
        pd.DataFrame or None: DataFrame if successful, None if 404 or error
    """
    url = BASE_URL.format(year=year, week=week)
    try:
        response = requests.get(url, timeout=REQUEST_TIMEOUT)
        
        # Gracefully skip 404s (future weeks or missing data)
        if response.status_code == 404:
            return None
        
        # Raise for other HTTP errors
        response.raise_for_status()
        
        # Decompress and read the gzip CSV
        with gzip.GzipFile(fileobj=io.BytesIO(response.content)) as gz:
            df = pd.read_csv(gz)
        
        return df
    
    except requests.exceptions.RequestException as e:
        # Log but don't crash on network errors
        print(f"Error downloading {year} week {week}: {e}")
        return None
    
    except Exception as e:
        # Catch any other errors (gzip, pandas, etc.)
        print(f"Error processing {year} week {week}: {e}")
        return None


def main():
    print("Downloading 1-Hour EUR/USD data from FXCM repository (2020-2026)...")
    print(f"Base URL: {BASE_URL}")
    print()
    
    all_dfs = []
    total_weeks = len(list(YEARS)) * len(list(WEEKS))
    
    # Progress bar for all years/weeks
    with tqdm(total=total_weeks, desc="Total Progress", unit="week") as pbar_total:
        for year in YEARS:
            for week in WEEKS:
                df = download_week(year, week)
                
                if df is not None:
                    all_dfs.append(df)
                
                # Pause between requests to avoid overloading server
                time.sleep(SLEEP_DURATION)
                pbar_total.update(1)
    
    print()
    if not all_dfs:
        print("ERROR: No data downloaded. Check network or URL structure.")
        return
    
    print(f"Downloaded {len(all_dfs)} weeks of data. Concatenating...")
    
    # Concatenate all DataFrames
    combined_df = pd.concat(all_dfs, ignore_index=True)
    print(f"Combined DataFrame shape: {combined_df.shape}")
    print(f"Columns before cleaning: {list(combined_df.columns)}")
    
    # Rename columns to lowercase and standardize
    combined_df.columns = [col.strip().lower() for col in combined_df.columns]
    print(f"Columns after lowercase: {list(combined_df.columns)}")
    
    # Handle FXCM Bid/Ask format: use midpoint between Bid and Ask
    if 'bidopen' in combined_df.columns and 'askopen' in combined_df.columns:
        print("Detected FXCM Bid/Ask format. Computing midpoints...")
        combined_df['open'] = (pd.to_numeric(combined_df['bidopen'], errors='coerce') + 
                               pd.to_numeric(combined_df['askopen'], errors='coerce')) / 2
        combined_df['high'] = (pd.to_numeric(combined_df['bidhigh'], errors='coerce') + 
                               pd.to_numeric(combined_df['askhigh'], errors='coerce')) / 2
        combined_df['low'] = (pd.to_numeric(combined_df['bidlow'], errors='coerce') + 
                              pd.to_numeric(combined_df['asklow'], errors='coerce')) / 2
        combined_df['close'] = (pd.to_numeric(combined_df['bidclose'], errors='coerce') + 
                                pd.to_numeric(combined_df['askclose'], errors='coerce')) / 2
        combined_df['volume'] = 0  # FXCM doesn't provide volume for forex
    else:
        # Standard OHLCV format - standardize column names
        rename_map = {}
        for col in combined_df.columns:
            if 'time' in col or 'date' in col:
                rename_map[col] = 'date'
            elif col == 'open':
                rename_map[col] = 'open'
            elif col == 'high':
                rename_map[col] = 'high'
            elif col == 'low':
                rename_map[col] = 'low'
            elif col == 'close':
                rename_map[col] = 'close'
            elif 'volume' in col or 'vol' in col:
                rename_map[col] = 'volume'
        
        if rename_map:
            combined_df = combined_df.rename(columns=rename_map)
    
    # Extract datetime column (handle various formats)
    date_col = None
    for col in ['datetime', 'date', 'time']:
        if col in combined_df.columns:
            date_col = col
            break
    
    if not date_col and 'date' not in combined_df.columns:
        # If no date column, check the first column
        date_col = combined_df.columns[0]
    
    if date_col and date_col != 'date':
        combined_df['date'] = combined_df[date_col]
    
    # Ensure required columns exist
    required_cols = ['date', 'open', 'high', 'low', 'close', 'volume']
    missing = [col for col in required_cols if col not in combined_df.columns]
    if missing:
        print(f"WARNING: Missing expected columns: {missing}")
        print(f"Available columns: {list(combined_df.columns)}")
    
    # Convert date to datetime
    print("Converting date column to datetime...")
    if 'date' not in combined_df.columns and 'datetime' in combined_df.columns:
        combined_df['date'] = combined_df['datetime']
    combined_df['date'] = pd.to_datetime(combined_df['date'], errors='coerce')
    
    # Convert OHLCV to numeric (handle any non-numeric values)
    for col in ['open', 'high', 'low', 'close']:
        if col in combined_df.columns:
            combined_df[col] = pd.to_numeric(combined_df[col], errors='coerce')
    
    if 'volume' not in combined_df.columns:
        combined_df['volume'] = 0
    
    # Drop NaN rows
    print("Dropping NaN rows...")
    combined_df = combined_df.dropna()
    print(f"After dropping NaN: {combined_df.shape[0]} rows")
    
    # Drop duplicates (keep first occurrence)
    print("Dropping duplicates...")
    combined_df = combined_df.drop_duplicates(subset=['date'], keep='first')
    print(f"After dropping duplicates: {combined_df.shape[0]} rows")
    
    # Sort by date chronologically
    print("Sorting by date...")
    combined_df = combined_df.sort_values('date').reset_index(drop=True)
    
    # Select and order final columns
    final_cols = ['date', 'open', 'high', 'low', 'close', 'volume']
    combined_df = combined_df[final_cols]
    
    # Export to CSV
    print(f"Exporting {combined_df.shape[0]} rows to {OUTPUT_FILE}...")
    combined_df.to_csv(OUTPUT_FILE, index=False)
    
    print()
    print("=" * 60)
    print("DOWNLOAD COMPLETE")
    print("=" * 60)
    print(f"File: {OUTPUT_FILE}")
    print(f"Rows: {combined_df.shape[0]}")
    print(f"Date range: {combined_df['date'].min()} to {combined_df['date'].max()}")
    print(f"Columns: {list(combined_df.columns)}")
    print()
    print(combined_df.head())


if __name__ == "__main__":
    main()
