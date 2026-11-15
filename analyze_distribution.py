#!/usr/bin/env python3
"""
Analyze train/test distribution shift and model behavior
"""
import pandas as pd
import numpy as np
from datetime import datetime

def analyze_periods():
    """Analyze data distribution across periods"""
    df_train = pd.read_csv('forex_data.csv')
    df_test = pd.read_csv('forex_data_test.csv')
    
    # Parse dates
    df_train['date'] = pd.to_datetime(df_train['date'])
    df_test['date'] = pd.to_datetime(df_test['date'])
    
    print("=" * 70)
    print("TRAIN/TEST DISTRIBUTION ANALYSIS")
    print("=" * 70)
    
    # Date ranges
    print(f"\nTRAIN Period: {df_train['date'].min()} to {df_train['date'].max()}")
    print(f"  Samples: {len(df_train):,}")
    print(f"  Duration: {(df_train['date'].max() - df_train['date'].min()).days} days")
    
    print(f"\nTEST Period:  {df_test['date'].min()} to {df_test['date'].max()}")
    print(f"  Samples: {len(df_test):,}")
    print(f"  Duration: {(df_test['date'].max() - df_test['date'].min()).days} days")
    
    # Volatility analysis (High-Low range as proxy)
    print("\n" + "=" * 70)
    print("VOLATILITY COMPARISON (High-Low Range)")
    print("=" * 70)
    
    df_train['hl_range'] = (df_train['high'] - df_train['low']) / df_train['close']
    df_test['hl_range'] = (df_test['high'] - df_test['low']) / df_test['close']
    
    train_vol = df_train['hl_range'].describe()
    test_vol = df_test['hl_range'].describe()
    
    print(f"\nTRAIN HL Range (volatility proxy):")
    print(f"  Mean: {train_vol['mean']:.6f}")
    print(f"  Std:  {train_vol['std']:.6f}")
    print(f"  Min:  {train_vol['min']:.6f}")
    print(f"  Max:  {train_vol['max']:.6f}")
    
    print(f"\nTEST HL Range (volatility proxy):")
    print(f"  Mean: {test_vol['mean']:.6f}")
    print(f"  Std:  {test_vol['std']:.6f}")
    print(f"  Min:  {test_vol['min']:.6f}")
    print(f"  Max:  {test_vol['max']:.6f}")
    
    print(f"\nVOLATILITY SHIFT: {((test_vol['mean'] - train_vol['mean']) / train_vol['mean'] * 100):.2f}%")
    
    # Returns distribution
    print("\n" + "=" * 70)
    print("RETURNS DISTRIBUTION (Close-to-Close)")
    print("=" * 70)
    
    train_returns = df_train['close'].pct_change().dropna()
    test_returns = df_test['close'].pct_change().dropna()
    
    print(f"\nTRAIN Returns:")
    print(f"  Mean: {train_returns.mean():.8f}")
    print(f"  Std:  {train_returns.std():.8f}")
    print(f"  Min:  {train_returns.min():.8f}")
    print(f"  Max:  {train_returns.max():.8f}")
    
    print(f"\nTEST Returns:")
    print(f"  Mean: {test_returns.mean():.8f}")
    print(f"  Std:  {test_returns.std():.8f}")
    print(f"  Min:  {test_returns.min():.8f}")
    print(f"  Max:  {test_returns.max():.8f}")
    
    # Price levels
    print("\n" + "=" * 70)
    print("PRICE LEVELS (Close)")
    print("=" * 70)
    
    print(f"\nTRAIN Close:")
    print(f"  Mean: {df_train['close'].mean():.6f}")
    print(f"  Min:  {df_train['close'].min():.6f}")
    print(f"  Max:  {df_train['close'].max():.6f}")
    
    print(f"\nTEST Close:")
    print(f"  Mean: {df_test['close'].mean():.6f}")
    print(f"  Min:  {df_test['close'].min():.6f}")
    print(f"  Max:  {df_test['close'].max():.6f}")
    
    print("\n" + "=" * 70)
    print("RECOMMENDATION")
    print("=" * 70)
    
    vol_shift = ((test_vol['mean'] - train_vol['mean']) / train_vol['mean'] * 100)
    
    if abs(vol_shift) > 20:
        print(f"\n⚠️  HIGH DISTRIBUTION SHIFT DETECTED ({vol_shift:.1f}%)")
        print("   - Model trained on {:.1f}x different volatility environment".format(
            test_vol['mean'] / train_vol['mean']
        ))
        print("   - Consider: Time-based strategy adaptive to volatility")
        print("   - Or: Include test period in training for better generalization")
    else:
        print(f"\n✓ Moderate volatility consistency ({vol_shift:.1f}%)")
        print("   - Issue is likely model overfitting, not distribution shift")
        print("   - Solution: Increase regularization or training iterations")

if __name__ == '__main__':
    analyze_periods()
