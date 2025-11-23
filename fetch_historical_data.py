#!/usr/bin/env python3
"""
Fetch Historical BTC Data for MVRV Calculations

Downloads historical BTCUSDT daily candle data from Binance API
and saves it to data/BTCUSDT_1d.csv in the required format.

Required for:
- MVRV ratio estimation
- Percentile calculations
- Historical analysis features
"""

import requests
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import time


def fetch_binance_klines(symbol='BTCUSDT', interval='1d', limit=1000, start_time=None):
    """
    Fetch historical klines (candlestick data) from Binance API

    Args:
        symbol: Trading pair (default: BTCUSDT)
        interval: Time interval (default: 1d for daily)
        limit: Number of candles to fetch (max 1000 per request)
        start_time: Start timestamp in milliseconds

    Returns:
        List of kline data
    """
    url = "https://api.binance.com/api/v3/klines"

    params = {
        'symbol': symbol,
        'interval': interval,
        'limit': limit
    }

    if start_time:
        params['startTime'] = start_time

    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Error fetching data: {e}")
        return None


def fetch_all_historical_data(days=1500):
    """
    Fetch all historical data by making multiple API calls

    Args:
        days: Number of days of history to fetch (default: 1500 = ~4 years)

    Returns:
        DataFrame with all historical data
    """
    print(f"Fetching {days} days of BTC historical data from Binance...")

    all_data = []

    # Calculate start time
    end_time = datetime.now()
    start_time = end_time - timedelta(days=days)
    current_time = start_time

    batch = 0
    while current_time < end_time:
        batch += 1
        start_ms = int(current_time.timestamp() * 1000)

        print(f"Batch {batch}: Fetching from {current_time.strftime('%Y-%m-%d')}...")

        klines = fetch_binance_klines(start_time=start_ms, limit=1000)

        if not klines:
            print("Failed to fetch data, retrying...")
            time.sleep(2)
            continue

        if len(klines) == 0:
            break

        all_data.extend(klines)

        # Get the last timestamp and move forward
        last_timestamp = klines[-1][0]  # Open time
        current_time = datetime.fromtimestamp(last_timestamp / 1000) + timedelta(days=1)

        # Rate limiting - be nice to Binance API
        time.sleep(0.5)

    print(f"\nFetched {len(all_data)} daily candles")

    # Convert to DataFrame
    df = pd.DataFrame(all_data, columns=[
        'open_time', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_volume', 'trades', 'taker_buy_base',
        'taker_buy_quote', 'ignore'
    ])

    # Convert timestamps to datetime
    df['time_period_start'] = pd.to_datetime(df['open_time'], unit='ms')
    df['time_period_end'] = pd.to_datetime(df['close_time'], unit='ms')

    # Convert price and volume columns to float
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = df[col].astype(float)

    # Select and reorder columns to match expected format
    df = df[[
        'time_period_start', 'time_period_end',
        'open', 'high', 'low', 'close', 'volume'
    ]]

    # Sort by date
    df = df.sort_values('time_period_end').reset_index(drop=True)

    return df


def main():
    """Main function to fetch and save historical data"""

    # Create data directory if it doesn't exist
    data_dir = Path(__file__).parent / 'data'
    data_dir.mkdir(exist_ok=True)

    output_file = data_dir / 'BTCUSDT_1d.csv'

    print("="*60)
    print("BTC Historical Data Fetcher")
    print("="*60)
    print()

    # Fetch data
    # Fetching ~4 years (1500 days) is enough for MVRV calculations
    # You can increase this if you want more history
    df = fetch_all_historical_data(days=1500)

    if df is None or len(df) == 0:
        print("\n❌ Failed to fetch data")
        return

    # Save to CSV
    print(f"\nSaving to {output_file}...")
    df.to_csv(output_file, index=False)

    print("\n" + "="*60)
    print("✅ SUCCESS!")
    print("="*60)
    print(f"Saved {len(df)} days of BTC data")
    print(f"Date range: {df['time_period_end'].min()} to {df['time_period_end'].max()}")
    print(f"File location: {output_file}")
    print(f"File size: {output_file.stat().st_size / 1024:.1f} KB")
    print()
    print("Your app now has the data needed for:")
    print("  ✓ MVRV ratio estimation")
    print("  ✓ Percentile calculations")
    print("  ✓ Historical analysis")
    print("="*60)


if __name__ == '__main__':
    main()
