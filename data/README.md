# Historical Data Directory

This directory contains historical Bitcoin price data used for MVRV calculations and percentile analysis.

## Required File

Place your historical data file here:
```
data/BTCUSDT_1d.csv
```

## File Format

The CSV file should contain daily BTC/USDT data with these columns:
- `time_period_start` - Start timestamp
- `time_period_end` - End timestamp
- `open` - Opening price
- `high` - High price
- `low` - Low price
- `close` - Closing price
- `volume` - Trading volume

## Data Requirements

For optimal MVRV estimation:
- **Minimum**: 200 days of data
- **Recommended**: 1,500+ days (~4 years)

## How to Get Data

### Option 1: Use the Fetcher Script (Recommended)
Run the included script to download data from Binance:
```bash
python fetch_historical_data.py
```

This will automatically download ~4 years of data and save it to `data/BTCUSDT_1d.csv`.

### Option 2: Manual Download
If you have your own historical data source, place the CSV file here with the format above.

## Notes

- The `.gitignore` file excludes `*.csv` files, so your data won't be committed to git
- The app will work without this data (using API fallback), but MVRV accuracy will be reduced
- Data is cached in-memory for performance
