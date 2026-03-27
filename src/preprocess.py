"""Preprocess script: download stock data, add technical indicators, save processed CSV.

Supports both single-ticker (AAPL) and multi-ticker (DOW30) workflows.
"""

import sys
from pathlib import Path

import pandas as pd

# Allow running as script from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.processor import DataProcessor


# DOW30 tickers (circa 2019-2020 composition)
DOW30_TICKERS = [
    "AAPL", "MSFT", "JPM", "V", "JNJ", "UNH", "HD", "INTC",
    "MRK", "PFE", "XOM", "CVX", "WMT", "BA", "MCD", "GS",
    "NKE", "AXP", "CRM", "NVDA", "GOOGL", "AMZN",
    "ABT", "TMO", "BAC",
]


def preprocess_single(processor: DataProcessor, ticker: str = "AAPL"):
    """Download and process a single ticker."""
    data = processor.download_data(
        tickers=[ticker],
        start_date="2009-01-01",
        end_date="2021-12-31",
    )
    df = data[ticker]
    df = processor.add_technical_indicators(df)
    df = df.ffill().dropna()

    output_path = processor.processed_dir / f"{ticker}_processed.csv"
    df.to_csv(output_path)
    print(f"Saved processed data ({len(df)} rows) to {output_path}")
    return df


def preprocess_dow30(processor: DataProcessor):
    """Download and process DOW30 constituent data into a single CSV.

    Output format: rows sorted by (date, tic), with columns:
    date, tic, open, high, low, close, volume, macd, rsi, cci, adx
    """
    all_frames = []
    failed = []

    data = processor.download_data(
        tickers=DOW30_TICKERS,
        start_date="2009-01-01",
        end_date="2020-09-30",
    )

    for ticker in DOW30_TICKERS:
        if ticker not in data:
            failed.append(ticker)
            continue
        df = data[ticker]
        if len(df) < 100:
            print(f"  Skipping {ticker}: only {len(df)} rows")
            failed.append(ticker)
            continue
        df = processor.add_technical_indicators(df)
        df = df.ffill().dropna()
        df["tic"] = ticker
        all_frames.append(df)

    if failed:
        print(f"Failed/skipped tickers: {failed}")

    # Combine all tickers
    combined = pd.concat(all_frames)
    combined.index.name = "date"
    combined = combined.reset_index()
    combined = combined.sort_values(["date", "tic"]).reset_index(drop=True)

    # Keep only dates where ALL tickers have data (aligned panel)
    tickers_available = combined["tic"].unique()
    n_tickers = len(tickers_available)
    date_counts = combined.groupby("date")["tic"].nunique()
    valid_dates = date_counts[date_counts == n_tickers].index
    combined = combined[combined["date"].isin(valid_dates)].reset_index(drop=True)

    output_path = processor.processed_dir / "dow30_daily.csv"
    combined.to_csv(output_path, index=False)
    print(f"\nSaved DOW30 data:")
    print(f"  Tickers: {n_tickers} ({list(tickers_available)})")
    print(f"  Rows: {len(combined)}")
    print(f"  Date range: {combined['date'].min()} to {combined['date'].max()}")
    print(f"  Output: {output_path}")
    return combined


def main():
    processor = DataProcessor(data_dir="data")

    if len(sys.argv) > 1 and sys.argv[1] == "--dow30":
        preprocess_dow30(processor)
    else:
        preprocess_single(processor)


if __name__ == "__main__":
    main()
