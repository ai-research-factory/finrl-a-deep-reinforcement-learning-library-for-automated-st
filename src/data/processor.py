"""Data processor for downloading stock data and adding technical indicators."""

import os
from pathlib import Path

import pandas as pd
import requests
import ta


API_BASE = "https://ai.1s.xyz/api/data/ohlcv"


class DataProcessor:
    """Downloads stock price data from ARF Data API and adds technical indicators."""

    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    def download_data(
        self,
        tickers: list[str],
        start_date: str,
        end_date: str,
    ) -> dict[str, pd.DataFrame]:
        """Download OHLCV data from ARF Data API for given tickers and date range.

        Args:
            tickers: List of ticker symbols (e.g. ['AAPL']).
            start_date: Start date string (e.g. '2009-01-01').
            end_date: End date string (e.g. '2021-12-31').

        Returns:
            Dictionary mapping ticker to its DataFrame.
        """
        results = {}
        for ticker in tickers:
            print(f"Downloading {ticker}...")
            params = {
                "ticker": ticker,
                "interval": "1d",
                "period": "max",
                "format": "json",
            }
            resp = requests.get(API_BASE, params=params, timeout=120)
            resp.raise_for_status()
            payload = resp.json()

            # API returns {"data": [...], ...}
            records = payload.get("data", payload)
            if isinstance(records, dict):
                records = [records]
            df = pd.DataFrame(records)
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = df.set_index("timestamp").sort_index()

            # Filter to requested date range
            df = df.loc[start_date:end_date]

            # Standardize column names
            col_map = {}
            for col in df.columns:
                col_map[col] = col.lower()
            df = df.rename(columns=col_map)

            # Ensure expected columns exist
            for col in ["open", "high", "low", "close", "volume"]:
                if col not in df.columns:
                    raise ValueError(f"Missing column '{col}' for {ticker}")

            # Save raw data
            raw_path = self.raw_dir / f"{ticker}.csv"
            df.to_csv(raw_path)
            print(f"  Saved {len(df)} rows to {raw_path}")

            results[ticker] = df
        return results

    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add MACD, RSI(14), CCI(14), and ADX(14) technical indicators.

        Args:
            df: DataFrame with columns: open, high, low, close, volume.

        Returns:
            DataFrame with added indicator columns.
        """
        df = df.copy()

        # MACD
        macd_indicator = ta.trend.MACD(close=df["close"])
        df["macd"] = macd_indicator.macd()

        # RSI (14-day)
        df["rsi"] = ta.momentum.RSIIndicator(close=df["close"], window=14).rsi()

        # CCI (14-day)
        df["cci"] = ta.trend.CCIIndicator(
            high=df["high"], low=df["low"], close=df["close"], window=14
        ).cci()

        # ADX (14-day)
        df["adx"] = ta.trend.ADXIndicator(
            high=df["high"], low=df["low"], close=df["close"], window=14
        ).adx()

        return df
