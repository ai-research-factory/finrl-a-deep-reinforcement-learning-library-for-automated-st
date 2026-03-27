# Cycle 2: Technical Findings — Real Data Pipeline

## Implementation

### DataProcessor (`src/data/processor.py`)
- `download_data(tickers, start_date, end_date)`: Fetches OHLCV data from the ARF Data API (`period=max`, filtered to requested date range). Saves raw CSV to `data/raw/{ticker}.csv`.
- `add_technical_indicators(df)`: Adds four technical indicators using the `ta` library:
  - **MACD** (default 12/26/9 windows)
  - **RSI** (14-day)
  - **CCI** (14-day)
  - **ADX** (14-day)

### Preprocess Script (`src/preprocess.py`)
Orchestrates the full pipeline: download AAPL data (2009-01-01 to 2021-12-31), add indicators, forward-fill NaN values, drop remaining leading NaN rows, save to `data/processed/AAPL_processed.csv`.

## Results

| Metric | Value |
|--------|-------|
| Rows (processed) | 3,248 |
| Date range | 2009-02-09 to 2021-12-31 |
| Columns | open, high, low, close, volume, macd, rsi, cci, adx |
| NaN values | 0 |

- 25 leading rows were dropped after forward-fill (from initial indicator warm-up windows), resulting in an effective start date of 2009-02-09 instead of 2009-01-02.
- The `ta` library's ADX indicator requires ~14 days of warm-up, which accounts for most of the dropped rows.

## Observations

- ARF Data API provides `period=max` option which covers the full history needed (2009–2021).
- Data quality appears good with no missing OHLCV values in the raw download.
- Forward-fill strategy is appropriate since NaN values only appear in the initial rows where indicators lack sufficient history.

## Dependencies Added
- `ta` — Technical analysis library for computing indicators
- `requests` — HTTP client for ARF Data API calls
