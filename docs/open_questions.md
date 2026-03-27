# Open Questions

## Data
- The ARF Data API `period=max` returns data from 1980 for AAPL, which covers our 2009-2021 range. For other DJIA tickers, max available history should be verified when expanding to multi-stock pipelines (Phase 6).
- Adjusted vs. unadjusted prices: The API returns what appears to be split-adjusted prices (AAPL ~$3 in 2009 suggests adjusted for subsequent splits). This is consistent with the paper's use of yfinance, which defaults to adjusted close.

## Technical Indicators
- The `ta` library's default MACD parameters (12/26/9) match the standard convention but the paper does not explicitly state which MACD parameters were used. We assume defaults.
- ADX warm-up period causes 25 rows to be dropped from the start of the dataset. This is a minor loss (~0.8% of data).

## Pipeline
- The current pipeline processes one ticker at a time. For Phase 6 (multi-stock portfolio), the `download_data` method already supports multiple tickers but the preprocess script will need to be extended.
