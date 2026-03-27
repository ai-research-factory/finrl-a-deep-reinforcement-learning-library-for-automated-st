# Open Questions

## Data
- The ARF Data API `period=max` returns data from 1980 for AAPL, which covers our 2009-2021 range. For other DJIA tickers, max available history should be verified when expanding to multi-stock pipelines (Phase 6).
- Adjusted vs. unadjusted prices: The API returns what appears to be split-adjusted prices (AAPL ~$3 in 2009 suggests adjusted for subsequent splits). This is consistent with the paper's use of yfinance, which defaults to adjusted close.

## Technical Indicators
- The `ta` library's default MACD parameters (12/26/9) match the standard convention but the paper does not explicitly state which MACD parameters were used. We assume defaults.
- ADX warm-up period causes 25 rows to be dropped from the start of the dataset. This is a minor loss (~0.8% of data).

## Pipeline
- Multi-ticker pipeline now works for 25 DOW30 components. 5 tickers are unavailable from the ARF Data API: PG, VZ, KO, DIS, CSCO (and other historical DOW30 members like MMM, IBM, CAT, TRV, WBA, DOW). This means we trade 25/30 stocks instead of 30.
- Panel alignment drops dates where any ticker is missing, resulting in 2,932 aligned trading days (2009-02-09 to 2020-09-30).

## Environment & Agent (Cycle 3)
- PPO hyperparameters use SB3 defaults (lr=3e-4, n_steps=2048, batch_size=64, gamma=0.99). The FinRL paper does not specify exact PPO hyperparameters, so defaults are used as a baseline.
- `hmax=100` (max shares per action) is a design choice — the paper suggests this as a reasonable limit but optimal values may vary by stock price level.
- The environment uses a single-step reward (change in portfolio value). The paper also mentions using Sharpe ratio as reward, which could be explored in future cycles.
- The large max drawdown (-40.39%) is largely attributed to the COVID-19 crash in the test period (March 2020). Walk-forward validation in Phase 4+ may help evaluate robustness outside of this extreme event.
