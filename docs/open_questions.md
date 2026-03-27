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
- The large max drawdown (-40.39%) is largely attributed to the COVID-19 crash in the test period (March 2020). Walk-forward validation helps evaluate robustness outside of this extreme event.

## Transaction Costs (Cycle 5)
- The FinRL paper uses a flat 0.1% transaction cost. We model this as separate `buy_cost_pct`, `sell_cost_pct`, `transaction_cost_pct`, and `slippage_pct` for more granular analysis.
- The risk-adjusted reward function uses a fixed `risk_penalty_coef=0.05`. The paper does not specify this parameter; it was chosen empirically. Future cycles could tune this via hyperparameter optimization.
- The `reward_scaling=1e-4` normalizes rewards for stable PPO training. This is an implementation detail not discussed in the paper.
- Cost impact on Sharpe ratio is modest (~3% reduction) partly because the PPO agent's trading frequency is moderate (~2700 trades over the test period). Higher-frequency strategies would show larger cost impact.

## Hyperparameter Optimization (Cycle 7)
- The Optuna search used only the first walk-forward fold for optimization. This is a deliberate choice to prevent overfitting to all folds, but means the optimal params may not generalize equally to all market regimes.
- Significant gap between 25K-step tuning Sharpe (1.81) and full 9-window walk-forward average Sharpe (1.13). The single-fold tuning result was optimistic; multi-window evaluation provides a more realistic estimate.
- The optimized learning rate (4.46e-5) is ~7x lower than the SB3 default (3e-4). Validated across 9 walk-forward windows — all achieved positive Sharpe, confirming the parameter works across market regimes.
- PPO (avg Sharpe 1.13) underperforms Buy & Hold (1.39) but matches Equal Weight (1.14) on average. This is consistent with a generally bullish market period (2011-2020) where passive strategies have a structural advantage.
- The `risk_penalty_coef=0.05` reward function modification (non-paper) was kept during optimization. The review recommends establishing a baseline with the paper's original reward (pure portfolio value change) for comparison. This is deferred to a future cycle.
- High Sharpe standard deviation (0.76) across windows indicates substantial regime dependence. The agent performs well in trending markets but struggles during turbulent periods (COVID crash in window 9: Sharpe 0.24, max DD -44.9%).
