# Cycle 3: Technical Findings — Integration and Single Backtest

## Implementation

### StockTradingEnv (`src/environment.py`)
- Gymnasium-compatible trading environment following the FinRL paper design.
- **State space**: `[cash, stock_prices (N), stock_holdings (N), tech_indicators (N×4)]` — total dimension = 1 + 25 + 25 + 100 = 151.
- **Action space**: Continuous `[-1, 1]^N`, scaled to `[-hmax, hmax]` shares per stock.
- **Reward**: Change in portfolio value at each step.
- **Transaction costs**: 0.1% buy/sell cost built into the environment execution.
- Sells execute before buys (to free up cash). Buys are capped by available cash.

### PPO Agent (`src/agents/ppo_agent.py`)
- Wrapper around Stable-Baselines3 PPO with `MlpPolicy`.
- Default hyperparameters: `lr=3e-4, n_steps=2048, batch_size=64, n_epochs=10, gamma=0.99`.
- Supports `train`, `predict`, `save`, and `load` methods.

### Data Pipeline (`src/preprocess.py`)
- Extended `preprocess.py` to support multi-ticker DOW30 download via `--dow30` flag.
- Downloads 25 DOW30 constituents from ARF Data API (2009-01-01 to 2020-09-30).
- Adds MACD, RSI(14), CCI(14), ADX(14) technical indicators per ticker.
- Aligns all tickers to common trading dates (panel data format).
- Output: `data/processed/dow30_daily.csv` with columns: `date, tic, open, high, low, close, volume, macd, rsi, cci, adx`.

### Backtest Runner (`src/run_single_backtest.py`)
- Loads `dow30_daily.csv`, splits into train (2009-02-09 to 2018-12-31) and test (2019-01-02 to 2020-09-30).
- Trains PPO for 100,000 timesteps on training environment.
- Runs deterministic policy on test environment, recording daily portfolio values.
- Saves `portfolio_value.csv`, `portfolio_value.png`, and `metrics.json`.

## Results

| Metric | Value |
|--------|-------|
| Initial Amount | $1,000,000.00 |
| Final Portfolio Value | $1,057,037.96 |
| Total Return | 5.70% |
| Sharpe Ratio | 0.2599 |
| Annualized Return | 3.23% |
| Max Drawdown | -40.39% |
| Hit Rate | 55.23% |
| Total Trades | 3,366 |

### Data Summary

| Metric | Value |
|--------|-------|
| Tickers | 25 of 30 DOW components |
| Training rows | 62,275 (2,491 dates × 25 stocks) |
| Test rows | 11,025 (441 dates × 25 stocks) |
| Features per stock | close, macd, rsi, cci, adx |

## Observations

1. **Positive but modest returns**: The PPO agent achieved a 5.70% total return over the ~21 month test period, which translates to a 3.23% annualized return. This is below buy-and-hold DOW performance over the same period but shows the agent learned a non-trivial policy.

2. **Large drawdown**: The -40.39% max drawdown largely coincides with the COVID-19 market crash in March 2020. The agent did not learn to de-risk ahead of the crash, which is expected given the unprecedented nature of the event.

3. **Moderate Sharpe ratio**: 0.26 Sharpe indicates the strategy generates positive excess returns but with high volatility. This is a baseline that can be improved with hyperparameter tuning and walk-forward validation.

4. **Transaction costs embedded**: Buy/sell costs (0.1% each) are built into the environment, so all returns are net of transaction costs.

5. **Walk-forward not yet implemented**: This cycle uses a single train/test split as specified. Walk-forward validation will be added in Phase 4+ for more robust evaluation.

6. **5 DOW30 tickers unavailable**: PG, VZ, KO, DIS, CSCO and other tickers not listed in the ARF Data API were excluded. 25 of 30 target components are included.

## Dependencies Added
- `stable-baselines3` — PPO and other DRL algorithms
- `gymnasium` — Environment interface (successor to OpenAI Gym)
