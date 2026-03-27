# Cycle 5: Transaction Cost Model - Technical Findings

## Implemented Changes

### 1. Transaction Cost Model (src/environment.py)
Added `transaction_cost_pct` and `slippage_pct` parameters to `StockTradingEnv.__init__()`. In the `step()` method, when a trade is executed, the total cost deducted from cash is:
- **Sells**: `sell_amount * (sell_cost_pct + transaction_cost_pct + slippage_pct)` deducted from proceeds
- **Buys**: `buy_amount * (buy_cost_pct + transaction_cost_pct + slippage_pct)` added to purchase cost

A `total_transaction_costs` counter tracks cumulative costs for reporting.

### 2. Risk-Adjusted Reward Function (src/environment.py)
Replaced the raw portfolio-value-change reward with a risk-penalized version:
```
reward = (daily_return - 0.05 * rolling_20d_volatility^2) / reward_scaling
```
This encourages the agent to maximize risk-adjusted returns, approximating Sharpe ratio optimization. See `docs/reward_function_design.md` for full design rationale.

### 3. Calmar and Sortino Ratios (src/backtest.py)
Extended `compute_metrics()` to return `sortinoRatio` (using downside deviation) and `calmarRatio` (annual return / |max drawdown|).

### 4. Walk-Forward Validation (src/run_single_backtest.py)
Replaced single train/test split with `WalkForwardValidator` (n_splits=10). Each window trains a fresh PPO model and evaluates on the subsequent OOS period. Results are saved to `reports/walk_forward_summary.csv`.

## Cost Analysis Results

A single PPO model was trained and evaluated under two cost configurations:

| Metric | Gross (0% costs) | Net (fee=0.1%, slip=0.05%) | Impact |
|--------|:-:|:-:|:-:|
| Sharpe Ratio | 0.5057 | 0.4908 | -0.0149 |
| Annual Return | 12.06% | 11.49% | -0.57% |
| Max Drawdown | -43.34% | -43.44% | -0.10% |
| Sortino Ratio | 0.4556 | 0.4443 | -0.0113 |
| Calmar Ratio | 0.2782 | 0.2646 | -0.0136 |
| Total Trades | 2717 | 2733 | - |
| Transaction Costs | $0 | $6,989 | - |

### Key Observations
- **Transaction costs reduce Sharpe ratio by ~3%** (0.5057 -> 0.4908), confirming that cost modeling matters for realistic evaluation.
- The $6,989 in total costs represents ~0.7% of the initial $1M portfolio, a modest but measurable drag on performance.
- Max drawdown is virtually unchanged (-43.34% vs -43.44%), as costs primarily affect the steady-state return rather than extreme events.
- The slight increase in trade count under the net configuration is due to the environment's affordability check being affected by costs, which marginally alters execution paths.

## Acceptance Criteria Verification
- `cost_comparison.json` generated with `gross_performance` and `net_performance` keys.
- Net Sharpe (0.4908) < Gross Sharpe (0.5057).
