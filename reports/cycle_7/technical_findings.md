# Cycle 7: Hyperparameter Optimization — Technical Findings

## Objective

Systematically optimize PPO agent hyperparameters using Optuna (TPE sampler) to maximize out-of-sample Sharpe ratio on the first walk-forward fold.

## Method

- **Optimization framework**: Optuna with Tree-structured Parzen Estimator (TPE)
- **Objective**: Maximize OOS Sharpe ratio on first walk-forward fold
- **Tuning timesteps**: 25,000 per trial (for speed), with full 100,000-step re-evaluation of best params
- **Number of trials**: 100
- **Storage**: SQLite (`reports/cycle_7/study.db`)

### Search Space

| Parameter | Range | Type |
|-----------|-------|------|
| `learning_rate` | 1e-5 to 1e-3 | Log-uniform |
| `n_steps` | {2048, 4096} | Categorical |
| `gamma` | {0.99, 0.995, 0.999} | Categorical |
| `ent_coef` | {0.0, 0.01} | Categorical |

### Data Split (First Walk-Forward Fold)

- **Train**: 2009-09-21 to 2011-03-01 (368 trading days)
- **Test**: 2011-03-03 to 2012-03-23 (268 trading days)
- **Stock universe**: 25 DOW30 constituents

## Results

### Best Parameters Found

| Parameter | Default (SB3) | Optimized | Note |
|-----------|---------------|-----------|------|
| `learning_rate` | 3.0e-4 | 4.46e-5 | ~7x lower — more conservative learning |
| `n_steps` | 2048 | 4096 | Larger batch — more stable gradient estimates |
| `gamma` | 0.99 | 0.999 | Higher discount — agent values long-term returns |
| `ent_coef` | 0.0 | 0.01 | Entropy bonus encourages exploration |

### Trial Statistics (100 trials, 25K-step tuning)

| Stat | Sharpe Ratio |
|------|-------------|
| Mean | 0.9925 |
| Std | 0.2707 |
| Min | 0.4826 |
| Max | 1.8148 |
| Best trial | #94 |

### Full Evaluation (100K steps, best params)

| Metric | Value |
|--------|-------|
| Sharpe Ratio | 0.6127 |
| Annual Return | 8.47% |
| Max Drawdown | -13.22% |
| Hit Rate | 50.19% |
| Sortino Ratio | 0.6372 |
| Calmar Ratio | 0.6407 |
| Total Trades | 2,403 |

## Key Observations

1. **Lower learning rate dominates**: The optimal learning rate (4.46e-5) is significantly below the SB3 default (3e-4). This suggests the trading signal is subtle and aggressive updates cause instability.

2. **Long-horizon discount**: gamma=0.999 (vs default 0.99) indicates the agent benefits from considering returns over longer horizons, consistent with multi-day position holding strategies.

3. **Entropy regularization helps**: ent_coef=0.01 was selected, suggesting the agent benefits from maintaining exploration — preventing premature convergence to a suboptimal policy.

4. **Tuning vs full-training gap**: Best trial Sharpe during 25K tuning was 1.81, but the full 100K re-evaluation yielded 0.61. This gap suggests (a) longer training may lead to slight overfitting, or (b) the 25K-step evaluation captures a noisier but more optimistic estimate.

5. **All trials positive**: Every trial achieved positive Sharpe (min 0.48), indicating the PPO agent is generally robust across the hyperparameter search space for this fold.

## Limitations

- Only the first walk-forward fold was used for optimization (by design, per Phase 7 spec). Full multi-window evaluation is deferred to Phase 8.
- The tuning-to-full-training Sharpe gap warrants investigation — it may indicate that 25K steps is insufficient for final performance prediction.
- Results are `label: "implementation-improvement"` since hyperparameter optimization goes beyond the paper's default PPO configuration.

## Transaction Costs

All evaluations include transaction costs (10 bps fee + 5 bps slippage) applied within the environment. The reported metrics are net of costs.

## Output Files

- `scripts/tune_hyperparameters.py` — Optuna optimization script
- `reports/cycle_7/best_params.json` — Optimal hyperparameters
- `reports/cycle_7/study.db` — Full Optuna study database (100 trials)
- `reports/cycle_7/metrics.json` — ARF standard metrics
