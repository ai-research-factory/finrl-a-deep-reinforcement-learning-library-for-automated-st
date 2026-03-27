# Performance Comparison: PPO vs Baselines (Cycle 7)

## Walk-Forward Averaged Metrics

| Metric | PPO (Optimized) | Buy & Hold | Equal Weight |
|--------|----------------|------------|--------------|
| Sharpe Ratio | 1.1263 | 1.3850 | 1.1418 |
| Annual Return | 12.37% | 20.13% | 15.27% |
| Max Drawdown | -0.1432 | -0.1332 | -0.1387 |
| Hit Rate | 55.18% | 57.43% | 56.31% |
| Sortino Ratio | 1.1302 | 1.3403 | 1.1094 |
| Calmar Ratio | 1.7151 | 2.6466 | 2.1415 |

## Notes

- All metrics are net of transaction costs (10 bps fee + 5 bps slippage).
- PPO uses optimized hyperparameters from Optuna (lr=4.46e-5, n_steps=4096, gamma=0.999, ent_coef=0.01).
- Walk-forward validation uses 10 rolling windows with min 252-day training period.
- Results averaged across 9 walk-forward windows.
- Label: `implementation-improvement` (hyperparameter optimization goes beyond paper defaults).
