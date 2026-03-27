# Reward Function Design

## Previous Design (Cycle 3)

The original reward function was simply the absolute change in portfolio value:

```
reward = new_portfolio_value - previous_portfolio_value
```

This encouraged the agent to maximize raw returns without considering risk. The agent had no incentive to avoid volatile strategies, which contributed to the large maximum drawdown (-40.39%) observed in Cycle 3.

## Updated Design (Cycle 5)

The reward function now incorporates a risk penalty based on recent return volatility:

```
daily_return = (new_value - prev_value) / prev_value
volatility = std(recent_20_day_returns)
reward = (daily_return - risk_penalty_coef * volatility^2) / reward_scaling
```

### Parameters
- `risk_penalty_coef` (default: 0.05): Controls the strength of the volatility penalty. Higher values make the agent more risk-averse.
- `reward_scaling` (default: 1e-4): Normalizes the reward magnitude for stable PPO training.
- Rolling window: 20 most recent daily returns for volatility estimation.

### Rationale
- **Risk adjustment**: By penalizing variance, the reward function approximates maximizing a mean-variance utility, which is aligned with maximizing the Sharpe ratio.
- **Rolling window**: Using a 20-day rolling window provides a local estimate of recent volatility, encouraging the agent to adapt its risk exposure dynamically.
- **Scaling**: Portfolio-value-based rewards can be very large in absolute terms, causing gradient instability. Normalizing by `reward_scaling` keeps rewards in a range suitable for PPO's value function approximation.

### Connection to Sharpe Ratio
The Sharpe ratio is defined as E[r] / std(r). Our reward `r - lambda * sigma^2` is the first-order approximation of maximizing the Sharpe ratio under mean-variance preferences. The agent learns to trade off expected return against volatility, which is the core of risk-adjusted performance optimization.
