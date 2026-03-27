"""Baseline trading strategies for comparison with DRL agents.

Implements Buy & Hold and Equal Weight (monthly rebalance) strategies
that can be evaluated on the same walk-forward framework as the PPO agent.
"""

import numpy as np
import pandas as pd

from src.backtest import BacktestConfig


def buy_and_hold(
    test_df: pd.DataFrame,
    stock_dim: int,
    initial_amount: float = 1_000_000.0,
    config: BacktestConfig | None = None,
) -> list[float]:
    """Buy & Hold: purchase equal-weight portfolio on day 1, hold until end.

    Args:
        test_df: Panel data with 'tic' and 'close' columns, date-indexed.
        stock_dim: Number of stocks.
        initial_amount: Starting capital.
        config: Backtest config for cost parameters.

    Returns:
        List of daily portfolio values.
    """
    cfg = config or BacktestConfig()
    cost_rate = (cfg.fee_bps + cfg.slippage_bps) / 10000

    dates = sorted(test_df.index.unique())
    first_day = test_df.loc[dates[0]]
    prices = first_day["close"].values

    # Allocate equal cash per stock, buy as many shares as possible
    cash_per_stock = initial_amount / stock_dim
    holdings = np.zeros(stock_dim)
    total_cost = 0.0
    for i in range(stock_dim):
        affordable = int(cash_per_stock / (prices[i] * (1 + cost_rate)))
        holdings[i] = affordable
        buy_amount = affordable * prices[i]
        total_cost += buy_amount * cost_rate

    remaining_cash = initial_amount - np.sum(holdings * prices) - total_cost

    portfolio_values = []
    for date in dates:
        day_data = test_df.loc[date]
        day_prices = day_data["close"].values
        pv = remaining_cash + np.sum(holdings * day_prices)
        portfolio_values.append(pv)

    return portfolio_values


def equal_weight_rebalance(
    test_df: pd.DataFrame,
    stock_dim: int,
    initial_amount: float = 1_000_000.0,
    rebalance_freq: int = 21,
    config: BacktestConfig | None = None,
) -> list[float]:
    """Equal Weight with periodic rebalancing (default: monthly ~21 trading days).

    Args:
        test_df: Panel data with 'tic' and 'close' columns, date-indexed.
        stock_dim: Number of stocks.
        initial_amount: Starting capital.
        rebalance_freq: Rebalance every N trading days.
        config: Backtest config for cost parameters.

    Returns:
        List of daily portfolio values.
    """
    cfg = config or BacktestConfig()
    cost_rate = (cfg.fee_bps + cfg.slippage_bps) / 10000

    dates = sorted(test_df.index.unique())
    holdings = np.zeros(stock_dim)
    cash = initial_amount
    portfolio_values = []

    for day_idx, date in enumerate(dates):
        day_data = test_df.loc[date]
        prices = day_data["close"].values

        # Rebalance on day 0 and every rebalance_freq days
        if day_idx % rebalance_freq == 0:
            # Sell all current holdings
            if day_idx > 0:
                sell_proceeds = np.sum(holdings * prices)
                sell_cost = sell_proceeds * cost_rate
                cash += sell_proceeds - sell_cost
                holdings = np.zeros(stock_dim)

            # Buy equal-weight
            target_per_stock = cash / stock_dim
            total_buy_cost = 0.0
            for i in range(stock_dim):
                affordable = int(target_per_stock / (prices[i] * (1 + cost_rate)))
                holdings[i] = affordable
                buy_amount = affordable * prices[i]
                total_buy_cost += buy_amount * cost_rate
            cash = cash - np.sum(holdings * prices) - total_buy_cost

        pv = cash + np.sum(holdings * prices)
        portfolio_values.append(pv)

    return portfolio_values
