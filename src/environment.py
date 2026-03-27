"""Stock trading environment compatible with Gymnasium for DRL-based trading."""

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces


class StockTradingEnv(gym.Env):
    """A stock trading environment following the FinRL paper design.

    State space: [cash, stock_prices, stock_holdings, technical_indicators]
    Action space: continuous actions representing shares to buy/sell per stock
    Reward: change in portfolio value
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        df: pd.DataFrame,
        stock_dim: int,
        hmax: int = 100,
        initial_amount: float = 1_000_000.0,
        buy_cost_pct: float = 0.001,
        sell_cost_pct: float = 0.001,
        tech_indicator_list: list[str] | None = None,
    ):
        """Initialize the trading environment.

        Args:
            df: DataFrame with columns including 'close' and technical indicators.
                For multi-stock, data should be stacked with a 'tic' column and
                sorted by (date, tic).
            stock_dim: Number of stocks being traded.
            hmax: Maximum number of shares to trade per action.
            initial_amount: Starting cash balance.
            buy_cost_pct: Transaction cost for buying (fraction).
            sell_cost_pct: Transaction cost for selling (fraction).
            tech_indicator_list: List of technical indicator column names.
        """
        super().__init__()

        self.df = df
        self.stock_dim = stock_dim
        self.hmax = hmax
        self.initial_amount = initial_amount
        self.buy_cost_pct = buy_cost_pct
        self.sell_cost_pct = sell_cost_pct
        self.tech_indicator_list = tech_indicator_list or []

        # Number of unique dates (each date has stock_dim rows)
        if stock_dim > 1:
            self.dates = sorted(df.index.unique())
        else:
            self.dates = list(df.index)
        self.terminal_day = len(self.dates) - 1

        # State: cash + prices + holdings + tech_indicators
        # For each stock: close price, holding, then each tech indicator
        state_dim = 1 + stock_dim + stock_dim + stock_dim * len(self.tech_indicator_list)

        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(stock_dim,), dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(state_dim,), dtype=np.float32
        )

        # Episode state
        self.day = 0
        self.cash = initial_amount
        self.holdings = np.zeros(stock_dim, dtype=np.float64)
        self.portfolio_values = [initial_amount]
        self.trades_count = 0

    def _get_prices(self, day_idx: int) -> np.ndarray:
        """Get closing prices for all stocks on a given day."""
        if self.stock_dim == 1:
            return np.array([self.df.iloc[day_idx]["close"]])
        date = self.dates[day_idx]
        day_data = self.df.loc[date]
        return day_data["close"].values

    def _get_tech_indicators(self, day_idx: int) -> np.ndarray:
        """Get technical indicators for all stocks on a given day."""
        if not self.tech_indicator_list:
            return np.array([])
        if self.stock_dim == 1:
            row = self.df.iloc[day_idx]
            return np.array([row[ind] for ind in self.tech_indicator_list])
        date = self.dates[day_idx]
        day_data = self.df.loc[date]
        indicators = []
        for ind in self.tech_indicator_list:
            indicators.extend(day_data[ind].values)
        return np.array(indicators)

    def _get_state(self) -> np.ndarray:
        """Construct the observation state vector."""
        prices = self._get_prices(self.day)
        tech = self._get_tech_indicators(self.day)
        state = np.concatenate([
            [self.cash],
            prices,
            self.holdings,
            tech,
        ])
        return state.astype(np.float32)

    def _get_portfolio_value(self) -> float:
        """Calculate total portfolio value (cash + stock holdings)."""
        prices = self._get_prices(self.day)
        return self.cash + np.sum(self.holdings * prices)

    def reset(self, seed=None, options=None):
        """Reset the environment to the initial state."""
        super().reset(seed=seed)
        self.day = 0
        self.cash = self.initial_amount
        self.holdings = np.zeros(self.stock_dim, dtype=np.float64)
        self.portfolio_values = [self.initial_amount]
        self.trades_count = 0
        return self._get_state(), {}

    def step(self, action: np.ndarray):
        """Execute one step in the environment.

        Actions are scaled from [-1, 1] to [-hmax, hmax] shares.
        Sells are executed first (to free up cash), then buys.
        """
        terminated = self.day >= self.terminal_day
        if terminated:
            return self._get_state(), 0.0, True, False, {}

        # Scale actions to share counts
        actions = (action * self.hmax).astype(int)
        prices = self._get_prices(self.day)

        # Sort: sell first (negative actions), then buy (positive actions)
        sell_indices = np.where(actions < 0)[0]
        buy_indices = np.where(actions > 0)[0]

        # Execute sells
        for idx in sell_indices:
            sell_qty = min(abs(actions[idx]), int(self.holdings[idx]))
            if sell_qty > 0:
                sell_amount = sell_qty * prices[idx]
                self.cash += sell_amount * (1 - self.sell_cost_pct)
                self.holdings[idx] -= sell_qty
                self.trades_count += 1

        # Execute buys
        for idx in buy_indices:
            buy_qty = actions[idx]
            buy_cost = buy_qty * prices[idx] * (1 + self.buy_cost_pct)
            # Only buy what we can afford
            affordable = int(self.cash / (prices[idx] * (1 + self.buy_cost_pct)))
            buy_qty = min(buy_qty, affordable)
            if buy_qty > 0:
                self.cash -= buy_qty * prices[idx] * (1 + self.buy_cost_pct)
                self.holdings[idx] += buy_qty
                self.trades_count += 1

        # Advance day
        self.day += 1
        new_value = self._get_portfolio_value()
        reward = new_value - self.portfolio_values[-1]
        self.portfolio_values.append(new_value)

        terminated = self.day >= self.terminal_day
        return self._get_state(), reward, terminated, False, {}
