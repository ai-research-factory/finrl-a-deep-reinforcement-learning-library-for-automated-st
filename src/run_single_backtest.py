"""Single backtest: train PPO on DOW30 data and evaluate on test period.

Usage:
    python -m src.run_single_backtest
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Allow running from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.environment import StockTradingEnv
from src.agents.ppo_agent import PPOAgent
from src.backtest import BacktestConfig, compute_metrics


TECH_INDICATORS = ["macd", "rsi", "cci", "adx"]
TRAIN_START = "2009-01-01"
TRAIN_END = "2018-12-31"
TEST_START = "2019-01-01"
TEST_END = "2020-09-30"
TOTAL_TIMESTEPS = 100_000
INITIAL_AMOUNT = 1_000_000.0


def load_and_split_data(data_path: str):
    """Load DOW30 data and split into train/test periods."""
    df = pd.read_csv(data_path)
    df["date"] = pd.to_datetime(df["date"])

    train_df = df[(df["date"] >= TRAIN_START) & (df["date"] <= TRAIN_END)].copy()
    test_df = df[(df["date"] >= TEST_START) & (df["date"] <= TEST_END)].copy()

    # Set date as index for the environment
    train_df = train_df.set_index("date")
    test_df = test_df.set_index("date")

    stock_dim = train_df["tic"].nunique()
    print(f"Stock dimension: {stock_dim}")
    print(f"Train: {train_df.index.min()} to {train_df.index.max()} ({len(train_df)} rows)")
    print(f"Test:  {test_df.index.min()} to {test_df.index.max()} ({len(test_df)} rows)")

    return train_df, test_df, stock_dim


def run_backtest():
    """Run the full train-and-test backtest pipeline."""
    data_path = "data/processed/dow30_daily.csv"

    if not Path(data_path).exists():
        print("DOW30 data not found. Running preprocessing...")
        from src.preprocess import preprocess_dow30
        from src.data.processor import DataProcessor
        processor = DataProcessor(data_dir="data")
        preprocess_dow30(processor)

    print("Loading data...")
    train_df, test_df, stock_dim = load_and_split_data(data_path)

    # Create training environment
    print("\nCreating training environment...")
    train_env = StockTradingEnv(
        df=train_df,
        stock_dim=stock_dim,
        hmax=100,
        initial_amount=INITIAL_AMOUNT,
        buy_cost_pct=0.001,
        sell_cost_pct=0.001,
        tech_indicator_list=TECH_INDICATORS,
    )

    # Train PPO agent
    print(f"\nTraining PPO agent for {TOTAL_TIMESTEPS} timesteps...")
    agent = PPOAgent(train_env)
    agent.train(total_timesteps=TOTAL_TIMESTEPS)

    # Save model
    model_path = "models/ppo_dow30"
    agent.save(model_path)
    print(f"Model saved to {model_path}")

    # Create test environment and run backtest
    print("\nRunning backtest on test data...")
    test_env = StockTradingEnv(
        df=test_df,
        stock_dim=stock_dim,
        hmax=100,
        initial_amount=INITIAL_AMOUNT,
        buy_cost_pct=0.001,
        sell_cost_pct=0.001,
        tech_indicator_list=TECH_INDICATORS,
    )

    obs, _ = test_env.reset()
    done = False
    while not done:
        action = agent.predict(obs)
        obs, reward, done, truncated, info = test_env.step(action)

    # Collect results
    portfolio_values = test_env.portfolio_values
    test_dates = sorted(test_df.index.unique())
    # portfolio_values has initial value + one value per step
    dates_for_pv = test_dates[: len(portfolio_values)]

    portfolio_df = pd.DataFrame({
        "date": dates_for_pv,
        "portfolio_value": portfolio_values[: len(dates_for_pv)],
    })

    # Save results
    reports_dir = Path("reports/cycle_3")
    reports_dir.mkdir(parents=True, exist_ok=True)

    csv_path = reports_dir / "portfolio_value.csv"
    portfolio_df.to_csv(csv_path, index=False)
    print(f"\nSaved portfolio values to {csv_path}")

    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(portfolio_df["date"], portfolio_df["portfolio_value"], linewidth=1.5)
    ax.set_title("Portfolio Value During Test Period (PPO Agent on DOW30)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Portfolio Value ($)")
    ax.grid(True, alpha=0.3)
    ax.ticklabel_format(style="plain", axis="y")
    plt.xticks(rotation=45)
    plt.tight_layout()

    png_path = reports_dir / "portfolio_value.png"
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    print(f"Saved portfolio plot to {png_path}")

    # Compute metrics
    pv_series = pd.Series(portfolio_values[: len(dates_for_pv)], index=dates_for_pv)
    daily_returns = pv_series.pct_change().dropna()

    config = BacktestConfig()
    metrics = compute_metrics(daily_returns)
    total_trades = test_env.trades_count

    # Build metrics.json
    final_value = portfolio_values[-1]
    total_return = (final_value / INITIAL_AMOUNT) - 1

    metrics_json = {
        "sharpeRatio": metrics["sharpeRatio"],
        "annualReturn": metrics["annualReturn"],
        "maxDrawdown": metrics["maxDrawdown"],
        "hitRate": metrics["hitRate"],
        "totalTrades": total_trades,
        "transactionCosts": {
            "feeBps": 10,
            "slippageBps": 5,
            "netSharpe": metrics["sharpeRatio"],  # costs built into env
        },
        "walkForward": {
            "windows": 0,
            "positiveWindows": 0,
            "avgOosSharpe": 0.0,
        },
        "customMetrics": {
            "phase": "single_backtest",
            "totalReturn": round(total_return, 4),
            "finalPortfolioValue": round(final_value, 2),
            "initialAmount": INITIAL_AMOUNT,
            "trainPeriod": f"{TRAIN_START} to {TRAIN_END}",
            "testPeriod": f"{TEST_START} to {TEST_END}",
            "totalTimesteps": TOTAL_TIMESTEPS,
            "stockDim": stock_dim,
        },
    }

    metrics_path = reports_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics_json, f, indent=2)
    print(f"Saved metrics to {metrics_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("BACKTEST RESULTS")
    print("=" * 60)
    print(f"Initial Amount:     ${INITIAL_AMOUNT:,.2f}")
    print(f"Final Value:        ${final_value:,.2f}")
    print(f"Total Return:       {total_return * 100:.2f}%")
    print(f"Sharpe Ratio:       {metrics['sharpeRatio']:.4f}")
    print(f"Annual Return:      {metrics['annualReturn'] * 100:.2f}%")
    print(f"Max Drawdown:       {metrics['maxDrawdown'] * 100:.2f}%")
    print(f"Hit Rate:           {metrics['hitRate'] * 100:.2f}%")
    print(f"Total Trades:       {total_trades}")
    print("=" * 60)

    return metrics_json


if __name__ == "__main__":
    run_backtest()
