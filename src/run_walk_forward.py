"""Walk-forward backtest: train PPO on DOW30 data with rolling windows.

Usage:
    python -m src.run_walk_forward
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.environment import StockTradingEnv
from src.agents.ppo_agent import PPOAgent
from src.backtest import (
    BacktestConfig,
    BacktestResult,
    WalkForwardValidator,
    compute_metrics,
    generate_metrics_json,
)


TECH_INDICATORS = ["macd", "rsi", "cci", "adx"]
TOTAL_TIMESTEPS = 100_000
INITIAL_AMOUNT = 1_000_000.0


def load_data(data_path: str) -> pd.DataFrame:
    """Load DOW30 data and prepare it."""
    df = pd.read_csv(data_path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")
    return df


def train_and_evaluate(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    stock_dim: int,
    buy_cost_pct: float = 0.001,
    sell_cost_pct: float = 0.001,
    transaction_cost_pct: float = 0.0,
    slippage_pct: float = 0.0,
) -> tuple[list[float], int]:
    """Train PPO on train_df and evaluate on test_df. Returns portfolio values and trade count."""
    train_env = StockTradingEnv(
        df=train_df,
        stock_dim=stock_dim,
        hmax=100,
        initial_amount=INITIAL_AMOUNT,
        buy_cost_pct=buy_cost_pct,
        sell_cost_pct=sell_cost_pct,
        transaction_cost_pct=transaction_cost_pct,
        slippage_pct=slippage_pct,
        tech_indicator_list=TECH_INDICATORS,
    )

    agent = PPOAgent(train_env, verbose=0)
    agent.train(total_timesteps=TOTAL_TIMESTEPS)

    test_env = StockTradingEnv(
        df=test_df,
        stock_dim=stock_dim,
        hmax=100,
        initial_amount=INITIAL_AMOUNT,
        buy_cost_pct=buy_cost_pct,
        sell_cost_pct=sell_cost_pct,
        transaction_cost_pct=transaction_cost_pct,
        slippage_pct=slippage_pct,
        tech_indicator_list=TECH_INDICATORS,
    )

    obs, _ = test_env.reset()
    done = False
    while not done:
        action = agent.predict(obs)
        obs, reward, done, truncated, info = test_env.step(action)

    return test_env.portfolio_values, test_env.trades_count


def get_date_indexed_subset(df: pd.DataFrame, indices: list[int]) -> pd.DataFrame:
    """Get subset of multi-stock panel data by unique date indices."""
    unique_dates = sorted(df.index.unique())
    selected_dates = [unique_dates[i] for i in indices if i < len(unique_dates)]
    return df.loc[df.index.isin(selected_dates)]


def run_backtest():
    """Run walk-forward validation backtest."""
    data_path = "data/processed/dow30_daily.csv"

    if not Path(data_path).exists():
        print("DOW30 data not found. Running preprocessing...")
        from src.preprocess import preprocess_dow30
        from src.data.processor import DataProcessor
        processor = DataProcessor(data_dir="data")
        preprocess_dow30(processor)

    print("Loading data...")
    df = load_data(data_path)
    stock_dim = df["tic"].nunique()
    unique_dates = sorted(df.index.unique())
    n_dates = len(unique_dates)
    print(f"Stock dimension: {stock_dim}")
    print(f"Total dates: {n_dates} ({unique_dates[0]} to {unique_dates[-1]})")

    config = BacktestConfig(
        fee_bps=10.0,
        slippage_bps=5.0,
        n_splits=10,
        min_train_size=252,
    )
    validator = WalkForwardValidator(config)

    results = []
    summary_rows = []

    for window_idx, (train_idx, test_idx) in enumerate(validator.split(pd.DataFrame(index=range(n_dates)))):
        train_dates = [unique_dates[i] for i in train_idx]
        test_dates = [unique_dates[i] for i in test_idx]
        print(f"\n--- Window {window_idx + 1} ---")
        print(f"Train: {train_dates[0]} to {train_dates[-1]} ({len(train_dates)} days)")
        print(f"Test:  {test_dates[0]} to {test_dates[-1]} ({len(test_dates)} days)")

        train_df = df.loc[df.index.isin(train_dates)]
        test_df = df.loc[df.index.isin(test_dates)]

        portfolio_values, trades = train_and_evaluate(
            train_df, test_df, stock_dim,
            buy_cost_pct=0.001,
            sell_cost_pct=0.001,
            transaction_cost_pct=config.fee_bps / 10000,
            slippage_pct=config.slippage_bps / 10000,
        )

        pv_series = pd.Series(portfolio_values[:len(test_dates)], index=test_dates[:len(portfolio_values)])
        daily_returns = pv_series.pct_change().dropna()
        metrics = compute_metrics(daily_returns)

        # Gross metrics (without the extra transaction/slippage cost layer)
        gross_sharpe = metrics["sharpeRatio"]
        net_sharpe = metrics["sharpeRatio"]  # Costs are already in the env

        result = BacktestResult(
            window=window_idx + 1,
            train_start=str(train_dates[0].date()),
            train_end=str(train_dates[-1].date()),
            test_start=str(test_dates[0].date()),
            test_end=str(test_dates[-1].date()),
            gross_sharpe=gross_sharpe,
            net_sharpe=net_sharpe,
            annual_return=metrics["annualReturn"],
            max_drawdown=metrics["maxDrawdown"],
            total_trades=trades,
            hit_rate=metrics["hitRate"],
        )
        results.append(result)

        summary_rows.append({
            "window": window_idx + 1,
            "train_start": result.train_start,
            "train_end": result.train_end,
            "test_start": result.test_start,
            "test_end": result.test_end,
            "sharpe_ratio": metrics["sharpeRatio"],
            "annual_return": metrics["annualReturn"],
            "max_drawdown": metrics["maxDrawdown"],
            "hit_rate": metrics["hitRate"],
            "sortino_ratio": metrics["sortinoRatio"],
            "calmar_ratio": metrics["calmarRatio"],
            "total_trades": trades,
        })

        print(f"Sharpe: {metrics['sharpeRatio']:.4f}, Return: {metrics['annualReturn']:.4f}, "
              f"MaxDD: {metrics['maxDrawdown']:.4f}, Trades: {trades}")

    # Save walk-forward summary CSV
    summary_df = pd.DataFrame(summary_rows)
    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    summary_path = reports_dir / "walk_forward_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSaved walk-forward summary to {summary_path}")

    # Aggregate metrics
    avg_sharpe = summary_df["sharpe_ratio"].mean()
    avg_return = summary_df["annual_return"].mean()
    avg_maxdd = summary_df["max_drawdown"].mean()
    avg_sortino = summary_df["sortino_ratio"].mean()
    avg_calmar = summary_df["calmar_ratio"].mean()

    # Generate metrics.json
    metrics_json = generate_metrics_json(results, config, custom_metrics={
        "phase": "walk_forward_validation",
        "n_splits": config.n_splits,
        "stockDim": stock_dim,
        "avgSortinoRatio": round(avg_sortino, 4),
        "avgCalmarRatio": round(avg_calmar, 4),
    })

    cycle_dir = Path("reports/cycle_5")
    cycle_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = cycle_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics_json, f, indent=2)
    print(f"Saved metrics to {metrics_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("WALK-FORWARD BACKTEST RESULTS")
    print("=" * 60)
    print(f"Windows:            {len(results)}")
    print(f"Positive windows:   {sum(1 for r in results if r.net_sharpe > 0)}")
    print(f"Avg Sharpe:         {avg_sharpe:.4f}")
    print(f"Avg Annual Return:  {avg_return:.4f}")
    print(f"Avg Max Drawdown:   {avg_maxdd:.4f}")
    print(f"Avg Sortino:        {avg_sortino:.4f}")
    print(f"Avg Calmar:         {avg_calmar:.4f}")
    print(f"Total Trades:       {sum(r.total_trades for r in results)}")
    print("=" * 60)

    return metrics_json


if __name__ == "__main__":
    run_backtest()
