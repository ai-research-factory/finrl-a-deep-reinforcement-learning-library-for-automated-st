"""Cost analysis: compare gross vs net performance with transaction costs.

Runs the same backtest with two configurations:
  1. Gross (no costs): transaction_cost_pct=0, slippage_pct=0
  2. Net (with costs): transaction_cost_pct=0.001, slippage_pct=0.0005

Usage:
    python -m src.run_cost_analysis
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

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

    train_df = train_df.set_index("date")
    test_df = test_df.set_index("date")

    stock_dim = train_df["tic"].nunique()
    return train_df, test_df, stock_dim


def run_single_config(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    stock_dim: int,
    label: str,
    buy_cost_pct: float = 0.0,
    sell_cost_pct: float = 0.0,
    transaction_cost_pct: float = 0.0,
    slippage_pct: float = 0.0,
) -> dict:
    """Train and evaluate a single cost configuration."""
    print(f"\n{'=' * 50}")
    print(f"Running: {label}")
    print(f"  buy_cost_pct={buy_cost_pct}, sell_cost_pct={sell_cost_pct}")
    print(f"  transaction_cost_pct={transaction_cost_pct}, slippage_pct={slippage_pct}")
    print(f"{'=' * 50}")

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

    portfolio_values = test_env.portfolio_values
    test_dates = sorted(test_df.index.unique())
    pv_series = pd.Series(
        portfolio_values[:len(test_dates)],
        index=test_dates[:len(portfolio_values)],
    )
    daily_returns = pv_series.pct_change().dropna()

    metrics = compute_metrics(daily_returns)
    final_value = portfolio_values[-1]
    total_return = (final_value / INITIAL_AMOUNT) - 1

    result = {
        "sharpeRatio": metrics["sharpeRatio"],
        "annualReturn": metrics["annualReturn"],
        "maxDrawdown": metrics["maxDrawdown"],
        "hitRate": metrics["hitRate"],
        "sortinoRatio": metrics["sortinoRatio"],
        "calmarRatio": metrics["calmarRatio"],
        "totalReturn": round(total_return, 4),
        "finalPortfolioValue": round(final_value, 2),
        "totalTrades": test_env.trades_count,
        "totalTransactionCosts": round(test_env.total_transaction_costs, 2),
    }

    print(f"  Sharpe: {metrics['sharpeRatio']:.4f}")
    print(f"  Annual Return: {metrics['annualReturn']:.4f}")
    print(f"  Max Drawdown: {metrics['maxDrawdown']:.4f}")
    print(f"  Total Return: {total_return:.4f}")
    print(f"  Total Trades: {test_env.trades_count}")
    print(f"  Total Costs: ${test_env.total_transaction_costs:,.2f}")

    return result


def evaluate_with_costs(
    agent,
    test_df: pd.DataFrame,
    stock_dim: int,
    label: str,
    buy_cost_pct: float = 0.0,
    sell_cost_pct: float = 0.0,
    transaction_cost_pct: float = 0.0,
    slippage_pct: float = 0.0,
) -> dict:
    """Evaluate a trained agent under a specific cost configuration."""
    print(f"\n  Evaluating: {label}")

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

    portfolio_values = test_env.portfolio_values
    test_dates = sorted(test_df.index.unique())
    pv_series = pd.Series(
        portfolio_values[:len(test_dates)],
        index=test_dates[:len(portfolio_values)],
    )
    daily_returns = pv_series.pct_change().dropna()
    metrics = compute_metrics(daily_returns)

    final_value = portfolio_values[-1]
    total_return = (final_value / INITIAL_AMOUNT) - 1

    result = {
        "sharpeRatio": metrics["sharpeRatio"],
        "annualReturn": metrics["annualReturn"],
        "maxDrawdown": metrics["maxDrawdown"],
        "hitRate": metrics["hitRate"],
        "sortinoRatio": metrics["sortinoRatio"],
        "calmarRatio": metrics["calmarRatio"],
        "totalReturn": round(total_return, 4),
        "finalPortfolioValue": round(final_value, 2),
        "totalTrades": test_env.trades_count,
        "totalTransactionCosts": round(test_env.total_transaction_costs, 2),
    }

    print(f"    Sharpe: {metrics['sharpeRatio']:.4f}, Return: {total_return:.4f}, "
          f"MaxDD: {metrics['maxDrawdown']:.4f}, Costs: ${test_env.total_transaction_costs:,.2f}")

    return result


def run_cost_analysis():
    """Run cost analysis comparing gross and net performance.

    Trains a SINGLE model, then evaluates the same policy under
    two cost configurations for a fair comparison.
    """
    data_path = "data/processed/dow30_daily.csv"

    if not Path(data_path).exists():
        print("DOW30 data not found. Running preprocessing...")
        from src.preprocess import preprocess_dow30
        from src.data.processor import DataProcessor
        processor = DataProcessor(data_dir="data")
        preprocess_dow30(processor)

    print("Loading data...")
    train_df, test_df, stock_dim = load_and_split_data(data_path)
    print(f"Stock dimension: {stock_dim}")

    # Train a single model (with moderate costs during training for realistic learning)
    print("\nTraining PPO agent...")
    train_env = StockTradingEnv(
        df=train_df,
        stock_dim=stock_dim,
        hmax=100,
        initial_amount=INITIAL_AMOUNT,
        buy_cost_pct=0.001,
        sell_cost_pct=0.001,
        transaction_cost_pct=0.0,
        slippage_pct=0.0,
        tech_indicator_list=TECH_INDICATORS,
    )
    agent = PPOAgent(train_env, verbose=0)
    agent.train(total_timesteps=TOTAL_TIMESTEPS)
    print("Training complete.")

    # Evaluate the SAME model under two cost configurations
    gross = evaluate_with_costs(
        agent, test_df, stock_dim,
        label="GROSS (no costs)",
        buy_cost_pct=0.0,
        sell_cost_pct=0.0,
        transaction_cost_pct=0.0,
        slippage_pct=0.0,
    )

    net = evaluate_with_costs(
        agent, test_df, stock_dim,
        label="NET (fee=0.1%, slippage=0.05%)",
        buy_cost_pct=0.001,
        sell_cost_pct=0.001,
        transaction_cost_pct=0.001,
        slippage_pct=0.0005,
    )

    # Build comparison report
    comparison = {
        "gross_performance": gross,
        "net_performance": net,
        "cost_config": {
            "gross": {
                "buy_cost_pct": 0.0,
                "sell_cost_pct": 0.0,
                "transaction_cost_pct": 0.0,
                "slippage_pct": 0.0,
            },
            "net": {
                "buy_cost_pct": 0.001,
                "sell_cost_pct": 0.001,
                "transaction_cost_pct": 0.001,
                "slippage_pct": 0.0005,
            },
        },
        "impact": {
            "sharpe_reduction": round(gross["sharpeRatio"] - net["sharpeRatio"], 4),
            "return_reduction": round(gross["annualReturn"] - net["annualReturn"], 4),
            "drawdown_increase": round(abs(net["maxDrawdown"]) - abs(gross["maxDrawdown"]), 4),
        },
    }

    # Save
    reports_dir = Path("reports/cycle_5")
    reports_dir.mkdir(parents=True, exist_ok=True)

    comparison_path = reports_dir / "cost_comparison.json"
    with open(comparison_path, "w") as f:
        json.dump(comparison, f, indent=2)
    print(f"\nSaved cost comparison to {comparison_path}")

    # Print comparison
    print("\n" + "=" * 60)
    print("COST IMPACT ANALYSIS")
    print("=" * 60)
    print(f"{'Metric':<25} {'Gross':>12} {'Net':>12} {'Impact':>12}")
    print("-" * 61)
    print(f"{'Sharpe Ratio':<25} {gross['sharpeRatio']:>12.4f} {net['sharpeRatio']:>12.4f} {comparison['impact']['sharpe_reduction']:>12.4f}")
    print(f"{'Annual Return':<25} {gross['annualReturn']:>12.4f} {net['annualReturn']:>12.4f} {comparison['impact']['return_reduction']:>12.4f}")
    print(f"{'Max Drawdown':<25} {gross['maxDrawdown']:>12.4f} {net['maxDrawdown']:>12.4f} {comparison['impact']['drawdown_increase']:>12.4f}")
    print(f"{'Total Trades':<25} {gross['totalTrades']:>12d} {net['totalTrades']:>12d}")
    print(f"{'Transaction Costs ($)':<25} {gross['totalTransactionCosts']:>12,.2f} {net['totalTransactionCosts']:>12,.2f}")
    print("=" * 60)

    return comparison


if __name__ == "__main__":
    run_cost_analysis()
