"""Walk-forward backtest: train PPO on DOW30 data with rolling windows.

Supports optimized hyperparameters from Optuna and baseline strategy comparison.

Usage:
    python -m src.run_walk_forward
    python -m src.run_walk_forward --cycle 7 --use-optimized-params
"""

import argparse
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
from src.baselines import buy_and_hold, equal_weight_rebalance
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


def load_optimized_params(cycle: int) -> dict | None:
    """Load optimized hyperparameters from best_params.json if available."""
    path = Path(f"reports/cycle_{cycle}/best_params.json")
    if not path.exists():
        return None
    with open(path) as f:
        params = json.load(f)
    return {
        "learning_rate": params.get("learning_rate", 3e-4),
        "n_steps": params.get("n_steps", 2048),
        "gamma": params.get("gamma", 0.99),
        "ent_coef": params.get("ent_coef", 0.0),
    }


def train_and_evaluate(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    stock_dim: int,
    config: BacktestConfig,
    ppo_params: dict | None = None,
) -> tuple[list[float], int]:
    """Train PPO on train_df and evaluate on test_df. Returns portfolio values and trade count."""
    env_kwargs = dict(
        stock_dim=stock_dim,
        hmax=100,
        initial_amount=INITIAL_AMOUNT,
        buy_cost_pct=0.001,
        sell_cost_pct=0.001,
        transaction_cost_pct=config.fee_bps / 10000,
        slippage_pct=config.slippage_bps / 10000,
        tech_indicator_list=TECH_INDICATORS,
    )

    train_env = StockTradingEnv(df=train_df, **env_kwargs)

    agent_kwargs = {"verbose": 0}
    if ppo_params:
        agent_kwargs.update(ppo_params)

    agent = PPOAgent(train_env, **agent_kwargs)
    agent.train(total_timesteps=TOTAL_TIMESTEPS)

    test_env = StockTradingEnv(df=test_df, **env_kwargs)
    obs, _ = test_env.reset()
    done = False
    while not done:
        action = agent.predict(obs)
        obs, reward, done, truncated, info = test_env.step(action)

    return test_env.portfolio_values, test_env.trades_count


def evaluate_baselines(
    test_df: pd.DataFrame,
    stock_dim: int,
    config: BacktestConfig,
    test_dates: list,
) -> dict:
    """Evaluate baseline strategies on the same test period."""
    bh_pv = buy_and_hold(test_df, stock_dim, INITIAL_AMOUNT, config)
    ew_pv = equal_weight_rebalance(test_df, stock_dim, INITIAL_AMOUNT, config=config)

    results = {}
    for name, pv_list in [("buy_and_hold", bh_pv), ("equal_weight", ew_pv)]:
        pv_series = pd.Series(pv_list[:len(test_dates)], index=test_dates[:len(pv_list)])
        daily_returns = pv_series.pct_change().dropna()
        metrics = compute_metrics(daily_returns)
        results[name] = metrics
    return results


def generate_sharpe_histogram(sharpes: list[float], baseline_sharpes: dict, output_path: Path):
    """Generate Sharpe ratio distribution histogram."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(sharpes, bins=max(5, len(sharpes) // 2), alpha=0.7, color="steelblue", edgecolor="black", label="PPO (per window)")
    ax.axvline(np.mean(sharpes), color="red", linestyle="--", linewidth=2, label=f"PPO Mean: {np.mean(sharpes):.3f}")
    ax.axvline(0, color="gray", linestyle=":", linewidth=1)

    colors = {"buy_and_hold": "green", "equal_weight": "orange"}
    for name, avg_sharpe in baseline_sharpes.items():
        label = name.replace("_", " ").title()
        ax.axvline(avg_sharpe, color=colors.get(name, "purple"), linestyle="-.", linewidth=2, label=f"{label} Avg: {avg_sharpe:.3f}")

    ax.set_xlabel("Sharpe Ratio (OOS)")
    ax.set_ylabel("Count")
    ax.set_title("Walk-Forward OOS Sharpe Ratio Distribution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved Sharpe distribution to {output_path}")


def generate_performance_comparison(
    ppo_results: list[dict],
    baseline_results: dict[str, list[dict]],
    output_path: Path,
):
    """Generate a markdown performance comparison table."""
    def avg_metrics(results_list):
        if not results_list:
            return {}
        keys = results_list[0].keys()
        return {k: np.mean([r[k] for r in results_list]) for k in keys}

    ppo_avg = avg_metrics(ppo_results)
    bh_avg = avg_metrics(baseline_results.get("buy_and_hold", []))
    ew_avg = avg_metrics(baseline_results.get("equal_weight", []))

    lines = [
        "# Performance Comparison: PPO vs Baselines (Cycle 7)",
        "",
        "## Walk-Forward Averaged Metrics",
        "",
        "| Metric | PPO (Optimized) | Buy & Hold | Equal Weight |",
        "|--------|----------------|------------|--------------|",
    ]

    metric_labels = [
        ("sharpeRatio", "Sharpe Ratio"),
        ("annualReturn", "Annual Return"),
        ("maxDrawdown", "Max Drawdown"),
        ("hitRate", "Hit Rate"),
        ("sortinoRatio", "Sortino Ratio"),
        ("calmarRatio", "Calmar Ratio"),
    ]
    for key, label in metric_labels:
        ppo_val = ppo_avg.get(key, 0)
        bh_val = bh_avg.get(key, 0)
        ew_val = ew_avg.get(key, 0)
        if key in ("annualReturn", "hitRate"):
            lines.append(f"| {label} | {ppo_val:.2%} | {bh_val:.2%} | {ew_val:.2%} |")
        else:
            lines.append(f"| {label} | {ppo_val:.4f} | {bh_val:.4f} | {ew_val:.4f} |")

    lines.extend([
        "",
        "## Notes",
        "",
        "- All metrics are net of transaction costs (10 bps fee + 5 bps slippage).",
        "- PPO uses optimized hyperparameters from Optuna (lr=4.46e-5, n_steps=4096, gamma=0.999, ent_coef=0.01).",
        "- Walk-forward validation uses 10 rolling windows with min 252-day training period.",
        f"- Results averaged across {len(ppo_results)} walk-forward windows.",
        "- Label: `implementation-improvement` (hyperparameter optimization goes beyond paper defaults).",
    ])

    output_path.write_text("\n".join(lines) + "\n")
    print(f"Saved performance comparison to {output_path}")


def run_backtest(cycle: int = 7, use_optimized_params: bool = False):
    """Run walk-forward validation backtest with baseline comparison."""
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

    # Load optimized params if requested
    ppo_params = None
    if use_optimized_params:
        ppo_params = load_optimized_params(cycle)
        if ppo_params:
            print(f"Using optimized params: {ppo_params}")
        else:
            print("No optimized params found, using defaults.")

    config = BacktestConfig(
        fee_bps=10.0,
        slippage_bps=5.0,
        n_splits=10,
        min_train_size=252,
    )
    validator = WalkForwardValidator(config)

    results = []
    summary_rows = []
    ppo_metrics_list = []
    baseline_results = {"buy_and_hold": [], "equal_weight": []}

    for window_idx, (train_idx, test_idx) in enumerate(validator.split(pd.DataFrame(index=range(n_dates)))):
        train_dates = [unique_dates[i] for i in train_idx]
        test_dates = [unique_dates[i] for i in test_idx]
        print(f"\n--- Window {window_idx + 1} ---")
        print(f"Train: {train_dates[0]} to {train_dates[-1]} ({len(train_dates)} days)")
        print(f"Test:  {test_dates[0]} to {test_dates[-1]} ({len(test_dates)} days)")

        train_df = df.loc[df.index.isin(train_dates)]
        test_df = df.loc[df.index.isin(test_dates)]

        # PPO agent
        portfolio_values, trades = train_and_evaluate(
            train_df, test_df, stock_dim, config, ppo_params=ppo_params,
        )

        pv_series = pd.Series(portfolio_values[:len(test_dates)], index=test_dates[:len(portfolio_values)])
        daily_returns = pv_series.pct_change().dropna()
        metrics = compute_metrics(daily_returns)
        ppo_metrics_list.append(metrics)

        net_sharpe = metrics["sharpeRatio"]
        result = BacktestResult(
            window=window_idx + 1,
            train_start=str(train_dates[0].date()),
            train_end=str(train_dates[-1].date()),
            test_start=str(test_dates[0].date()),
            test_end=str(test_dates[-1].date()),
            gross_sharpe=net_sharpe,
            net_sharpe=net_sharpe,
            annual_return=metrics["annualReturn"],
            max_drawdown=metrics["maxDrawdown"],
            total_trades=trades,
            hit_rate=metrics["hitRate"],
        )
        results.append(result)

        # Baselines
        bl_metrics = evaluate_baselines(test_df, stock_dim, config, test_dates)
        for name in baseline_results:
            baseline_results[name].append(bl_metrics[name])

        summary_rows.append({
            "window": window_idx + 1,
            "train_start": result.train_start,
            "train_end": result.train_end,
            "test_start": result.test_start,
            "test_end": result.test_end,
            "ppo_sharpe": metrics["sharpeRatio"],
            "ppo_annual_return": metrics["annualReturn"],
            "ppo_max_drawdown": metrics["maxDrawdown"],
            "ppo_hit_rate": metrics["hitRate"],
            "ppo_sortino": metrics["sortinoRatio"],
            "ppo_calmar": metrics["calmarRatio"],
            "ppo_trades": trades,
            "bh_sharpe": bl_metrics["buy_and_hold"]["sharpeRatio"],
            "bh_annual_return": bl_metrics["buy_and_hold"]["annualReturn"],
            "bh_max_drawdown": bl_metrics["buy_and_hold"]["maxDrawdown"],
            "ew_sharpe": bl_metrics["equal_weight"]["sharpeRatio"],
            "ew_annual_return": bl_metrics["equal_weight"]["annualReturn"],
            "ew_max_drawdown": bl_metrics["equal_weight"]["maxDrawdown"],
        })

        print(f"PPO  Sharpe: {metrics['sharpeRatio']:.4f}, Return: {metrics['annualReturn']:.4f}, MaxDD: {metrics['maxDrawdown']:.4f}")
        print(f"B&H  Sharpe: {bl_metrics['buy_and_hold']['sharpeRatio']:.4f}")
        print(f"EW   Sharpe: {bl_metrics['equal_weight']['sharpeRatio']:.4f}")

    # Save walk-forward summary CSV
    summary_df = pd.DataFrame(summary_rows)
    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    summary_path = reports_dir / "walk_forward_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSaved walk-forward summary to {summary_path}")

    # Output directory for this cycle
    cycle_dir = Path(f"reports/cycle_{cycle}")
    cycle_dir.mkdir(parents=True, exist_ok=True)

    # Sharpe distribution stats
    ppo_sharpes = [m["sharpeRatio"] for m in ppo_metrics_list]
    bh_sharpes = [m["sharpeRatio"] for m in baseline_results["buy_and_hold"]]
    ew_sharpes = [m["sharpeRatio"] for m in baseline_results["equal_weight"]]

    # Generate Sharpe distribution histogram
    baseline_avg_sharpes = {
        "buy_and_hold": np.mean(bh_sharpes) if bh_sharpes else 0,
        "equal_weight": np.mean(ew_sharpes) if ew_sharpes else 0,
    }
    generate_sharpe_histogram(ppo_sharpes, baseline_avg_sharpes, cycle_dir / "sharpe_distribution.png")

    # Generate performance comparison report
    generate_performance_comparison(ppo_metrics_list, baseline_results, cycle_dir / "performance_comparison.md")

    # Generate metrics.json with walk-forward stats
    custom = {
        "phase": "walk_forward_with_optimized_params" if use_optimized_params else "walk_forward_validation",
        "label": "implementation-improvement" if use_optimized_params else "paper-reproduction",
        "n_splits": config.n_splits,
        "stockDim": stock_dim,
        "avgSortinoRatio": round(float(np.mean([m["sortinoRatio"] for m in ppo_metrics_list])), 4),
        "avgCalmarRatio": round(float(np.mean([m["calmarRatio"] for m in ppo_metrics_list])), 4),
        "sharpeDistribution": {
            "mean": round(float(np.mean(ppo_sharpes)), 4),
            "std": round(float(np.std(ppo_sharpes)), 4),
            "min": round(float(np.min(ppo_sharpes)), 4),
            "max": round(float(np.max(ppo_sharpes)), 4),
        },
        "baselineComparison": {
            "buyAndHold": {
                "avgSharpe": round(float(np.mean(bh_sharpes)), 4),
                "avgReturn": round(float(np.mean([m["annualReturn"] for m in baseline_results["buy_and_hold"]])), 4),
                "avgMaxDD": round(float(np.mean([m["maxDrawdown"] for m in baseline_results["buy_and_hold"]])), 4),
            },
            "equalWeight": {
                "avgSharpe": round(float(np.mean(ew_sharpes)), 4),
                "avgReturn": round(float(np.mean([m["annualReturn"] for m in baseline_results["equal_weight"]])), 4),
                "avgMaxDD": round(float(np.mean([m["maxDrawdown"] for m in baseline_results["equal_weight"]])), 4),
            },
        },
    }
    if use_optimized_params and ppo_params:
        custom["optimizedParams"] = ppo_params

    metrics_json = generate_metrics_json(results, config, custom_metrics=custom)

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
    print(f"Avg PPO Sharpe:     {np.mean(ppo_sharpes):.4f} +/- {np.std(ppo_sharpes):.4f}")
    print(f"Avg B&H Sharpe:     {np.mean(bh_sharpes):.4f}")
    print(f"Avg EW  Sharpe:     {np.mean(ew_sharpes):.4f}")
    print(f"Total Trades:       {sum(r.total_trades for r in results)}")
    print("=" * 60)

    return metrics_json


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Walk-forward backtest")
    parser.add_argument("--cycle", type=int, default=7, help="Cycle number for output directory")
    parser.add_argument("--use-optimized-params", action="store_true", help="Load optimized params from best_params.json")
    args = parser.parse_args()
    run_backtest(cycle=args.cycle, use_optimized_params=args.use_optimized_params)
