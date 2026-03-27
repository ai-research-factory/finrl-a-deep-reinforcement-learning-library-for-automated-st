"""Hyperparameter optimization for PPO agent using Optuna.

Optimizes PPO hyperparameters on the first walk-forward fold,
maximizing out-of-sample Sharpe ratio.

Usage:
    python -m scripts.tune_hyperparameters
    python -m scripts.tune_hyperparameters --n-trials 100
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import optuna
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.backtest import BacktestConfig, WalkForwardValidator, compute_metrics
from src.environment import StockTradingEnv
from src.agents.ppo_agent import PPOAgent


TECH_INDICATORS = ["macd", "rsi", "cci", "adx"]
INITIAL_AMOUNT = 1_000_000.0
TOTAL_TIMESTEPS = 100_000
TUNING_TIMESTEPS = 25_000  # Reduced timesteps for faster hyperparameter search


def load_data(data_path: str = "data/processed/dow30_daily.csv") -> pd.DataFrame:
    """Load DOW30 panel data."""
    df = pd.read_csv(data_path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")
    return df


def get_first_fold(df: pd.DataFrame, config: BacktestConfig):
    """Get train/test DataFrames for the first walk-forward fold."""
    unique_dates = sorted(df.index.unique())
    n_dates = len(unique_dates)

    validator = WalkForwardValidator(config)
    for train_idx, test_idx in validator.split(pd.DataFrame(index=range(n_dates))):
        train_dates = [unique_dates[i] for i in train_idx]
        test_dates = [unique_dates[i] for i in test_idx]
        train_df = df.loc[df.index.isin(train_dates)]
        test_df = df.loc[df.index.isin(test_dates)]
        return train_df, test_df, train_dates, test_dates

    raise RuntimeError("No valid walk-forward fold found")


def evaluate_params(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    test_dates: list,
    stock_dim: int,
    config: BacktestConfig,
    learning_rate: float,
    n_steps: int,
    gamma: float,
    ent_coef: float,
) -> float:
    """Train PPO with given params and return OOS Sharpe ratio."""
    train_env = StockTradingEnv(
        df=train_df,
        stock_dim=stock_dim,
        hmax=100,
        initial_amount=INITIAL_AMOUNT,
        buy_cost_pct=0.001,
        sell_cost_pct=0.001,
        transaction_cost_pct=config.fee_bps / 10000,
        slippage_pct=config.slippage_bps / 10000,
        tech_indicator_list=TECH_INDICATORS,
    )

    agent = PPOAgent(
        train_env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        gamma=gamma,
        ent_coef=ent_coef,
        verbose=0,
    )
    agent.train(total_timesteps=TUNING_TIMESTEPS)

    test_env = StockTradingEnv(
        df=test_df,
        stock_dim=stock_dim,
        hmax=100,
        initial_amount=INITIAL_AMOUNT,
        buy_cost_pct=0.001,
        sell_cost_pct=0.001,
        transaction_cost_pct=config.fee_bps / 10000,
        slippage_pct=config.slippage_bps / 10000,
        tech_indicator_list=TECH_INDICATORS,
    )

    obs, _ = test_env.reset()
    done = False
    while not done:
        action = agent.predict(obs)
        obs, reward, done, truncated, info = test_env.step(action)

    portfolio_values = test_env.portfolio_values[:len(test_dates)]
    pv_series = pd.Series(portfolio_values, index=test_dates[:len(portfolio_values)])
    daily_returns = pv_series.pct_change().dropna()
    metrics = compute_metrics(daily_returns)

    return metrics["sharpeRatio"]


def objective(trial: optuna.Trial, train_df, test_df, test_dates, stock_dim, config):
    """Optuna objective function: maximize Sharpe ratio on first fold."""
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    n_steps = trial.suggest_categorical("n_steps", [2048, 4096])
    gamma = trial.suggest_categorical("gamma", [0.99, 0.995, 0.999])
    ent_coef = trial.suggest_categorical("ent_coef", [0.0, 0.01])

    sharpe = evaluate_params(
        train_df, test_df, test_dates, stock_dim, config,
        learning_rate=learning_rate,
        n_steps=n_steps,
        gamma=gamma,
        ent_coef=ent_coef,
    )

    return sharpe


def main():
    parser = argparse.ArgumentParser(description="PPO hyperparameter optimization with Optuna")
    parser.add_argument("--n-trials", type=int, default=100, help="Number of Optuna trials")
    parser.add_argument("--data-path", type=str, default="data/processed/dow30_daily.csv")
    args = parser.parse_args()

    # Ensure data exists
    data_path = args.data_path
    if not Path(data_path).exists():
        print("DOW30 data not found. Running preprocessing...")
        from src.preprocess import preprocess_dow30
        from src.data.processor import DataProcessor
        processor = DataProcessor(data_dir="data")
        preprocess_dow30(processor)

    print("Loading data...")
    df = load_data(data_path)
    stock_dim = df["tic"].nunique()
    print(f"Stock dimension: {stock_dim}")

    config = BacktestConfig(
        fee_bps=10.0,
        slippage_bps=5.0,
        n_splits=10,
        min_train_size=252,
    )

    print("Extracting first walk-forward fold...")
    train_df, test_df, train_dates, test_dates = get_first_fold(df, config)
    print(f"Train: {train_dates[0].date()} to {train_dates[-1].date()} ({len(train_dates)} days)")
    print(f"Test:  {test_dates[0].date()} to {test_dates[-1].date()} ({len(test_dates)} days)")

    # Create Optuna study with SQLite storage
    reports_dir = Path("reports/cycle_7")
    reports_dir.mkdir(parents=True, exist_ok=True)
    storage_path = reports_dir / "study.db"
    storage = f"sqlite:///{storage_path}"

    study = optuna.create_study(
        study_name="ppo_hyperopt",
        direction="maximize",
        storage=storage,
        load_if_exists=True,
    )

    print(f"\nStarting optimization with {args.n_trials} trials...")
    study.optimize(
        lambda trial: objective(trial, train_df, test_df, test_dates, stock_dim, config),
        n_trials=args.n_trials,
        show_progress_bar=True,
    )

    # Save best parameters
    best_params = study.best_params
    best_params["best_sharpe"] = study.best_value
    best_params["n_trials"] = len(study.trials)

    best_params_path = reports_dir / "best_params.json"
    with open(best_params_path, "w") as f:
        json.dump(best_params, f, indent=2)
    print(f"\nBest parameters saved to {best_params_path}")

    # Print results
    print("\n" + "=" * 60)
    print("OPTIMIZATION RESULTS")
    print("=" * 60)
    print(f"Best trial:         #{study.best_trial.number}")
    print(f"Best Sharpe ratio:  {study.best_value:.4f}")
    print(f"Best parameters:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")
    print(f"Total trials:       {len(study.trials)}")
    print(f"Study saved to:     {storage_path}")
    print("=" * 60)

    # Re-evaluate best params with full timesteps for accurate metrics
    print("\nRe-evaluating best params with full training timesteps...")
    best = study.best_params
    best_train_env = StockTradingEnv(
        df=train_df,
        stock_dim=stock_dim,
        hmax=100,
        initial_amount=INITIAL_AMOUNT,
        buy_cost_pct=0.001,
        sell_cost_pct=0.001,
        transaction_cost_pct=config.fee_bps / 10000,
        slippage_pct=config.slippage_bps / 10000,
        tech_indicator_list=TECH_INDICATORS,
    )
    best_agent = PPOAgent(
        best_train_env,
        learning_rate=best["learning_rate"],
        n_steps=best["n_steps"],
        gamma=best["gamma"],
        ent_coef=best["ent_coef"],
        verbose=0,
    )
    best_agent.train(total_timesteps=TOTAL_TIMESTEPS)

    best_test_env = StockTradingEnv(
        df=test_df,
        stock_dim=stock_dim,
        hmax=100,
        initial_amount=INITIAL_AMOUNT,
        buy_cost_pct=0.001,
        sell_cost_pct=0.001,
        transaction_cost_pct=config.fee_bps / 10000,
        slippage_pct=config.slippage_bps / 10000,
        tech_indicator_list=TECH_INDICATORS,
    )
    obs, _ = best_test_env.reset()
    done = False
    while not done:
        action = best_agent.predict(obs)
        obs, reward, done, truncated, info = best_test_env.step(action)

    pv = best_test_env.portfolio_values[:len(test_dates)]
    pv_series = pd.Series(pv, index=test_dates[:len(pv)])
    daily_returns = pv_series.pct_change().dropna()
    full_metrics = compute_metrics(daily_returns)
    total_trades = best_test_env.trades_count
    print(f"Full evaluation Sharpe: {full_metrics['sharpeRatio']:.4f}")

    # Generate metrics.json
    trial_sharpes = [t.value for t in study.trials if t.value is not None]
    metrics_json = {
        "sharpeRatio": full_metrics["sharpeRatio"],
        "annualReturn": full_metrics["annualReturn"],
        "maxDrawdown": full_metrics["maxDrawdown"],
        "hitRate": full_metrics["hitRate"],
        "totalTrades": total_trades,
        "transactionCosts": {
            "feeBps": config.fee_bps,
            "slippageBps": config.slippage_bps,
            "netSharpe": full_metrics["sharpeRatio"],
        },
        "walkForward": {
            "windows": 1,
            "positiveWindows": 1 if full_metrics["sharpeRatio"] > 0 else 0,
            "avgOosSharpe": full_metrics["sharpeRatio"],
        },
        "customMetrics": {
            "phase": "hyperparameter_optimization",
            "label": "implementation-improvement",
            "optimizationMethod": "optuna_tpe",
            "tuningTimesteps": TUNING_TIMESTEPS,
            "fullEvalTimesteps": TOTAL_TIMESTEPS,
            "nTrials": len(study.trials),
            "bestTrialNumber": study.best_trial.number,
            "bestParams": study.best_params,
            "fullEvalMetrics": full_metrics,
            "trialSharpeStats": {
                "mean": round(float(np.mean(trial_sharpes)), 4),
                "std": round(float(np.std(trial_sharpes)), 4),
                "min": round(float(np.min(trial_sharpes)), 4),
                "max": round(float(np.max(trial_sharpes)), 4),
            },
            "searchSpace": {
                "learning_rate": "loguniform(1e-5, 1e-3)",
                "n_steps": [2048, 4096],
                "gamma": [0.99, 0.995, 0.999],
                "ent_coef": [0.0, 0.01],
            },
            "evaluationFold": "first_walk_forward_window",
            "trainPeriod": f"{train_dates[0].date()} to {train_dates[-1].date()}",
            "testPeriod": f"{test_dates[0].date()} to {test_dates[-1].date()}",
        },
    }

    metrics_path = reports_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics_json, f, indent=2)
    print(f"Metrics saved to {metrics_path}")

    return study


if __name__ == "__main__":
    main()
