# FinRL: A Deep Reinforcement Learning Library for Automated Stock Trading in Quantitative Finance

## Project ID
proj_592c901a

## Taxonomy
ReinforcementLearning

## Current Cycle
2

## Objective
Implement, validate, and iteratively improve the paper's approach with production-quality standards.


## Design Brief
### Problem
Applying deep reinforcement learning (DRL) to quantitative finance is challenging due to the complexities of financial markets, such as low signal-to-noise ratios and non-stationarity. Furthermore, there is a lack of standardized frameworks that bridge the gap between DRL research and practical financial applications. This makes it difficult for researchers and practitioners to develop, test, and deploy DRL-based trading strategies in a robust and reproducible manner.

This paper introduces FinRL, an open-source Python library designed to address these challenges. It provides a complete, end-to-end framework that includes simulated stock trading environments compatible with OpenAI Gym, implementations of various state-of-the-art DRL algorithms, and comprehensive backtesting modules. The goal of FinRL is to lower the barrier to entry for using DRL in finance, enabling users to streamline the development, training, and evaluation of automated trading strategies.

### Datasets
{"name":"Dow Jones Industrial Average (DJIA) Components","source":"yfinance API","method":"yfinance.download(tickers, start='2009-01-01', end='2021-01-01')","tickers":["AXP","AMGN","AAPL","BA","CAT","CSCO","CVX","GS","HD","HON","IBM","INTC","JNJ","KO","JPM","MCD","MMM","MRK","MSFT","NKE","PG","TRV","UNH","CRM","VZ","V","WBA","WMT","DIS","DOW"]}

### Targets
Maximize the risk-adjusted return of a trading portfolio. The primary optimization objective for the reinforcement learning agent is the reward function, which is defined as the change in portfolio value at each step. The ultimate evaluation metric is the Sharpe Ratio on out-of-sample data.

### Model
FinRL is not a single model but a framework. Its core components are:
1.  **Stock Trading Environment**: An OpenAI Gym-compatible environment that simulates the stock market. The state space includes market information (prices, technical indicators) and portfolio information (stock holdings, cash balance). The action space represents the trading decision (buy/sell/hold). The reward is the change in portfolio value.
2.  **DRL Agents**: The framework integrates with the `stable-baselines3` library to provide access to common DRL algorithms such as A2C, PPO, and DDPG. These agents learn a policy that maps states to actions to maximize the cumulative reward.
3.  **Data Processor**: A module for fetching data from APIs like `yfinance` and preprocessing it by adding technical indicators.

### Training
The paper uses a chronological train-validation-test split. For example, data from 2009-01-01 to 2018-12-31 is used for training, 2019-01-01 to 2020-12-31 for validation (e.g., hyperparameter tuning), and 2021-01-01 onwards for out-of-sample testing. The paper also advocates for a walk-forward validation approach for more robust evaluation, where the model is periodically retrained as new data becomes available.

### Evaluation
The primary evaluation is conducted through backtesting on an out-of-sample test set. Key performance metrics include:
- Cumulative Return
- Annualized Return
- Annualized Volatility
- Sharpe Ratio
- Max Drawdown
- Calmar Ratio
The performance of the DRL agent is compared against baseline strategies, primarily a market-neutral benchmark (e.g., holding the DJIA index) and a simple buy-and-hold strategy for the traded assets.


## データ取得方法（共通データ基盤）

**合成データの自作は禁止。以下のARF Data APIからデータを取得すること。**

### ARF Data API
```bash
# OHLCV取得 (CSV形式)
curl -o data/aapl_1d.csv "https://ai.1s.xyz/api/data/ohlcv?ticker=AAPL&interval=1d&period=5y"
curl -o data/btc_1h.csv "https://ai.1s.xyz/api/data/ohlcv?ticker=BTC/USDT&interval=1h&period=1y"
curl -o data/nikkei_1d.csv "https://ai.1s.xyz/api/data/ohlcv?ticker=^N225&interval=1d&period=10y"

# JSON形式
curl "https://ai.1s.xyz/api/data/ohlcv?ticker=AAPL&interval=1d&period=5y&format=json"

# 利用可能なティッカー一覧
curl "https://ai.1s.xyz/api/data/tickers"
```

### Pythonからの利用
```python
import pandas as pd
API = "https://ai.1s.xyz/api/data/ohlcv"
df = pd.read_csv(f"{API}?ticker=AAPL&interval=1d&period=5y")
df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.set_index("timestamp")
```

### ルール
- **リポジトリにデータファイルをcommitしない** (.gitignoreに追加)
- 初回取得はAPI経由、以後はローカルキャッシュを使う
- data/ディレクトリは.gitignoreに含めること



## ★ 今回のタスク (Cycle 2)


### Phase 2: 実データパイプラインの構築

**ゴール**: yfinanceから実際の株価データを取得し、前処理して保存するパイプラインを構築する。

**具体的な作業指示**:
1. `src/data/processor.py`に`DataProcessor`クラスを作成します。
2. `DataProcessor`に`download_data`メソッドを追加します。このメソッドはティッカーリスト、開始日、終了日を引数に取り、`yfinance.download`を使用してデータを取得し、`data/raw/{ticker}.csv`として保存します。ティッカー`['AAPL']`、期間`2009-01-01`から`2021-12-31`でテストしてください。
3. `DataProcessor`に`add_technical_indicators`メソッドを追加します。このメソッドは価格データフレームを受け取り、`ta`ライブラリを使用してMACD、RSI (14日)、CCI (14日)、ADX (14日)を計算し、列として追加します。
4. `src/preprocess.py`スクリプトを作成します。このスクリプトは`DataProcessor`を使い、`AAPL`のデータをダウンロードし、テクニカル指標を追加して、`data/processed/AAPL_processed.csv`に保存します。NaN値は前方フィル(`ffill`)で処理してください。

**期待される出力ファイル**:
- src/data/processor.py
- src/preprocess.py
- data/processed/AAPL_processed.csv

**受入基準 (これを全て満たすまで完了としない)**:
- 基準1: `python src/preprocess.py`が成功し、`data/processed/AAPL_processed.csv`が生成される。
- 基準2: 生成されたCSVファイルに`macd`, `rsi`, `cci`, `adx`の列が含まれている。










## 全体Phase計画 (参考)

✓ Phase 1: コア環境とエージェントの実装 — 合成データ上で動作する、単一銘柄の取引環境と基本的なA2Cエージェントを実装する。
→ Phase 2: 実データパイプラインの構築 — yfinanceから実際の株価データを取得し、前処理して保存するパイプラインを構築する。
  Phase 3: バックテストと評価フレームワークの実装 — 学習済みエージェントのパフォーマンスを評価するためのバックテストエンジンと評価指標計算を実装する。
  Phase 4: 取引コストの導入 — 取引環境とバックテストに手数料とスリッページを組み込み、コストの影響を評価する。
  Phase 5: 複数アルゴリズムの比較 — 論文で言及されている主要なDRLアルゴリズム（A2C, PPO, DDPG）を実装し、性能を比較する。
  Phase 6: 複数銘柄ポートフォリオ環境への拡張 — 複数の株式を同時に取引できるポートフォリオ管理環境を実装する。
  Phase 7: ハイパーパラメータ最適化 — Optunaを用いてPPOエージェントの主要なハイパーパラメータを最適化する。
  Phase 8: ウォークフォワード検証の実装 — より現実的な評価のために、ウォークフォワード検証フレームワークを実装し実行する。
  Phase 9: 総合レポートと可視化 — 全フェーズの結果をまとめた総合的なテクニカルレポートを生成する。
  Phase 10: エグゼクティブサマリーと仕上げ — 非技術者向けの要約を作成し、コードベースの品質を向上させる。


## 評価原則
- **主指標**: Sharpe ratio (net of costs) on out-of-sample data
- **Walk-forward必須**: 単一のtrain/test splitでの最終評価は不可
- **コスト必須**: 全メトリクスは取引コスト込みであること
- **安定性**: Walk-forward窓の正の割合を報告
- **ベースライン必須**: 必ずナイーブ戦略と比較

## 再現モードのルール（論文忠実度の維持）

このプロジェクトは**論文再現**が目的。パフォーマンス改善より論文忠実度を優先すること。

### パラメータ探索の制約
- **論文で既定されたパラメータをまず実装し、そのまま評価すること**
- パラメータ最適化を行う場合、**論文既定パラメータの近傍のみ**を探索（例: 論文が12ヶ月なら [6, 9, 12, 15, 18] ヶ月）
- 論文と大きく異なるパラメータ（例: 月次論文に対して日次10営業日）で良い結果が出ても、それは「論文再現」ではなく「独自探索」
- 独自探索で得た結果は `customMetrics` に `label: "implementation-improvement"` として記録し、論文再現結果と明確に分離

### データ条件の忠実度
- 論文のデータ頻度（日次/月次/tick）にできるだけ合わせる
- ユニバース規模が論文より大幅に小さい場合、その制約を `docs/open_questions.md` に明記
- リバランス頻度・加重方法も論文に合わせる



## 禁止事項
- 未来情報を特徴量やシグナルに使わない
- 全サンプル統計でスケーリングしない (train-onlyで)
- テストセットでハイパーパラメータを調整しない
- コストなしのgross PnLだけで判断しない
- 時系列データにランダムなtrain/test splitを使わない
- APIキーやクレデンシャルをコミットしない
- **新しい `scripts/run_cycle_N.py` や `scripts/experiment_cycleN.py` を作成しない。既存の `src/` 内ファイルを修正・拡張すること**
- **合成データを自作しない。必ずARF Data APIからデータを取得すること**
- **「★ 今回のタスク」以外のPhaseの作業をしない。1サイクル=1Phase**
- **論文が既定するパラメータから大幅に逸脱した探索を「再現」として報告しない**

## Git / ファイル管理ルール
- **データファイル(.csv, .parquet, .h5, .pkl, .npy)は絶対にgit addしない**
- `__pycache__/`, `.pytest_cache/`, `*.pyc` がリポジトリに入っていたら `git rm --cached` で削除
- `git add -A` や `git add .` は使わない。追加するファイルを明示的に指定する
- `.gitignore` を変更しない（スキャフォールドで設定済み）
- データは `data/` ディレクトリに置く（.gitignore済み）
- 学習済みモデルは `models/` ディレクトリに置く（.gitignore済み）

## 出力ファイル
以下のファイルを保存してから完了すること:
- `reports/cycle_2/metrics.json` — 下記スキーマに従う（必須）
- `reports/cycle_2/technical_findings.md` — 実装内容、結果、観察事項

### metrics.json 必須スキーマ
```json
{
  "sharpeRatio": 0.0,
  "annualReturn": 0.0,
  "maxDrawdown": 0.0,
  "hitRate": 0.0,
  "totalTrades": 0,
  "transactionCosts": { "feeBps": 10, "slippageBps": 5, "netSharpe": 0.0 },
  "walkForward": { "windows": 0, "positiveWindows": 0, "avgOosSharpe": 0.0 },
  "customMetrics": {}
}
```
- 全フィールドを埋めること。Phase 1-2で未実装のメトリクスは0.0/0で可。
- `customMetrics`に論文固有の追加メトリクスを自由に追加してよい。
- `docs/open_questions.md` — 未解決の疑問と仮定
- `README.md` — 今回のサイクルで変わった内容を反映して更新（セットアップ手順、主要な結果、使い方など）
- `docs/open_questions.md` に以下も記録:
  - ARF Data APIで問題が発生した場合（エラー、データ不足、期間の短さ等）
  - CLAUDE.mdの指示で不明確な点や矛盾がある場合
  - 環境やツールの制約で作業が完了できなかった場合

## 標準バックテストフレームワーク

`src/backtest.py` に以下が提供済み。ゼロから書かず、これを活用すること:
- `WalkForwardValidator` — Walk-forward OOS検証のtrain/test split生成
- `calculate_costs()` — ポジション変更に基づく取引コスト計算
- `compute_metrics()` — Sharpe, 年率リターン, MaxDD, Hit rate算出
- `generate_metrics_json()` — ARF標準のmetrics.json生成

```python
from src.backtest import WalkForwardValidator, BacktestConfig, calculate_costs, compute_metrics, generate_metrics_json
```

## Key Commands
```bash
pip install -e ".[dev]"
pytest tests/
python -m src.cli run-experiment --config configs/default.yaml
```

Commit all changes with descriptive messages.
