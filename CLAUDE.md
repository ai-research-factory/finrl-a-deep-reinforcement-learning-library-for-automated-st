# FinRL: A Deep Reinforcement Learning Library for Automated Stock Trading in Quantitative Finance

## Project ID
proj_592c901a

## Taxonomy
ReinforcementLearning

## Current Cycle
7

## Objective
Implement, validate, and iteratively improve the paper's approach with production-quality standards.


## Design Brief
### Problem
Deep Reinforcement Learning (DRL) presents a promising avenue for developing adaptive, automated trading strategies. However, its application in quantitative finance is challenging due to the need for specialized, realistic trading environments, robust backtesting frameworks, and accessible DRL algorithm implementations. Existing tools often lack a comprehensive, end-to-end structure, forcing practitioners to build significant infrastructure from scratch. This paper introduces FinRL, an open-source Python library designed to bridge this gap. FinRL provides a full-stack framework that includes modules for data fetching and processing, a simulated stock trading environment compatible with standard RL libraries, a suite of pre-implemented DRL agents, and a rigorous backtesting engine. The goal is to democratize the use of DRL in finance, enabling faster development, evaluation, and deployment of automated trading strategies.

### Datasets
{"name":"Dow Jones 30 Constituents (DOW)","source":"yfinance API","details":"Daily OHLCV data for the 30 stocks in the Dow Jones Industrial Average. The paper uses the period from 2009-01-01 to 2020-09-30. Tickers can be sourced from a standard list of DOW 30 components for that era."}

### Targets
The primary objective is to train a DRL agent to learn a trading policy that maximizes the risk-adjusted return, as measured by the Sharpe Ratio, on a portfolio of stocks.

### Model
FinRL employs a three-layer architecture. The first layer is a Data Processor for fetching financial data and engineering features (e.g., technical indicators). The second layer is a custom trading environment, modeled after OpenAI Gym, which simulates the market. The state space in this environment includes stock prices, technical indicators, and current portfolio holdings. The action space represents the number of shares to buy or sell for each stock. The reward function is defined as the change in portfolio value at each step. The third layer consists of DRL agents (e.g., PPO, A2C, DDPG) from libraries like Stable Baselines3, which learn a policy by interacting with the environment.

### Training
The historical data is split chronologically into a training set and a testing set. The paper uses 2009-01-01 to 2018-12-31 for training and 2019-01-01 to 2020-09-30 for out-of-sample testing. The DRL agent is trained on the training data for a specified number of timesteps. For more robust evaluation, a walk-forward validation approach is recommended, where the model is periodically retrained on newer data and tested on the subsequent period.

### Evaluation
The trained agent's performance is evaluated on the out-of-sample test set. Key performance metrics include Annualized Return, Annualized Volatility, Sharpe Ratio, and Maximum Drawdown. The agent's performance is benchmarked against traditional strategies like a passive Buy-and-Hold strategy on the same portfolio and the performance of the Dow Jones Industrial Average (DJIA) index.


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



## ★ 今回のタスク (Cycle 7)


### Phase 7: ハイパーパラメータ最適化

**ゴール**: Optunaを用いてPPOエージェントの主要なハイパーパラメータを体系的に探索し、最適な組み合わせを見つける。

**具体的な作業指示**:
1. `scripts/tune_hyperparameters.py`を新規作成します。
2. `Optuna`ライブラリを使い、最適化スタディを作成します。目的関数は、ウォークフォワードの最初のfold（学習データで学習し、検証データで評価）のシャープレシオを最大化することとします。
3. 探索するハイパーパラメータは、`learning_rate` (1e-5 to 1e-3)、`n_steps` (2048, 4096)、`gamma` (0.99, 0.995, 0.999)、`ent_coef` (0.0, 0.01)とします。
4. 最適化を100試行実行し、見つかった最適なパラメータを`reports/cycle_7/best_params.json`に保存します。Optunaのスタディ結果も`reports/cycle_7/study.db`に保存します。

**期待される出力ファイル**:
- scripts/tune_hyperparameters.py
- reports/cycle_7/best_params.json
- reports/cycle_7/study.db

**受入基準 (これを全て満たすまで完了としない)**:
- `best_params.json`が生成され、最適化された`learning_rate`などの値が含まれている。
- `study.db`が生成されている。





## スコア推移
Cycle 2: 45% → Cycle 4: 50% → Cycle 6: 45%
改善速度: 0.0%/cycle ⚠ 停滞気味 — アプローチの転換を検討





## レビューからのフィードバック
### レビュー改善指示
1. 【最重要】Walk-forward検証をプロジェクトの唯一の公式な評価手法として確立すること。具体的には、(1) `src/run_single_backtest.py` を `src/run_walk_forward.py` にリネームする。 (2) このスクリプトをデフォルトの実行パスとし、Walk-forwardの各OOS期間の結果を集計したサマリー（`reports/walk_forward_summary.csv`）を生成する。 (3) 次のサイクルの `reports/cycle_N/metrics.json` は、このWalk-forward結果の**平均値と標準偏差**（例: `avgOosSharpe`, `stdOosSharpe`）を主要メトリクスとして報告するように `generate_metrics_json` 関数を修正し、`walkForward`セクション（`windows`, `positiveWindows`等）を正しく埋めること。
2. 【重要】ベースライン戦略を実装し、比較評価を行うこと。`src/backtest.py` に、期間開始時に均等配分で購入し最後まで保持する「バイアンドホールド戦略」と、定期的に均等ウェイトにリバランスする「均等配分戦略」のバックテストロジックを追加する。`src/run_walk_forward.py` の中で、これらのベースライン戦略も同じWalk-forwardの枠組みで評価し、結果をPPOエージェントと比較する表を `reports/cycle_N/technical_findings.md` に必ず含めること。
3. 【推奨】論文忠実度と独自改善を明確に分離すること。リワード関数の変更 (`risk_penalty_coef`) は論文からの逸脱である。まず、論文で提示されているリワード関数（ポートフォリオ価値の変化）に戻した状態でWalk-forward検証を実行し、論文再現のベースライン性能を確定させる。その上で、リスク調整後リワード関数を使った場合の結果を「改善実験」として別途報告し、その効果を比較分析するドキュメント（例: `docs/reward_function_ablation.md`）を作成すること。
### マネージャー指示 (次のアクション)
1. 【最優先】Walk-forward検証の完全な実行と報告: `run_cost_analysis.py`の使用を中止し、`src/run_single_backtest.py`に実装済みのWalk-forward検証を公式な評価手法として採用する。`config/backtest_dow_30.yml`等の設定ファイルで`n_windows`を10以上に設定してバックテストを再実行し、`reports/cycle_7/metrics.json`の`walkForward`セクションに結果が正しく記録されることを必須とする。
2. 【重要】ベースライン戦略の導入と比較: `src/baselines.py`を新規作成し、「Buy and Hold」戦略と「均等配分ポートフォリオ（月次リバランス）」戦略を実装する。これらのベースライン戦略もPPOエージェントと同一のWalk-forward検証フレームワーク (`src/run_single_backtest.py`を使用) で評価し、Sharpeレシオ、年間リターン、最大ドローダウンをまとめた比較表を`reports/cycle_7/performance_comparison.md`として生成する。
3. 【推奨】Walk-forward結果の安定性分析: Walk-forwardの各検証ウィンドウにおけるSharpeレシオの分布（平均、標準偏差、最小値、最大値）を計算し、`reports/cycle_7/metrics.json`に追記する。さらに、この分布をヒストグラムとして`reports/cycle_7/sharpe_distribution.png`に出力し、戦略性能が特定の期間に過度に依存していないか（安定しているか）を評価する。


## 全体Phase計画 (参考)

✓ Phase 1: コア環境とエージェントの実装 — 合成データ上で動作する、基本的な株式取引環境とPPOエージェントを実装する。
✓ Phase 2: 実データパイプラインの構築 — yfinanceからDOW30の株価データを取得し、テクニカル指標を追加して前処理を行うパイプラインを構築する。
✓ Phase 3: 統合とシングルバックテスト — 実データ、環境、エージェントを統合し、単一の学習・テスト期間でバックテストを実行する。
✓ Phase 4: 評価指標とベースライン比較 — 標準的な財務評価指標を計算するモジュールを実装し、エージェントの性能をベースライン戦略と比較する。
✓ Phase 5: 取引コストモデルの導入 — 取引環境に手数料とスリッページをモデル化し、コストがパフォーマンスに与える影響を評価する。
✓ Phase 6: ウォークフォワード検証の実装 — 単一の学習・テスト分割ではなく、より頑健なウォークフォワード検証フレームワークを実装する。
→ Phase 7: ハイパーパラメータ最適化 — Optunaを用いてPPOエージェントの主要なハイパーパラメータを体系的に探索し、最適な組み合わせを見つける。
  Phase 8: 最適化パラメータでの再評価 — 見つかった最適なハイパーパラメータを使用して、完全なウォークフォワード検証を再実行し、パフォーマンスの向上を確認する。
  Phase 9: 代替DRLアルゴリズムとの比較 — 別のDRLアルゴリズム（A2C）を実装し、PPOとの性能をウォークフォワード検証で比較する。
  Phase 10: 状態空間の拡張と影響分析 — マクロ経済指標を状態空間に追加し、それがエージェントのパフォーマンスに与える影響を分析する。
  Phase 11: 最終レポートと可視化 — 全フェーズの結果を統合し、包括的なテクニカルレポートと主要な可視化を生成する。
  Phase 12: コードの仕上げとエグゼクティブサマリー — プロジェクトのコード品質を向上させ、非技術者向けの要約を作成する。


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
- `reports/cycle_7/metrics.json` — 下記スキーマに従う（必須）
- `reports/cycle_7/technical_findings.md` — 実装内容、結果、観察事項

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
