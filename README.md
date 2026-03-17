# Optopus

Optopus はメタヒューリスティクスのライブラリです。MaxCut・TSP・SAT・QUBO などの組合せ最適化問題に対して、局所探索・タブー探索・焼きなまし法などのアルゴリズムを簡単に適用できます。

## 基本的な使い方

```rust
use optopus::prelude::*;

// 1. 問題インスタンスを作成（またはファイルから読み込み）
let mut mc = MaxCut::new();
mc.add_weight(0, 1, 1.0);
mc.add_weight(0, 2, 1.0);
mc.add_weight(1, 2, 1.0);
// または: let mc = MaxCut::load_from_file("data/g1.txt")?;

// 2. 探索状態を初期化
let mut state = SearchState::new(&mc);

// 3. ヒューリスティックを設定して実行
let ls = LocalSearch::<MaxCutFlipNeighbor>::new(
    StopCondition::iterations(1_000_000)
);
ls.run(&mut state).unwrap();

// 4. 結果を取得
println!("best objective = {}", state.best_solution.objective);
```

## 終了条件

`StopCondition` はビルダーメソッドで簡単に設定できます:

```rust
use optopus::heuristic::StopCondition;
use std::time::Duration;

// 反復回数で止める
StopCondition::iterations(1_000_000)

// 実行時間で止める
StopCondition::duration(Duration::from_secs(30))

// 複数条件を組み合わせる
StopCondition::iterations(1_000_000)
    .with_duration(Duration::from_secs(30))
```

## 利用可能なアルゴリズム

| アルゴリズム | 型 |
|---|---|
| 局所探索 | `LocalSearch<N>` |
| タブー探索 | `TabuSearch<N>` |
| 焼きなまし法 | `SimulatedAnnealing<N>` |
| Bang-Bang SA | `BangBangSimulatedAnnealing<N>` |
| ランダムウォーク | `RandomWalk<N>` |
| 逐次合成 | `Sequential<P>` |
| BLS (MaxCut専用) | `BreakoutLocalSearchForMaxCut` |

## 対応問題

| 問題 | 型 | 近傍 |
|---|---|---|
| Max Cut | `MaxCut` | `MaxCutFlipNeighbor`, `MaxCutSwapNeighbor` |
| SAT | `Sat` | `SatFlipNeighbor`, `SatSwapNeighbor` |
| TSP | `TspWithCoordinates` | `TspTwoOptNeighbor`, `TspRelocateNeighbor` |
| QUBO | `Qubo` | `QuboFlipNeighbour`, `QuboSwapNeighbour` |
| Formula | `FormulaProblem` | `FormulaFlipNeighbor`, `FormulaSwapNeighbor` |

## サンプル

```bash
# Max Cut: LocalSearch と TabuSearch を比較
cargo run --example max_cut

# 独自問題の定義方法
cargo run --example custom_problem
```

## 独自問題の定義

`ProblemTrait`・`MoveToNeigbor`・`Rankable` を実装することで、任意の問題に既存のアルゴリズムを適用できます。詳細は [examples/custom_problem.rs](examples/custom_problem.rs) を参照してください。

## エラーハンドリング

ヒューリスティックの `run` / `run_once` は `Result<(), OptError>` を返します:

```rust
use optopus::error::OptError;

match ls.run(&mut state) {
    Ok(()) => println!("success"),
    Err(OptError::InvalidState(msg)) => eprintln!("invalid state: {}", msg),
    Err(e) => eprintln!("error: {}", e),
}
```
