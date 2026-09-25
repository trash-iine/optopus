# PopulationAnnealing

**API:** [`PopulationAnnealing`](../../api/optopus/heuristic/struct.PopulationAnnealing.html)

Population Annealing Monte Carlo (PAMC) は `population_size` 個のレプリカからなる集団を持ち、共有の逆温度 `β` を上げながら冷却し、
温度のステップごとに集団をリサンプリングします。

解が [`Evaluate`](../traits.md) を実装している任意の問題で動き、Metropolis の提案は move 型 `N` として与えます。

## 例 { #example }

```rust
use optopus::heuristic::PopulationAnnealing;
use optopus::prelude::*;

let mut rng = seeded_rng(42);
let mc = MaxCut::new(Graph::erdos_renyi(800, 0.02, &mut rng));
let mut state = SearchState::new_with_seed(&mc, 42);

let mut pa = PopulationAnnealing::<MaxCut, MaxCutFlipNeighbor>::new(
    StopCondition::iterations(100_000),
    /* population_size = */ 50,
    /* initial_beta    = */ 0.1,
    /* delta_beta      = */ 0.02,
    /* sweeps_per_step = */ 50,
    /* reset_period    = */ Some(400),
);
pa.run(&mut state)?;
println!("cut weight = {}", state.best_solution.objective);
```

`PopulationAnnealing` は prelude に入っていないので、`optopus::heuristic` からインポートしてください。

## アルゴリズムの概要 { #algorithm-sketch }

集団は `population_size` 個のランダムな解で初期化されます。各 `run_once` が一つのアニーリングステップで、手法の定義どおりの順序で進みます。

1. リサンプリング。レプリカ `j` は期待値 `τ_j = exp(−Δβ (E_j − E_min)) / Z · R` 個の複製を得ます。
   `E_j` は `Evaluate` から読み、数値的な安定性のために `E_min` だけずらします。エネルギーの低いレプリカが優先的に複製され、
   集団はちょうど `population_size` 個に戻ります。
2. `β` を `delta_beta` だけ進めます。ただし `reset_period` ステップごとに `initial_beta` に戻し、
   集団が収束した後に多様性を取り戻します。全体の最良解はリセットをまたいで残ります。
3. Metropolis スイープ。新しい `β` で各レプリカを `sweeps_per_step` 回スイープします。提案された move は、
   [SA](simulated_annealing.md) が使うのと同じ `boltzmann_accept` ヘルパーを通して確率 `min(1, exp(-β · ΔE))` で受理されます。

最初のリサンプリングの前にスイープはありません。手法はランダムな集団から始まり、それは高温ではスイープが作るはずの分布にすでになっています。

`state.iteration` はステップごとに `sweeps_per_step` だけ進むので、最良解到達時間や anytime trajectory がほかのヒューリスティクスと比べて意味を持ちます。
乱数はすべて決まった順序で `state.rng` を通るので、シード付きの実行はビット単位で再現可能です。

### スイープの長さ { #sweep-length }

スイープは系全体を一回なめることなので、既定では `N` の近傍にある move の数だけ提案します。この数はエピソードごとに一度数えます。
単一変数の move なら変数の数で、コストは一度だけの O(n) です。2-opt のような対の move では近傍は O(n²) で、
数えるためだけにすべての move を作っては捨てることになるので、長さを固定してください。

```rust
let pa = PopulationAnnealing::<Tsp, TspTwoOptNeighbor>::new(
    StopCondition::duration(std::time::Duration::from_secs(30)),
    50, 0.1, 0.02, 50, Some(400),
).with_sweep_length(1_000);
```

## コンストラクタ { #constructor }

```rust
PopulationAnnealing::<P, N>::new(
    stop_condition: StopCondition,
    population_size: usize,        // R、レプリカ数
    initial_beta: f64,             // 開始時の逆温度
    delta_beta: f64,               // ステップごとの増分
    sweeps_per_step: usize,        // ステップごと、レプリカごとの Metropolis スイープ回数
    reset_period: Option<usize>,   // None = リセットしない
) -> Self
```

`population_size < 2`、`initial_beta <= 0`、`delta_beta <= 0`、`sweeps_per_step == 0` のいずれかなら panic します。
`with_sweep_length` は `0` で panic します。

`clear()` は集団を捨て、`β` とステップカウンタをリセットします。そのため新しいエピソードは `initial_beta` からアニーリングをやり直します。

## ベンチマーク設定 { #benchmark-config }

```toml
[[heuristics]]
kind = "PopulationAnnealing"
neighbor = "Flip"
population_size = 50
initial_beta = 0.1        # 任意 (値は既定値)
delta_beta = 0.02         # 任意 (値は既定値)
sweeps_per_step = 50      # 任意 (値は既定値)
reset_period = 400        # 任意 (値は既定値。0 でリセットを無効化)
sweep_length = 2000       # 任意 (既定は近傍の大きさ)
[heuristics.stop_condition]
max_duration_secs = 30.0
```

## 参考文献 { #references }

- Wang, Machta, Katzgraber. "Population annealing: Theory and application in
  spin glasses." Phys. Rev. E 92, 063307, 2015
  ([arXiv:1508.05647](https://arxiv.org/abs/1508.05647))。ここで実装しているアルゴリズムです。
- Machta, J. "Population annealing with weighted averages: A Monte Carlo method
  for rough free-energy landscapes." Phys. Rev. E 82, 026704, 2010.
