# SimulatedAnnealing

**API:** [`SimulatedAnnealing`](../../api/optopus/heuristic/struct.SimulatedAnnealing.html)

一様ランダムに近傍を一つ選び、Boltzmann 確率 `exp(−worsening / T)` で受理します。改善する move は常に受理されます。
温度は各ステップの後に `cooling_rate` 倍されます。

## 例 { #example }

```rust
use optopus::prelude::*;

let mc = MaxCut::new(Graph::from_edges([(0, 1, 1.0), (0, 2, 1.0), (1, 2, 1.0)]));
let mut state = SearchState::new(&mc);
let mut sa = SimulatedAnnealing::<MaxCutFlipNeighbor>::new(
    StopCondition::iterations(100_000),
    /* initial_temperature = */ 1.0,
    /* cooling_rate        = */ 0.9999,
);
sa.run(&mut state)?;
println!("cut weight = {}", state.best_solution.objective);
```

## コンストラクタ { #constructor }

```rust
SimulatedAnnealing::<N>::new(
    stop_condition: StopCondition,
    initial_temperature: f64,
    cooling_rate: f64,
) -> Self
```

`N` は `MoveToNeighbor<P> + Evaluate` (つまり `Evaluate<f64>`) を満たす必要があります。
悪化量は `Evaluable::minimized()` から読むので、もとの目的関数の向きは自動的に扱われます。

`clear()` は現在の温度を `initial_temperature` に戻します。

## 受理規則 { #acceptance-rule }

共通のヘルパー `boltzmann_accept(delta: Evaluable<f64>, T: f64)` は、`delta` がスコアを改善するなら `true` を返し、
そうでなければ一様乱数を引いて `exp(−worsening / T)` と比べます。

## ベンチマーク設定 { #benchmark-config }

```toml
[[heuristics]]
kind = "SimulatedAnnealing"
neighbor = "Flip"             # 必須。使える値は問題ごとに決まる
initial_temperature = 1.0     # 必須
cooling_rate = 0.9999         # 必須。各ステップの後に T に掛けられる
[heuristics.stop_condition]
max_iteration = 100_000
```

## BangBangSimulatedAnnealing

[`BangBangSimulatedAnnealing`](../../api/optopus/heuristic/struct.BangBangSimulatedAnnealing.html)
は温度スケジュールが振動する変種です。

```rust
BangBangSimulatedAnnealing::<N>::new(
    stop_condition: StopCondition,
    initial_temperature: f64,
    cooling_rate: f64,
    min_wave_threshold: f64,
    max_wave_threshold: f64,
)
```

温度は `min_wave_threshold` を下回るまで乗算的に下がり、次に `cooling_rate` で割ることで `max_wave_threshold`
を超えるまで上がり、これを繰り返します。のこぎり波の形によって、探索が貪欲になりすぎたときにときどき探索性を注入し直します。

`BangBangSimulatedAnnealing` には専用の `kind` はありません。ベンチマークランナーが作るのは通常のスケジュールだけで、
振動するほうは Rust API からだけ使えます。

## 参考文献 { #references }

- Kirkpatrick, S., Gelatt, C. D., and Vecchi, M. P. "Optimization by Simulated
  Annealing." Science, 220(4598), 671-680, 1983.
- Cerny, V. "Thermodynamical Approach to the Traveling Salesman Problem: An
  Efficient Simulation Algorithm." *Journal of Optimization Theory and
  Applications*, 45(1), 41-51, 1985.
