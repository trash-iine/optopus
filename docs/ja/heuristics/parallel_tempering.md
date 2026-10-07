# ParallelTempering

**API:** [`ParallelTempering`](../../api/optopus/heuristic/struct.ParallelTempering.html)

Parallel tempering (レプリカ交換モンテカルロ法) は、`num_replicas` 個の探索のコピーを固定した温度で走らせ、隣り合う温度のコピー同士に場所を交換させます。
高温のレプリカが良い谷で見つけた配置ははしごを下って低温で磨かれ、低温の端で行き詰まった配置ははしごを上って抜け出せる温度まで行きます。

解が [`Evaluate`](../traits.md) を実装している任意の問題で動き、Metropolis の提案を移動型 `N` として与えます。[PopulationAnnealing](population_annealing.md) と同じ要件です。

## 例 { #example }

```rust
use optopus::heuristic::ParallelTempering;
use optopus::prelude::*;

let mc = MaxCut::new(Graph::erdos_renyi(800, 0.02, &mut seeded_rng(42)));
let mut state = SearchState::new_with_seed(&mc, 42);

let mut pt = ParallelTempering::<MaxCut, MaxCutFlipNeighbor>::new(
    StopCondition::duration(std::time::Duration::from_secs(10)),
    /* num_replicas        = */ 20,
    /* beta_min            = */ 0.1,
    /* beta_max            = */ 5.0,
    /* sweeps_per_exchange = */ 1,
);
pt.run(&mut state)?;
println!("cut weight = {}", state.best_solution.objective);
println!("exchanges accepted = {:?}", pt.exchange_acceptance());
```

`ParallelTempering` は prelude に入っていないので、`optopus::heuristic` から import します。

## アルゴリズムの概略 { #algorithm-sketch }

逆温度は `beta_min` から `beta_max` までの等比数列で、各レプリカはランダムな解から始まります。`run_once` 1 回が 1 ラウンドです。

1. Metropolis sweep。各レプリカを自分の温度で `sweeps_per_exchange` 回 sweep します。提案された移動は確率 `min(1, exp(−β · ΔE))` で受理されます。population annealing と同じ sweep です。
2. 交換。温度 `k` と `k + 1` のレプリカを確率 `min(1, exp((β_k − β_{k+1})(E_k − E_{k+1})))` で入れ替えます。あるラウンドは偶数番目の組を、次のラウンドは奇数番目の組を試すので、どの組も 1 ラウンドおきに試され、一つのレプリカが同時に二つの交換に入ることはありません。

冷却はしません。はしごは固定で、これが population annealing との違いです。population annealing のレプリカは上がっていく一つの温度を共有し、良いものに向けて再標本化されます。

`state.iteration` は 1 ラウンドに `sweeps_per_exchange` 進み、現在解はそのラウンドの最良のレプリカです。乱数はすべて決まった順で `state.rng` から引くので、seed を固定した実行はビット単位で再現します。

## はしごの選び方 { #choosing-the-ladder }

温度は目的関数の単位を持ちます。`beta_max` は最も低温のレプリカが降下法になる程度に低温に、`beta_min` は最も高温のレプリカが自由に動ける程度に高温にします。
その間は、隣同士がエネルギーで重なる程度に密でなければなりません。そうでないと交換がほとんど受理されず、配置ははしごを渡れません。
`exchange_acceptance()` はそれまでに受理された交換の割合を返し、0 に近ければレプリカを増やすか範囲を狭めるべきだということです。

## コンストラクタ { #constructor }

```rust
ParallelTempering::<P, N>::new(
    stop_condition: StopCondition,
    num_replicas: usize,         // 2 以上
    beta_min: f64,               // 最高温、正
    beta_max: f64,               // 最低温、beta_min より大きい
    sweeps_per_exchange: usize,  // 1 以上
) -> Self
```

`with_sweep_length(n)` は、近傍を実行ごとに一度数える代わりに 1 sweep あたりの提案数を固定します。population annealing と同じく、近傍が二次の大きさになるペアの移動で効きます。
`betas()` ははしごを返します。

## ベンチマーク設定 { #benchmark-config }

```toml
[[heuristics]]
kind = "ParallelTempering"
neighbor = "Flip"
num_replicas = 20
beta_min = 0.1              # 省略可 (既定値を表示)
beta_max = 5.0              # 省略可 (既定値を表示)
sweeps_per_exchange = 1     # 省略可 (既定値を表示)
# sweep_length = 800        # 省略可、既定は近傍の大きさ
[heuristics.stop_condition]
max_duration_secs = 30.0
```

範囲は設定を読んだ時点で検査されるので、不正なはしごは実行が始まる前に失敗します。

## 参考文献 { #references }

- Hukushima, K. and Nemoto, K. "Exchange Monte Carlo method and application to
  spin glass simulations." *Journal of the Physical Society of Japan*, 65(6),
  1604-1608, 1996.
- Wang, W., Machta, J. and Katzgraber, H. G. "Comparing Monte Carlo methods for
  finding ground states of Ising spin glasses. Population annealing, simulated
  annealing, and parallel tempering." *Physical Review E*, 92, 013303, 2015
  (arXiv:1412.2104).
