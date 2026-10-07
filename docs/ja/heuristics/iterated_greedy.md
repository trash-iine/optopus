# IteratedGreedy

**API:** [`IteratedGreedy`](../../api/optopus/heuristic/struct.IteratedGreedy.html)

Iterated Greedy は、現在解からランダムにいくつかの要素を取り出し、それぞれを最も安い位置に戻し、局所探索をかけ、一定の温度で受理を判定します。
Ruiz と Stützle が [Permutation Flow Shop](../problems/flow_shop.md) のために提案したもので、この短いループは今も知られている中で最も強い手法の一つです。

[`Ruinable`](../traits.md) を実装し、その解が [`Evaluate`](../traits.md) を実装している任意の問題で動きます。設定ファイルからはフローショップで使えます。

## 例 { #example }

```rust
use optopus::prelude::*;

let fs = FlowShop::load_file("data/instances/flow_shop/taillard/ta051.txt")?;
let mut state = SearchState::new_with_seed(&fs, 42);

iterated_greedy_for_flow_shop(
    StopCondition::duration(std::time::Duration::from_secs(10)),
    /* removal_count      = */ 4,
    /* temperature_factor = */ 0.4,
)
.run(&mut state)?;
println!("makespan = {}", state.best_solution.objective);
```

`iterated_greedy_for_flow_shop` はコンストラクタに、フローショップが渡す二つのもの、局所探索としての全体挿入降下と論文の温度を加えたものです。
同じ探索を書き下し、別の問題で使うと次のようになります。

```rust
use optopus::prelude::*;
use optopus::problem::tsp::AnchoredTourDescent;

let mut ig = IteratedGreedy::<Tsp>::new(
    StopCondition::iterations(10_000),
    /* removal_count  = */ 3,
    /* temperature_of = */ |_: &Tsp| 1.0,
)
.with_local_repair(Box::new(AnchoredTourDescent::new()));
```

## アルゴリズムの概略 { #algorithm-sketch }

`run_once` は 1 回ごとに候補を一つ作ります。

1. 破壊。現在解から `removal_count` 個の要素を一様ランダムに取り出す。
2. 構築。それぞれをランダムな順で最も安い位置に戻す。
3. 降下。`LocalRepair` が与えられていれば実行する。
4. 受理。候補が現在解より悪くなければ受理し、そうでなければ確率 `exp(−Δ / T)` で受理する。`T` は実行の間ずっと同じです。

大域最良解は現在解とは別に保持されるので、悪い解を受理しても何も失いません。

## ALNS との違い { #how-it-differs-from-alns }

どちらも同じトレイトの上の ruin and recreate で、[ALNS](alns.md) もフローショップで動きます。
ALNS は三つの破壊と二つの修復のオペレータをルーレットで選び、要素の一定割合を取り出し、冷却します。
Iterated Greedy はランダムな破壊一つ、貪欲な修復一つ、取り出す個数を固定し、冷却しません。
Taillard の 50 × 20 インスタンスでは、同じ時間で Iterated Greedy の方が強くなります。

## 温度 { #temperature }

温度は目的関数の単位を持つので、`IteratedGreedy::new` はそれをインスタンスの関数として受け取り、各実行の開始時に一度だけ読みます。
フローショップでは Ruiz と Stützle が、平均処理時間の 10 分の 1 に係数を掛けた `T = factor · Σp / (n · m · 10)` としており、
`iterated_greedy_for_flow_shop` が `temperature_factor` からこれを計算します。

## コンストラクタ { #constructor }

```rust
IteratedGreedy::<P>::new(
    stop_condition: StopCondition,
    removal_count: usize,                       // 1 反復で取り出す要素数
    temperature_of: impl Fn(&P) -> f64 + 'static,
) -> Self
```

`removal_count` が 0 なら panic し、実行の最初の反復で温度が正でなければ panic します。
`with_local_repair(Box<dyn LocalRepair<P>>)` で降下を設定します。`clear()` は温度を捨てるので、次の実行は自分のインスタンスから温度を読み直します。

## ベンチマーク設定 { #benchmark-config }

```toml
[[heuristics]]
kind = "IteratedGreedy"
removal_count = 4           # 省略可 (既定値を表示)
temperature_factor = 0.4    # 省略可 (既定値を表示)
[heuristics.stop_condition]
max_duration_secs = 30.0
```

既定値は Ruiz と Stützle が調整した値です。

## 参考文献 { #references }

- Ruiz, R. and Stützle, T. "A simple and effective iterated greedy algorithm
  for the permutation flowshop scheduling problem." *European Journal of
  Operational Research*, 177(3), 2033-2049, 2007.
