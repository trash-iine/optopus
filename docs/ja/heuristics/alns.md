# AdaptiveLargeNeighborhoodSearch

**API:** [`AdaptiveLargeNeighborhoodSearch`](../../api/optopus/heuristic/struct.AdaptiveLargeNeighborhoodSearch.html)

Adaptive Large Neighborhood Search は、現在解の一部を壊して (ruin) 作り直します (recreate)。
オペレータの組は、最近の成績を追跡する重みを持つルーレットで選びます。

[`Ruinable`](../traits.md) を実装し、その解が [`Evaluate`](../traits.md) を実装している任意の問題で動きます。
現在それに当たるのは [CVRP](../problems/vrp.md) と [TSP](../problems/tsp.md) です。
`Ruinable` は問題に必要な形、つまり有限の資源を奪い合うコンテナに要素が割り当てられているという形を表し、すでにその形を持つ問題の族を名指ししています。
巡回路はコンテナが一つの場合で、それが何を犠牲にするかは [オペレータ](#operators) で説明します。

## 例 { #example }

```rust
use optopus::prelude::*;

let vrp = Vrp::load_file("data/instances/vrp/demo16.vrp")?;
let mut state = SearchState::new(&vrp);

let mut alns = alns_for_vrp(
    StopCondition::iterations(10_000),
    /* removal_fraction = */ 0.15,
    /* cooling_rate     = */ 0.9995,
);
alns.run(&mut state)?;

let sol = &state.best_solution;
println!("total distance = {}", sol.distance);
for (vehicle, route) in sol.routes.iter().enumerate() {
    println!("vehicle {vehicle}: depot -> {route:?} -> depot");
}
```

自前の move 集合を持つので、`neighbor` の型パラメータは取りません。`demo16.vrp` はリポジトリに入っている 15 顧客のテスト用インスタンスです。

`alns_for_vrp` はコンストラクタとアンカー付きのルート降下を組み合わせたもので、これがあるからこそ ruin-and-recreate がルート上で効果を上げます。
同じ組を展開して書き、builder を一つつなげると次のようになります。

```rust
use optopus::prelude::*;
use optopus::problem::vrp::AnchoredRouteDescent;

let mut alns = AdaptiveLargeNeighborhoodSearch::<Vrp>::new(
    StopCondition::iterations(10_000),
    /* removal_fraction = */ 0.15,
    /* cooling_rate     = */ 0.9995,
)
.with_local_repair(Box::new(AnchoredRouteDescent::new()))
.with_max_removal(20);
```

この形を使うのは、最後の行のように [builder](#constructor) をつなげたいときや、VRP に独自の [`LocalRepair`](../traits.md) を渡したいときです。
組み合わせ関数のない問題もこの形を使い、自前の修復を与えるか、`with_local_repair` を付けずに素の ruin-and-recreate にします。

TSP での組み合わせは `alns_for_tsp` です。これは再挿入された都市をアンカーとする Or-opt と 2-opt の降下
`AnchoredTourDescent` と探索を組み合わせます。

```rust
use optopus::prelude::*;

let tsp = Tsp::load_file("data/instances/tsp/berlin52.tsp")?;
let mut state = SearchState::new(&tsp);

alns_for_tsp(StopCondition::iterations(10_000), 0.15, 0.9995).run(&mut state)?;
println!("tour length = {}", state.best_solution.objective);
```

## アルゴリズムの概要 { #algorithm-sketch }

各 `run_once` は候補を一つ作ります。

1. オペレータの選択。適応的な重みに対するルーレットで、破壊オペレータと修復オペレータを一つずつ選びます。
2. 破壊。現在解から `removal_fraction · n` 個の要素を取り除きます。
3. 修復。それらをすべて挿入し直します。
4. 降下。作り直した解に対し、再挿入した要素をアンカーとして問題の `LocalRepair` を走らせます (下を参照)。
5. 受理。問題の `partial_energy` (CVRP の容量ペナルティを含めた目的関数) に対する焼きなまし法の基準で判定します。
   温度は 5% 悪い解がおよそ 0.5 の確率で受理されるように初期化され、反復ごとに `cooling_rate` で冷却されます。
6. 採点。オペレータの組に報酬を与えます。新しい全体の最良解なら `4`、現在解より良ければ `2`、悪い解が受理されたら `1`、それ以外は `0` です。
   100 反復ごとに、その区間の平均スコアが反応係数 `0.1` で重みに混ぜ込まれます。

破壊と修復の一歩は単一の `MoveToNeighbor` ではないので、このヒューリスティクスは `state.apply` を通さず、`state.solution` を直接操作します。

### オペレータ { #operators }

| 破壊 | 取り除くもの |
|---|---|
| Random | 一様に引いた `k` 個の要素。 |
| Worst | 取り除いたときの gain が最も大きい `k` 個の要素。つまり寄り道のコストが最も大きいもの。 |
| Shaw | ランダムに選んだ種の要素と最も関連の深い `k` 個の要素。 |

| 修復 | 挿入の仕方 |
|---|---|
| Greedy | 取り除いた各要素を、ランダムな順序で最も安い位置に挿入する。 |
| Regret-2 | regret が最も大きい要素から挿入する。regret は、最も安い位置と、別のコンテナの中で最も安い位置との差。 |

regret は位置の間ではなくコンテナの間で測ります。二番目に安い場所はほとんどいつも同じコンテナの隣の位置で、どの要素でも差がほぼ 0 になり、
regret-2 が greedy と区別できなくなってしまうからです。

それぞれの問題はこれを次のように解釈します。

| | CVRP | TSP |
|---|---|---|
| 要素、コンテナ | 顧客、車両 | 都市、巡回路 |
| 関連度 | `distance(a, b) + \|demand(a) − demand(b)\|` | `distance(a, b)` |
| 挿入コスト | 寄り道と容量ペナルティ。そのため、どのルートも満杯でも挿入先は常にある | 寄り道 |
| Regret-2 | ルート間 | コンテナが一つでは定義できないので、何も順位付けしない固定の順序で候補を挿入する。探索を担うのは破壊オペレータと greedy |
| `LocalRepair` | `AnchoredRouteDescent`、granular なルート降下 | `AnchoredTourDescent`、最近傍に対する Or-opt と 2-opt |

## コンストラクタ { #constructor }

```rust
AdaptiveLargeNeighborhoodSearch::<P>::new(
    stop_condition: StopCondition,
    removal_fraction: f64,   // 反復ごとに壊す要素の割合
    cooling_rate: f64,       // 反復ごとの幾何的な冷却係数
) -> Self
```

`removal_fraction` か `cooling_rate` が `(0, 1]` の外にあれば panic します。

それ以外はすべて既定値が公開された builder なので、`new` の引数は三つのままです。

| builder | 設定するもの | 既定値 |
|---|---|---|
| `with_local_repair(Box<dyn LocalRepair<P>>)` | 修復後のアンカー付き局所探索 | なし。`alns_for_vrp` と `alns_for_tsp` が設定する |
| `with_scoring(best, better, accept)` | オペレータの組が得る報酬 | `4.0 / 2.0 / 1.0` |
| `with_adaptation(segment_len, reaction)` | 採点区間あたりの反復数と、その平均を重みに混ぜる割合 | `100` / `0.1` |
| `with_max_removal(n)` | 反復ごとに取り除く数の上限 | `50` |

重みは区間平均の凸結合なので、意味を持つのは三つの報酬の比だけです。

二つの局所修復も同じ形の builder を持っています。`AnchoredRouteDescent::new()` と `AnchoredTourDescent::new()`
は公開された既定値を与え、それぞれ次のものをつなげられます。

| builder | 設定するもの | 既定値 |
|---|---|---|
| `with_granularity(k)` | 各 move が考える最近傍の相手の数 | ルートでは `20`、巡回路では `10` |
| `with_ring(r)` | 各アンカーと一緒になめる相手の数。`0` ならアンカーだけ | `5` |
| `with_max_passes(p)` | 修復一回あたりのなめるリストのパス数 | `4` |

既定値は各降下の関連定数です (`AnchoredTourDescent::DEFAULT_GRANULARITY` など。リングは
`AnchoredSweep::DEFAULT_RING`)。調整した降下は、`alns_for_vrp` や `alns_for_tsp` が作るはずだったものの代わりに
`with_local_repair` に渡します。

`clear()` はオペレータの重みと温度をリセットしますが、builder で設定したものは残します。
`LocalRepair` 自身がインスタンスから導いたキャッシュ (両方の降下が持つ候補リスト) を残すか捨てるかは、その `LocalRepair` の判断です。

## ベンチマーク設定 { #benchmark-config }

```toml
[[heuristics]]
kind = "AdaptiveLargeNeighborhoodSearch"
removal_fraction = 0.15    # 任意 (値は既定値)
cooling_rate = 0.9995      # 任意 (値は既定値)
[heuristics.stop_condition]
max_duration_secs = 30.0
```

設定できるのはこの二つのキーだけです。上の builder は探索のものも降下のものも Rust からしか使えず、TOML からは使えません。
そのため、それらを書いた設定は受け付けられますが無視されます。

## 参考文献 { #references }

- Ropke, S. and Pisinger, D. "An Adaptive Large Neighborhood Search Heuristic
  for the Pickup and Delivery Problem with Time Windows." *Transportation
  Science*, 40(4), 455-472, 2006.
- Shaw, P. "Using Constraint Programming and Local Search Methods to Solve
  Vehicle Routing Problems." In CP 1998, pp. 417-431. Springer, 1998.
  (Shaw removal です。)
