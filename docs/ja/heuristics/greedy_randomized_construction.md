# GreedyRandomizedConstruction

**API:** [`GreedyRandomizedConstruction`](../../api/optopus/heuristic/struct.GreedyRandomizedConstruction.html)

GRASP の構築フェーズです。現在解を空にし、要素を一つずつ入れて新しい解を作ります。
各ステップで、まだ外にある要素それぞれを最も安い配置で価格付けし、制限付き候補リスト、つまりコストが最安から `alpha` の範囲にある要素から一つを一様に引きます。

```text
cost ≤ c_min + alpha · (c_max − c_min)
```

`alpha = 0` は純粋な貪欲法で、`alpha = 1` は要素をランダムな順に取り、それぞれを最も安い位置に置きます。

[`Ruinable`](../traits.md) を実装し、その解が [`Evaluate`](../traits.md) を実装している任意の問題で動きます。
設定ファイルからは TSP、VRP、[フローショップ](../problems/flow_shop.md)、[Bin Packing](../problems/bin_packing.md) で使えます。

## GRASP は合成で書く { #grasp-is-a-composition }

1 回の実行で解をちょうど一つ作って止まるので、これは探索ではなく合成の部品です。
構築と降下を繰り返して最良を残す GRASP 本体は、これを第 2 ステップにした [`Iterated`](meta.md) です。

```rust
use optopus::prelude::*;

let fs = FlowShop::load_file("data/instances/flow_shop/taillard/ta051.txt")?;
let budget = StopCondition::duration(std::time::Duration::from_secs(10));
let mut state = SearchState::new_with_seed(&fs, 42);

Iterated::<FlowShop>::new(
    budget.clone(),
    Box::new(LocalSearch::<FlowShopInsertNeighbor>::new(budget.clone())),
    Box::new(GreedyRandomizedConstruction::<FlowShop>::new(budget, /* alpha = */ 0.2)),
)
.run(&mut state)?;
println!("makespan = {}", state.best_solution.objective);
```

[`examples/compose_grasp.rs`](https://github.com/trash-iine/optopus/blob/main/examples/compose_grasp.rs) はこれをフローショップと Bin Packing で実行し、
Bin Packing では探索に variable neighborhood descent を使って、multi-start 降下と比べます。

## 効くところ { #where-it-pays }

貪欲の規則は、問題の `insertion_cost` が意味するところの「最も安い挿入から」なので、構築の質はその規則の質で決まります。

- フローショップでは最も安い挿入は NEH の規則そのもので、Taillard の 50 × 20 インスタンスで GRASP は multi-start 降下に勝ちます。
- Bin Packing では最も安く入れられる品目は開いている bin に収まるものなので、大きい品目が最後まで残り、それぞれが bin を開きます。First Fit Decreasing の逆です。そこでは GRASP は multi-start 降下に負け、ALNS が両方を大きく上回ります。
- 巡回路と 2-opt では multi-start 降下とほぼ互角です。

各ステップで残りの全要素を価格付けし直すので、`n` 要素の解を作るのは O(n² · buckets · places) です。数百要素ならミリ秒単位、数千要素では遅くなります。

## コンストラクタ { #constructor }

```rust
GreedyRandomizedConstruction::<P>::new(
    stop_condition: StopCondition,   // 実行を早く終えることはあるが、2 回作らせることはない
    alpha: f64,                      // [0, 1]
) -> Self
```

`alpha` が `[0, 1]` の外なら panic します。オペレータ本体は `building_blocks::search` の自由関数 `randomized_greedy_insertion` で、
ALNS の修復オペレータと並んでおり、解の一部をこの方法で作り直したい探索から使えます。

## ベンチマーク設定 { #benchmark-config }

設定ファイルでの GRASP も同じ合成です。

```toml
[[heuristics]]
kind = "Iterated"
[heuristics.stop_condition]
max_duration_secs = 30.0

[[heuristics.steps]]
kind = "LocalSearch"
neighbor = "Relocate"

[[heuristics.steps]]
kind = "GreedyRandomizedConstruction"
alpha = 0.2                 # 省略可 (既定値を表示)、[0, 1] の範囲か検査される
```

## 参考文献 { #references }

- Feo, T. A. and Resende, M. G. C. "Greedy randomized adaptive search
  procedures." *Journal of Global Optimization*, 6(2), 109-133, 1995.
- Laguna, M., Martí, R., Martínez-Gavara, A., Pérez-Peló, S. and Resende,
  M. G. C. "Greedy randomized adaptive search procedures with path
  relinking. An analytical review of designs and implementations."
  *European Journal of Operational Research*, 327(3), 717-734, 2025.
