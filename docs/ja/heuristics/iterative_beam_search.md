# IterativeBeamSearch

**API:** [`IterativeBeamSearch`](../../api/optopus/heuristic/struct.IterativeBeamSearch.html)

Iterative beam search は、空の状態から要素を 1 つずつ入れて解を組み立て、各ステップで良い部分解を数個だけ残します。組み立て終わったら、残す数を 2 倍にしてもう一度行います。
実行は貪欲な構築から始まり、後半の時間をより幅の広い探索に使います。

[`Ruinable`](../traits.md) を実装し、その `Partial` が `Clone` で、解が [`Evaluate`](../traits.md) を実装している任意の問題で動きます。
設定ファイルからは [TSP](../problems/tsp.md)、[VRP](../problems/vrp.md)、[フローショップ](../problems/flow_shop.md)、[Bin Packing](../problems/bin_packing.md) で使えます。

[`BeamSearch`](beam_search.md) とは別の探索です。そちらは完成した解のビームを持ち、近傍の move で各解を展開します。

## 例 { #example }

```rust
use optopus::prelude::*;

let fs = FlowShop::load_file("data/instances/flow_shop/taillard/ta051.txt")?;
let mut state = SearchState::new_with_seed(&fs, 42);

let mut search = IterativeBeamSearch::<FlowShop>::new(StopCondition::iterations(8));
search.run(&mut state)?;
println!("makespan = {}", state.best_solution.objective);
```

8 回のパスが、幅 1、2、4 と広がって 128 まで走ります。

## アルゴリズムの概要 { #algorithm-sketch }

`run_once` は、現在の幅 `D` でのパス 1 回です。

1. 現在解を空にし、その要素の順序を引きます。
2. 空の部分解 1 つからビームを始めます。
3. 順序の次の要素について、ビームにあるすべての部分解のすべての配置を `insertion_cost` で価格付けします。
4. `partial_energy` が最も低くなる配置を `D` 個残し、実際に挿入します。これが次のビームです。
5. すべての要素を入れ終えたら、ビームの中の最良がそのパスの解です。これが現在解になり、実行全体の最良解は通常どおり保持されます。
6. `D` に `growth` を掛けます。

1 回のパスは要素 1 つにつき `D · n` 個の配置を価格付けするので、コストは要素数の 2 乗と幅に比例して増えます。
部分解は、その中で 2 つ目以降に残った配置の数だけコピーされ、要素 1 つにつき最大 `D` 回です。

幅 1 のパスは、ランダムな順序での貪欲な挿入です。幅の広いパスは、同じ探索に余裕を足したものではありません。順序を引き直すので、解はパスごとに必ず良くなるわけではなく、平均として良くなります。

反復数による停止はパスとパスの間で効きます。時間制限は要素と要素の間でもパスを打ち切り、そのパスは捨てられます。

幅を広げながら再スタートする枠組みは、参考文献にあるフローショップ向けの探索のものです。分岐の仕方は異なります。
そちらは、まだスケジュールしていない任意のジョブを列の端に追加し、下界とアイドル時間でノードを順位付けします。
こちらは、次に入れる要素を引いた順序で固定し、任意の位置に挿入し、部分解のエネルギーで順位付けします。これはどの `Ruinable` な問題でも答えられる量です。

## コンストラクタ { #constructor }

```rust
IterativeBeamSearch::<P>::new(stop_condition: StopCondition) -> Self
```

`stop_condition` は実行の終わりを決めます。[停止条件](../guide/stop_conditions.md)を参照してください。

| Builder | 設定するもの | 既定値 |
|---|---|---|
| `with_initial_width(width)` | 最初のパスの幅 | `1` |
| `with_growth(growth)` | パスごとに幅に掛ける係数。1 以上 | `2.0` |

係数を 1 にすると、すべてのパスが最初の幅のまま、毎回新しい順序で走ります。`clear()` は幅を最初の値に戻します。

## ベンチマーク設定 { #benchmark-config }

```toml
[[heuristics]]
kind = "IterativeBeamSearch"
initial_width = 1     # 省略可(既定値)
growth = 2.0          # 省略可(既定値)
[heuristics.stop_condition]
max_duration_secs = 30.0
```

## 参考文献 { #references }

- Ow, P. S. and Morton, T. E. "Filtered Beam Search in Scheduling."
  *International Journal of Production Research*, 26(1), 35-62, 1988.
- Libralesso, L., Focke, P. A., Secardin, A. and Jost, V. "Iterative beam
  search algorithms for the permutation flowshop." *European Journal of
  Operational Research*, 301(1), 217-234, 2022.
