# WeightedLocalSearch

**API:** [`WeightedLocalSearch`](../../api/optopus/heuristic/struct.WeightedLocalSearch.html)

Weighted Local Search は、問題の制約の一つひとつに重みを持たせ、重み付きのコストを下る探索です。降下が止まるたびに、そのとき破れている制約の重みを上げます。
探索が何度も破る制約は他より重くなるので、どの制約が難しいかを教えなくても、探索はその制約を満たす方向へ押されます。

[`WeightedConstraints`](../traits.md#core-trait-reference) を実装した任意の問題で動きます。設定ファイルからは、節を制約とする [MaxSAT](../problems/sat.md) と、辺を制約とする [vertex cover](../problems/vertex_cover.md) および [graph coloring](../problems/graph_coloring.md) で使えます。

## 例 { #example }

```rust
use optopus::prelude::*;

let vc = VertexCover::load_file("data/instances/max_cut/G22")?;
let mut state = SearchState::new_with_seed(&vc, 42);

WeightedLocalSearch::<VertexCover>::new(StopCondition::duration(
    std::time::Duration::from_secs(10),
))
.run(&mut state)?;
println!("cover size = {}", state.best_solution.cover_size);
```

## アルゴリズムの概略 { #algorithm-sketch }

重みはすべて整数で、1 から始まります。解の重み付きコストは、目的関数値に、破れている制約の重みを足したものです。move の gain は、その move がこのコストをどれだけ下げるかです。

`run_once` は 1 回で 1 ステップ進みます。

1. 問題の move を走査し、タブーメモリが許すものの中で、正の gain が最大の move を選ぶ。同じ値のものからは一様に引く。
2. そのような move がなければ、解は重み付きの局所最適にある。確率 `1 − smoothing` で、破れている制約すべての重みに 1 を足す。それ以外のときは、満たされていて重みが 1 より大きい制約すべてから 1 を引く。
3. 続いて、問題が示す脱出 move のうち最も良いもので局所最適を離れる。破れている制約がある間は、一様に引いた破れている制約一つを満たす move がこれに当たる。破れている制約がなくなれば、目的関数を良くする方向へ進む move が当たる。被覆から頂点を外す move や、最小の色クラスから頂点を出す move である。タブーメモリが禁じている脱出 move は、すべてが禁じられているときに限って選ぶ。

適用した move は、`tabu_tenure` から引いたステップ数の間、元に戻せません。重みは、1000 と制約数の大きい方を超えません。

最良解は問題自身の順序で保持され、重みはそこに入りません。そのため、この探索が報告する値は他のどのヒューリスティクスとも比べられます。重みは探索のもので、`run` のたびに 1 から始まります。

実行は停止条件で終わります。脱出 move のない重み付き局所最適でも終わります。これは何も破っておらず、目的関数を良くする方向へ進む move が一つもない解です。MaxSAT では、すべての節を満たす割当がこれに当たります。

## コンストラクタ { #constructor }

```rust
WeightedLocalSearch::<P>::new(stop_condition: StopCondition) -> Self
```

`with_tabu_tenure((min, max))` はタブー期間の範囲を設定します。設定しなければ `(3, 12)` で、空の範囲では panic します。`with_smoothing(p)` は smoothing の確率を設定します。設定しなければ `0.0003` で、`[0, 1]` の外では panic します。`weights()` は、実行中の、または最後の実行の重みを返します。

## 問題に追加する { #adding-it-to-a-problem }

問題は `WeightedConstraints` を実装して参加します。探索が使う move と、`Weights` 型を指定します。`Weights` は制約ごとの重みと、その重みのもとで move に値を付けるために問題が持つキャッシュを保持します。

| メソッド | すること |
|---|---|
| `unit_weights` | すべて 1 の重みを、ある解に対して値付けした状態で作る。 |
| `constraint_count` | 制約の数を返す。重みの上限がこれで決まる。 |
| `weighted_gain` | move に値を付ける。重み付きコストを下げるとき正。 |
| `apply_weighted` | move を解に適用し、その結果に合わせて重みの値付けを更新する。 |
| `raise_violated` | 破れている制約すべての重みに、上限まで 1 を足す。 |
| `lower_satisfied` | 満たされていて重みが 1 より大きい制約すべてから 1 を引く。 |
| `escape_moves` | 重み付き局所最適から出る move を詰める。 |

探索はステップごとにすべての move を走査するので、`weighted_gain` はキャッシュした値を読むようにします。`Sat` は変数ごとに重み付きの flip gain を持ち、`VertexCover` は頂点ごとに、その頂点だけが覆っている辺の重みを持ちます。`GraphColoring` は頂点と色の組ごとに、その色の隣接頂点への辺の重みを持ちます。いずれも、move を適用するのと同じ走査でキャッシュを更新します。

## ベンチマーク設定 { #benchmark-config }

```toml
[[heuristics]]
kind = "WeightedLocalSearch"
tabu_tenure = [3, 12]     # 任意 (既定値を表示)
smoothing = 0.0003        # 任意 (既定値を表示)
[heuristics.stop_condition]
max_duration_secs = 30.0
```

move は問題が指定するので、`neighbor` は取りません。

## 参考文献 { #references }

- Thornton, J., Pham, D. N., Bain, S. and Ferreira Jr., V. "Additive versus
  Multiplicative Clause Weighting for SAT." *Proc. AAAI-04*, 191-196, 2004.
- Lin, P., Cai, S., Zou, M. and Lin, J. "New Characterizations and Efficient
  Local Search for General Integer Linear Programming." 2023.
  [arXiv:2305.00188](https://arxiv.org/abs/2305.00188)
