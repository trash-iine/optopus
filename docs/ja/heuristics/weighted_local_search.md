# WeightedLocalSearch

**API:** [`WeightedLocalSearch`](../../api/optopus/heuristic/struct.WeightedLocalSearch.html)

Weighted Local Search は、問題の制約の一つひとつに重みを持たせ、行き詰まるたびに、そのとき破れている制約の重みを上げる探索です。
探索が何度も破る制約は他より重くなるので、どの制約が難しいかを教えなくても、探索はその制約を満たす方向へ押されます。何も破らない解を得た後は、その解の目的関数を良くすることと、そのせいで破れたものを直すことを繰り返します。

[`WeightedConstraints`](../traits.md#core-trait-reference) を実装した任意の問題で動きます。設定ファイルからは、節を制約とする [MaxSAT](../problems/sat.md) と、辺を制約とする [vertex cover](../problems/vertex_cover.md) および [graph coloring](../problems/graph_coloring.md) で使えます。

## 例 { #example }

```rust
use optopus::prelude::*;

let vc = VertexCover::load_file("data/instances/max_cut/G22")?;
let mut state = SearchState::new_with_seed(&vc, 42);

WeightedLocalSearch::<VertexCover>::new(
    StopCondition::duration(std::time::Duration::from_secs(10)),
    (3, 12),
    0.0003,
)
.run(&mut state)?;
println!("cover size = {}", state.best_solution.cover_size);
```

## アルゴリズムの概要 { #algorithm-sketch }

重みはすべて整数で、1 から始まります。制約ごとの重みのほかに、目的関数にも重みがあります。ここでいう目的関数は、制約を除いた部分のコスト(被覆の大きさや使う色の数など)です。問題自身の目的関数が破れた制約に課すペナルティは含みません。

制約の tight move は、その制約の変数を動かして、制約をできるだけきつくする move です。破れている制約では、それを満たす move です。満たされている制約では、満たしたまま余裕を使い切る move です。両端とも被覆に入っている辺から頂点を外す move や、真のリテラルが他にもある節で真のリテラルを偽にする move がこれに当たります。

`run_once` は 1 回で 1 ステップ進みます。手元の解によって、次の三つのモードのどれかで動きます。

1. 何も破らない解がまだ見つかっていない間は、それを探す。破れている制約を三つ引き、その tight move に、満たす制約の重みから破る制約の重みを引いた値を付ける。タブーメモリが許すもののうち、値が正で最大の move を適用する。
2. 解が何も破っていない間は、目的関数を良くする。何も破らずに目的関数を最も下げる move を適用する。そのような move がなければ、問題が示す目的関数の脱出 move のうち、値が最も良いものを適用する。被覆から頂点を外す move や、最小の色クラスから頂点を出す move で、何かを破る。値の付け方は三つ目のモードと同じ。
3. それ以外のときは、これまでの最良より良い解へ向けて直す。move の値には、最良の目的関数値より下に行くなら目的関数の重みを足し、行かないなら引く。一つ目のモードと同じく破れている制約を三つ引き、その tight move に正の値のものがなければ、満たされている制約を三十引いてその tight move を試す。

一つ目か三つ目のモードで、引いた move に正の値のものがなければ、重みを変えます。確率 `1 − smoothing` で、破れている制約すべての重みに 1 を足します。何も破らない解が見つかった後なら、解が最良より良くないときは目的関数の重みにも 1 を足します。それ以外のときは、満たされていて重みが 1 より大きい制約すべてから 1 を引き、解が最良より良ければ目的関数の重みからも 1 を引きます。その後、破れている制約を一様に一つ引き、その tight move のうち値が最大のものを適用します。ここと目的関数の脱出 move では、タブーメモリが禁じている move は、すべてが禁じられているときに限って選びます。

引いた制約の tight move が、1 ステップで値を付ける数より多いときは、その数だけ一様に引いて値を付けます。破れている制約では 2000、満たされている制約では 350、重みを変えたときは 150 です。そのため、二つ目のモードを除けば、1 ステップのコストは問題の大きさによらずほぼ一定です。二つ目のモードはすべての move を見ます。

適用した move は、`tabu_tenure` から引いたステップ数の間、元に戻せません。制約の重みは 1000 と制約数の大きい方を超えず、目的関数の重みはその十分の一を超えません。最良解が 1,500,000 ステップ良くならなければ、探索をやり直します。変数ごとに最良解か新しい解の値を等確率で取り、重みはすべて 1 に戻します。

最良解は問題自身の順序で保持され、重みはそこに入りません。そのため、この探索が報告する値は他のどのヒューリスティクスとも比べられます。重みとタブーメモリは探索のもので、`run` のたびに初めから始まります。

実行は停止条件で終わります。何も破っておらず、目的関数を良くする方向へ進む move が一つもない解でも終わります。MaxSAT では、すべての節を満たす割当がこれに当たります。

## コンストラクタ { #constructor }

```rust
WeightedLocalSearch::<P>::new(
    stop_condition: StopCondition,
    tabu_tenure: (u64, u64),
    smoothing: f64,
) -> Self
```

調整済みの値は `(3, 12)` と `0.0003` です。空のタブー期間の範囲と、`[0, 1]` の外の smoothing では panic します。`with_violated_samples(3, 2000)`、`with_satisfied_samples(30, 350)`、`with_random_constraint_samples(150)` は、1 ステップで引く制約の数と、値を付ける move の数を設定します。示した値が既定値です。`weights()` は、実行中の、または最後の実行の `ConstraintWeights` を返します。そこから、制約ごとの重み、破れている制約、問題のキャッシュを読めます。`objective_weight()` は目的関数の重みを返します。

## 問題に追加する { #adding-it-to-a-problem }

問題は `WeightedConstraints` を実装して参加します。探索が使う move と、`Prices` 型を指定します。`Prices` は、重みのもとで move に値を付けるために問題が持つキャッシュです。重みそのもの、破れている制約の集合、重みを上げ下げする規則は探索の側が持ち、重みが変わったときに問題を呼び戻します。

| メソッド | すること |
|---|---|
| `unit_prices` | すべての重みが 1 のときのキャッシュと、ある解が破っている制約の集合を作る。集合の上限(`bound`)が制約の数になる。 |
| `weight_gain` | move に値を付ける。満たす制約の重みから、破る制約の重みを引いた値。 |
| `apply_weighted` | move を解に適用し、その結果に合わせてキャッシュと破れている制約の集合を更新する。 |
| `reweight` | 制約一つの重みが変わった後に、キャッシュを直す。 |
| `tight_moves` | 制約一つの tight move を詰める。 |
| `objective_escapes` | 何を破るかによらず、目的関数を良くする方向へ一歩進む move を詰める。省略でき、既定では何も詰めない。そのときは MaxSAT と同じく、何も破らない解で探索が終わる。 |
| `objective` / `objective_change` | 制約を除いた目的関数と、move によるその変化。省略でき、既定では 0。目的関数が制約だけでできている問題向け。 |
| `restart_from` | 最良解を受け取り、やり直しの出発点を返す。省略でき、既定では新しい解。組み込みの問題は、最良解と新しい解を変数ごとに混ぜる。 |

二つ目のモードはすべての move を見るので、`weight_gain` はキャッシュした値を読むようにします。`Sat` は変数ごとに重み付きの flip gain を持ち、`VertexCover` は頂点ごとに、もう一方の端が集合の外にある辺の重みを持ちます。その頂点が集合の中にあればそれだけが覆っている辺、外にあれば入ることで覆う未被覆の辺です。`GraphColoring` は頂点と色の組ごとに、その色の隣接頂点への辺の重みを持ちます。いずれも、move を適用するのと同じ走査でキャッシュを更新します。

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
