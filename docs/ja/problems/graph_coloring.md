# Graph Coloring

**API:** [`GraphColoring`](../../api/optopus/problem/graph_coloring/struct.GraphColoring.html)

無向グラフ `G = (V, E)` の正しい彩色とは、すべての辺の両端点が異なる色になるように、各頂点 `v` に色 `c_v` を割り当てることです。
使う色の種類数を最小化します。

```text
minimize  |{c_v : v ∈ V}|   subject to   c_i ≠ c_j  for every edge (i,j) ∈ E
```

達成できる最小の数が彩色数 `χ(G)` です。グラフが 3 彩色可能かを判定するだけでもすでに NP 完全です。

パレットはグラフから導いた `k = max_degree + 1` 色に固定されているので、正しい彩色は必ず存在します
(Brooks の定理によれば、ほとんどすべてのグラフでさらに厳しい上界が成り立ちます)。`k` を別に入力することはなく、
実行はそのうち何色を使えたかを報告します。

実行可能性は [Vertex Cover](vertex_cover.md) と同じくソフトに扱います。ソルバはペナルティを加えた目的関数を最適化し、
`penalty_weight = n + 1` とすることで、衝突を一つ解消することが色数のどんな変化よりも常に得になり、どの最適解も正しい彩色になります。

```text
objective(c) = colors_used(c) + penalty_weight · conflicts(c)
```

ここで `conflicts` は両端点が同じ色の辺の数です。

## 例 { #example }

探索を実行し、彩色を読み出します。

```rust
use optopus::prelude::*;

let gc = GraphColoring::new(Graph::from_edges([(0, 1, 1.0), (1, 2, 1.0), (0, 2, 1.0)]));
let mut state = SearchState::new(&gc);
LocalSearch::<GraphColoringRecolorNeighbor>::new(StopCondition::iterations(10_000))
    .run(&mut state)
    .unwrap();

let sol = &state.best_solution;
println!("colors used = {}, conflicts = {}", sol.colors_used, sol.conflicts);
println!("coloring = {:?}", sol.colors); // colors[v] は頂点 v の色
```

## 解 { #solution }

[`GraphColoringSolution`](../../api/optopus/problem/graph_coloring/struct.GraphColoringSolution.html)
は割り当て `colors` (`colors[v]` は `0..k` の範囲の `c_v`)、上で定義したペナルティ付きの `objective`、
空でない色クラスの数 `colors_used`、制約違反の数 `conflicts` を持ちます。さらに頂点ごとに、各色の隣接頂点がいくつあるか
(TabuCol の Γ 行列) をキャッシュしていて、これによってどの move の gain も O(1) で求まります。

## 近傍 { #neighbors }

| 型 | move | 反復コスト |
|---|---|---|
| [`GraphColoringRecolorNeighbor`](../../api/optopus/problem/graph_coloring/struct.GraphColoringRecolorNeighbor.html) | 頂点一つに別の色を与える。設定では `neighbor = "Flip"` で選ぶ。 | `iter + 1` |
| [`GraphColoringSwapNeighbor`](../../api/optopus/problem/graph_coloring/struct.GraphColoringSwapNeighbor.html) | 異なる色の二頂点の色を交換する。`colors_used` は変わらない。 | `iter + 2` |

一頂点の再彩色だけでは、色数の項は平坦な地形になります。衝突は強く修正されますが、色クラスを一つ空にするにはそのクラスのすべての頂点を一つずつ動かす必要があり、
最後の一つを動かすまで何の報酬もありません。探索はすぐに正しい彩色に達し、その後ゆっくりと色数を減らしていくと考えてください。

## 交叉 { #crossover }

- `GraphColoringUniformCrossover`。頂点ごとにランダムに親を選びます。色のラベルは両親の間でそろっていないので、
  子の色数と衝突は一から計算し直します。

## ファイル形式 { #file-format }

Graph Coloring は [MaxCut のグラフ形式](max_cut.md#file-format) をそのまま使います。
辺の重みは無視されます (どの辺も衝突の数に等しく寄与します)。

```rust
use optopus::prelude::*;

let gc = GraphColoring::load_file("data/instances/graph_coloring/example.txt")?;
```

## 参考文献 { #references }

- Hertz, A., and de Werra, D. "Using tabu search techniques for graph
  coloring." *Computing* 39, pp. 345-351, 1987. (TabuCol。解がキャッシュする Γ 行列の起源です。)
- Karp, R. M. "Reducibility Among Combinatorial Problems." In *Complexity of
  Computer Computations*, pp. 85-103. Plenum Press, 1972. (彩色数は Karp の
  21 個の NP 完全問題の一つです。)
