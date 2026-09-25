# Vertex Cover

**API:** [`VertexCover`](../../api/optopus/problem/vertex_cover/struct.VertexCover.html)

無向グラフ `G = (V, E)` に対し、頂点被覆とは、すべての辺が少なくとも一方の端点を `S` に持つような部分集合
`S ⊆ V` のことです。そのような部分集合の大きさを最小化します。言い換えると、各頂点 `v` に二値の所属
`x_v ∈ {0,1}` (`v ∈ S` のときに限り `x_v = 1`) を選び、すべての辺が被覆されるという条件の下で最小化します。

```text
minimize  Σ_v x_v   subject to   x_i + x_j ≥ 1  for every edge (i,j) ∈ E
```

Vertex Cover は NP 困難です。補集合をとることで Independent Set や Clique と等価になります。

実行可能性はソフトに扱います。ソルバが実際に最適化するのはペナルティを加えた目的関数で、
`penalty_weight` はその最適解が必ず実行可能 (被覆されない辺がない) になるだけ大きく選ばれています。

```text
objective(x) = cover_size(x) + penalty_weight · uncovered_edges(x)
```

## 例 { #example }

探索を実行し、被覆をなす頂点を読み出します。

```rust
use optopus::prelude::*;

let vc = VertexCover::new(Graph::from_edges([(0, 1, 1.0), (1, 2, 1.0), (0, 2, 1.0)]));
let mut state = SearchState::new(&vc);
LocalSearch::<VertexCoverFlipNeighbor>::new(StopCondition::iterations(10_000))
    .run(&mut state)
    .unwrap();

let sol = &state.best_solution;
println!("cover size = {}", sol.cover_size);
let cover: Vec<usize> = sol
    .x
    .iter()
    .enumerate()
    .filter(|&(_, &in_cover)| in_cover)
    .map(|(v, _)| v)
    .collect();
println!("cover = {cover:?}"); // すべての辺を被覆するために選ばれた頂点
```

## 解 { #solution }

[`VertexCoverSolution`](../../api/optopus/problem/vertex_cover/struct.VertexCoverSolution.html)
は上の定義の所属 `x` (`x[v]` は頂点 `v` の被覆への所属 `x_v`) と、上で定義したペナルティ付きの
`objective` を持ちます。`cover_size` は `Σ x_v = |S|`、`uncovered_edges` は制約違反の数です。

## 近傍 { #neighbors }

| 型 | move | 反復コスト |
|---|---|---|
| [`VertexCoverFlipNeighbor`](../../api/optopus/problem/vertex_cover/struct.VertexCoverFlipNeighbor.html) | 頂点一つの所属を反転する。 | `iter + 1` |
| [`VertexCoverSwapNeighbor`](../../api/optopus/problem/vertex_cover/struct.VertexCoverSwapNeighbor.html) | 被覆に入っている頂点と入っていない頂点を入れ替える。 | `iter + 2` |

## 交叉 { #crossover }

- `VertexCoverUniformCrossover`。頂点ごとにランダムに親を選びます。
- `VertexCover` は `SubProblemExtractable` を実装しています。両親で一致している頂点は固定され、残りの頂点が部分インスタンスになります。

## ファイル形式 { #file-format }

Vertex Cover は [MaxCut のグラフ形式](max_cut.md#file-format) をそのまま使います。
辺の重みは無視されます (どの辺も被覆の制約に等しく寄与します)。

```rust
use optopus::prelude::*;

let vc = VertexCover::new(Graph::load_from_file("data/instances/max_cut/G1")?);
```

## 参考文献 { #references }

- Karp, R. M. "Reducibility Among Combinatorial Problems." In *Complexity of
  Computer Computations*, pp. 85-103. Plenum Press, 1972. (Vertex Cover は
  Karp の 21 個の NP 完全問題の一つです。)
