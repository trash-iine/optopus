# MaxCut

**API:** [`MaxCut`](../../api/optopus/problem/max_cut/struct.MaxCut.html)

頂点集合 `V`、辺集合 `E`、辺の重み `w` を持つ重み付き無向グラフ `G = (V, E, w)` が与えられたとき、
`V` を互いに素な二つの集合に分け、分割をまたぐ辺の重みの合計を最大化する問題です。
言い換えると、各頂点 `i` にどちら側に入るかを表す二値ラベル `x_i ∈ {0, 1}` を割り当てます。
辺 `(i, j)` がカットされるのはちょうど `x_i ≠ x_j` のときです。

```text
maximize  Σ_{(i,j)∈E} w_ij · [x_i ≠ x_j]        (x ∈ {0,1}^|V|)
```

## 例 { #example }

探索を実行し、見つかった分割を読み出します。

```rust
use optopus::prelude::*;

let mc = MaxCut::from_edges([(0, 1, 1.0), (0, 2, 1.0), (1, 2, 2.0)]);
let mut state = SearchState::new(&mc);
LocalSearch::<MaxCutFlipNeighbor>::new(StopCondition::iterations(10_000))
    .run(&mut state)
    .unwrap();

let sol = &state.best_solution;
println!("cut weight = {}", sol.objective);
for (v, &side) in sol.x.iter().enumerate() {
    println!("vertex {v} is on side {}", side as u8); // `v` が二つの集合のどちらに入ったか
}
```

`MaxCut::from_edges` は `MaxCut::new(Graph::from_edges(...))` の便利なラッパーです。
どちらも重複した辺は集合として扱い、最後に書いたものが残ります。

## 解 { #solution }

[`MaxCutSolution`](../../api/optopus/problem/max_cut/struct.MaxCutSolution.html)
は上の定義の分割を表します。`x[v]` は頂点 `v` の側 (`false`/`true`) です。

## 近傍 { #neighbors }

| 型 | TOML の `neighbor` | move |
|---|---|---|
| `MaxCutFlipNeighbor` | `"Flip"` | 頂点を一つ反対側に移す。`iter + 1`。 |
| `MaxCutSwapNeighbor` | `"Swap"` | 反対側にある二つの頂点を入れ替える。`iter + 2`。 |

## 交叉 { #crossover }

- `MaxCutUniformCrossover`。頂点ごとにランダムに親を選びます。
- `MaxCut` は `SubProblemExtractable` も実装しているので、`SubProblemBasedCrossover`
  が使えます。両親で一致している頂点は固定され、一致しない頂点が部分 MaxCut インスタンスになります。
  その辺には、固定された近傍へ向かうバイアス項が含まれます。

## ファイル形式 { #file-format }

`Graph::load_from_file` は、ヘッダ行一つとそれに続く辺の行を受け取ります。頂点は 1 始まりです。

```text
N M
i j w
i j w
...
```

- `N` は頂点数、`M` は辺の数です。
- `w` は省略でき、省略すると `1.0` になります。
- 頂点は内部で 0 始まりに変換されます。

```rust
use optopus::prelude::*;

let mc = MaxCut::new(Graph::load_from_file("data/instances/max_cut/G1")?);
```

## 最適値が分かっているインスタンス { #instances-with-a-known-optimum }

[`PlantedMaxCut`](../../api/optopus/problem/max_cut/struct.PlantedMaxCut.html)
は選んだ解を中心にインスタンスを作ります。そのため最適値は最良既知値ではなく、構成から厳密に分かります。

```rust
use optopus::common::seeded_rng;
use optopus::problem::{PlantedMaxCut, TileProbs2d};

let planted = PlantedMaxCut::tile_planting_2d(
    40, // 40 x 40 のトーラス、次数 4
    TileProbs2d::new(0.35, 0.0, 0.65),
    &mut seeded_rng(1),
);
planted.verify().unwrap(); // 記録された最適値がインスタンスから計算した値と一致する
// planted.optimum はどの実行も超えられない上限
```

| コンストラクタ | トポロジー | 難しさのつまみ |
|---|---|---|
| `tile_planting_2d(l, TileProbs2d, rng)` | 正方格子のトーラス、次数 4 | クラスの混合比 `p1`/`p2`/`p3` |
| `tile_planting_3d(l, TileProbs3d, rng)` | 立方格子のトーラス、次数 6 | クラスの混合比 `p_2fp`/`p_4fp` |
| `wishart(n, alpha, WishartCouplers, rng)` | 完全グラフ | `alpha = M / n`、`(0, 1)` の範囲 |

- すべてのインスタンスにはゲージ変換がかかっています。どの構成法も本来はすべてがそろった状態を埋め込み、それは簡単に見つかってしまいます。
  ランダムなゲージによって最適解を任意の分割に移します。これは符号付きグラフ上のスイッチングなので、解のラベルは付け替わりますが、
  フラストレーションの構造、したがって難しさはそのままです。
- 整数の重みにすると「最適値に達した」かどうかが判定できます。タイルプランティングと
  `WishartCouplers::Discrete` は整数の重みを作るので、`f32` の目的関数値は厳密です。
  `WishartCouplers::Gaussian` はそうではなく、実行は丸め誤差の範囲までしか評価できません。
  `verify()` はこの区別を強制し、最適値が往復で一致しなくなったインスタンスを拒否します。
- `alpha` は 1 未満でなければならず、有用な値は `n` に依存します。埋め込んだベクトルは Wishart 結合行列の核に入っていて、
  その次元は `n - M` です。`alpha >= 1` では核が埋め込んだベクトルだけに縮み、固有値分解で多項式時間に復元できてしまうので、
  コンストラクタはこれを拒否します。そこから十分離れたところでは、「必ず解ける」と「決して解けない」の境界は核の次元がおよそ 32 で一定のところにあり、
  `n = 48 / 64 / 96 / 256` で `alpha = 0.35 / 0.50 / 0.65 / 0.90` に当たります。どの大きさでも小さい `alpha`
  のほうが難しい側です。easy-hard-easy の形から想像されるような易しい側ではありません。ここでの基準が、物理でいう基底状態ではなく厳密な最適値に達することだからです。
  `chook` の既定値 `alpha = 0.75` は四つの大きさすべてで易しいです。

インスタンス群の生成は `examples/generate_hard_maxcut.rs` にあります。
[`data/instances/README.md`](https://github.com/trash-iine/optopus/blob/main/data/instances/README.md) を参照してください。

## 補足 { #notes }

- `MaxCutSolution` は `x`、`gain`、`objective` の三つのフィールドを持ちます。詳細は
  [rustdoc](../../api/optopus/problem/max_cut/struct.MaxCutSolution.html) にあります。

## 参考文献 { #references }

- Karp, R. M. "Reducibility Among Combinatorial Problems." In *Complexity of
  Computer Computations*, pp. 85-103. Plenum Press, 1972. (Max Cut は Karp の
  21 個の NP 完全問題の一つです。)
- 標準的なベンチマーク集合は Gset グラフ (G1–G81) で、`rudy`
  グラフ生成器で作られ、Y. Ye によって配布されています。
- Perera, D. et al. "Chook, A comprehensive suite for generating binary
  optimization problems with planted solutions."
  [arXiv:2005.14344](https://arxiv.org/abs/2005.14344). `PlantedMaxCut`
  が従っている参照実装です。
- Perera, D., Hamze, F., Raymond, J., Weigel, M. and Katzgraber, H. G.
  "Computational hardness of spin-glass problems with tile-planted solutions."
  Phys. Rev. E 101, 023316 (2020).
  [arXiv:1907.10809](https://arxiv.org/abs/1907.10809)
- Hamze, F., Raymond, J., Pattison, C. A., Biswas, K. and Katzgraber, H. G.
  "Wishart planted ensemble: A tunably rugged pairwise Ising model with a
  first-order phase transition." Phys. Rev. E 101, 052102 (2020).
  [arXiv:1906.00275](https://arxiv.org/abs/1906.00275)
