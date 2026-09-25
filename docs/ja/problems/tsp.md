# TSP

**API:** [`Tsp`](../../api/optopus/problem/tsp/struct.Tsp.html)

`n` 個の都市と、都市 `i` と `j` の距離 `d(i, j)` が与えられたとき、すべての都市をちょうど一度ずつ訪れて出発点に戻る最短の巡回路を求めます。
巡回路は `{1, ..., n}` の置換 `π` で、`π(k)` は `k` 番目に訪れる都市です。このハミルトン閉路の総延長を最小化します。

```text
minimize  Σ_{k=1}^{n} d(π(k), π(k mod n + 1))    (π a permutation of the n cities)
```

このクレートの `Tsp` は距離そのものを
[`DistanceStore`](../../api/optopus/common/distance_store/struct.DistanceStore.html) に保持します。
`Vrp` が使うのと同じストアです。インスタンスは二次元座標と TSPLIB の標準的な距離式の一つから作るか
(下の [辺の重みの種類](#edge-weight-types) を参照)、距離行列を直接与えて作ります。
どの距離をメモリに保持するかは構築時に選びます ([距離の保持方法](#distance-storage) を参照)。

## 例 { #example }

探索を実行し、見つかった訪問順を読み出します。

```rust
use optopus::prelude::*;

let tsp = Tsp::new(
    "demo".to_string(),
    vec![(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)],  // 正方形の配置
);
let mut state = SearchState::new(&tsp);
LocalSearch::<TspTwoOptNeighbor>::new(StopCondition::iterations(10_000))
    .run(&mut state)
    .unwrap();

let sol = &state.best_solution;
println!("tour length = {}", sol.objective);
println!("visiting order = {:?}", sol.tour); // 訪問順に並んだ都市のインデックス
```

`Tsp::new` の既定は `EdgeWeightType::Continuous` です。下の距離式から選ぶには
`Tsp::with_edge_weight_type` を使います。

## 距離の保持方法 { #distance-storage }

| コンストラクタ | 保持するもの | 使う場面 |
|---|---|---|
| `Tsp::new(name, coords)`, `Tsp::with_edge_weight_type(name, coords, ewt)` | `n × n` の行列全体、`8n²` バイト | `n` がせいぜい数千 |
| `Tsp::with_nearest_neighbors(name, coords, ewt, k)` | 各都市の近い順に `k` 個の近傍、`n × k` 個の距離 | 行列が収まらない |
| `Tsp::from_distance_matrix(name, matrix)` | 与えた行列そのもの | 距離がユークリッド距離でない、または座標がない |

近傍リスト方式のインスタンスでも、すべての組について `distance(i, j)` に答えます。
リストにない組は座標から計算するので、値は行列全体を持つ場合と同じで、遠い組のコストが高くなるだけです。
Lin-Kernighan とアンカー付きの巡回路降下が求める候補リストは、`k` が足りていれば保持している行からそのまま取り出されます。

`Tsp::from_distance_matrix` は `Vec<Vec<f64>>` を受け取り、正方でない行列を拒否します。
このインスタンスには座標がないので、`coordinates()` と `edge_weight_type()` は `None` を返します。
対称な行列を与えてください。近傍は 2-opt の move を交換する4本の辺から評価しますが、これは区間を反転しても長さが変わらない場合にしか成り立ちません。
非対称な行列では、報告される gain が巡回路の長さからずれていきます。

```rust
use optopus::prelude::*;

let matrix = vec![
    vec![0.0, 3.0, 7.0],
    vec![3.0, 0.0, 5.0],
    vec![7.0, 5.0, 0.0],
];
let tsp = Tsp::from_distance_matrix("triangle".to_string(), matrix)?;
```

`Tsp::load_file` は都市数が `Tsp::DIST_MATRIX_MAX_N` (2000) 以下のファイルでは行列全体を保持し、
それを超えると都市ごとに `Tsp::NEAREST_NEIGHBORS_ABOVE_CAP` 個 (20) の近傍に切り替えます。

## 解 { #solution }

[`TspSolution`](../../api/optopus/problem/tsp/struct.TspSolution.html) は
上の定義の置換 `π` を `tour` として (`tour[k]` は `π(k)`、つまり `k` 番目に訪れる都市)、
巡回路の長さ `Σ d(π(k), π(k+1))` を `objective` として持ちます。

## 近傍 { #neighbors }

| 型 | move | 反復コスト |
|---|---|---|
| `TspTwoOptNeighbor` | 2-opt。二本の辺の間にある巡回路の区間を反転する。 | `iter + 1` |
| `TspRelocateNeighbor` | 都市を一つ取り除き、別の位置に挿入し直す。 | `iter + 1` |

## 交叉 { #crossover }

- `TspOrderCrossover`。Order Crossover (OX) です。一方の親から連続する区間をコピーし、残りの位置をもう一方の親の順序で埋めます。

## Ruin and recreate { #ruin-and-recreate }

`Tsp` は都市を要素、巡回路を唯一のコンテナとして `Ruinable` を実装しているので、
[AdaptiveLargeNeighborhoodSearch](../heuristics/alns.md) がその上で動きます。
`alns_for_tsp` はこの探索を `AnchoredTourDescent` と組み合わせます。これは、ruin が再挿入したばかりの都市とその近傍に対する
Or-opt と 2-opt の降下で、その粒度、リング、パス数は ALNS のページで説明している builder で設定します。

## 辺の重みの種類 { #edge-weight-types }

`EdgeWeightType` で距離式を選びます。

| 種類 | 式 | TSPLIB のキー |
|---|---|---|
| `Continuous` | 素のユークリッド距離 (丸めなし) | (`new` の既定) |
| `Euc2d` | `nint(sqrt(dx² + dy²))` | `EUC_2D` |
| `Ceil2d` | `ceil(sqrt(dx² + dy²))` | `CEIL_2D` |
| `Att` | TSPLIB の擬似ユークリッド距離 | `ATT` |
| `Geo` | TSPLIB の大円距離 (DDD.MM → ラジアン、R = 6378.388 km) | `GEO` |

## ファイル形式 (TSPLIB) { #file-format-tsplib }

```text
NAME: <name>
TYPE: TSP
COMMENT: ...
DIMENSION: N
EDGE_WEIGHT_TYPE: EUC_2D | CEIL_2D | ATT | GEO
NODE_COORD_SECTION
1 x1 y1
2 x2 y2
...
EOF
```

`TYPE`、`COMMENT` などの未知のヘッダキーは読み飛ばされます。ヘッダキーの順序は任意です。
座標の行は 1 始まりで、内部で 0 始まりに変換されます。重みの種類 `EXPLICIT` には対応していません。

```rust
use optopus::prelude::*;

let tsp = Tsp::load_file("data/instances/tsp/att48.tsp")?;
```

## 参考文献 { #references }

- Reinelt, G. "TSPLIB, A Traveling Salesman Problem Library." *ORSA Journal
  on Computing*, 3(4), 376-384, 1991. (ファイル形式と標準インスタンス集合を定義しています。)
