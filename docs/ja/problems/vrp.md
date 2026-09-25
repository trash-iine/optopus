# CVRP

**API:** [`Vrp`](../../api/optopus/problem/vrp/struct.Vrp.html)

Capacitated Vehicle Routing Problem (CVRP) では、デポ (顧客 `0`) と `n` 人の顧客 `1, ..., n` があり、
各顧客は整数の需要 `q_i` を持ち、各組の間に距離 `d(i, j)` があります。顧客は共通の容量 `Q` を持つ `K` 台の同種の車両群が担当し、
各車両はデポから出発してデポに戻ります。顧客を高々 `K` 本のルート `R_1, ..., R_K` (それぞれ1台の車両が訪れる顧客の列) に分割し、
すべての顧客がちょうど一度ずつ訪問され、どのルートも需要の合計が `Q` を超えないようにします。総移動距離を最小化します。

```text
minimize  Σ_{k=1}^{K} distance(R_k)
subject to  R_1, …, R_K partition {1, …, n},  Σ_{i∈R_k} q_i ≤ Q for every route
```

CVRP は TSP を一般化したもので、容量無制限の車両1台にすればちょうど TSP に戻ります。
また、物流やラストワンマイル配送の計画で広く使われる配送経路問題の族の基本形でもあります。

容量はソフトな制約で、[Vertex Cover](vertex_cover.md) とまったく同じようにペナルティで扱います。
`penalty_weight` はどんな巡回路の長さよりも大きく選ばれているので、実行可能解が存在する限り、ペナルティを加えた目的関数の最適解はすべて実行可能です。

```text
objective = distance + penalty_weight · Σ_k max(0, load(R_k) − Q)
```

## 例 { #example }

探索を実行し、各車両のルートを読み出します。

```rust
use optopus::prelude::*;

let vrp = Vrp::new(
    "demo",
    vec![(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)],  // [0] がデポ
    vec![0, 1, 1, 1],                                      // 需要。[0] は無視される
    2,                                                      // 容量
    2,                                                      // num_vehicles (0 = 自動)
);
let mut state = SearchState::new(&vrp);
LocalSearch::<VrpRelocateNeighbor>::new(StopCondition::iterations(10_000))
    .run(&mut state)
    .unwrap();

let sol = &state.best_solution;
println!("total distance = {}", sol.distance);
for (vehicle, route) in sol.routes.iter().enumerate() {
    println!("vehicle {vehicle}: depot -> {route:?} -> depot"); // 顧客のインデックスだけ。デポは暗黙
}
```

最も近い整数に丸めた `EUC_2D` 距離 (CVRPLIB の慣習) を使うには、同じ引数で `Vrp::with_rounding` を使います。

### 距離の保持方法 { #distance-storage }

距離は
[`DistanceStore`](../../api/optopus/common/distance_store/struct.DistanceStore.html) に保持されます。
`Tsp` が使うのと同じストアです。

| コンストラクタ | 保持するもの | 使う場面 |
|---|---|---|
| `Vrp::new(...)`, `Vrp::with_rounding(...)` | `nodes × nodes` の行列全体、`8 nodes²` バイト | インスタンスのノード数がせいぜい数千 |
| `Vrp::with_nearest_neighbors(name, coords, demands, capacity, num_vehicles, rounded, k)` | 各ノードの近い順に `k` 個の近傍、`nodes × k` 個の距離 | 行列が収まらない |
| `Vrp::from_distance_matrix(name, matrix, demands, capacity, num_vehicles)` | 与えた行列そのもの | 距離がユークリッド距離でない、または座標がない |

近傍リスト方式のインスタンスでも、すべての組について `distance(i, j)` に答えます。
リストにない組は座標から計算するので、値は行列全体を持つ場合と同じで、遠い組のコストが高くなるだけです。
降下が作る granular な候補リストは `distance` を読むので、どのストアでも動きます。

`Vrp::from_distance_matrix` はノード `0` をデポとする `Vec<Vec<f64>>` を受け取り、正方でない行列を拒否します。
2-opt の gain は交換する4本の辺だけから区間の反転を評価するので、対称な行列を与えてください。
このインスタンスには座標がないので、`coordinates()` と `rounded()` は `None` を返します。

`Vrp::load_file` はノード数が `Vrp::DIST_MATRIX_MAX_N` (2000) 以下のファイルでは行列全体を保持し、
それを超えるとノードごとに `Vrp::NEAREST_NEIGHBORS_ABOVE_CAP` 個 (20) の近傍に切り替えます。

### 車両数 { #fleet-size }

`num_vehicles = 0` を渡すと、first-fit-decreasing に 10% の余裕を加えて車両数を決めます。
余裕を持たせるのは、距離が最適な解は最小台数より数台多く使うのが普通だからです。遠くの顧客を専用のルートに分けるほうが、そこへ寄り道するより安いことがあります。
使わない車両にコストはかかりませんが、車両が足りないと最適解を失います。

## 解 { #solution }

[`VrpSolution`](../../api/optopus/problem/vrp/struct.VrpSolution.html) は
上の定義の分割を表します。`routes` は `R_1, ..., R_K`、キャッシュされた `route_loads` はルートごとの
`Σ_{i∈R_k} q_i`、`distance` は `Σ_k distance(R_k)`、`overload` は `Σ_k max(0, load(R_k) − Q)`、
`objective` は上で定義したペナルティ付きの値です。
使わない車両は空のルート (距離 `0`) になります。各ルートは訪れる顧客 (`1..=n`) だけを並べ、デポ (インデックス `0`) は両端に暗黙に置かれます。

## 近傍 { #neighbors }

| 型 | move | 範囲 |
|---|---|---|
| `VrpRelocateNeighbor` | 顧客を一人、別のルートのある位置に移す。 | ルート間のみ |
| `VrpSwapNeighbor` | 二つのルートの間で顧客を二人交換する。 | ルート間のみ |
| `VrpTwoOptNeighbor` | 一つのルートの中の区間を反転する。 | ルート内のみ |

これらの move は gain に `penalty_weight` を組み込んでいることに注意してください。
[HybridGeneticSearchForVrp](../heuristics/hgs.md) のように実行中に容量ペナルティを調整する必要があるヒューリスティクスはこれらを使えず、
自前の move 評価を用意します。

## 交叉 { #crossover }

- `VrpOrderCrossover`。両親を大きな巡回路に平たくし、Order Crossover (`common::order_crossover`) を適用してから、
  [`split_giant_tour`](../../api/optopus/problem/vrp/fn.split_giant_tour.html)
  (Prins の Split) で子を `num_vehicles` 本のルートに復号し直します。Split は、OX が作った顧客の順序に対して区切り位置を最適に選ぶ動的計画法です。
  そのため子は、同じ順序のほかのどんな区切り方 (親自身が持っていた分割を含む) よりも悪くなりません。

ここでは Split を上の固定された `penalty_weight` の下で復号するので、動的計画法が最小化するのはちょうど子の `objective` です。
このオペレータと [HybridGeneticSearchForVrp](../heuristics/hgs.md) の組み換えの段階を分けているのはこのペナルティだけです。
HGS は探索の進行に合わせて調整し直す容量ペナルティで同じ復号器を動かします (そしてその後に granular な局所降下を行います)。

## Ruin and recreate { #ruin-and-recreate }

`Vrp` は顧客を要素、車両をコンテナ、容量をそれらが奪い合う資源として `Ruinable` を実装しているので、
[AdaptiveLargeNeighborhoodSearch](../heuristics/alns.md) がその上で動きます。
`alns_for_vrp` はこの探索を `AnchoredRouteDescent` と組み合わせます。これは ruin が再挿入したばかりの顧客の周りで走らせる
granular なルート降下で、その粒度、リング、パス数は ALNS のページで説明している builder で設定します。

## ファイル形式 (CVRPLIB) { #file-format-cvrplib }

```text
NAME : <name>
COMMENT : (... Min no of trucks: K ...)
TYPE : CVRP
DIMENSION : N
EDGE_WEIGHT_TYPE : EUC_2D
CAPACITY : Q
NODE_COORD_SECTION
1 x1 y1
...
DEMAND_SECTION
1 0
...
DEPOT_SECTION
1
-1
EOF
```

対応しているのは `EUC_2D` だけで、常に最も近い整数に丸めます。`COMMENT` に `No of trucks: K` があればそれを読み、
なければ上で説明したとおりに車両数を決めます。`DEPOT_SECTION` で指定されたノードはインデックス `0` に振り直されます。

```rust
use optopus::prelude::*;

let vrp = Vrp::load_file("data/instances/vrp/X-n101-k25.vrp")?;
```

## 参考文献 { #references }

- Uchoa, E., Pecin, D., Pessoa, A., Poggi, M., Vidal, T., and Subramanian, A.
  "New Benchmark Instances for the Capacitated Vehicle Routing Problem."
  European Journal of Operational Research, 257(3), 845-858, 2017.
  (CVRPLIB の "X" セットです。)
- Prins, C. "A Simple and Effective Evolutionary Algorithm for the Vehicle
  Routing Problem." Computers & Operations Research, 31(12), 1985-2002, 2004.
  (Split です。)
