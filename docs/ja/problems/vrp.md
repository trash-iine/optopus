# VRP

**API:** [`Vrp`](../../api/optopus/problem/vrp/struct.Vrp.html)

デポ (ノード `0`) と `n` 人の顧客 `1, ..., n` があり、各顧客は整数の需要 `q_i`、サービス時間を持ち、
各組の間に距離 `d(i, j)` があります。顧客を担当するのは一つ以上の
[車両タイプ](../../api/optopus/problem/vrp/struct.VehicleType.html) からなるフリートで、
タイプごとに容量、速度、固定費と距離比例費、最小使用台数、ルート時間の上限を持ちます。
各タイプは `max_count` 台分の車両 *スロット* を提供します。顧客をスロット 1 本につき 1 本のルートに分割し、
すべての顧客がちょうど一度ずつ訪問されるようにします。最小化するのは、時間の項 (全ルートの所要時間の合計か、最も長い 1 本)
とフリートの費用の和で、容量、ルート時間、最小使用台数の違反にはペナルティがかかります。

```text
route_time(slot) = route_distance(slot) / speed(type of slot) + Σ_{c in route} service_time(c)
time_component   = TotalTime => Σ_slots route_time(slot)  |  Makespan => max_slots route_time(slot)
objective = time_component
          + cost_weight · (Σ fixed_cost(used slots) + Σ variable_cost_per_distance(slot) · route_distance(slot))
          + penalty_weight · (overload + time_excess + min_count_shortfall)
```

Capacitated VRP (CVRP) は、速度 `1`、サービス時間なし、費用なし、ルート時間の上限なしの 1 タイプのインスタンスで、
通常のコンストラクタはすべてこれを作ります。その目的関数はよく知られた次の形に戻ります。

```text
objective = distance + penalty_weight · Σ_k max(0, load(R_k) − Q)
```

CVRP は TSP を一般化したもので、容量無制限の車両 1 台にすればちょうど TSP に戻ります。
また、物流やラストワンマイル配送の計画で広く使われる配送経路問題の族の基本形でもあります。

どの制約もソフトで、[Vertex Cover](vertex_cover.md) とまったく同じようにペナルティで扱います。
`penalty_weight` は目的関数のペナルティ以外の部分すべての上界に 1 を足した値なので、実行可能解が存在する限り、
ペナルティを加えた目的関数の最適解はすべて実行可能です。この上界はあえて緩く取ってあり、必要なのは厳密に支配することだけです。
積載超過と最小台数の不足は整数なので、違反があれば必ず重み 1 個分以上のコストになります。
ルート時間の超過は連続量なので、1 時間単位に満たない違反はその分だけ小さく評価されます。

## スロットは車両タイプに固定の連続ブロックで結び付く { #slots-are-bound-to-vehicle-types-in-fixed-contiguous-blocks }

どの解もスロット 1 本につきルートがちょうど 1 本で、`num_slots` はフリート全体の `Σ max_count` です。
各車両タイプはスロットの連続ブロックを持ちます。タイプ `0` がスロット `0..max_count[0]`、タイプ `1` が次のブロック、という具合で、
この対応は構築時に一度決まり、その後は変わりません。つまりスロットのタイプは *インスタンス* の一部であって解の一部ではなく、
顧客を別タイプのスロットへ移したり交換したりすることが、そのままその顧客の車両タイプの変更になります。

同じタイプの空きスロットどうしは区別がつかないので、新しいルートを開く move が見るのはそのタイプの最初の空きスロット 1 本だけです
(`Vrp::idle_representatives`)。ほかのスロットは同じルートに別のラベルを付けたものにしかなりません。

## 例 { #example }

CVRP のインスタンスで探索を実行し、各車両のルートを読み出します。

```rust
use optopus::prelude::*;

// ファイルから読むなら let vrp = Vrp::load_file("data/instances/vrp/demo16.vrp")?;
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
println!("total distance = {}", sol.total_distance());
for (vehicle, route) in sol.routes.iter().enumerate() {
    println!("vehicle {vehicle}: depot -> {route:?} -> depot"); // 顧客のインデックスだけ。デポは暗黙
}
```

最も近い整数に丸めた `EUC_2D` 距離 (CVRPLIB の慣習) を使うには、同じ引数で `Vrp::with_rounding` を使います。

異種フリートは `Vrp::with_fleet` で作ります。

```rust
use optopus::prelude::*;

let fleet = Vrp::with_fleet(
    "demo",
    vec![(0.0, 0.0), (2.0, 0.0), (0.0, 3.0), (-2.0, -1.0)], // [0] がデポ
    vec![0, 4, 3, 2],                                        // 需要。[0] は無視される
    vec![0.0, 1.0, 1.0, 0.5],                                // サービス時間。[0] は無視される
    vec![
        VehicleType::new("truck", 6, 1.0, 2)  // 容量、速度、max_count
            .with_costs(10.0, 0.1)            // fixed_cost、variable_cost_per_distance
            .with_min_count(1)
            .with_max_route_time(20.0),
        VehicleType::new("van", 3, 2.0, 1).with_costs(2.0, 0.4),
    ],
    ObjectiveMode::TotalTime,
    1.0,     // cost_weight
    false,   // rounded (EUC_2D で最も近い整数に丸めるか)
);

let mut state = SearchState::new(&fleet);
LocalSearch::<VrpRelocateNeighbor>::new(StopCondition::iterations(10_000))
    .run(&mut state)
    .unwrap();

let sol = &state.best_solution;
println!("objective = {}", sol.objective);
for (slot, route) in sol.routes.iter().enumerate() {
    let vt = fleet.vehicle_type_of_slot(slot);
    println!("slot {slot} ({}): depot -> {route:?} -> depot", vt.name);
}
```

`vehicle_types` は宣言順にスロットを並べます。ここでは `num_slots()` が `3` で (truck のスロット 2 本、続いて van のスロット 1 本)、
`type_of_slot(2) == 1` が van です。`VehicleType::new` は `(name, capacity, speed, max_count)` を取り、費用も最小台数もルート時間の上限もなしから始めます。
`with_costs` / `with_min_count` / `with_max_route_time` で残りを設定します。

### 距離の保持方法 { #distance-storage }

距離は
[`DistanceStore`](../../api/optopus/building_blocks/instance/struct.DistanceStore.html) に保持されます。
`Tsp` が使うのと同じストアです。

| コンストラクタ | 保持するもの | 使う場面 |
|---|---|---|
| `Vrp::new(...)`、`Vrp::with_rounding(...)`、`Vrp::with_fleet(...)` | ノード数が `Vrp::DIST_MATRIX_MAX_N` (2000) までなら `nodes × nodes` の全行列、それを超えるとノードあたり `Vrp::NEAREST_NEIGHBORS_ABOVE_CAP` (20) 個の近傍 | 通常の場合 |
| `Vrp::with_nearest_neighbors(name, coords, demands, capacity, num_vehicles, rounded, k)` | 各ノードの最近傍 `k` 個、`nodes × k` 個の距離 | 行列が載らない場合 |
| `Vrp::from_distance_matrix(name, matrix, demands, capacity, num_vehicles)` | 与えられた行列そのもの | 距離がユークリッドでない、または座標がない場合 |

最近傍だけを持つインスタンスでも、`distance(i, j)` はどの組についても答えます。
リストに入っていない組は座標から計算するので、値は全行列の場合と同じで、遠い組だけ余分にコストがかかります。
降下が作る granular な候補リストは `distance` を読むので、どのストアでも動きます。

`Vrp::from_distance_matrix` はノード `0` をデポとする `Vec<Vec<f64>>` を取り、正方でない行列は拒否します。
2-opt の gain は区間の反転を交換される 4 本の辺だけから計算するので、対称な行列を渡してください。
このインスタンスは座標を持たないので、`coordinates()` と `rounded()` は `None` を返します。

### 車両数 { #fleet-size }

CVRP のコンストラクタに `num_vehicles = 0` を渡すと、first-fit-decreasing に 10% の余裕を足した台数になります。
余裕を持たせるのは、距離が最適な解が最小台数より数台多く使うことがよくあるからです。
遠い顧客を単独のルートに切り出すほうが、寄り道するより安くつくことがあります。
使わない車両にコストはかかりませんが、足りない車両は最適解そのものを失わせます。

## 目的関数のモード { #objective-mode }

[`ObjectiveMode`](../../api/optopus/problem/vrp/enum.ObjectiveMode.html) は、スロットごとの `route_time` を
時間の項がどう集計するかを選びます。

- `TotalTime`。全スロットにわたる和で、通常の配送経路問題の目的関数です。
- `Makespan`。最も長い 1 本のルートの所要時間で、「最後の車両が戻るのはいつか」という min-max の目的関数です。

`Makespan` は加法的ではないので、この下での探索は move が変える量を足し合わせて価格付けすることができません。
それでもどのヒューリスティクスも動きます。増分更新は move を価格付けする時点で新しい最大値を解決し、そのコストは顧客数ではなくフリートの台数で抑えられます。

## 解 { #solution }

[`VrpSolution`](../../api/optopus/problem/vrp/struct.VrpSolution.html) は `routes` にスロット 1 本につき 1 エントリを持ち
(`routes.len() == num_slots()`。使っていない車両は空のルートです)、各ルートは訪れる顧客 (`1..=n`) だけを並べ、
デポは両端に暗黙に置かれます。加えて、目的関数に現れる量をスロットごとまたは合計でキャッシュしています
(一覧は API ページにあります。`total_distance()` はルートの距離を合計します)。
`Vrp::solution_from_routes` は素の `Vec<Vec<usize>>` の分割からキャッシュをすべて計算し直すもので、
増分更新はすべてこれを基準に検証されています。`Vrp::validate_routes` は、分割がちょうど `num_slots()` 本のルートで
顧客 `1..=n` をちょうど一度ずつ訪れているかを確認します。

`Distance` (GA の多様性に使います) は broken-pairs の隣接カウントです。どの車両が走るかは見ないので、
同じ trip を別の車両タイプで走る二つの解は距離 `0` になります。ここでの多様性は trip の集合が違うことを指します。

| フィールド | 型 | 意味 |
|---|---|---|
| `routes` | `Vec<Vec<usize>>` | スロットごとのルート。顧客を訪問順に並べ、デポは両端に暗黙 |
| `objective` | `f64` | ペナルティ付きの目的関数。小さいほど良い |
| `total_cost` | `f64` | 使った車両の固定費と距離に比例する費用の合計 |
| `total_time` | `f64` | ルートの所要時間の合計 |
| `makespan` | `f64` | 最も長いルートの所要時間 |
| `overload` | `i64` | 容量を超えた需要のルートごとの合計。実行可能なら `0` |
| `time_excess` | `f64` | 車両ごとの上限を超えた所要時間の合計。実行可能なら `0` |

`Vrp::load_file` は CVRPLIB ファイルのノードを 0 から数え直すので、ファイルのノード `k` は `k - 1` になります。ファイルのノード 1 であるデポは `0` に、id が `k` の顧客は `routes` の中で `k - 1` になります。残りのキャッシュされたフィールドは rustdoc にあります。

## 近傍 { #neighbors }

| 型 | move | 範囲 |
|---|---|---|
| `VrpRelocateNeighbor` | 顧客を一人、別のスロットのある位置に移す。 | スロット間のみ |
| `VrpSwapNeighbor` | 二つのスロットの間で顧客を二人交換する。 | スロット間のみ |
| `VrpTwoOptNeighbor` | 一つのスロットのルートの中の区間を反転する。 | ルート内のみ |

スロットのタイプが違う相手へ顧客を移したり交換したりすると、その顧客は移動先のタイプで価格も時間も容量も付け直されます。
使用中のスロットを変えられるのは relocate だけなので、固定費、`used_count`、`min_count_shortfall` を動かすのも relocate だけです。

これらの move は gain に `penalty_weight` を組み込んでいます。
[HybridGeneticSearchForVrp](../heuristics/hgs.md) のように実行中にペナルティを調整する必要があるヒューリスティクスは、
問題が共有するルート機構を通して同じ編集を自前のペナルティで価格付けします。

組み込みの move はすべて `Evaluate` と `EnabledTabu` を実装しているので、`SimulatedAnnealing` や `TabuSearch` を含め、move の型を取るどのヒューリスティクスでも動きます。

## 交叉 { #crossover }

- `VrpOrderCrossover`。両親を大きな巡回路に平たくし、Order Crossover (`building_blocks::representation::order_crossover`) を適用してから、
  [`split_giant_tour`](../../api/optopus/problem/vrp/fn.split_giant_tour.html)
  (Prins の Split) で子を `num_slots()` 本のルートに復号し直します。Split は、OX が作った顧客の順序に対して区切り位置を最適に選ぶ動的計画法で、
  各ルートはそれを走るスロットの車両タイプで価格付けされます。
  そのため子は、同じ順序のほかのどんな区切り方 (親自身が持っていた分割を含む) よりも悪くなりません。

ここでは Split を上の固定された `penalty_weight` の下で復号するので、動的計画法が最小化するのはちょうど子の `objective` です。
例外は二つあり、どちらも明記してあります。最小台数の不足は分割全体の性質なので区切りごとには価格付けしません。
また `Makespan` の下では、ルートにわたる最大値とルートにわたる和を足したものはスカラーの最短路にならないので、
加法的な total time を代理として復号します。
このオペレータと [HybridGeneticSearchForVrp](../heuristics/hgs.md) の組み換えの段階を分けているのはこのペナルティだけです。
HGS は探索の進行に合わせて調整し直すペナルティで同じ復号器を動かします (そしてその後に granular な局所降下を行います)。

## Ruin and recreate { #ruin-and-recreate }

`Vrp` は顧客を要素、車両スロットをコンテナとして `Ruinable` を実装しているので、
[AdaptiveLargeNeighborhoodSearch](../heuristics/alns.md) がどのフリートの上でも動きます。
`alns_for_vrp` はこの探索を `AnchoredRouteDescent` と組み合わせます。これは ruin が再挿入したばかりの顧客の周りで走らせる
granular なルート降下で、その粒度、リング、パス数は ALNS のページで説明している builder で設定します。

## ファイル形式 { #file-formats }

`Vrp::load_file` は、パスが `.toml` で終わるときは TOML のフリート形式を、それ以外は CVRPLIB を読みます。

### CVRPLIB { #cvrplib }

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
出来上がるのは `Vrp::new` が作るのと同じ 1 タイプのインスタンスです。

```rust
use optopus::prelude::*;

let vrp = Vrp::load_file("data/instances/vrp/X-n101-k25.vrp")?;
# Ok::<(), optopus::error::OptError>(())
```

### TOML { #toml }

独自のスキーマで、どのテーブルでも未知のキーを拒否します。任意キーの綴り間違いは、黙って無視される既定値ではなくパースエラーになります。

```toml
name = "demo_fleet"                 # 任意。省略するとファイル名の語幹
objective_mode = "TotalTime"        # 必須: "TotalTime" | "Makespan"
cost_weight = 1.0                   # 任意。既定値 1.0
# rounded を省略すると false (EUC_2D の丸めをしない素のユークリッド距離)

[depot]
x = 0.0
y = 0.0

[[vehicle_types]]
name = "truck"
capacity = 6
speed = 1.0
fixed_cost = 10.0                   # 任意。既定値 0.0
variable_cost_per_distance = 0.1    # 任意。既定値 0.0
min_count = 1                       # 任意。既定値 0
max_count = 2                       # 必須: このタイプがスロットを 2 本提供する
max_route_time = 40.0               # 任意。既定値は上限なし (キーごと省略する)

[[vehicle_types]]
name = "van"
capacity = 3
speed = 2.0
fixed_cost = 2.0
variable_cost_per_distance = 0.4
max_count = 2                       # 必須: このタイプがスロットをさらに 2 本提供する
# min_count と max_route_time を省略すると 0 と上限なし

[[customers]]
id = 1                              # id はちょうど 1..=n。抜けも重複も不可
x = 4.0
y = 1.0
demand = 2
service_time = 1.0                  # 任意。既定値 0.0

[[customers]]
id = 2
x = 5.0
y = -2.0
demand = 3
# service_time を省略すると 0.0
```

`vehicle_types` は宣言順にスロットを並べます。このフリートは truck のスロット 2 本 (`0`、`1`) のあとに van のスロット 2 本 (`2`、`3`) で、
`num_slots() == 4` です。リポジトリに入っている `data/instances/vrp/demo_fleet.toml` は、このフリートで顧客 6 人を扱うインスタンスです。

```rust
use optopus::prelude::*;

let fleet = Vrp::load_file("data/instances/vrp/demo_fleet.toml")?;
# Ok::<(), optopus::error::OptError>(())
```

ベンチマークの config では、どちらの形式も `problem = "Vrp"` です。

```toml
[[instances]]
path = "data/instances/vrp/demo_fleet.toml"
problem = "Vrp"

[[heuristics]]
kind = "LocalSearch"
neighbor = "Relocate"          # Relocate | Swap | TwoOpt
[heuristics.stop_condition]
max_iteration = 2000
```

## ヒューリスティクス { #heuristics }

CVRP で動くヒューリスティクスは、どれも異種フリートで動きます。三つの近傍を使う汎用のもの、
`VrpOrderCrossover` を使う `GeneticAlgorithm`、そして問題専用の二つ、
[AdaptiveLargeNeighborhoodSearch](../heuristics/alns.md) と
[HybridGeneticSearch](../heuristics/hgs.md) です。後者二つのルート機構は、どの編集もそれが触れるスロットの車両タイプで価格付けします。

## 参考文献 { #references }

- Uchoa, E., Pecin, D., Pessoa, A., Poggi, M., Vidal, T., and Subramanian, A.
  "New Benchmark Instances for the Capacitated Vehicle Routing Problem."
  European Journal of Operational Research, 257(3), 845-858, 2017.
  (CVRPLIB の "X" セットです。)
- Prins, C. "A Simple and Effective Evolutionary Algorithm for the Vehicle
  Routing Problem." Computers & Operations Research, 31(12), 1985-2002, 2004.
  (Split です。)
