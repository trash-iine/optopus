# ベンチマーク

**API:** [`BenchmarkConfig`](../../api/optopus/benchmark/struct.BenchmarkConfig.html)

Optopus には CLI のベンチマークランナーが付いています。TOML の設定を受け取り、各ヒューリスティクスを各インスタンスで N 回並列に実行し、
TOML のレポートを書き出します。

## CLI

```sh
cargo run --release -- path/to/config.toml
```

出力は `result/<config_stem>_<timestamp>.toml` に書かれます。

## 設定のスキーマ { #config-schema }

```toml
num_runs = 10                          # (インスタンス, ヒューリスティクス) の組ごとの繰り返し回数
seed = 42                              # 任意のマスターシード。設定すると再実行が
                                       # ビット単位で一致する (各実行が自分のシードを導出する)

[[instances]]
path = "data/instances/max_cut/G*"     # ファイルパスまたは glob (Gset のファイルには拡張子がない)
problem = "MaxCut"                     # MaxCut | Qubo | Sat | Tsp | VertexCover | JobShop | Vrp | GraphColoring

[[heuristics]]
kind = "LocalSearch"                   # 下の kind 一覧を参照
neighbor = "Flip"                      # Flip | Swap | TwoOpt | Relocate
[heuristics.stop_condition]
max_iteration = 100_000                # フィールドは任意の組み合わせ。どれか一つを満たせば止まる
max_duration_secs = 30.0
max_failed_update = 5_000
```

`[[instances]]` と `[[heuristics]]` はそれぞれ複数書けます。ランナーはその直積を実行します。

## ヒューリスティクスの kind { #heuristic-kinds }

下の値はどれも `[[heuristics]]` ブロックの `kind` タグとして使えます。それぞれのリンク先のアルゴリズムのページに、
受け取るフィールド (必須、任意、既定値) が書かれています。この表は索引にすぎません。

| `kind` | 対象 |
|---|---|
| [`LocalSearch`](../heuristics/local_search.md#benchmark-config) | すべて |
| [`TabuSearch`](../heuristics/tabu_search.md#benchmark-config) | すべて |
| [`SimulatedAnnealing`](../heuristics/simulated_annealing.md#benchmark-config) | すべて |
| [`LateAcceptanceHillClimbing`](../heuristics/late_acceptance.md#benchmark-config) | すべて |
| [`BeamSearch`](../heuristics/beam_search.md#benchmark-config) | すべて |
| [`RandomWalk`](../heuristics/random_walk.md#benchmark-config) | すべて |
| [`ReinforcementLearningSearch`](../heuristics/rl_search.md#benchmark-config) | すべて |
| [`PopulationAnnealing`](../heuristics/population_annealing.md#benchmark-config) | すべて |
| [`Sequential` / `Iterated` / `VariableNeighborhoodSearch` / `Restart`](../heuristics/meta.md#benchmark-config) | すべて |
| [`GeneticAlgorithm`](../heuristics/genetic_algorithm.md#benchmark-config) | すべて |
| [`BreakoutLocalSearch`](../heuristics/breakout_local_search.md#benchmark-config) | 現在は MaxCut |
| [`LinKernighanHelsgaun`](../heuristics/lkh.md#benchmark-config) | TSP のみ |
| [`WalkSat`](../heuristics/walksat.md#benchmark-config) | SAT のみ |
| [`AdaptiveLargeNeighborhoodSearch`](../heuristics/alns.md#benchmark-config) | VRP, TSP |
| [`HybridGeneticSearch`](../heuristics/hgs.md#benchmark-config) | VRP のみ |

未知の kind や必須フィールドの欠落は、実行が始まる前のパースの時点でエラーになります。

## すべての kind に共通のフィールド { #fields-shared-by-every-kind }

`stop_condition` は `max_iteration`、`max_duration_secs`、`max_failed_update` を任意に組み合わせて受け取り、
どれか一つを満たした時点でヒューリスティクスが止まります。これは [停止条件](stop_conditions.md) の builder の TOML 版で、
各上限が何を数えるかはそのページに書かれています。

`steps` はヒューリスティクスの表を入れ子にした配列です (`[[heuristics.steps]]`、下の例を参照)。
どの位置がどの役割を担うかは kind ごとに決まっていて、その kind のページに書かれています。

`neighbor` は問題ごとに決まります。

| 問題 | 使える neighbor |
|---|---|
| MaxCut, QUBO, SAT, VertexCover | `Flip`, `Swap` |
| TSP | `TwoOpt`, `Relocate` |
| JobShop | `Swap`, `Relocate` |
| VRP | `Relocate`, `Swap`, `TwoOpt` |

問題専用の kind (`BreakoutLocalSearch`、`LinKernighanHelsgaun`、
`AdaptiveLargeNeighborhoodSearch`、`WalkSat` など) は自前の move 集合を持つので、`neighbor` を取りません。

## 入れ子の例 (TOML で書く ILS) { #nested-example-ils-in-toml }

```toml
[[heuristics]]
kind = "Iterated"
[heuristics.stop_condition]
max_iteration = 1_000_000

[[heuristics.steps]]                   # 探索フェーズ
kind = "LocalSearch"
neighbor = "Flip"
[heuristics.steps.stop_condition]
max_failed_update = 1

[[heuristics.steps]]                   # 摂動フェーズ
kind = "RandomWalk"                    # 無条件のランダム move = ランダム化のキック
neighbor = "Flip"
[heuristics.steps.stop_condition]
max_iteration = 200
```

## 出力レポート { #output-report }

各実行は `BenchmarkReport` を生成します。

```text
BenchmarkReport
├── timestamp: String
├── config_file: String
└── results: Vec<InstanceHeuristicResult>
    ├── instance_path: String
    ├── problem: ProblemKind
    ├── heuristic: HeuristicConfig
    ├── summary: Summary
    │   ├── num_successful_runs: usize
    │   ├── best_objective / avg_objective / worst_objective: f64
    │   ├── std_objective: f64                  (母標準偏差)
    │   ├── best_time_to_best_secs / avg_time_to_best_secs: f64
    │   ├── avg_total_time_secs: f64
    │   ├── avg_initial_objective / avg_improvement: Option<f64>
    │   └── avg_n_accepted / avg_n_rejected / avg_acceptance_rate
    │       / avg_n_best_updates: Option<f64>
    └── runs: Vec<SingleRunResult>
        ├── run_index: usize
        ├── status: String                       ("success" | "error: …")
        ├── best_objective: f64
        ├── best_iteration: u64
        ├── time_to_best_secs / total_time_secs: f64
        ├── initial_objective / improvement: Option<f64>
        ├── n_accepted / n_rejected / n_best_updates: Option<u64>
        ├── seed: Option<u64>                    (この実行に導出されたシード)
        ├── solution: Vec<usize>                 (0 始まり、問題ごとの符号化)
        └── trajectory: Vec<(f64, f64)>          (改善ごとの (elapsed_secs, objective))
```

`trajectory` は anytime 曲線で、問題の最適化の向きに単調です。ベンチマークビューアが描くのはこれです。
エラーで終わった実行では `Option` のフィールドは出力されません。

解の符号化は次のとおりです。

| 問題 | `solution` |
|---|---|
| MaxCut | カットの片側にある頂点のインデックス |
| QUBO | 1 に設定された変数のインデックス |
| SAT | `true` に設定された変数のインデックス |
| TSP | 都市の訪問順 |
| VertexCover | 被覆に含まれる頂点のインデックス |
| JobShop | 作業の列 (ジョブのインデックスを、それぞれ `n_machines` 回ずつ並べたもの) |
| VRP | すべてのルートをデポ (`0`) を区切りにして平たくしたもの。`0, r0…, 0, r1…, 0` (空のルートは省略) |
