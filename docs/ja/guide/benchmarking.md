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
                                       # ビット単位で一致する (各実行が自分のシードを導出する)。
                                       # ただし max_duration_secs で打ち切られる実行は除く

[[instances]]
path = "data/instances/max_cut/G*"     # ファイルパスまたは glob (Gset のファイルには拡張子がない)
problem = "MaxCut"                     # MaxCut | Qubo | Sat | Tsp | VertexCover | JobShop | FlowShop | Qap | BinPacking | Vrp | GraphColoring
# initial_solution = [0, 3, 5]        # 任意。すべての実行をここから始める (下を参照)

[[heuristics]]
kind = "LocalSearch"                   # 下の kind 一覧を参照
neighbor = "Flip"                      # Flip | Swap | TwoOpt | Relocate
[heuristics.stop_condition]
max_iteration = 100_000                # フィールドは任意の組み合わせ。どれか一つを満たせば止まる
max_duration_secs = 30.0
max_failed_update = 5_000
```

`[[instances]]` と `[[heuristics]]` はそれぞれ複数書けます。ランナーはその直積を実行します。

`max_duration_secs` による停止は、時計がその時間に達したときの iteration で終わり、その位置は再実行のたびに変わります。
そのためシードを固定しても、時間で止まる実行が同じ結果を返すのは、どちらの停止よりも前に最良解を見つけていた場合だけです。
各実行に時間予算を与えつつビット単位で再実行したいときは、後と同じように実行どうしがコアを分け合う状態で、config を一度 `max_duration_secs` のまま実行します。そのうえで、各実行が進んだ iteration 数である `n_accepted + n_rejected` の最小値以下の `max_iteration` に置き換えてください。

各実行のシードは、マスターシードと 3 つの位置から導出されます。`[[instances]]` が一致したファイルを順に並べたときのインスタンスの位置 (glob の一致はパス順)、`[[heuristics]]` の中でのヒューリスティクスの位置、実行の番号です。
パスは使わないので、同じインスタンスを別のパス (たとえば新しい一時ファイル) から読んでも、実行は同じ結果を繰り返します。
インスタンスやヒューリスティクスを途中に挿入したり、glob が一致するファイルが 1 つ増えたりすると、それより後ろのシードがずれます。
同じ config で別の実行が欲しいときは、別の `seed` を与えてください。

`initial_solution` を書くと、そのインスタンスのすべての実行をランダムな解ではなくその解から始めます。あるベンチマークの最良解から次のベンチマークを続けられます。
書き方はレポートの `solution` と同じで ([出力レポート](#output-report) の符号化の表を参照)、インスタンスに合わない解を与えると各実行が `initial solution error: …` で失敗します。1 つのインスタンスを指すので、`path` が 2 つ以上のファイルに一致すると config エラーになります。
自前で新しい解を引くヒューリスティクス (`PopulationAnnealing`、`ParallelTempering`、`GeneticAlgorithm`、最初のリスタート以降の `Restart`、`GreedyRandomizedConstruction`) では、最初の暫定解になるだけです。また、同じシードでもランダムな開始とは別の乱数を引きます。

`Benchmark::evaluate_solution(problem, path, solution)` は、同じ符号化の解を何も実行せずに採点します。返すのは、その解で終わった実行が報告する `best_objective` です。

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
| [`ParallelTempering`](../heuristics/parallel_tempering.md#benchmark-config) | すべて |
| [`Sequential` / `Iterated` / `VariableNeighborhoodSearch` / `Restart`](../heuristics/meta.md#benchmark-config) | すべて |
| [`GeneticAlgorithm`](../heuristics/genetic_algorithm.md#benchmark-config) | すべて |
| [`BreakoutLocalSearch`](../heuristics/breakout_local_search.md#benchmark-config) | 現在は MaxCut |
| [`LinKernighanHelsgaun`](../heuristics/lkh.md#benchmark-config) | TSP のみ |
| [`WalkSat`](../heuristics/walksat.md#benchmark-config) | SAT のみ |
| [`AdaptiveLargeNeighborhoodSearch`](../heuristics/alns.md#benchmark-config) | VRP, TSP, FlowShop, BinPacking |
| [`IteratedGreedy`](../heuristics/iterated_greedy.md#benchmark-config) | FlowShop のみ |
| [`GreedyRandomizedConstruction`](../heuristics/greedy_randomized_construction.md#benchmark-config) | VRP, TSP, FlowShop, BinPacking |
| [`HybridGeneticSearch`](../heuristics/hgs.md#benchmark-config) | VRP のみ |
| [`BranchAndBound`](../heuristics/branch_and_bound.md#benchmark-config) | MaxCut, Qubo, Sat, VertexCover |

未知の kind や必須フィールドの欠落は、実行が始まる前のパースの時点でエラーになります。

[VRP](../problems/vrp.md) のインスタンスは CVRPLIB か、パスが `.toml` で終わるときは異種フリートです。
上の表で片方に使える `kind` は、もう片方でもそのまま使えます。

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
| JobShop, FlowShop | `Swap`, `Relocate` |
| Qap | `Swap` |
| BinPacking | `Relocate`, `Swap` |
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
        ├── proven_optimal / dual_bound: Option  (BranchAndBound のみ)
        ├── solution: Vec<usize>                 (0 始まり、問題ごとの符号化)
        └── trajectory: Vec<(f64, f64)>          (改善ごとの (elapsed_secs, objective))
```

`trajectory` は anytime 曲線で、問題の最適化の向きに単調です。ベンチマークビューアが描くのはこれです。
エラーで終わった実行では `Option` のフィールドは出力されません。
`proven_optimal` と `dual_bound` が出るのはトップレベルの `BranchAndBound` だけです。前者はその実行が `best_objective` の最適性を証明したかどうか、後者はまだ除外されていない解が到達しうる最良の目的関数値で、証明が済むと `best_objective` に一致します。

解の符号化は次のとおりです。

| 問題 | `solution` |
|---|---|
| MaxCut | カットの片側にある頂点のインデックス |
| QUBO | 1 に設定された変数のインデックス |
| SAT | `true` に設定された変数のインデックス |
| TSP | 都市の訪問順 |
| VertexCover | 被覆に含まれる頂点のインデックス |
| JobShop | 作業の列 (ジョブのインデックスを、それぞれ `n_machines` 回ずつ並べたもの) |
| FlowShop | ジョブの順序 |
| Qap | 各施設の場所 |
| BinPacking | 各品目の bin |
| VRP | すべてのルートをデポ (`0`) を区切りにして平たくしたもの。`0, r0…, 0, r1…, 0`。車両が 1 種類なら空いているスロットは省略します。複数の車種がある車両群ではすべてのスロットを並べ、空いているものは `0, 0` になるので、どのルートをどの車種が走ったかが符号化に残ります |
