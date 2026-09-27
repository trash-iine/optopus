# Optopus ドキュメント

組合せ最適化問題のためのメタヒューリスティクス最適化ライブラリです。Rust で書かれています。

- [API リファレンス (rustdoc)](../api/optopus/index.html)。公開されているすべての型、トレイト、メソッドのシグネチャとドキュメントコメントがあります。API リファレンスは英語のみです。
- [ベンチマークビューア](../benchmarks/viewer.html)。対応するすべての問題について、ヒューリスティクスを横並びで比較できます。絞り込みと並べ替えができます。

## はじめに

- [クイックスタート](quickstart.md)。最小の実行例とファイルローダ。
- [基本概念](concepts.md)。設計方針、三つのユースケース、主要なパターン。
- [SearchState](search_state.md)。状態が何を保持するか、ヒューリスティクスがそれをどう進めるか、サブランのクローン方式、縮約をまたぐ方法。
- [コアトレイト](traits.md)。最低限必要なトレイトと、ヒューリスティクスごとに追加で必要なトレイト。

## ガイド

- [停止条件](guide/stop_conditions.md)
- [ベンチマーク](guide/benchmarking.md)。TOML スキーマと CLI
- [エラー処理](guide/error_handling.md)
- [独自の問題を定義する](guide/custom_problem.md)
- [整数変数で問題を書く](guide/integer_modeling.md)。組み込みの問題をすべて `IntegerProblem` で書く
- [独自のヒューリスティクスを定義する](guide/custom_heuristic.md)
- [学習した摂動方策で BLS を動かす](guide/learned_perturbation.md)

## リファレンス

### 問題

- [概要](problems/README.md)
- [MaxCut](problems/max_cut.md) と、その厳密な [kernelization](problems/max_cut_kernel.md)
- [QUBO](problems/qubo.md)
- [MaxSAT](problems/sat.md)
- [TSP](problems/tsp.md)
- [Vertex Cover](problems/vertex_cover.md)
- [Job Shop Scheduling](problems/job_shop_scheduling.md)
- [CVRP](problems/vrp.md)
- [Graph Coloring](problems/graph_coloring.md)
- [Formula](problems/formula.md)

### ヒューリスティクス

- [概要](heuristics/README.md)
- [Local Search](heuristics/local_search.md)
- [Simulated Annealing](heuristics/simulated_annealing.md) (Bang-Bang 版を含む)
- [Late Acceptance Hill Climbing](heuristics/late_acceptance.md)
- [Tabu Search](heuristics/tabu_search.md)
- [Random Walk](heuristics/random_walk.md)
- [Beam Search](heuristics/beam_search.md)
- [Reinforcement Learning Search](heuristics/rl_search.md)
- [Genetic Algorithm](heuristics/genetic_algorithm.md) (`Crossover` トレイトを含む)
- [Population Annealing](heuristics/population_annealing.md)
- [Adaptive Large Neighborhood Search](heuristics/alns.md)
- [Breakout Local Search](heuristics/breakout_local_search.md)
- [Meta-heuristics](heuristics/meta.md) (Sequential, Iterated (ILS), VNS, Restart)
- [Lin-Kernighan-Helsgaun (TSP)](heuristics/lkh.md)
- [WalkSAT (MaxSAT)](heuristics/walksat.md)
- [Hybrid Genetic Search (CVRP)](heuristics/hgs.md)
