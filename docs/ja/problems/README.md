# 問題

**API:** [`optopus::problem`](../../api/optopus/problem/index.html)

組み込みの各問題は `ProblemTrait` と、関係するすべてのヒューリスティクスに組み込めるだけの追加のトレイトを実装しています。

| 問題 | 向き | 解 | 近傍 | 交叉 | ローダ |
|---|---|---|---|---|---|
| [MaxCut](max_cut.md) | 最大化 | `MaxCutSolution` | Flip / Swap | `MaxCutUniformCrossover` | `Graph::load_from_file` |
| [QUBO](qubo.md) | 最小化 | `QuboSolution` | Flip / Swap | `QuboUniformCrossover` | `Qubo::load_file` |
| [MaxSAT](sat.md) | 最大化 | `SatSolution` | Flip / Swap | `SatUniformCrossover` | `Sat::load_file` (DIMACS CNF) |
| [TSP](tsp.md) | 最小化 | `TspSolution` | TwoOpt / Relocate | `TspOrderCrossover` | `Tsp::load_file` (TSPLIB) |
| [Vertex Cover](vertex_cover.md) | 最小化 | `VertexCoverSolution` | Flip / Swap | `VertexCoverUniformCrossover` | `Graph::load_from_file` |
| [Job Shop Scheduling](job_shop_scheduling.md) | 最小化 | `JobShopSolution` | Swap / Relocate | `JobShopPpxCrossover` | `JobShopScheduling::load_file` |
| [CVRP](vrp.md) | 最小化 | `VrpSolution` | Relocate / Swap / TwoOpt | `VrpOrderCrossover` | `Vrp::load_file` (CVRPLIB) |
| [Graph Coloring](graph_coloring.md) | 最小化 | `GraphColoringSolution` | Flip (再彩色) / Swap | `GraphColoringUniformCrossover` | `GraphColoring::load_file` |
| [Formula](formula.md) | 設定可能 | `FormulaSolution` | Flip / Swap | `FormulaUniformCrossover` | (なし、`Expr` の AST から作る) |
| [整数変数](integer.md) | 設定可能 | `IntSolution` | `IntChangeNeighbor` | (なし) | (なし、`IntegerProblem` を実装する) |

型名は `optopus::prelude` からエクスポートされています。`Solution` 構造体のフィールド、ファイル形式、
その問題が実装している任意のトレイトは各ページを参照してください。

探索ではなく厳密なインスタンス縮約を提供している問題も一つあります。
[MaxCutKernel](max_cut_kernel.md) は最適値を保つことが証明された規則で疎な MaxCut インスタンスを縮め、
その結果はどのヒューリスティクスでも探索できます。
