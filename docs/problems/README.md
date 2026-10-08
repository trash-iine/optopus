# Problems

**API:** [`optopus::problem`](../api/optopus/problem/index.html)

Each built-in problem implements `ProblemTrait` plus enough additional traits
to plug into every relevant heuristic. Its moves run under every base
heuristic, as the [Heuristics](../heuristics/README.md#base) page lists.

| Problem | Direction | Solution | Neighbors | Crossover | Loader |
|---|---|---|---|---|---|
| [MaxCut](max_cut.md) | Maximize | `MaxCutSolution` | Flip / Swap | `MaxCutUniformCrossover` | `MaxCut::load_file` |
| [QUBO](qubo.md) | Minimize | `QuboSolution` | Flip / Swap | `QuboUniformCrossover` | `Qubo::load_file` |
| [MaxSAT](sat.md) | Maximize | `SatSolution` | Flip / Swap | `SatUniformCrossover` | `Sat::load_file` (DIMACS CNF) |
| [TSP](tsp.md) | Minimize | `TspSolution` | TwoOpt / Relocate | `TspOrderCrossover` | `Tsp::load_file` (TSPLIB) |
| [Vertex Cover](vertex_cover.md) | Minimize | `VertexCoverSolution` | Flip / Swap | `VertexCoverUniformCrossover` | `VertexCover::load_file` |
| [Job Shop Scheduling](job_shop_scheduling.md) | Minimize | `JobShopSolution` | Swap / Relocate | `JobShopPpxCrossover` | `JobShopScheduling::load_file` |
| [VRP](vrp.md) | Minimize | `VrpSolution` | Relocate / Swap / TwoOpt | `VrpOrderCrossover` | `Vrp::load_file` (CVRPLIB, or a TOML fleet file) |
| [Graph Coloring](graph_coloring.md) | Minimize | `GraphColoringSolution` | Flip (recolor) / Swap | `GraphColoringUniformCrossover` | `GraphColoring::load_file` |
| [Formula](formula.md) | Configurable | `FormulaSolution` | Change / Swap / Reverse | `IntCrossover` | (none, built from an `Expr`) |
| [Integer variables](integer.md) | Configurable | `IntSolution` | Change / Swap / Reverse | `IntCrossover` | (none, built from a closure) |

Type names are exported from `optopus::prelude`. See each page for the
`Solution` struct fields, the file format, and which optional traits the
problem implements.

One problem also ships an exact instance reduction rather than a search:
[MaxCutKernel](max_cut_kernel.md) shrinks a sparse MaxCut instance by rules
that provably preserve the optimum, and any heuristic can search the result.
