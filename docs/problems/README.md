# Problems

Each built-in problem implements `ProblemTrait` plus enough additional traits
to plug into every relevant heuristic.

| Problem | Direction | Solution | Neighbors | Crossover | Loader |
|---|---|---|---|---|---|
| [MaxCut](max_cut.md) | Maximize | `MaxCutSolution` | Flip / Swap | `MaxCutUniformCrossover` | `Graph::load_from_file` |
| [QUBO](qubo.md) | Minimize | `QuboSolution` | Flip / Swap | `QuboUniformCrossover` | `Qubo::load_file` |
| [MaxSAT](sat.md) | Maximize | `SatSolution` | Flip / Swap | `SatUniformCrossover` | `Sat::load_file` (DIMACS CNF) |
| [TSP 2D](tsp.md) | Minimize | `TspSolution` | TwoOpt / Relocate | `TspOrderCrossover` | `TspWithCoordinates::load_file` (TSPLIB) |
| [Vertex Cover](vertex_cover.md) | Minimize | `VertexCoverSolution` | Flip / Swap | `VertexCoverUniformCrossover` | `Graph::load_from_file` |
| [Job Shop Scheduling](job_shop_scheduling.md) | Minimize | `JobShopSolution` | Swap / Relocate | `JobShopPpxCrossover` | `JobShopScheduling::load_file` |
| [Formula](formula.md) | Configurable | `FormulaSolution` | Flip / Swap | `FormulaUniformCrossover` | (none — built from `Expr` AST) |

Type names are exported from `optopus::prelude`. See each page for the
`Solution` struct fields, the file format, and which optional traits the
problem implements.
