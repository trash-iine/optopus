# Optopus

A metaheuristic optimization library for combinatorial problems.
Provides a uniform interface for applying local search, tabu search, simulated
annealing, beam search, genetic algorithms, and more to MaxCut, QUBO, MaxSAT,
TSP, Vertex Cover, Job Shop Scheduling, VRP (homogeneous or heterogeneous
fleet), Graph Coloring, and user-defined problems.

## Quick Start

```bash
cargo run --example max_cut
```

```rust
use optopus::prelude::*;

// Or from a file, let mc = MaxCut::load_file("data/instances/max_cut/G1")?;
let mc = MaxCut::new(Graph::from_edges([
    (0, 1, 1.0),
    (0, 2, 1.0),
    (1, 2, 1.0),
]));

let mut state = SearchState::new(&mc);
let mut ls = LocalSearch::<MaxCutFlipNeighbor>::new(
    StopCondition::iterations(1_000_000),
);
ls.run(&mut state).unwrap();

println!("best cut = {}", state.best_solution.objective);
// state.best_solution.x holds the partition. docs/problems/max_cut.md lists every field.
```

Every heuristic stops on a `StopCondition`, such as
`StopCondition::iterations(100_000)`,
`StopCondition::duration(std::time::Duration::from_secs(10))` or
`StopCondition::failed_updates(1_000)`. How to combine them, and their TOML
form, are collected in [Stop conditions](docs/guide/stop_conditions.md).

See [`docs/quickstart.md`](docs/quickstart.md) for a longer tour, including
file-based loading. The documentation is also available in Japanese, starting
at [`docs/ja/README.md`](docs/ja/README.md).

## Supported Problems

| Problem | Type | Neighbors |
|---|---|---|
| [Max Cut](docs/problems/max_cut.md) | `MaxCut` | `MaxCutFlipNeighbor`, `MaxCutSwapNeighbor` |
| [QUBO](docs/problems/qubo.md) | `Qubo` | `QuboFlipNeighbor`, `QuboSwapNeighbor` |
| [MaxSAT](docs/problems/sat.md) | `Sat` | `SatFlipNeighbor`, `SatSwapNeighbor` |
| [TSP](docs/problems/tsp.md) | `Tsp` | `TspTwoOptNeighbor`, `TspRelocateNeighbor` |
| [Vertex Cover](docs/problems/vertex_cover.md) | `VertexCover` | `VertexCoverFlipNeighbor`, `VertexCoverSwapNeighbor` |
| [Job Shop Scheduling](docs/problems/job_shop_scheduling.md) | `JobShopScheduling` | `JobShopSwapNeighbor`, `JobShopRelocateNeighbor` |
| [VRP](docs/problems/vrp.md) | `Vrp` | `VrpRelocateNeighbor`, `VrpSwapNeighbor`, `VrpTwoOptNeighbor` |
| [Graph Coloring](docs/problems/graph_coloring.md) | `GraphColoring` | `GraphColoringRecolorNeighbor`, `GraphColoringSwapNeighbor` |
| [Formula](docs/problems/formula.md) | `FormulaProblem` | `IntChangeNeighbor`, `IntSwapNeighbor` |
| [Integer variables](docs/problems/integer.md) | `IntegerProblem` | `IntChangeNeighbor`, `IntSwapNeighbor`, `IntReverseNeighbor` |

## Available Heuristics

| Algorithm | Type |
|---|---|
| [Local Search](docs/heuristics/local_search.md) | `LocalSearch<N>` |
| [Simulated Annealing](docs/heuristics/simulated_annealing.md) | `SimulatedAnnealing<N>`, `BangBangSimulatedAnnealing<N>` |
| [Late Acceptance Hill Climbing](docs/heuristics/late_acceptance.md) | `LateAcceptanceHillClimbing<N>` |
| [Tabu Search](docs/heuristics/tabu_search.md) | `TabuSearch<N>` |
| [Random Walk](docs/heuristics/random_walk.md) | `RandomWalk<N>` |
| [Beam Search](docs/heuristics/beam_search.md) | `BeamSearch<P, N>` |
| [Reinforcement Learning Search](docs/heuristics/rl_search.md) | `ReinforcementLearningSearch<N>` |
| [Genetic Algorithm](docs/heuristics/genetic_algorithm.md) | `GeneticAlgorithm<P, C>` |
| [Population Annealing](docs/heuristics/population_annealing.md) | `PopulationAnnealing<P, N>` |
| [Adaptive Large Neighborhood Search](docs/heuristics/alns.md) | `AdaptiveLargeNeighborhoodSearch<P>` |
| [Breakout Local Search](docs/heuristics/breakout_local_search.md) | `BreakoutLocalSearch<P, S>` |
| [Sequential / Iterated / VNS / Restart](docs/heuristics/meta.md) | `Sequential<P>`, `Iterated<P>`, `VariableNeighborhoodSearch<P>`, `Restart<P>` |
| [Lin-Kernighan-Helsgaun (TSP)](docs/heuristics/lkh.md) | `LinKernighanHelsgaunForTsp` |
| [WalkSAT (MaxSAT)](docs/heuristics/walksat.md) | `WalkSatForSat` |
| [Hybrid Genetic Search (CVRP)](docs/heuristics/hgs.md) | `HybridGeneticSearchForVrp` |

## Benchmark CLI

The crate also builds a CLI benchmark runner: describe instances, heuristics,
and stop conditions in a TOML config, and get aggregated
best/avg/worst/std/time results.

```bash
cargo run --release -- path/to/config.toml
# report is written to result/<config_stem>_<timestamp>.toml
```

See [`docs/guide/benchmarking.md`](docs/guide/benchmarking.md) for the config
schema and [`docs/benchmarks/`](docs/benchmarks/) for results on standard
instance sets.

## Documentation

The rendered documentation site is at
<https://trash-iine.github.io/optopus/>:

- [API reference (rustdoc)](https://trash-iine.github.io/optopus/api/optopus/index.html)
  — full signatures and doc comments for every public type and trait
- [Benchmark viewer](https://trash-iine.github.io/optopus/benchmarks/viewer.html)
  — cross-heuristic results on standard instance sets

The same pages as Markdown in this repository, indexed in
[`docs/README.md`](docs/README.md):

- [`docs/quickstart.md`](docs/quickstart.md) — getting started, file loaders
- [`docs/concepts.md`](docs/concepts.md) — design philosophy and key patterns
- [`docs/search_state.md`](docs/search_state.md) — `SearchState`: the state every heuristic drives
- [`docs/traits.md`](docs/traits.md) — core traits reference
- [`docs/problems/`](docs/problems/) — supported problems
- [`docs/heuristics/`](docs/heuristics/) — available algorithms
- Guides in [`docs/guide/`](docs/guide/):
  - [`stop_conditions.md`](docs/guide/stop_conditions.md), iteration, time and stagnation limits
  - [`benchmarking.md`](docs/guide/benchmarking.md), the TOML config and the CLI runner
  - [`custom_problem.md`](docs/guide/custom_problem.md), defining your own problem and adding tabu to its moves
  - [`integer_modeling.md`](docs/guide/integer_modeling.md), writing a problem over integer variables without new types
  - [`custom_heuristic.md`](docs/guide/custom_heuristic.md), implementing `Heuristic` for your own algorithm
  - [`error_handling.md`](docs/guide/error_handling.md), the `OptError` cases
  - [`learned_perturbation.md`](docs/guide/learned_perturbation.md), driving BLS with a learned perturbation policy
- [`docs/benchmarks/`](docs/benchmarks/) — performance reports on standard instance sets

## Examples

```bash
cargo run --example max_cut             # MaxCut: LocalSearch and TabuSearch
cargo run --example beam_search         # MaxCut: BeamSearch
cargo run --example custom_problem      # define your own problem
cargo run --example custom_heuristic    # define your own heuristic
```

The last two are walked through in
[`docs/guide/custom_problem.md`](docs/guide/custom_problem.md) and
[`docs/guide/custom_heuristic.md`](docs/guide/custom_heuristic.md).

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or <https://www.apache.org/licenses/LICENSE-2.0>)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or <https://opensource.org/licenses/MIT>)

at your option.

Benchmark instance data under `data/instances/` has its own provenance and
licensing — see [`data/instances/README.md`](data/instances/README.md).

### Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in the work by you, as defined in the Apache-2.0 license, shall be
dual licensed as above, without any additional terms or conditions.
