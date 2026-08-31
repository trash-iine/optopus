# Optopus

A metaheuristic optimization library for combinatorial problems.
Provides a uniform interface for applying local search, tabu search, simulated
annealing, beam search, genetic algorithms, and more to MaxCut, QUBO, MaxSAT,
TSP, Vertex Cover, Job Shop Scheduling, CVRP, and user-defined problems.

## Quick Start

```bash
cargo run --example max_cut
```

```rust
use optopus::prelude::*;

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
```

See [`docs/quickstart.md`](docs/quickstart.md) for a longer tour, including
file-based loading.

## Supported Problems

| Problem | Type | Neighbors |
|---|---|---|
| [Max Cut](docs/problems/max_cut.md) | `MaxCut` | `MaxCutFlipNeighbor`, `MaxCutSwapNeighbor` |
| [QUBO](docs/problems/qubo.md) | `Qubo` | `QuboFlipNeighbor`, `QuboSwapNeighbor` |
| [MaxSAT](docs/problems/sat.md) | `Sat` | `SatFlipNeighbor`, `SatSwapNeighbor` |
| [TSP](docs/problems/tsp.md) | `TspWithCoordinates` | `TspTwoOptNeighbor`, `TspRelocateNeighbor` |
| [Vertex Cover](docs/problems/vertex_cover.md) | `VertexCover` | `VertexCoverFlipNeighbor`, `VertexCoverSwapNeighbor` |
| [Job Shop Scheduling](docs/problems/job_shop_scheduling.md) | `JobShopScheduling` | `JobShopSwapNeighbor`, `JobShopRelocateNeighbor` |
| [CVRP](docs/problems/vrp.md) | `Vrp` | `VrpRelocateNeighbor`, `VrpSwapNeighbor`, `VrpTwoOptNeighbor` |
| [Formula](docs/problems/formula.md) | `FormulaProblem` | `FormulaFlipNeighbor`, `FormulaSwapNeighbor` |

## Available Heuristics

| Algorithm | Type |
|---|---|
| [Local Search](docs/heuristics/local_search.md) | `LocalSearch<N>` |
| [Simulated Annealing](docs/heuristics/simulated_annealing.md) | `SimulatedAnnealing<N>`, `BangBangSimulatedAnnealing<N>` |
| [Late Acceptance Hill Climbing](docs/heuristics/late_acceptance.md) | `LateAcceptanceHillClimbing<N>` |
| [Tabu Search](docs/heuristics/tabu_search.md) | `TabuSearch<N>` |
| [Random Walk](docs/heuristics/random_walk.md) | `RandomWalk<N>` |
| [Beam Search](docs/heuristics/beam_search.md) | `BeamSearch<P, N>` |
| [RL Search](docs/heuristics/rl_search.md) | `RlSearch<N>` |
| [Genetic Algorithm](docs/heuristics/genetic_algorithm.md) | `GeneticAlgorithm<P, C>` |
| [Sequential / Iterated / VNS / Restart](docs/heuristics/meta.md) | `Sequential<P>`, `Iterated<P>`, `VariableNeighborhoodSearch<P>`, `Restart<P>` |
| [Breakout Local Search (MaxCut)](docs/heuristics/breakout_local_search.md) | `BreakoutLocalSearchForMaxCut` |
| [Population Annealing (MaxCut)](docs/heuristics/population_annealing.md) | `PopulationAnnealingForMaxCut` |
| [Lin-Kernighan-Helsgaun (TSP)](docs/heuristics/lkh.md) | `LinKernighanHelsgaunForTsp` |
| [WalkSAT (MaxSAT)](docs/heuristics/walksat.md) | `WalkSatForSat` |
| [Hybrid Genetic Search (CVRP)](docs/heuristics/hgs.md) | `HybridGeneticSearchForVrp` |
| [Adaptive Large Neighborhood Search (CVRP)](docs/heuristics/alns.md) | `AdaptiveLargeNeighborhoodSearchForVrp` |

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

The same pages as Markdown in this repository:

- [`docs/quickstart.md`](docs/quickstart.md) — getting started, file loaders
- [`docs/concepts.md`](docs/concepts.md) — design philosophy and key patterns
- [`docs/search_state.md`](docs/search_state.md) — `SearchState`: the state every heuristic drives
- [`docs/traits.md`](docs/traits.md) — core traits reference
- [`docs/problems/`](docs/problems/) — supported problems
- [`docs/heuristics/`](docs/heuristics/) — available algorithms
- [`docs/guide/`](docs/guide/) — stop conditions, benchmarking, custom problem/heuristic, error handling
- [`docs/benchmarks/`](docs/benchmarks/) — performance reports on standard instance sets

## Examples

```bash
cargo run --example max_cut             # MaxCut: LocalSearch and TabuSearch
cargo run --example beam_search         # MaxCut: BeamSearch
cargo run --example custom_problem      # define your own problem
cargo run --example custom_heuristic    # define your own heuristic
```

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
