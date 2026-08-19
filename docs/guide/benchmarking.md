# Benchmarking

**API:** [`BenchmarkConfig`](../api/optopus/benchmark/struct.BenchmarkConfig.html) · [`HeuristicConfig`](../api/optopus/benchmark/enum.HeuristicConfig.html) · [`BenchmarkReport`](../api/optopus/benchmark/struct.BenchmarkReport.html)

Optopus ships with a CLI benchmark runner that takes a TOML config, runs each
heuristic on each instance N times in parallel, and writes a TOML report.

## CLI

```sh
cargo run --release -- path/to/config.toml
```

Output is written to `result/<config_stem>_<timestamp>.toml`.

## Config schema

```toml
num_runs = 10                          # repetitions per (instance, heuristic) pair
seed = 42                              # optional master seed; when set, reruns are
                                       # bit-identical (each run derives its own seed)

[[instances]]
path = "data/instances/max_cut/G*"     # file path or glob (Gset files have no extension)
problem = "MaxCut"                     # MaxCut | Qubo | Sat | Tsp | VertexCover | JobShop | Vrp

[[heuristics]]
kind = "LocalSearch"                   # see kinds below
neighbor = "Flip"                      # Flip | Swap | TwoOpt | Relocate
[heuristics.stop_condition]
max_iteration = 100_000                # any subset of fields; ANY-met semantics
max_duration_secs = 30.0
max_failed_update = 5_000
```

- [`BenchmarkConfig`](../api/optopus/benchmark/struct.BenchmarkConfig.html)
- [`ProblemKind`](../api/optopus/benchmark/enum.ProblemKind.html)
- [`HeuristicConfig`](../api/optopus/benchmark/enum.HeuristicConfig.html)

Multiple `[[instances]]` and `[[heuristics]]` blocks are allowed; the runner
takes the Cartesian product.

## Heuristic kinds

Every value below is a valid `kind` tag for a `[[heuristics]]` block. Each links
to the fields it takes — required, optional, and their defaults — on the
algorithm's own page; this table is only the index.

| `kind` | Applies to |
|---|---|
| [`LocalSearch`](../heuristics/local_search.md#benchmark-config) | all |
| [`TabuSearch`](../heuristics/tabu_search.md#benchmark-config) | all |
| [`SimulatedAnnealing`](../heuristics/simulated_annealing.md#benchmark-config) | all |
| [`LateAcceptanceHillClimbing`](../heuristics/late_acceptance.md#benchmark-config) | all |
| [`RandomWalk`](../heuristics/random_walk.md#benchmark-config) | all |
| [`RlSearch`](../heuristics/rl_search.md#benchmark-config) | all |
| [`Sequential` / `Iterated` / `VariableNeighborhoodSearch` / `Restart`](../heuristics/meta.md#benchmark-config) | all |
| [`GeneticAlgorithm`](../heuristics/genetic_algorithm.md#benchmark-config) | all |
| [`BreakoutLocalSearch`](../heuristics/breakout_local_search.md#benchmark-config) | MaxCut only |
| [`RlBreakoutLocalSearch`](../heuristics/rl_breakout_local_search.md#benchmark-config) | MaxCut only |
| [`PopulationAnnealingForMaxCut`](../heuristics/population_annealing.md#benchmark-config) | MaxCut only |
| [`LinKernighanHelsgaun`](../heuristics/lkh.md#benchmark-config) | TSP only |
| [`WalkSat`](../heuristics/walksat.md#benchmark-config) | SAT only |
| [`AdaptiveLargeNeighborhoodSearch`](../heuristics/alns.md#benchmark-config) | VRP only |
| [`HybridGeneticSearch`](../heuristics/hgs.md#benchmark-config) | VRP only |

Unknown kinds and missing required fields fail at parse time, before any run
starts. [`BeamSearch`](../heuristics/beam_search.md) has no `kind` — it is
reachable from the Rust API only.

## Fields shared by every kind

`stop_condition` accepts any subset of `max_iteration`, `max_duration_secs`,
`max_failed_update`; the heuristic stops when any of them is met. It is the TOML
form of the builder in [Stop Conditions](stop_conditions.md), which explains what
each limit counts.

`steps` is a nested array of heuristic tables (`[[heuristics.steps]]`, see the
example below). Which slot plays which role is per kind, and is documented with
that kind.

`neighbor` is per-problem:

| Problem | Valid neighbors |
|---|---|
| MaxCut, QUBO, SAT, VertexCover | `Flip`, `Swap` |
| TSP | `TwoOpt`, `Relocate` |
| JobShop | `Swap`, `Relocate` |
| VRP | `Relocate`, `Swap`, `TwoOpt` |

The problem-specific kinds (`BreakoutLocalSearch`, `LinKernighanHelsgaun`,
`AdaptiveLargeNeighborhoodSearch`, `WalkSat`, …) take no `neighbor` — they own
their move sets.

## Nested example: ILS in TOML

```toml
[[heuristics]]
kind = "Iterated"
[heuristics.stop_condition]
max_iteration = 1_000_000

[[heuristics.steps]]                   # search phase
kind = "LocalSearch"
neighbor = "Flip"
[heuristics.steps.stop_condition]
max_failed_update = 1

[[heuristics.steps]]                   # perturbation phase
kind = "RandomWalk"                    # unconditional random move = randomizing kick
neighbor = "Flip"
[heuristics.steps.stop_condition]
max_iteration = 200
```

**API:** [`HeuristicConfig::Iterated`](../api/optopus/benchmark/enum.HeuristicConfig.html#variant.Iterated)

## Output report

Each run produces a `BenchmarkReport`:

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
    │   ├── std_objective: f64                  (population std)
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
        ├── seed: Option<u64>                    (this run's derived seed)
        ├── solution: Vec<usize>                 (0-indexed, problem-specific encoding)
        └── trajectory: Vec<(f64, f64)>          (elapsed_secs, objective) per improvement
```

`trajectory` is the anytime curve, monotone in the problem's optimization
direction — it is what the benchmark viewer plots. The `Option` fields are
absent for runs that ended in an error.

Solution encoding:

| Problem | `solution` |
|---|---|
| MaxCut | vertex indices on the cut side |
| QUBO | variable indices set to 1 |
| SAT | variable indices set to `true` |
| TSP | city visit order |
| VertexCover | vertex indices in the cover |
| JobShop | operation sequence (job indices, each repeated `n_machines` times) |
| VRP | all routes flattened with the depot (`0`) as separator: `0, r0…, 0, r1…, 0` (empty routes omitted) |
