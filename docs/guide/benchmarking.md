# Benchmarking

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
path = "data/instances/max_cut/G*.txt"           # file path or glob
problem = "MaxCut"                     # MaxCut | Qubo | Sat | Tsp | VertexCover | JobShop

[[heuristics]]
kind = "LocalSearch"                   # see kinds below
neighbor = "Flip"                      # Flip | Swap | TwoOpt | Relocate
[heuristics.stop_condition]
max_iteration = 100_000                # any subset of fields; ANY-met semantics
max_duration_secs = 30.0
max_failed_update = 5_000
```

Multiple `[[instances]]` and `[[heuristics]]` blocks are allowed; the runner
takes the Cartesian product.

## Heuristic kinds and required fields

| `kind` | Applies to | Required | Optional |
|---|---|---|---|
| `LocalSearch` | all | `neighbor` | — |
| `TabuSearch` | all | `neighbor`, `tabu_tenure` | — |
| `SimulatedAnnealing` | all | `neighbor`, `initial_temperature`, `cooling_rate` | — |
| `LateAcceptanceHillClimbing` | all | `neighbor`, `history_length` | — |
| `BreakoutLocalSearch` | MaxCut only | `tabu_tenure`, `t`, `l0`, `p0`, `q` | — |
| `RlBreakoutLocalSearch` | MaxCut only | `tabu_tenure`, `t`, `l0` | `strength_bins` (`[1.0, 2.0, 4.0]`), `learning_rate` (0.1), `softmax_temperature` (1.0), `exploration` (0.05), `policy_weights` |
| `LinKernighanHelsgaun` | TSP only | — | `num_neighbors` (default 5), `max_depth` (default 5) |
| `RlSearch` | all | `neighbor` | `learning_rate` (0.01), `softmax_temperature` (1.0), `reward_shaping` (`Raw`\|`Normalized`\|`BestImprovement`, default `Normalized`), `policy_weights`, `max_candidates` |
| `Sequential` | all | `steps` | — |
| `Iterated` | all | `steps` (`[0]` = search, `[1]` = perturbation) | — |
| `Restart` | all | `steps` (single inner), `restart_condition` | — |
| `GeneticAlgorithm` | all | `population_size` (≥ 2), `steps` (`[0]` = mutation, optional `[1]` = init_improvement) | `crossover_kind` (per-problem default: `Uniform`, `Order` for TSP, `Ppx` for JobShop), `parent_selection` (`Tournament` default \| `DistantTopK`), `parent_top_k` (required when `DistantTopK`) |

`tabu_tenure` is a `(min, max)` pair, e.g. `tabu_tenure = [5, 10]`.
`stop_condition` accepts any subset of `max_iteration`, `max_duration_secs`,
`max_failed_update`.

`neighbor` is per-problem:

| Problem | Valid neighbors |
|---|---|
| MaxCut, QUBO, SAT, VertexCover | `Flip`, `Swap` |
| TSP | `TwoOpt`, `Relocate` |
| JobShop | `Swap`, `Relocate` |

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
kind = "SimulatedAnnealing"            # high temperature = randomizing kick
neighbor = "Flip"
initial_temperature = 5.0
cooling_rate = 0.99
[heuristics.steps.stop_condition]
max_iteration = 200
```

> `RandomWalk` is exposed in the library API but does not currently have a CLI
> `kind`. In TOML configs, a short high-temperature `SimulatedAnnealing` phase
> (as above) serves as the perturbation.

## Output report

Each run produces a `BenchmarkReport`:

```text
BenchmarkReport
├── timestamp: String
├── config_file: String
└── results: Vec<InstanceHeuristicResult>
    ├── instance_path: String
    ├── heuristic: HeuristicConfig
    ├── summary: Summary
    │   ├── num_successful_runs: usize
    │   ├── best_objective / avg_objective / worst_objective: f64
    │   ├── std_objective: f64                  (population std)
    │   ├── best_time_to_best_secs / avg_time_to_best_secs: f64
    │   └── avg_total_time_secs: f64
    └── runs: Vec<SingleRunResult>
        ├── run_index: usize
        ├── status: String                       ("success" | "error: …")
        ├── best_objective: f64
        ├── best_iteration: u64
        ├── time_to_best_secs / total_time_secs: f64
        └── solution: Vec<usize>                 (0-indexed, problem-specific encoding)
```

Solution encoding:

| Problem | `solution` |
|---|---|
| MaxCut | vertex indices on the cut side |
| QUBO | variable indices set to 1 |
| SAT | variable indices set to `true` |
| TSP | city visit order |
| VertexCover | vertex indices in the cover |
| JobShop | operation sequence (job indices, each repeated `n_machines` times) |
