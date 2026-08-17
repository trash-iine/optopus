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

Multiple `[[instances]]` and `[[heuristics]]` blocks are allowed; the runner
takes the Cartesian product.

## Heuristic kinds and required fields

| `kind` | Applies to | Required | Optional |
|---|---|---|---|
| `LocalSearch` | all | `neighbor` | — |
| `TabuSearch` | all | `neighbor`, `tabu_tenure` | — |
| `SimulatedAnnealing` | all | `neighbor`, `initial_temperature`, `cooling_rate` | — |
| `LateAcceptanceHillClimbing` | all | `neighbor`, `history_length` | — |
| `RandomWalk` | all | `neighbor` (give it a `stop_condition` — an empty one never terminates) | — |
| `BreakoutLocalSearch` | MaxCut only | `tabu_tenure`, `t`, `l0`, `p0`, `q` | — |
| `RlBreakoutLocalSearch` | MaxCut only | `tabu_tenure`, `t`, `l0` | `strength_bins` (`[1.0, 2.0, 4.0]`), `learning_rate` (0.1), `softmax_temperature` (1.0), `exploration` (0.05), `policy_weights` |
| `PopulationAnnealingForMaxCut` | MaxCut only | `population_size` (≥ 2) | `initial_beta` (0.1), `delta_beta` (0.02), `sweeps_per_step` (50), `reset_period` (400; `0` disables), `cluster_moves` (true) |
| `LinKernighanHelsgaun` | TSP only | — | `num_neighbors` (default 5), `max_depth` (default 5) |
| `AdaptiveLargeNeighborhoodSearch` | VRP only | — | `removal_fraction` (0.15), `cooling_rate` (0.9995) |
| `HybridGeneticSearch` | VRP only | — | `min_population_size` (25), `generation_size` (40), `granularity` (20), `target_feasible` (0.2), `restart_generations` (20000) |
| `WalkSat` | SAT only | — | `noise` (0.3), `adaptive_noise` (false) |
| `RlSearch` | all | `neighbor` | `learning_rate` (0.01), `softmax_temperature` (1.0), `reward_shaping` (`Raw`\|`Normalized`\|`BestImprovement`, default `Normalized`), `policy_weights`, `max_candidates`; `discount` is still accepted for config compatibility but ignored with a warning |
| `Sequential` | all | `steps` | — |
| `Iterated` | all | `steps` (`[0]` = search, `[1]` = perturbation) | — |
| `VariableNeighborhoodSearch` | all | `steps` (`[0]` = search, `[1..]` = shakes N_1..N_kmax) | — |
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

> `RandomWalk` never stops on its own, so always give it a `stop_condition`.
> A short high-temperature `SimulatedAnnealing` phase works as a perturbation too.

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
