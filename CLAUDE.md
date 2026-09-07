# Optopus — Codebase Guide

A metaheuristic optimization library for combinatorial problems, written in Rust.

Three orthogonal concerns are kept strictly separate. Problems say what to
optimize, heuristics say how to search, and `SearchState` carries the iteration
count, timing, RNG, and the current and best solutions. Any heuristic works with
any problem, and no problem-specific code lives in the heuristic layer. The CLI
entry `src/main.rs` turns a TOML config into a benchmark run and a TOML report.

## Where to look

This file is the map: where things live, what to touch when adding something,
and the conventions this repo holds itself to. It deliberately does not explain
the algorithms or the traits, because three other places do that better.

| Question | Read |
|---|---|
| What does this heuristic do, and what does it need? | `docs/heuristics/README.md`, then the page it links |
| What does this problem look like, and what is its file format? | `docs/problems/` |
| What do the core traits require? | `docs/traits.md` and `src/trait_defs/` |
| How does `SearchState` work? | `docs/search_state.md` |
| Why is it built this way, and what has already been tried? | `decisions/README.md` |
| How do I use the library? | the rustdoc, and the site built from `docs/` |

`decisions/` holds one file per measured decision with an index. Read that index
before re-proposing anything that looks like an obvious improvement; several of
them were tried and priced already. `docs/` and the rustdoc are for users of the
library and carry none of that material.

## Extension recipes

### Add a new problem to the benchmark (3 sites, all small)

1. `ProblemKind` variant in `src/benchmark/config.rs`
2. `with_problem` arm in `src/benchmark/problems.rs`
3. One impl block in `src/benchmark/problems.rs`: `BenchmarkProblem` (load_instance) + `BenchmarkSolution` (objective/encode) + `ConfigurableProblem` (`NAME`, `MINIMIZE`, `VALID_NEIGHBORS`, `with_neighbor` registry, optional `build_special_heuristic`, `build_crossover`)

Plus the library side: `src/problem/<name>/{mod,problem,neighbor,crossover}.rs` (private mods + `pub use`), re-exports in `src/problem/mod.rs` (all types including the crossover) and `src/prelude.rs` (problem / solution / neighbor types; most crossovers are exported only from `problem/mod.rs`). Each move's `MoveToNeighbor` impl also needs `fn tabu_policy(&self) -> Option<&dyn EnabledTabu> { Some(self) }` next to its `EnabledTabu` impl. Without it the move compiles and silently has no tabu list, and `trait_defs/tabu.rs` has the test that pins every built-in move.

### Add a new base metaheuristic

Implement `Heuristic<P>` in `src/heuristic/<name>.rs`, re-export via `heuristic/mod.rs` + prelude, then add one `HeuristicConfig` variant in `src/benchmark/config.rs` and follow the compile errors (one arm in `BaseBuilder::visit` in `src/benchmark/factory.rs`). The base-heuristic dispatch is written once, not per problem.

## Module Map

```
src/
├── lib.rs / main.rs / prelude.rs / error.rs (OptError)
├── benchmark/
│   ├── config.rs             ProblemKind, NeighborKind, HeuristicConfig, BenchmarkConfig
│   ├── factory.rs            ConfigNeighbor, NeighborVisitor, ConfigurableProblem,
│   │                         BaseBuilder, build_heuristic (the single generic factory)
│   ├── problems.rs           all per-problem registration + with_problem
│   ├── runner.rs             run_from_config, run loop, per-run seed derivation
│   └── report.rs             SingleRunResult, Summary, BenchmarkReport
├── search_state/mod.rs       SearchState<'a, P>, SearchStateCloneType
├── trait_defs/               core traits, re-exported via search_state and prelude
│   ├── rankable.rs           Rankable, rank_cmp, filter_best, Distance
│   ├── problem.rs            ProblemTrait
│   ├── neighbor.rs           MoveToNeighbor
│   ├── evaluate.rs           Evaluable, Evaluate
│   ├── crossover.rs          Crossover, SubProblemExtractable
│   ├── tabu.rs               EnabledTabu (object safe on purpose)
│   ├── binary.rs             BinaryProblem
│   └── reduction.rs          ProblemReduction
├── common/                   shared data structures and helpers; put new shared code here
│   ├── graph/                Graph (mod.rs), random and lattice generators (generator.rs),
│   │                         seeded_rng
│   ├── binary.rs             uniform_binary_crossover, hamming_distance,
│   │                         lift_binary_solution, lift_compact_binary_solution,
│   │                         apply_swap_as_two_flips
│   ├── tabu.rs               TabuKey (Var / Pair / Triple), TabuMemory
│   ├── permutation.rs        order_crossover (OX)
│   └── parse.rs              InstanceLines
├── heuristic/
│   ├── mod.rs                Heuristic trait, StopCondition
│   ├── local_search.rs / simulated_annealing.rs (+BangBang) / tabu_search.rs
│   ├── late_acceptance.rs / beam_search.rs / random_walk.rs / restart.rs
│   ├── sequential.rs         Sequential<P>, Iterated<P> (ILS lives here too)
│   ├── variable_neighborhood_search.rs
│   ├── genetic_algorithm.rs  GeneticAlgorithm<P, C>, ParentSelection
│   ├── population_annealing.rs  PopulationAnnealing<P, N>
│   ├── crossover.rs          SubProblemBasedCrossover<P>
│   ├── reinforcement_learning/  RlSearch<N>
│   └── specific/             one directory per problem once it has several
│       ├── max_cut/          bls.rs (BreakoutLocalSearch, also descend / kick /
│       │                     externally_driven for examples/rl_bls.rs),
│       │                     best_swap.rs (the one operator with no generic
│       │                     equivalent)
│       ├── vrp/              ops/ (pricing fns, RouteState, granular.rs, Descent),
│       │                     alns.rs, hgs/ (mod.rs driver, population.rs)
│       ├── lkh_for_tsp.rs
│       └── walksat_for_sat.rs
└── problem/                  each holds problem, solution, neighbors, crossover
    ├── max_cut/              + kernel.rs (MaxCutKernel, the one ProblemReduction),
    │                         planted.rs (PlantedMaxCut)
    ├── qubo/ sat/ tsp_2d/ vertex_cover/ job_shop_scheduling/
    ├── vrp/                  + split.rs (split_giant_tour), adjacency.rs (RouteAdjacency)
    └── binary_optimization/  FormulaProblem, Expr
```

## Benchmarking (`src/benchmark/`)

A TOML config becomes a `BenchmarkConfig`, each heuristic runs on each instance
N times (rayon-parallel), and a `BenchmarkReport` is written as timestamped TOML
under `result/`. `docs/guide/benchmarking.md` is the user-facing guide.

```toml
num_runs = 10
seed = 42                      # optional: makes every run bit-reproducible
[[instances]]
path = "data/instances/max_cut/G*"   # globs supported (Gset files have no extension)
problem = "MaxCut"             # MaxCut | Qubo | Sat | Tsp | VertexCover | JobShop | Vrp
[[heuristics]]
kind = "LocalSearch"           # see the table below
neighbor = "Flip"              # Flip | Swap | TwoOpt | Relocate
[heuristics.stop_condition]
max_iteration = 100000         # max_duration_secs / max_failed_update also supported
```

`HeuristicConfig` is an internally-tagged enum (`#[serde(tag = "kind")]`), so
each `kind` declares its own required fields, and missing fields or unknown
kinds fail at parse time.

| `kind` | Applies to | Required fields | Optional fields |
|---|---|---|---|
| `LocalSearch` | all | | |
| `TabuSearch` | all | `tabu_tenure = [min, max]` | |
| `SimulatedAnnealing` | all | `initial_temperature`, `cooling_rate` | |
| `LateAcceptanceHillClimbing` | all | `history_length` | |
| `RandomWalk` | all | a `stop_condition` (an empty one never terminates) | |
| `RlSearch` | all | | `learning_rate`, `softmax_temperature`, `reward_shaping`, `policy_weights`, `max_candidates` |
| `BreakoutLocalSearch` | MaxCut | `tabu_tenure`, `t`, `l0`, `p0`, `q` | |
| `PopulationAnnealing` | all | `neighbor`, `population_size` | `initial_beta`, `delta_beta`, `sweeps_per_step`, `reset_period`, `sweep_length` |
| `LinKernighanHelsgaun` | TSP | | `num_neighbors`, `max_depth` |
| `WalkSat` | SAT | | `noise`, `adaptive_noise` |
| `AdaptiveLargeNeighborhoodSearch` | VRP | | `removal_fraction`, `cooling_rate` |
| `HybridGeneticSearch` | VRP | | `min_population_size`, `generation_size`, `granularity`, `target_feasible`, `restart_generations` |
| `Sequential` | all | `steps` | |
| `Iterated` | all | `steps[0]` = search, `steps[1]` = perturbation | |
| `VariableNeighborhoodSearch` | all | `steps[0]` = search, `steps[1..]` = shakes | |
| `Restart` | all | `steps[0]`, `restart_condition` | |
| `GeneticAlgorithm` | all | `population_size` | `crossover_kind`, `parent_selection`, `parent_top_k` |

`RlSearch` still parses `discount` and ignores it with a warning, since
single-step REINFORCE has none.

A `SingleRunResult` carries `best_objective`, `best_iteration`, timing, the
per-run `seed`, `solution` (0-indexed), and `trajectory`, the
`(elapsed_secs, objective)` anytime curve made monotone in the problem's
direction. `Summary` aggregates those across runs.

MaxCut instance suites: besides the G-set, `examples/generate_dense_maxcut.rs`,
`generate_sparse_maxcut.rs` and `generate_hard_maxcut.rs` produce suites locally
from fixed seeds, so they are never committed. The hard suite has an exact
optimum by construction; see `decisions/0010`.

## Documentation site (`docs/` + `mkdocs.yml`)

`docs/` is the source of the Pages site (mkdocs-material). The **API reference is
part of that site**: `cargo doc --no-deps --lib` (the `--lib` avoids the lib/bin
output-filename collision) is copied to `docs/api/` — gitignored, generated —
so every page's `**API:**` line links it with a plain relative path that
`mkdocs build --strict` verifies. Both workflows build it that way; CI's `docs`
job adds `RUSTDOCFLAGS=-D warnings`, so a broken intra-doc link fails the PR.
To build the site locally:

```bash
cargo doc --no-deps --lib && rm -rf docs/api && cp -r target/doc docs/api
mkdocs build --strict --site-dir /tmp/site   # or `mkdocs serve`
```

Every page carries exactly one `**API:**` line, directly under the `# ` title,
holding exactly one link: the rustdoc item that page is about (a problem's
problem type, a heuristic's heuristic type; a module index only on the pages
that survey a whole module — `problems/README.md`, `heuristics/README.md`,
`traits.md`). Sibling types are reached from that item's module page, not from
a second link — a page that needs to point at one from its body links it
inline in the prose, never as another `**API:**` line. Adding a problem or
heuristic means adding that one line to the new page too — `--strict` catches a
wrong path, not a missing line.

