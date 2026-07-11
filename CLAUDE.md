# Optopus — Codebase Guide

A metaheuristic optimization library for combinatorial problems, written in Rust.

**Design philosophy:** three orthogonal concerns kept strictly separate:
- **Problems** — what to optimize (MaxCut, QUBO, MaxSAT, TSP, VertexCover, JobShop, custom formula)
- **Heuristics** — how to search (LocalSearch, SA, TabuSearch, GA, RlSearch, …)
- **SearchState** — iteration count, timing, RNG, current and best solutions

Any heuristic works with any problem; no problem-specific code lives in the heuristic layer.
CLI entry `src/main.rs`: TOML config → benchmark run → TOML output (via `BenchmarkReport::write_to_dir`).

## Library Concept (3 use cases)

1. **Existing problem × existing heuristic** — run `LocalSearch`, `SimulatedAnnealing`, `TabuSearch`, etc. on MaxCut / QUBO / SAT / TSP / VertexCover / JobShop in a few lines via `use optopus::prelude::*`.
2. **Apply existing heuristics to a new problem** — implement just three traits (`ProblemTrait` + `Rankable` on `Solution` + `MoveToNeighbor`) and every heuristic works as-is. Add `Evaluate<f64>` for SA/LAHC/RlSearch, `EnabledTabu` for TabuSearch, `BinaryProblem` to reuse the generic binary machinery in `src/common/`.
3. **Combine heuristics and run benchmarks** — compose components with `Sequential` / `Iterated` / `Restart` / `GeneticAlgorithm`, write a TOML config, and get aggregated best/avg/worst/std/time results.

## Extension recipes

**Add a new problem to the benchmark (3 sites, all small):**
1. `ProblemKind` variant in `src/benchmark/config.rs`
2. `with_problem` arm in `src/benchmark/problems.rs`
3. One impl block in `src/benchmark/problems.rs`: `BenchmarkProblem` (load_instance) + `BenchmarkSolution` (objective/encode) + `ConfigurableProblem` (`NAME`, `MINIMIZE`, `VALID_NEIGHBORS`, `with_neighbor` registry, optional `build_special_heuristic`, `build_crossover`)

Plus the library side: `src/problem/<name>/{mod,problem,neighbor,crossover}.rs` (private mods + `pub use`), re-exports in `src/problem/mod.rs` (all types including the crossover) and `src/prelude.rs` (problem / solution / neighbor types; most crossovers are exported only from `problem/mod.rs`).

**Add a new base metaheuristic:** implement `Heuristic<P>` in `src/heuristic/<name>.rs`, re-export via `heuristic/mod.rs` + prelude, then add one `HeuristicConfig` variant in `src/benchmark/config.rs` and follow the compile errors (one arm in `BaseBuilder::visit` in `src/benchmark/factory.rs`). The base-heuristic dispatch is written once, not per problem.

## Module Map

```
src/
├── lib.rs / main.rs / prelude.rs / error.rs (OptError)
├── benchmark/
│   ├── mod.rs                public re-exports
│   ├── config.rs             ProblemKind, NeighborKind, HeuristicConfig (tagged enum),
│   │                         StopConditionConfig, BenchmarkConfig, validate_config
│   ├── factory.rs            ConfigNeighbor, NeighborVisitor, ConfigurableProblem trait,
│   │                         BaseBuilder, build_heuristic (the single generic factory)
│   ├── problems.rs           ALL per-problem registration + with_problem (ProblemVisitor)
│   ├── runner.rs             Benchmark::run_from_config, run loop, per-run seed derivation
│   └── report.rs             SingleRunResult / Summary / BenchmarkReport (+ write_to_dir)
├── search_state/
│   └── mod.rs                SearchState<'a, P>, SearchStateCloneType
├── trait_defs/               core traits (re-exported via search_state & prelude)
│   ├── rankable.rs           Rankable, rank_cmp, filter_best, Distance
│   ├── problem.rs            ProblemTrait
│   ├── neighbor.rs           MoveToNeighbor
│   ├── evaluate.rs           Evaluable, Evaluate
│   ├── crossover.rs          Crossover, SubProblemExtractable
│   ├── tabu.rs               EnabledTabu
│   └── binary.rs             BinaryProblem (unlocks the shared binary machinery)
├── common/                   shared data structures & helpers (put new shared code here)
│   ├── graph.rs              Graph (used by MaxCut / VertexCover)
│   ├── binary.rs             uniform_binary_crossover, hamming_distance,
│   │                         lift_binary_solution / lift_compact_binary_solution,
│   │                         apply_swap_as_two_flips
│   ├── tabu.rs               VarTabuMap, is_var_enabled, add_var_to_tabu
│   ├── gain_index.rs         GainIndex (improving-move index)
│   └── parse.rs              InstanceLines (file-loader scaffold with FileLoad errors)
├── heuristic/
│   ├── mod.rs                Heuristic trait, StopCondition
│   ├── local_search.rs / simulated_annealing.rs (+BangBang) / tabu_search.rs
│   ├── late_acceptance.rs    LateAcceptanceHillClimbing<N>
│   ├── beam_search.rs / random_walk.rs
│   ├── sequential.rs         Sequential<P>, Iterated<P>  ← ILS lives here too
│   ├── restart.rs            Restart<P>
│   ├── genetic_algorithm.rs  GeneticAlgorithm<P, C>, ParentSelection
│   ├── crossover.rs          SubProblemBasedCrossover<P>
│   ├── reinforcement_learning/  RlSearch<N> (REINFORCE policy over move features)
│   └── specific/
│       ├── bls_for_max_cut.rs   BreakoutLocalSearchForMaxCut
│       └── lkh_for_tsp.rs       LinKernighanHelsgaunForTsp
└── problem/
    ├── max_cut/              MaxCut, MaxCutSolution, {Flip,Swap}Neighbor, UniformCrossover
    ├── qubo/                 Qubo, QuboSolution, {Flip,Swap}Neighbor, UniformCrossover
    ├── sat/                  Sat, SatSolution, {Flip,Swap}Neighbor, UniformCrossover
    ├── tsp_2d/               TspWithCoordinates, TspSolution, {TwoOpt,Relocate}Neighbor, OrderCrossover
    ├── vertex_cover/         VertexCover, VertexCoverSolution, {Flip,Swap}Neighbor, UniformCrossover
    ├── job_shop_scheduling/  JobShopScheduling, JobShopSolution, {Swap,Relocate}Neighbor, PpxCrossover
    └── binary_optimization/  FormulaProblem, Expr, Formula{Flip,Swap}Neighbor, FormulaUniformCrossover
```

## Core Traits (`src/trait_defs/`)

These live in `src/trait_defs/` and are re-exported via `crate::search_state::*` and the `prelude`. Internal code imports them from `crate::trait_defs`; the `search_state` re-export is kept for the public API.

- **`Rankable`**: `is_better_than(&Self) -> bool`. Implemented by every `Solution`; the optimization direction is baked into the problem. The `filter_best(iter)` helper returns the set of tied-best items.
- **`ProblemTrait`**: `type Solution: Clone + Rankable; fn new_solution(&self, rng) -> Solution`.
- **`MoveToNeighbor<P>`**: a single one-step move.
  ```rust
  fn iter(prob, sol) -> impl Iterator<Item = Self> + Send;     // lazy
  fn apply_to_solution(&self, prob, sol) -> Result<(), OptError>;
  fn move_to_be_better_than(&self, prob, src, other) -> bool;  // default: clone + apply
  fn random_neighbor(prob, sol, rng) -> Option<Self>;          // default: reservoir over iter()
  fn apply_to_iteration(&self, iter: u64) -> u64;              // default: iter + 1
  ```
  The two defaults are slow-but-correct and emit a one-shot `tracing::warn!` when hit;
  every built-in move overrides both (O(1) gain compare; direct O(1)/O(n) sampler used
  each step by SA / LAHC / RandomWalk).
- **`Evaluable<T>` / `Evaluate<T>`** (default `T = f64`): `Maximize(T)` / `Minimize(T)` carries the direction of an objective delta. `Evaluable<f64>::worsening_amount()` normalizes both directions to "positive = worse" (used by `boltzmann_accept`). Required for SA / LAHC / RlSearch. QUBO also exposes `Evaluate<Coefficient = i32>` for integer gains.
- **`Crossover<P>`**: `crossover(&mut self, prob, sol1, sol2, rng) -> Result<Solution, OptError>` (exactly two parents; RNG passed in for reproducibility; `Err` only when the operator genuinely cannot produce an offspring, e.g. an inner sub-heuristic failed).
- **`EnabledTabu`**: `type TabuMap: Default`, `is_move_enabled(map, iter)`, `add_to_tabu_map(map, iter, tenure, rng)`. The tenure is sampled from the passed RNG (`&mut state.rng`) so seeded runs are bit-reproducible. Required by TabuSearch.
- **`SubProblemExtractable`**: `extract_sub_problem(sol1, sol2) -> Self`, `lift_solution(sol1, sol2, sub_sol)`. Variables that agree in both parents are fixed; the disagreeing variables form the sub-problem. Binary problems delegate lifting to `common::lift_binary_solution` (shared index space: MaxCut, VertexCover) or `common::lift_compact_binary_solution` (compacted indices: SAT, Formula); QUBO keeps its own bias-folding variant.
- **`BinaryProblem`**: `type Flip`, `variable_indices()`, `variable(sol, i)`, `flip_move(sol, i)` — implemented by all binary problems; unlocks the shared machinery in `src/common/binary.rs`.

## SearchState (`src/search_state/mod.rs`)

```rust
pub struct SearchState<'a, P: ProblemTrait> {
    pub instance: &'a P,
    pub solution: P::Solution,        // current
    pub best_solution: P::Solution,   // global best
    pub initial_solution: P::Solution,
    pub iteration: u64, pub best_iteration: u64,
    pub best_time: Instant,
    pub n_accepted: u64, pub n_rejected: u64, pub n_best_updates: u64,
    pub rng: SmallRng,                // ALL randomness flows through this
    pub(crate) start_*: ...,          // sub-run merge anchors — hands off
}
```

**Key methods**: `new(problem)`, `new_with_seed(problem, seed)`, `with_solution(problem, sol)` / `with_solution_and_seed(problem, sol, seed)` (warm start), `apply(neighbor)` (apply + iter + best update), `apply_move_only(neighbor)` (defer best update), `update_best()`, `progress_iteration()`, `random_neighbor::<N>(context)` (uniform random move or `InvalidState` error), `run_sub(heuristic, clone_type)` (the sub-run triad below), `is_neighbor_better_than_{current,best}(n)`, `duration()`.

**Reproducibility**: all randomness (initial solutions, move selection, tabu tenures, BLS perturbations) flows through `state.rng`. `clone_for_new_run` forks the RNG so meta-heuristic composition stays deterministic under a fixed seed.

**Sub-run clone/merge pattern** (used by every meta-heuristic to isolate phases):
```rust
state.run_sub(inner_heuristic.as_mut(), SearchStateCloneType::ClearBest)?;
// Simple    — keeps all counters and best (sets start_iteration to current)
// ClearBest — resets best and timers to current state (iteration = 0)
// StartBest — restarts from best_solution                (iteration = 0)
```

## Heuristic Algorithms (`src/heuristic/`)

**`Heuristic<P>` trait**: `clear()`, `stop_condition() -> &StopCondition`, `run_once(state)`, `run(state)` (default: `clear` → loop `run_once`). `is_done(state)` has a default that delegates to `stop_condition()`; heuristics with extra termination logic (LocalSearch, LKH: "stop at a local optimum") override it on top.

**Conventions**: constructors take `stop_condition` first; invalid arguments panic with a `# Panics` doc section; `PhantomData` fields are named `_neighbor`.

**`StopCondition`** (builder; stops when *any* condition is met):
```rust
StopCondition::iterations(1_000_000)
    .with_duration(Duration::from_secs(30))
    .with_failed_updates(10_000)
// also: StopCondition::duration(d), StopCondition::failed_updates(n)
```

### Base
| Type | Description |
|---|---|
| `LocalSearch<N>` | Greedy best-improving; halts at a local optimum |
| `SimulatedAnnealing<N>` | Random neighbor, `exp(-Δ/T)` acceptance, multiplicative cooling (requires `Evaluate<f64>`) |
| `BangBangSimulatedAnnealing<N>` | Oscillating temperature between `min_wave_threshold` and `max_wave_threshold` |
| `LateAcceptanceHillClimbing<N>` | LAHC: accepts a move if it is no worse than the score `history_length` steps ago (requires `Evaluate<f64>`) |
| `TabuSearch<N>` | Best non-tabu neighbor; aspiration overrides tabu when global best is improved; tenure ∈ `(min, max)` sampled from `state.rng` |
| `RandomWalk<N>` | Uniform random move with unconditional acceptance (useful as a perturbation) |
| `BeamSearch<P, N>` | Maintains top-`k` candidates; expands the full neighborhood of every beam member each iteration |
| `RlSearch<N>` | REINFORCE policy-gradient move selection over hand-crafted move features; weights persist across episodes (requires `Evaluate<f64>`) |

### Meta
| Type | Description |
|---|---|
| `Sequential<P>` | Runs a `Vec<Box<dyn Heuristic<P>>>` in order (each step on a `ClearBest` clone) |
| `Iterated<P>` | ILS: alternates `search` and `perturbation` (`Box<dyn Heuristic<P>>`); lives in `sequential.rs` |
| `Restart<P>` | Runs the inner heuristic; when `restart_condition` triggers, replaces `solution` with a fresh random one (best is preserved) |
| `GeneticAlgorithm<P, C>` | 2-parent selection (`Tournament` or `DistantTopK`) → `Crossover<P>` → mutation (`Heuristic<P>`) → worst-replacement; tracks `best_idx` incrementally |

### Crossover
- `SubProblemBasedCrossover<P>` (`crossover.rs`): builds a sub-problem from disagreeing variables, solves it with `sub_heuristic`, then lifts the result. Requires `P: SubProblemExtractable`.
- `*UniformCrossover`: per-variable random parent selection; all binary problems delegate to `common::uniform_binary_crossover`.
- `TspOrderCrossover` (OX) for TSP; `JobShopPpxCrossover` for JobShop.

### Problem-specific
- `BreakoutLocalSearchForMaxCut` (`specific/bls_for_max_cut.rs`): greedy local search plus adaptive perturbation (strong / weak flip / swap / plateau cluster), with probabilities decaying via the non-improvement counter `omega`. The plateau operators flip zero-gain vertices (tracked by the opt-in `zero_gain` index) without changing the objective — key on large sparse Gset instances.
- `RlBreakoutLocalSearchForMaxCut` (`specific/rl_bls_for_max_cut.rs`): same `BlsOps` machinery, but a contextual softmax bandit picks perturbation type (5 ops incl. both plateau variants) × strength; weights persist across `Restart`/`Iterated` episodes.
- `LinKernighanHelsgaunForTsp` (`specific/lkh_for_tsp.rs`): LK-style variable-depth moves with candidate lists; stops at a local optimum.

## Problem Types (`src/problem/`)

Binary solutions all name the assignment vector `x: Vec<bool>`.

| Problem | Direction | Solution | Neighbors | Crossover | Notes |
|---|---|---|---|---|---|
| **MaxCut** | Max | `x`, `gain: Vec<f32>`, `objective: f32` | Flip / Swap | Uniform | format: `N M / i j w` (1-indexed); optional `positive_gain` index (advanced) |
| **QUBO** | Min | `x`, `gain: Vec<i32>`, `objective: i32` | Flip / Swap | Uniform | `Coefficient = i32`; `SubProblemExtractable` (bias folding); optional `negative_gain` index (advanced) |
| **MaxSAT** | Max | `x` (0-indexed), `n_satisfied: usize`, `gain: Vec<i64>` | Flip / Swap | Uniform | DIMACS CNF |
| **TSP 2D** | Min | `tour: Vec<usize>`, `objective: f64` | TwoOpt / Relocate | Order (OX) | TSPLIB (EUC_2D / CEIL_2D / ATT / GEO); lazy distance matrix for `n ≤ 2000` (`DIST_MATRIX_MAX_N`), move gains computed on the fly from it |
| **VertexCover** | Min | `x`, `gain: Vec<i32>`, `objective` (penalty-augmented), `cover_size`, `uncovered_edges` | Flip / Swap | Uniform | same edge-list format as MaxCut |
| **JobShop** | Min | `operations: Vec<usize>`, `objective` (makespan) | Swap / Relocate | Ppx | `n_jobs n_machines` header + one job per line |
| **FormulaProblem** | Configurable (`OptDirection`) | `x`, `score: f64` (always higher-is-better), `gain: Vec<f64>` | Flip / Swap | Uniform + `SubProblemExtractable` | see below; **library-only** (no instance file format, so intentionally absent from `ProblemKind`) |

**FormulaProblem details**: AST `Expr = Const(f64) | Var(usize) | Neg | Add(Vec) | Mul(Vec)` with `+ - * /` operators. Constraints: `Comparison { lhs, rel: ConstraintRel, rhs, penalty_weight }` (Lt / Gt / Le / Ge / Eq) or `Clamp { expr, lo, hi, penalty_weight }`. A pre-compiled polynomial (`CompiledPoly`) gives O(d) gain deltas; `interaction_neighbors` tracks which variables' gains may change on each flip.

## Benchmarking (`src/benchmark/`)

TOML config → `BenchmarkConfig` → run each heuristic on each instance N times (rayon-parallel) → `BenchmarkReport` → timestamped TOML in `result/`.

```toml
num_runs = 10
seed = 42                      # optional: makes every run bit-reproducible
[[instances]]
path = "data/instances/max_cut/G*"   # globs supported (Gset files have no extension)
problem = "MaxCut"             # MaxCut | Qubo | Sat | Tsp | VertexCover | JobShop
[[heuristics]]
kind = "LocalSearch"           # see list below
neighbor = "Flip"              # Flip | Swap | TwoOpt | Relocate
[heuristics.stop_condition]
max_iteration = 100000         # max_duration_secs / max_failed_update also supported
```

`HeuristicConfig` is an internally-tagged enum (`#[serde(tag = "kind")]`), so each `kind` declares exactly its own required fields; missing fields and unknown kinds fail at parse time.

**Supported `kind` values**: `LocalSearch`, `TabuSearch` (`tabu_tenure = [min, max]`), `SimulatedAnnealing` (`initial_temperature`, `cooling_rate`), `LateAcceptanceHillClimbing` (`history_length`), `RlSearch` (optional `learning_rate` / `discount` / `softmax_temperature` / `reward_shaping` / `policy_weights` / `max_candidates`), `BreakoutLocalSearch` (MaxCut only; `tabu_tenure`, `t`, `l0`, `p0`, `q`, optional `plateau_prob`), `RlBreakoutLocalSearch` (MaxCut only; `tabu_tenure`, `t`, `l0`, optional `strength_bins` / `learning_rate` / `softmax_temperature` / `exploration` / `policy_weights`), `LinKernighanHelsgaun` (TSP only; optional `num_neighbors`, `max_depth`), and the meta-heuristics `Sequential` / `Iterated` / `Restart` / `GeneticAlgorithm` (nested `steps` array; `Iterated` uses `steps[0] = search, steps[1] = perturbation`; `Restart` also requires `restart_condition`; GA requires `population_size`, optional `crossover_kind` / `parent_selection` / `parent_top_k`).

**`Summary` fields**: `num_successful_runs`, `best/avg/worst/std_objective`, `best/avg_time_to_best_secs`, `avg_total_time_secs`, plus averaged `initial_objective` / `improvement` / acceptance counters. Each `SingleRunResult` carries `best_objective: f64`, `best_iteration: u64`, timing, the per-run `seed`, and `solution: Vec<usize>` (0-indexed encoding).

## Key Design Patterns

1. **Gain-based incremental updates** — binary/formula solutions cache per-variable `gain`; applying a move only refreshes the affected neighbors in O(degree). MaxCut and QUBO additionally offer optional `positive_gain` / `negative_gain` indexes (advanced) to enumerate only improving moves — used by problem-specific heuristics like BLS, not needed for standard use. TSP instead computes move gains on the fly from the lazily built distance matrix; JobShop re-decodes per candidate (and evaluates candidates with rayon on large instances, order-preserving so results are thread-count independent).
2. **Sub-run clone/merge** — every meta-heuristic uses `state.run_sub(inner, clone_type)` (`clone_for_new_run` → run → `update_state`). The global iteration counter advances monotonically across all phases.
3. **Seeded reproducibility** — all randomness flows through `state.rng` (`SmallRng`); `EnabledTabu::add_to_tabu_map` and `Crossover::crossover` take the RNG explicitly. With `seed` set in the benchmark config, reruns are bit-identical (enforced by e2e tests).
4. **Tabu abstraction via trait** — `TabuSearch` is generic over `N: EnabledTabu`. Each move type owns its `TabuMap`; binary problems share `VarTabuMap` + helpers from `common/tabu.rs`.
5. **Always-higher-is-better in `FormulaProblem`** — for `Maximize`, `score = objective − penalty`; for `Minimize`, `score = −objective − penalty`. Heuristics only need the higher-is-better convention.
6. **`MoveToNeighbor::iter()` is lazy** — `LocalSearch` selects with `max_by` in O(n) without collecting; `TabuSearch` uses `max_by` plus aspiration; only `BeamSearch` materializes all candidates.
7. **Config factory is generic, per-problem code is registration-only** — `build_heuristic` + `BaseBuilder` (in `benchmark/factory.rs`) contain the only base-heuristic dispatch; each problem contributes a small `ConfigurableProblem` impl in `benchmark/problems.rs`. Shared code goes in `src/common/`, not in `problem/` or at the top level.
