# Optopus — Codebase Guide

A metaheuristic optimization library for combinatorial problems, written in Rust.

**Design philosophy:** three orthogonal concerns kept strictly separate:
- **Problems** — what to optimize (MaxCut, QUBO, SAT, TSP, custom formula)
- **Heuristics** — how to search (LocalSearch, SA, TabuSearch, GA, …)
- **SearchState** — iteration count, timing, current and best solutions

Any heuristic works with any problem; no problem-specific code lives in the heuristic layer.
CLI entry `src/main.rs`: TOML config → benchmark run → TOML output.

## Library Concept (3 use cases)

1. **Existing problem × existing heuristic** — run `LocalSearch`, `SimulatedAnnealing`, `TabuSearch`, etc. on MaxCut / QUBO / SAT / TSP in a few lines via `use optopus::prelude::*`.
2. **Apply existing heuristics to a new problem** — implement just three traits (`ProblemTrait` + `Rankable` on `Solution` + `MoveToNeighbor`) and every heuristic and the benchmark pipeline work as-is. Add `Evaluate<f64>` for SA/LAHC, `EnabledTabu` for TabuSearch, or `CdclEncodable` for CDCL.
3. **Combine heuristics and run benchmarks** — compose components with `Sequential` / `Iterated` / `Restart` / `GeneticAlgorithm`, write a TOML config, and get aggregated best/avg/worst/std/time results.

## Module Map

```
src/
├── lib.rs / main.rs / prelude.rs / error.rs (OptError) / benchmark.rs
├── search_state/
│   └── mod.rs                SearchState<'a, P>, SearchStateCloneType
├── trait_defs/               core traits (re-exported via search_state & prelude)
│   ├── rankable.rs           Rankable, filter_best, Distance
│   ├── problem.rs            ProblemTrait
│   ├── neighbor.rs           MoveToNeighbor
│   ├── evaluate.rs           Evaluable, Evaluate
│   ├── crossover.rs          Crossover, SubProblemExtractable
│   └── tabu.rs               EnabledTabu
├── heuristic/
│   ├── mod.rs                Heuristic / ParallelHeuristic trait, StopCondition
│   ├── local_search.rs / simulated_annealing.rs (+BangBang) / tabu_search.rs
│   ├── late_acceptance.rs    LateAcceptanceHillClimbing<N>
│   ├── beam_search.rs / random_walk.rs
│   ├── sequential.rs         Sequential<P>, Iterated<P>  ← ILS lives here too
│   ├── restart.rs            Restart<P>
│   ├── genetic_algorithm.rs  GeneticAlgorithm<P, C>
│   ├── crossover.rs          SubProblemBasedCrossover<P>
│   └── specific/
│       ├── bls_for_max_cut.rs   BreakoutLocalSearchForMaxCut
│       └── cdcl.rs              CdclSolver<P>
└── problem/
    ├── max_cut/              MaxCut, MaxCutSolution, MaxCut{Flip,Swap}Neighbor, MaxCutUniformCrossover
    ├── qubo/                 Qubo, QuboSolution, Qubo{Flip,Swap}Neighbor, QuboUniformCrossover
    ├── sat/                  Sat, SatSolution, Sat{Flip,Swap}Neighbor, SatUniformCrossover
    ├── tsp_2d/               TspWithCoordinates, TspSolution, Tsp{TwoOpt,Relocate}Neighbor, TspOrderCrossover
    └── binary_optimization/  FormulaProblem, Expr, Formula{Flip,Swap}Neighbor, FormulaUniformCrossover
```

## Core Traits (`src/trait_defs/`)

These live in `src/trait_defs/` and are re-exported via `crate::search_state::*` and the `prelude`, so `crate::search_state::ProblemTrait` and `use optopus::prelude::*` keep working.


- **`Rankable`**: `is_better_than(&Self) -> bool`. Implemented by every `Solution`; the optimization direction is baked into the problem. The `filter_best(iter)` helper returns the set of tied-best items.
- **`ProblemTrait`**: `type Solution: Clone + Rankable; fn new_solution(&self, rng) -> Solution`.
- **`MoveToNeighbor<P>`**: a single one-step move.
  ```rust
  fn iter(prob, sol) -> impl Iterator<Item = Self> + Send;     // lazy
  fn apply_to_solution(&self, prob, sol) -> Result<(), OptError>;
  fn move_to_be_better_than(&self, prob, src, other) -> bool;  // default: clone + apply
  fn apply_to_iteration(&self, iter: u64) -> u64;              // default: iter + 1
  ```
- **`Evaluable<T>` / `Evaluate<T>`** (default `T = f64`): `Maximize(T)` / `Minimize(T)` carries the direction of an objective delta. `Evaluable<f64>::worsening_amount()` normalizes both directions to "positive = worse" (used by `boltzmann_accept`). Required for SA / LAHC. QUBO also exposes `Evaluate<Coefficient = i32>` for integer gains.
- **`Crossover<P>`**: `crossover(&mut self, prob, sol1, sol2) -> Solution` (exactly two parents; uses an internal RNG). `apply_to_crossover_count(count) -> u64` (default `count + 1`) lets operators report a dynamic cost. `&mut self` allows stateful operators (e.g. `SubProblemBasedCrossover` runs an inner sub-heuristic).
- **`EnabledTabu`**: `type TabuMap: Default`, `is_move_enabled(map, iter)`, `add_to_tabu_map(map, iter, tenure: (u64, u64))`. Required by TabuSearch.
- **`SubProblemExtractable`**: `extract_sub_problem(sol1, sol2) -> Self`, `lift_solution(sol1, sol2, sub_sol)`. Variables that agree in both parents are fixed; the disagreeing variables form the sub-problem.
- **`CdclEncodable`**: lets a problem be solved by the CDCL engine via a CNF encoding. Methods: `cdcl_num_vars`, `cdcl_clauses() -> &[Vec<i64>]` (DIMACS-style, 1-indexed), `solution_from_assignment`, `count_satisfied`, `assignment_from_solution`. Implemented by MaxCut / QUBO / SAT.

## SearchState (`src/search_state/mod.rs`)

```rust
pub struct SearchState<'a, P: ProblemTrait> {
    pub instance: &'a P,
    pub solution: P::Solution,        // current
    pub best_solution: P::Solution,   // global best
    pub iteration: u64, pub best_iteration: u64,
    pub best_time: Instant,
    pub crossover_count: u64,         // updated by GA / Crossover
    pub(crate) start_iteration: u64, pub(crate) start_time: Instant,  // sub-run mgmt
}
```

**Key methods**: `new(problem)`, `with_solution(problem, sol)` (warm start), `apply(neighbor)` (apply + iter + best update), `apply_crossover(op, sol1, sol2) -> Solution`, `update_best()`, `progress_iteration()`, `is_neighbor_better_than_{current,best}(n)`, `get_best_move_par_chunks(iter, chunk_size)` (rayon-parallel), `duration()`.

**Sub-run clone/merge pattern** (used by every meta-heuristic to isolate phases):
```rust
let mut sub = state.clone_for_new_run(SearchStateCloneType::ClearBest);
// Simple    — keeps all counters and best (sets start_iteration to current)
// ClearBest — resets best and timers to current state (iteration = 0)
// StartBest — restarts from best_solution                (iteration = 0)
inner_heuristic.run(&mut sub)?;
state.update_state(sub);   // merges best into parent and accumulates iterations
```

## Heuristic Algorithms (`src/heuristic/`)

**`Heuristic<P>` trait**: `clear()`, `is_done(state)`, `run_once(state)`, `run(state)` (default: `clear` → loop `run_once`). The sub-trait `ParallelHeuristic<P>` adds `run_once_par` / `run_par` (default delegates to the sequential versions).

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
| `LocalSearch<N>` | Greedy best-improving; halts at a local optimum (`max_failed_update = 1`) |
| `SimulatedAnnealing<N>` | Random neighbor, `exp(-Δ/T)` acceptance, multiplicative cooling (requires `Evaluate<f64>`) |
| `BangBangSimulatedAnnealing<N>` | Oscillating temperature between `min_threshold` and `max_threshold` |
| `LateAcceptanceHillClimbing<N>` | LAHC: accepts a move if it is no worse than the score `history_length` steps ago (requires `Evaluate<f64>`) |
| `TabuSearch<N>` | Best non-tabu neighbor; aspiration overrides tabu when global best is improved; tenure ∈ `(min, max)` |
| `RandomWalk<N>` | Uniform random move with unconditional acceptance (useful as a perturbation) |
| `BeamSearch<P, N>` | Maintains top-`k` candidates; expands the full neighborhood of every beam member each iteration |

### Meta
| Type | Description |
|---|---|
| `Sequential<P>` | Runs a `Vec<Box<dyn Heuristic<P>>>` in order (each step on a `ClearBest` clone) |
| `Iterated<P>` | ILS: alternates `search` and `perturbation` (`Box<dyn Heuristic<P>>`); lives in `sequential.rs` |
| `Restart<P>` | Runs the inner heuristic; when `restart_condition` triggers, replaces `solution` with a fresh random one (best is preserved) |
| `GeneticAlgorithm<P, C>` | 2-parent tournament → `Crossover<P>` → mutation (`Heuristic<P>`) → worst-replacement; tracks `best_idx` incrementally |

### Crossover
- `SubProblemBasedCrossover<P>` (`crossover.rs`): builds a sub-problem from disagreeing variables, solves it with `sub_heuristic`, then lifts the result. Requires `P: SubProblemExtractable`. `apply_to_crossover_count` returns the cost as `sub_state.iteration`.
- `*UniformCrossover`: per-variable random parent selection; one per problem (MaxCut / QUBO / SAT / Formula).
- `TspOrderCrossover` (OX): order crossover for TSP.

### Problem-specific
- `BreakoutLocalSearchForMaxCut` (`specific/bls_for_max_cut.rs`): greedy local search plus adaptive perturbation (strong / weak flip / swap), with probabilities decaying via the non-improvement counter `omega`.
- `CdclSolver<P>` (`specific/cdcl.rs`, `P: CdclEncodable`): CDCL with VSIDS and Luby restart for MaxSAT. Defaults: `CDCL_DEFAULT_VAR_DECAY = 0.95`, `CDCL_DEFAULT_RESTART_BASE = 100`, `CDCL_DEFAULT_SNAPSHOT_INTERVAL = 50`.

## Problem Types (`src/problem/`)

| Problem | Direction | Solution | Neighbors | Crossover | Notes |
|---|---|---|---|---|---|
| **MaxCut** | Max | `cut: Vec<bool>`, `gain: Vec<f32>`, `objective: f32` | Flip / Swap | `MaxCutUniformCrossover` | format: `N M / i j w` (1-indexed); implements `CdclEncodable`; optional `positive_gain` index (advanced) |
| **QUBO** | Min | `x: Vec<bool>`, `gain: Vec<i32>`, `objective: i32` | Flip / Swap | `QuboUniformCrossover` | `Coefficient = i32`; `SubProblemExtractable` (bias folding); `CdclEncodable`; auto-converts MaxCut format; optional `negative_gain` index (advanced) |
| **MaxSAT** | Max | `x: Vec<bool>` (0-indexed), `n_satisfied: usize`, `gain: Vec<i64>` | Flip / Swap | `SatUniformCrossover` | DIMACS CNF; `CdclEncodable` |
| **TSP 2D** | Min | `tour: Vec<usize>`, `objective: f64`, `gain: HashMap<(edge, edge), f64>` (cached 2-opt gains) | TwoOpt / Relocate | `TspOrderCrossover` (OX) | TSPLIB |
| **FormulaProblem** | Configurable (`OptDirection`) | `x: Vec<bool>`, `score: f64` (always higher-is-better), `gain: Vec<f64>` | Flip / Swap | `FormulaUniformCrossover` + `SubProblemExtractable` | see below |

**FormulaProblem details**: AST `Expr = Const(f64) | Var(usize) | Neg | Add(Vec) | Mul(Vec)` with `+ - * /` operators. Constraints: `Comparison { lhs, rel: ConstraintRel, rhs, penalty_weight }` (Lt / Gt / Le / Ge / Eq) or `Clamp { expr, lo, hi, penalty_weight }`. A pre-compiled polynomial (`CompiledPoly`) gives O(d) gain deltas; `interaction_neighbors` tracks which variables' gains may change on each flip (variables sharing a monomial in the objective, plus all variables co-appearing in the same constraint expression).

## Benchmarking (`src/benchmark.rs`)

TOML config → `BenchmarkConfig` → run each heuristic on each instance N times (rayon-parallel) → `BenchmarkReport` → timestamped TOML in `result/` (`SingleRunResult` / `Summary` / `InstanceHeuristicResult`).

```toml
num_runs = 10
[[instances]]
path = "data/instances/max_cut/G*.txt"   # globs supported
problem = "MaxCut"             # MaxCut | Qubo | Sat | Tsp
[[heuristics]]
kind = "LocalSearch"           # see list below
neighbor = "Flip"              # Flip | Swap | TwoOpt | Relocate
[heuristics.stop_condition]
max_iteration = 100000         # max_duration_secs / max_failed_update also supported
```

**Supported `kind` values**: `LocalSearch`, `TabuSearch`, `SimulatedAnnealing`, `LateAcceptanceHillClimbing`, `Cdcl`, `BreakoutLocalSearch` (MaxCut only), and the meta-heuristics `Sequential` / `Iterated` / `Restart` (which take a nested `steps` array; `Iterated` uses `steps[0] = search, steps[1] = perturbation`; `Restart` also requires `restart_condition`).
Optional fields: `tabu_tenure: (u64, u64)`, `initial_temperature`, `cooling_rate`, `t / l0 / p0 / q` (BLS), `history_length` (LAHC), `var_decay / restart_base / snapshot_interval` (CDCL).

**`Summary` fields**: `num_successful_runs`, `best/avg/worst/std_objective`, `best/avg_time_to_best_secs`, `avg_total_time_secs`. Each `SingleRunResult` carries `best_objective: f64`, `best_iteration: u64`, `time_to_best_secs: f64`, `total_time_secs: f64`, and `solution: Vec<usize>` (0-indexed encoding).

## Key Design Patterns

1. **Gain-based incremental updates** — every solution caches per-variable `gain`. Applying a move only refreshes the affected neighbors in O(degree). MaxCut and QUBO additionally offer optional `positive_gain` / `negative_gain` indexes (advanced) to enumerate only improving moves in O(|improving|) — these are used by problem-specific heuristics like BLS, not needed for standard use. `FormulaProblem` uses `interaction_neighbors` to refresh only the variables whose gain can actually change.
2. **Sub-run clone/merge** — every meta-heuristic uses `clone_for_new_run` → run inner → `update_state` to merge the best back. The global iteration counter advances monotonically across all phases.
3. **Tabu abstraction via trait** — `TabuSearch` is generic over `N: EnabledTabu`. Each move type owns its `TabuMap`, decoupling tabu policy from the search algorithm.
4. **Always-higher-is-better in `FormulaProblem`** — for `Maximize`, `score = objective − penalty`; for `Minimize`, `score = −objective − penalty`. Heuristics only need the higher-is-better convention.
5. **`MoveToNeighbor::iter()` is lazy** — `LocalSearch` selects with `max_by` in O(n) without collecting; `TabuSearch` uses `filter_best` plus aspiration; `RandomWalk` uses `.choose()`. Only `BeamSearch` materializes all candidates (and `get_best_move_par_chunks` evaluates them in parallel).
