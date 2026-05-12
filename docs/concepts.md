# Concepts

## Design philosophy

Optopus separates three orthogonal concerns:

- **Problems** — *what* to optimize (MaxCut, QUBO, MaxSAT, TSP, Vertex Cover,
  Job Shop, custom formula).
- **Heuristics** — *how* to search (Local Search, SA, Tabu Search, GA, …).
- **`SearchState`** — iteration count, timing, current and best solutions.

Any heuristic works with any problem; no problem-specific code lives in the
heuristic layer. The CLI in `src/main.rs` reads a TOML benchmark config and
runs each heuristic on each instance.

## Three use cases

1. **Existing problem × existing heuristic** — run `LocalSearch`,
   `SimulatedAnnealing`, `TabuSearch`, etc. on MaxCut / QUBO / MaxSAT / TSP /
   Vertex Cover / Job Shop in a few lines via `use optopus::prelude::*`.
2. **Apply existing heuristics to a new problem** — implement the three core
   traits (`ProblemTrait`, `Rankable` on the solution, `MoveToNeighbor`) and
   every heuristic and the benchmark pipeline work as-is. Add `Evaluate<f64>`
   for SA / LAHC / RL Search, `EnabledTabu` for Tabu Search, `Crossover` for
   Genetic Algorithm. See [custom problem](guide/custom_problem.md).
3. **Compose heuristics and benchmark them** — use `Sequential`, `Iterated`,
   `Restart`, or `GeneticAlgorithm` to combine algorithms; describe a comparison
   in TOML and run the CLI to get aggregated statistics.

## `SearchState`

```rust
pub struct SearchState<'a, P: ProblemTrait> {
    pub instance: &'a P,
    pub solution: P::Solution,        // current
    pub best_solution: P::Solution,   // global best
    pub iteration: u64,
    pub best_iteration: u64,
    pub best_time: Instant,
    // pub(crate) start_iteration / start_time — sub-run management
}
```

Common methods:

| Method | What it does |
|---|---|
| `SearchState::new(problem)` | Random initial solution. |
| `SearchState::with_solution(problem, sol)` | Warm start from a known solution. |
| `apply(neighbor)` | Apply move + advance iteration + refresh best. |
| `apply_move_only(neighbor)` | Apply move + advance iteration; do **not** refresh best (used during perturbation phases). |
| `progress_iteration()` | Advance iteration with no move applied. |
| `update_best()` | Refresh best from current solution. |
| `is_neighbor_better_than_current(n)` / `_best(n)` | Lookahead checks. |
| `get_best_move_par_chunks(iter, chunk_size)` | Parallel best-move scan via Rayon. |
| `duration()` | Elapsed time since the current sub-run started. |
| `clone_for_new_run(kind)` + `update_state(sub)` | Sub-run isolation pattern (see below). |

### Sub-run clone/merge pattern

Every meta-heuristic uses this pattern to isolate a phase and then merge its
best solution back:

```rust
let mut sub = state.clone_for_new_run(SearchStateCloneType::ClearBest);
inner_heuristic.run(&mut sub)?;
state.update_state(sub);   // merges best back, accumulates iteration count
```

`SearchStateCloneType` variants:

| Variant | Solution | Best | Counters |
|---|---|---|---|
| `Simple` | current | retained | `start_iteration = iteration`; clocks unchanged |
| `ClearBest` | current | reset to current | `iteration = 0`, clocks reset |
| `StartBest` | best | retained | `iteration = 0`, clocks reset |

`update_state` panics if the sub-state references a different problem instance
and accumulates the sub-run's iteration delta into the parent counter.

## Core traits

All of the traits below live in `optopus::search_state` (re-exported from the
prelude). The first three are the minimum needed to support every heuristic;
the rest unlock specific algorithms.

| Trait | Required by | Key signature |
|---|---|---|
| `ProblemTrait` | every heuristic | `type Solution: Clone + Rankable; fn new_solution(&self, rng) -> Solution` |
| `Rankable` (on `Solution`) | every heuristic | `fn is_better_than(&self, other: &Self) -> bool` |
| `MoveToNeighbor<P>` | every heuristic | `fn iter(prob, sol) -> impl Iterator<Self> + Send`<br>`fn apply_to_solution(&self, prob, sol) -> Result<()>`<br>`fn move_to_be_better_than(&self, prob, src, other) -> bool` (default: clone + apply)<br>`fn apply_to_iteration(&self, iter) -> u64` (default: `iter + 1`) |
| `Rankable` (on the neighbor) | `LocalSearch`, `BeamSearch`, `RandomWalk`, `TabuSearch` | same signature; selects the best move among candidates |
| `Evaluate<T>` (returns `Evaluable<T>`) | `SimulatedAnnealing`, `LateAcceptanceHillClimbing`, `RLSearch` | `fn evaluate(&self) -> Evaluable<T>` (default `T = f64`); `Evaluable::Maximize(T)` / `Minimize(T)` carries the optimization direction. `Evaluable<f64>::worsening_amount()` normalizes both directions to "positive = worse" (used by `boltzmann_accept`). |
| `EnabledTabu` | `TabuSearch` | `type TabuMap: Default;`<br>`fn is_move_enabled(&self, map, iter) -> bool;`<br>`fn add_to_tabu_map(&self, map, iter, tenure: (u64, u64))` |
| `Crossover<P>` | `GeneticAlgorithm` | `fn crossover(&mut self, prob, sol1, sol2) -> P::Solution` (`&mut self` lets stateful operators run a sub-heuristic) |
| `SubProblemExtractable` | `SubProblemBasedCrossover` | `fn extract_sub_problem(&self, sol1, sol2) -> Self;`<br>`fn lift_solution(&self, sol1, sol2, sub_solution) -> Self::Solution` |
| `Distance` (on `Solution`) | `GeneticAlgorithm::ParentSelection::HammingTopK` | `fn distance(&self, other: &Self) -> usize` |

For QUBO, gain values are integers, so the relevant evaluators are
`Evaluate<i32>` (and `Evaluable<i32>`).

## Key design patterns

- **Gain-based incremental updates.** Every solution caches a per-variable
  `gain`. Applying a move only refreshes the affected neighbors in O(degree).
  MaxCut / QUBO additionally expose optional `positive_gain` / `negative_gain`
  indices for problem-specific solvers.
- **Sub-run clone/merge.** Each meta-heuristic runs an inner heuristic on a
  cloned state and merges only the best solution back. The global iteration
  counter advances monotonically across phases.
- **Tabu abstraction via trait.** `TabuSearch` is generic over `N: EnabledTabu`.
  Each move type owns its own `TabuMap`, decoupling tabu policy from the
  search algorithm.
- **Lazy `MoveToNeighbor::iter()`.** `LocalSearch` selects with `max_by` in
  O(n) without collecting; `TabuSearch` uses `filter_best` plus aspiration;
  `RandomWalk` uses `.choose()`. Only `BeamSearch` materializes all candidates,
  and `get_best_move_par_chunks` evaluates them in parallel.
- **Always-higher-is-better internals (`FormulaProblem`).** For `Maximize`,
  `score = objective − penalty`; for `Minimize`, `score = −objective − penalty`.
  Heuristics always treat higher `score` as better.
