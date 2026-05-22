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

`SearchState<'a, P>` is the shared scratch-pad that flows through every
heuristic: it owns the current solution, the global best, the iteration
counter, and timing. Heuristics never inspect a problem directly — they
mutate `SearchState`.

Full struct, methods, and clone variants: see
[SearchState API](search_state.md).

### Sub-run clone/merge pattern

Every meta-heuristic isolates a phase on a cloned state, then merges only the
best solution back. This is what lets `Sequential`, `Iterated`, `Restart`,
and `GeneticAlgorithm` compose freely while keeping iteration counts monotonic.

```rust
let mut sub = state.clone_for_new_run(SearchStateCloneType::ClearBest);
inner_heuristic.run(&mut sub)?;
state.update_state(sub);   // merges best back, accumulates iteration count
```

The three clone semantics (`Simple` / `ClearBest` / `StartBest`) are tabulated
in [SearchState API](search_state.md#searchstateclonetype-variants).

## Core traits

A problem only needs three traits to plug into every heuristic:
`ProblemTrait`, `Rankable` (on the solution), and `MoveToNeighbor`. Implement
additional traits to unlock specific algorithms: `Evaluate<f64>` for SA /
LAHC / RL Search, `EnabledTabu` for Tabu Search, `Crossover` for Genetic
Algorithm, `Distance` for some GA parent-selection strategies.

Full signatures and the per-heuristic requirement matrix: see
[Core traits](traits.md#core-trait-reference).

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
