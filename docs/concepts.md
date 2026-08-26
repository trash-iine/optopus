# Concepts

## Design philosophy

Optopus separates three orthogonal concerns:

- **Problems**: *what* to optimize (MaxCut, TSP, ...)
- **Heuristics**: *how* to optimize (Local Search, SA, ...).
- **Search state**: iteration count, timing, current and best solutions.

Every heuristic runs under the search state, recording its behavior and enable to compare heuristics in the same condition.

## Three use cases

1. **Existing problem × existing heuristic** — run `LocalSearch`,
   `SimulatedAnnealing`, ... on MaxCut, TSP, ... in a few lines via `use optopus::prelude::*`.
2. **Apply existing heuristics to a new problem** — implement the three core
   traits (`ProblemTrait` on the problem, `Rankable` on the solution and on the
   move, `MoveToNeighbor` on the solution move across the problem) and
   `LocalSearch`, `RandomWalk`, `BeamSearch` and every meta-heuristic work as-is.
   The rest are unlocked one trait at a time: `Evaluate<f64>` for SA / LAHC /
   RL Search, `EnabledTabu` for Tabu Search, `Distance` + a `Crossover` for
   Genetic Algorithm. 
   Full signatures and the per-heuristic requirement matrix: see
   [Core traits](traits.md#core-trait-reference).
3. **Compose heuristics and benchmark them** — use `Sequential`, `Iterated`,
   `VariableNeighborhoodSearch`, `Restart`, or `GeneticAlgorithm` to combine
   algorithms; describe a comparison in TOML and run the CLI to get aggregated
   statistics.

## `SearchState`

`SearchState<'a, P>` is the shared scratch-pad that flows through every
heuristic: it owns the current solution, the global best, the iteration
counter, and timing. Heuristics never inspect a problem directly. they
mutate `SearchState`.

Full struct, methods, and clone variants: see
[SearchState API](search_state.md).

### Sub-run clone/merge pattern

Every meta-heuristic isolates a phase on a cloned state, then merges only the
best solution back. This is what lets `Sequential`, `Iterated`,
`VariableNeighborhoodSearch`, `Restart`, and `GeneticAlgorithm` compose freely
while keeping iteration counts monotonic.

```rust
let mut sub = state.clone_for_new_run(SearchStateCloneType::ClearBest);
inner_heuristic.run(&mut sub)?;
state.update_state(sub);   // merges best back, accumulates iteration count
```

The three clone semantics (`Simple` / `ClearBest` / `StartBest`) are tabulated
in [SearchState API](search_state.md#searchstateclonetype-variants).

