# Sequential / Iterated / Restart

Three meta-heuristics that compose other heuristics via the **sub-run
clone/merge pattern**:

```rust
let mut sub = state.clone_for_new_run(SearchStateCloneType::ClearBest);
inner.run(&mut sub)?;
state.update_state(sub);   // merges best back, accumulates iteration count
```

The global iteration counter advances monotonically across all phases.

## Sequential

Runs a list of heuristics in order. Each one operates on a fresh
`ClearBest` clone; results are merged back between steps.

```rust
Sequential::<P>::new(
    stop_condition: StopCondition,
    heuristics: Vec<Box<dyn Heuristic<P>>>,
) -> Self

// Or build incrementally:
seq.push_heuristic(Box::new(...));
```

The outer `stop_condition` is checked between sub-heuristics; the inner
heuristics each carry their own stop condition.

## Iterated

Iterated Local Search (ILS) pattern. Alternates a `search` phase with a
`perturbation` phase:

```rust
Iterated::<P>::new(
    stop_condition: StopCondition,
    search: Box<dyn Heuristic<P>>,
    perturbation: Box<dyn Heuristic<P>>,
) -> Self
```

Cycle: `search` → check outer `stop_condition` → `perturbation` → repeat.
Both phases run on `ClearBest` clones; the global best survives.

A typical pairing: `search = LocalSearch`, `perturbation = RandomWalk` for a
few iterations.

## Restart

Runs an inner heuristic; whenever `restart_condition` is satisfied (typically
`max_failed_update`), replaces `state.solution` with a fresh random solution.
`state.best_solution` is preserved across restarts.

```rust
Restart::<P>::new(
    stop_condition: StopCondition,
    heuristic: Box<dyn Heuristic<P>>,
    restart_condition: StopCondition,
) -> Self
```

## Example

```rust
use optopus::prelude::*;

let mc = MaxCut::new(Graph::from_edges([(0, 1, 1.0), (1, 2, 1.0), (0, 2, 1.0)]));
let mut state = SearchState::new(&mc);

let ils = Iterated::<MaxCut>::new(
    StopCondition::iterations(10_000),
    Box::new(LocalSearch::<MaxCutFlipNeighbor>::new(StopCondition::failed_updates(1))),
    Box::new(RandomWalk::<MaxCutFlipNeighbor>::new(StopCondition::iterations(5))),
);

let mut solver = Restart::new(
    StopCondition::iterations(100_000),
    Box::new(ils),
    StopCondition::failed_updates(1_000),
);
solver.run(&mut state)?;
# Ok::<(), optopus::error::OptError>(())
```
