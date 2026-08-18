# Sequential / Iterated / VariableNeighborhoodSearch / Restart

Four meta-heuristics that compose other heuristics via the **sub-run
clone/merge pattern**:

```rust
let mut sub = state.clone_for_new_run(SearchStateCloneType::ClearBest);
inner.run(&mut sub)?;
state.update_state(sub);   // merges best back, accumulates iteration count
```

The global iteration counter advances monotonically across all phases. The
trait-level details and the three `SearchStateCloneType` variants are in
[concepts.md](../concepts.md#sub-run-clonemerge-pattern) and the
[SearchState API](../search_state.md#searchstateclonetype-variants).

Which one to reach for:

- **`Sequential`** — a fixed pipeline of phases, in order (say `LocalSearch` to
  clean up a random start, then `TabuSearch`).
- **`Iterated`** — escape local optima by alternating search with a perturbation
  (ILS). The default choice when one search stalls.
- **`VariableNeighborhoodSearch`** — the same idea with several shake
  neighborhoods of growing strength, escalating only when the current one fails.
- **`Restart`** — throw the incumbent away and draw a fresh random solution once
  progress stops; the global best survives.

For a population instead of a single incumbent, see
[GeneticAlgorithm](genetic_algorithm.md).

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
heuristics each carry their own stop condition. The cycle re-runs from the top
once it reaches the end of the list.

```rust
use optopus::prelude::*;

let mc = MaxCut::new(Graph::from_edges([(0, 1, 1.0), (1, 2, 1.0), (0, 2, 1.0)]));
let mut state = SearchState::new(&mc);

let mut seq = Sequential::<MaxCut>::new(
    StopCondition::iterations(100_000),
    vec![
        Box::new(LocalSearch::<MaxCutFlipNeighbor>::new(
            StopCondition::failed_updates(1),
        )),
        Box::new(TabuSearch::<MaxCutFlipNeighbor>::new(
            StopCondition::failed_updates(500),
            (5, 10),
            None,
        )),
    ],
);
seq.run(&mut state)?;
# Ok::<(), optopus::error::OptError>(())
```

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

```rust
use optopus::prelude::*;

let mc = MaxCut::new(Graph::from_edges([(0, 1, 1.0), (1, 2, 1.0), (0, 2, 1.0)]));
let mut state = SearchState::new(&mc);

let mut ils = Iterated::<MaxCut>::new(
    StopCondition::iterations(100_000),
    Box::new(LocalSearch::<MaxCutFlipNeighbor>::new(
        StopCondition::failed_updates(1),
    )),
    Box::new(RandomWalk::<MaxCutFlipNeighbor>::new(
        StopCondition::iterations(5),
    )),
);
ils.run(&mut state)?;
# Ok::<(), optopus::error::OptError>(())
```

## VariableNeighborhoodSearch

Basic Variable Neighborhood Search (VNS). Keeps an ordered list of shake
heuristics `N_1..N_kmax` (typically `RandomWalk` with growing budgets) and a
local `search`:

```rust
VariableNeighborhoodSearch::<P>::new(
    stop_condition: StopCondition,
    search: Box<dyn Heuristic<P>>,
    shakes: Vec<Box<dyn Heuristic<P>>>,   // must be non-empty
) -> Self
```

Cycle: snapshot the incumbent → shake in `N_k` → `search` → if the result
improves the incumbent, keep it and reset `k`; otherwise restore the incumbent
and advance `k` (wrapping around after the last neighborhood). The global best
survives either way.

```rust
use optopus::prelude::*;

let mc = MaxCut::new(Graph::from_edges([(0, 1, 1.0), (1, 2, 1.0), (0, 2, 1.0)]));
let mut state = SearchState::new(&mc);

let mut vns = VariableNeighborhoodSearch::<MaxCut>::new(
    StopCondition::iterations(100_000),
    Box::new(LocalSearch::<MaxCutFlipNeighbor>::new(
        StopCondition::failed_updates(1),
    )),
    vec![
        Box::new(RandomWalk::<MaxCutFlipNeighbor>::new(
            StopCondition::iterations(5),
        )),
        Box::new(RandomWalk::<MaxCutFlipNeighbor>::new(
            StopCondition::iterations(20),
        )),
        Box::new(RandomWalk::<MaxCutFlipNeighbor>::new(
            StopCondition::iterations(50),
        )),
    ],
);
vns.run(&mut state)?;
# Ok::<(), optopus::error::OptError>(())
```

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

Since the inner slot is any `Heuristic<P>`, the usual shape is a `Restart`
around an `Iterated`:

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

## Benchmark config

All four are one `kind` with a nested `steps` array; what each slot means is the
only difference:

```toml
[[heuristics]]
kind = "Iterated"            # Sequential | Iterated | VariableNeighborhoodSearch | Restart
[heuristics.stop_condition]
max_duration_secs = 30.0

[[heuristics.steps]]         # steps[0]
kind = "LocalSearch"
neighbor = "Flip"
[heuristics.steps.stop_condition]
max_failed_update = 1

[[heuristics.steps]]         # steps[1]
kind = "RandomWalk"
neighbor = "Flip"
[heuristics.steps.stop_condition]
max_iteration = 200
```

| `kind` | `steps` | Extra fields |
|---|---|---|
| `Sequential` | run in order, repeated until the outer stop condition | — |
| `Iterated` | `[0]` = search, `[1]` = perturbation | — |
| `VariableNeighborhoodSearch` | `[0]` = search, `[1..]` = shakes `N_1..N_kmax` | — |
| `Restart` | `[0]` = the inner heuristic | `restart_condition` (required, same shape as `stop_condition`) |

`Restart` is the only one with an extra table of its own:

```toml
[[heuristics]]
kind = "Restart"
[heuristics.restart_condition]     # required — when to reseed with a random solution
max_failed_update = 1_000
[heuristics.stop_condition]
max_duration_secs = 30.0

[[heuristics.steps]]               # steps[0] = the inner heuristic
kind = "TabuSearch"
neighbor = "Flip"
tabu_tenure = [5, 150]
[heuristics.steps.stop_condition]
max_iteration = 10_000
```

Steps nest arbitrarily deep: a `Restart` around an `Iterated` is the two blocks
above written inside each other, with the `steps` of the inner one indented one
level further as `[[heuristics.steps.steps]]`. See the full
[ILS example](../guide/benchmarking.md#nested-example-ils-in-toml).

## References

- Lourenco, H. R., Martin, O. C., and Stutzle, T. "Iterated Local Search."
  In Glover, F. and Kochenberger, G. A. (eds.), *Handbook of Metaheuristics*,
  pp. 320-353. Springer, 2003.
