# TabuSearch

At each step, pick the strictly best move that is not currently tabu, then
mark it tabu for a tenure drawn uniformly from `tabu_tenure = (min, max)`.

A tabu move is still selectable when it satisfies the **aspiration criterion**:
the resulting solution would be strictly better than the current global best.

## Constructor

```rust
TabuSearch::<N>::new(
    stop_condition: StopCondition,
    tabu_tenure: (u64, u64),
    tabu_map: Option<N::TabuMap>,
) -> Self
```

`N` must satisfy `MoveToNeighbor<P> + Clone + EnabledTabu + Rankable`.

`tabu_map` lets you inject a pre-warmed map (e.g. inherited from a previous
phase). Passing `None` starts from `N::TabuMap::default()`.

**Panics** if `tabu_tenure.0 > tabu_tenure.1`.

`clear()` resets the tabu map to its default value.

## Tabu map abstraction

Each neighbor type owns its `TabuMap` and the policy for inserting / querying
it via the `EnabledTabu` trait — `TabuSearch` is generic over the neighbor
and never knows what's stored. This lets QUBO/MaxCut/SAT key by variable
index, TSP by edge pair, Job Shop by swap position, etc.

`borrow_tabu_map`, `borrow_mut_tabu_map`, `take_tabu_map`, and `set_tabu_map`
let you inspect or transfer state between runs.

## Example

```rust
use optopus::prelude::*;

let mc = MaxCut::new(Graph::from_edges([(0, 1, 1.0), (0, 2, 1.0), (1, 2, 1.0)]));
let mut state = SearchState::new(&mc);
let mut ts = TabuSearch::<MaxCutFlipNeighbor>::new(
    StopCondition::iterations(10_000),
    /* tabu_tenure = */ (5, 10),
    None,
);
ts.run(&mut state)?;
# Ok::<(), optopus::error::OptError>(())
```
