# RandomWalk

Sample a uniformly random neighbor and apply it unconditionally — no
acceptance test, no comparison. The best solution encountered along the walk
is still tracked in `state.best_solution`.

## Constructor

```rust
RandomWalk::<N>::new(stop_condition: StopCondition) -> Self
```

`N` must satisfy `MoveToNeighbor<P> + Rankable`.

## When to use

`RandomWalk` is rarely useful on its own; its main role is as the
**perturbation** phase of [`Iterated`](meta.md#iterated): a few random moves
push the search out of a local optimum so the next greedy phase can climb a
different basin.

## Example

```rust
use optopus::prelude::*;

let mc = MaxCut::new(Graph::from_edges([(0, 1, 1.0), (0, 2, 1.0), (1, 2, 1.0)]));
let mut state = SearchState::new(&mc);
let mut rw = RandomWalk::<MaxCutFlipNeighbor>::new(StopCondition::iterations(10));
rw.run(&mut state)?;
# Ok::<(), optopus::error::OptError>(())
```

## Benchmark config

```toml
[[heuristics]]
kind = "RandomWalk"
neighbor = "Flip"        # required; the valid values are per-problem
[heuristics.stop_condition]
max_iteration = 200      # give it one — a random walk never stops on its own
```

An empty `stop_condition` never terminates, which matters most where
`RandomWalk` is normally used: as a nested perturbation step. See the
[ILS example](../guide/benchmarking.md#nested-example-ils-in-toml).
