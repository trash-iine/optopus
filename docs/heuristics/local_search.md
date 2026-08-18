# LocalSearch

Greedy best-improving hill climbing: at each step, evaluate every move in the
neighborhood, apply the strictly best one, and stop as soon as no improving
move exists (a local optimum).

## Constructor

```rust
LocalSearch::<N>::new(stop_condition: StopCondition) -> Self
```

`N` must satisfy `MoveToNeighbor<P> + Rankable`. The selection uses `max_by`
on a lazy `iter()` — no allocation per step.

## Behavior

- `max_failed_update` is forced to `Some(1)`. If you pass any other value it
  is overwritten and a warning is logged: a local-optimum step *is* a
  failed update by definition.
- The stop condition still applies (iterations / duration), so combine
  `LocalSearch` with `Restart` or `Iterated` for budgets larger than a single
  hill climb.

## Example

```rust
use optopus::prelude::*;

let mc = MaxCut::new(Graph::from_edges([(0, 1, 1.0), (0, 2, 1.0), (1, 2, 1.0)]));
let mut state = SearchState::new(&mc);
let mut ls = LocalSearch::<MaxCutFlipNeighbor>::new(StopCondition::iterations(1_000));
ls.run(&mut state)?;
# Ok::<(), optopus::error::OptError>(())
```

## Benchmark config

```toml
[[heuristics]]
kind = "LocalSearch"
neighbor = "Flip"        # required; the valid values are per-problem
[heuristics.stop_condition]
max_iteration = 100_000
```

`max_failed_update` is forced to `1` whatever the config says (see
[Behavior](#behavior)), so the useful budget keys here are `max_iteration` and
`max_duration_secs` — and they only bound a single hill climb. For a real
budget, nest `LocalSearch` inside [`Restart` or `Iterated`](meta.md).

## References

- Aarts, E. and Lenstra, J. K. (eds.) *Local Search in Combinatorial
  Optimization*. Princeton University Press, 2003.
