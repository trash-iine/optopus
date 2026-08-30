# LocalSearch

**API:** [`LocalSearch`](../api/optopus/heuristic/struct.LocalSearch.html)

Greedy best-improving hill climbing: at each step, evaluate every move in the
neighborhood, apply the strictly best one, and stop as soon as no improving
move exists (a local optimum).

## Example

```rust
use optopus::prelude::*;

let mc = MaxCut::new(Graph::from_edges([(0, 1, 1.0), (0, 2, 1.0), (1, 2, 1.0)]));
let mut state = SearchState::new(&mc);
let mut ls = LocalSearch::<MaxCutFlipNeighbor>::new(StopCondition::iterations(1_000));
ls.run(&mut state)?;
println!("cut weight = {}", state.best_solution.objective);
# Ok::<(), optopus::error::OptError>(())
```

## Algorithm sketch

Each `run_once`:

1. **Enumerate** the neighborhood through the lazy `N::iter`, keeping only the
   moves that are strictly better than the current solution.
2. **Select** the best of them with `max_by` over `rank_cmp`. Nothing is
   collected, so a step costs no allocation; ties go to the last one the
   iterator yields, which is arbitrary but harmless for hill climbing.
3. **Apply** it — or, when the filter left nothing, raise the local-optimum
   flag `is_done` reads and advance the iteration counter.

## Constructor

```rust
LocalSearch::<N>::new(stop_condition: StopCondition) -> Self
```

`N` must satisfy `MoveToNeighbor<P> + Rankable`.

`clear()` drops the local-optimum flag that `is_done` reads, so a second `run`
climbs again instead of reporting itself done at once — which is what makes
`LocalSearch` reusable as the search phase of a meta-heuristic.

## Behavior

For `StopCondition`,

- `max_failed_update` is forced to `Some(1)`. If you pass any other value it
  is overwritten and a warning is logged: a local-optimum step is a
  failed update by definition.
- The stop condition still applies (iterations / duration), so combine
  `LocalSearch` with `Restart` or `Iterated` for budgets larger than a single
  hill climb.

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
