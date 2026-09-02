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

`LocalSearch` halts at a local optimum whether or not the `StopCondition` says
anything: `is_done` also reports the iteration that found no improving move. So

- an empty `StopCondition::new(None, None, None)` reads exactly as "descend
  until nothing improves" — that is how
  [`BreakoutLocalSearchForMaxCut`](breakout_local_search.md) drives its descent;
- anything you do set is an *additional* budget on top of that, taken as
  written. `max_failed_update` included — it used to be rewritten to `Some(1)`
  here, which was redundant with the local-optimum test at best and wrong at
  worst, since `is_done` reads it as `iteration - best_iteration >= 1` and so a
  descent starting *below* the incumbent best satisfied it before taking a
  single move;
- a budget only ever bounds a single hill climb, so combine `LocalSearch` with
  `Restart` or `Iterated` for anything larger.

## Benchmark config

```toml
[[heuristics]]
kind = "LocalSearch"
neighbor = "Flip"        # required; the valid values are per-problem
[heuristics.stop_condition]
max_iteration = 100_000
```

Every key is honoured as written (see [Behavior](#behavior)), but all of them
only bound a single hill climb — the run ends at the local optimum regardless.
For a real budget, nest `LocalSearch` inside [`Restart` or `Iterated`](meta.md).

## References

- Aarts, E. and Lenstra, J. K. (eds.) *Local Search in Combinatorial
  Optimization*. Princeton University Press, 2003.
