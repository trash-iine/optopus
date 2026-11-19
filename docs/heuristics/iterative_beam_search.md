# IterativeBeamSearch

**API:** [`IterativeBeamSearch`](../api/optopus/heuristic/struct.IterativeBeamSearch.html)

Iterative beam search builds a solution from nothing, one element at a time,
keeping the best few partial solutions at every step, and then does it again
with twice as many. A run starts with a greedy construction and spends its
later time on wider and wider searches.

It runs on any problem implementing [`Ruinable`](../traits.md) whose
`Partial` is `Clone` and whose solution implements
[`Evaluate`](../traits.md). In a config that is [TSP](../problems/tsp.md),
[VRP](../problems/vrp.md), the [flow shop](../problems/flow_shop.md) and
[bin packing](../problems/bin_packing.md).

It is a different search from [`BeamSearch`](beam_search.md), which keeps a
beam of complete solutions and expands each by the moves of a neighborhood.

## Example

```rust
use optopus::prelude::*;

let fs = FlowShop::load_file("data/instances/flow_shop/taillard/ta051.txt")?;
let mut state = SearchState::new_with_seed(&fs, 42);

let mut search = IterativeBeamSearch::<FlowShop>::new(StopCondition::iterations(8));
search.run(&mut state)?;
println!("makespan = {}", state.best_solution.objective);
```

Eight passes run at the widths 1, 2, 4 and so on up to 128.

## Algorithm sketch

Each `run_once` is one pass at the current width `D`.

1. Empty the incumbent and draw an order for its elements.
2. Start the beam with the empty partial solution.
3. For the next element in the order, price every placement in every partial
   solution of the beam with `insertion_cost`.
4. Keep the `D` placements that leave the lowest `partial_energy`, and carry
   them out. They are the next beam.
5. When every element is placed, the best of the beam is the solution of the
   pass. It becomes the incumbent, and the best solution of the run is kept
   as usual.
6. Multiply `D` by `growth`.

A pass prices `D · n` placements per element, so its cost grows with the
square of the number of elements and linearly with the width. A partial
solution is copied for each placement kept beyond the first in it, up to `D`
copies per element.

The pass at width one is a greedy insertion in a random order. A wider pass
is not the same search with more room, since its order is drawn anew, so the
solutions do not improve pass by pass. They improve on average.

An iteration count stops the run between passes. A time limit also ends a
pass between two elements, and that pass is discarded.

The widening restart is the one the flow shop search in the references uses.
The branching differs. That search appends any job not yet scheduled to an
end of the sequence and ranks a node by a bound with the idle time. This one
fixes the next element by the drawn order, inserts it anywhere, and ranks by
the energy of the partial solution, which is what every `Ruinable` problem can
answer.

## Constructor

```rust
IterativeBeamSearch::<P>::new(stop_condition: StopCondition) -> Self
```

`stop_condition` decides when the run ends, see [Stop
conditions](../guide/stop_conditions.md).

| Builder | Sets | Default |
|---|---|---|
| `with_initial_width(width)` | the width of the first pass | `1` |
| `with_growth(growth)` | the factor the width grows by each pass, at least one | `2.0` |

A growth of one keeps every pass at the initial width, each with a new order.
`clear()` returns the width to its initial value.

## Benchmark config

```toml
[[heuristics]]
kind = "IterativeBeamSearch"
initial_width = 1     # optional (default shown)
growth = 2.0          # optional (default shown)
[heuristics.stop_condition]
max_duration_secs = 30.0
```

## References

- Ow, P. S. and Morton, T. E. "Filtered Beam Search in Scheduling."
  *International Journal of Production Research*, 26(1), 35-62, 1988.
- Libralesso, L., Focke, P. A., Secardin, A. and Jost, V. "Iterative beam
  search algorithms for the permutation flowshop." *European Journal of
  Operational Research*, 301(1), 217-234, 2022.
