# GreedyRandomizedConstruction

**API:** [`GreedyRandomizedConstruction`](../api/optopus/heuristic/struct.GreedyRandomizedConstruction.html)

The construction phase of GRASP. It empties the incumbent and builds a new
solution one element at a time. At every step each element still out is priced
at its cheapest placement, and one is drawn uniformly from the restricted
candidate list, the elements whose cost is within `alpha` of the cheapest,

```text
cost ≤ c_min + alpha · (c_max − c_min)
```

`alpha = 0` is pure greedy and `alpha = 1` takes the elements in a random order,
each at its cheapest place.

It runs on any problem implementing [`Ruinable`](../traits.md) whose solution
implements [`Evaluate`](../traits.md). In a config that is TSP, VRP, the
[flow shop](../problems/flow_shop.md) and [bin packing](../problems/bin_packing.md).

## GRASP is a composition

A run builds exactly one solution and stops, so this is a step for a
composition rather than a search. GRASP itself, a construction and a descent
repeated with the best kept, is [`Iterated`](meta.md) with this as its second
step.

```rust
use optopus::prelude::*;

let fs = FlowShop::load_file("data/instances/flow_shop/taillard/ta051.txt")?;
let budget = StopCondition::duration(std::time::Duration::from_secs(10));
let mut state = SearchState::new_with_seed(&fs, 42);

Iterated::<FlowShop>::new(
    budget.clone(),
    Box::new(LocalSearch::<FlowShopInsertNeighbor>::new(budget.clone())),
    Box::new(GreedyRandomizedConstruction::<FlowShop>::new(
        StopCondition::new(None, None, None),
        /* alpha = */ 0.2,
    )),
)
.run(&mut state)?;
println!("makespan = {}", state.best_solution.objective);
```

[`examples/compose_grasp.rs`](https://github.com/trash-iine/optopus/blob/main/examples/compose_grasp.rs)
runs this on the flow shop and on bin packing, with a variable neighborhood
descent as the search there, against multi-start descent.

## Where it pays

The greedy rule is cheapest insertion first, whatever the problem's
`insertion_cost` makes of that, so the construction is as good as that rule.

- On the flow shop the cheapest insertion is the rule NEH uses, and GRASP beats
  multi-start descent on Taillard's 50 × 20 instances.
- On bin packing the cheapest item to insert is one that fits an open bin, so
  the large items wait until last and each opens a bin, the opposite of First
  Fit Decreasing. There GRASP loses to multi-start descent, and ALNS is far
  ahead of both.
- On a tour with 2-opt it is about level with multi-start descent.

Every step reprices every element left, so building a solution of `n` elements
is O(n² · buckets · places). That is milliseconds on a few hundred elements and
slow on thousands.

## Constructor

```rust
GreedyRandomizedConstruction::<P>::new(
    stop_condition: StopCondition,   // can end a run early, never makes it build twice
    alpha: f64,                      // in [0, 1]
) -> Self
```

Panics if `alpha` is outside `[0, 1]`. The operator itself is the free function
`randomized_greedy_insertion` in `building_blocks::search`, beside the repair
operators of ALNS, for a search that wants to recreate part of a solution
this way.

## Benchmark config

GRASP in a config is the same composition.

```toml
[[heuristics]]
kind = "Iterated"
[heuristics.stop_condition]
max_duration_secs = 30.0

[[heuristics.steps]]
kind = "LocalSearch"
neighbor = "Relocate"

[[heuristics.steps]]
kind = "GreedyRandomizedConstruction"
alpha = 0.2                 # optional (default shown), checked to be in [0, 1]
```

## References

- Feo, T. A. and Resende, M. G. C. "Greedy randomized adaptive search
  procedures." *Journal of Global Optimization*, 6(2), 109-133, 1995.
- Laguna, M., Martí, R., Martínez-Gavara, A., Pérez-Peló, S. and Resende,
  M. G. C. "Greedy randomized adaptive search procedures with path
  relinking. An analytical review of designs and implementations."
  *European Journal of Operational Research*, 327(3), 717-734, 2025.
