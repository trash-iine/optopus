# WeightedLocalSearch

**API:** [`WeightedLocalSearch`](../api/optopus/heuristic/struct.WeightedLocalSearch.html)

Weighted Local Search gives every constraint of a problem a weight, descends
on the weighted cost, and raises the weights of the constraints it finds
violated each time the descent stops. A constraint the search keeps failing
comes to outweigh the others, so the search is pushed to satisfy it without
being told which constraints are the hard ones.

Runs on any problem implementing
[`WeightedConstraints`](../traits.md#core-trait-reference). In a config it is
registered for [MaxSAT](../problems/sat.md), whose constraints are the clauses,
and for [vertex cover](../problems/vertex_cover.md) and
[graph coloring](../problems/graph_coloring.md), whose constraints are the
edges.

## Example

```rust
use optopus::prelude::*;

let vc = VertexCover::load_file("data/instances/max_cut/G22")?;
let mut state = SearchState::new_with_seed(&vc, 42);

WeightedLocalSearch::<VertexCover>::new(StopCondition::duration(
    std::time::Duration::from_secs(10),
))
.run(&mut state)?;
println!("cover size = {}", state.best_solution.cover_size);
```

## Algorithm sketch

Every weight is an integer and starts at one. The weighted cost of a solution
is its objective plus the weights of the constraints it violates, and the gain
of a move is how far it lowers that cost.

Each `run_once` makes one step.

1. Scan the problem's moves and take the one with the largest positive gain
   among those the tabu memory allows, drawn uniformly among equals.
2. If there is none, the solution is a weighted local optimum. With
   probability `1 − smoothing` add one to the weight of every violated
   constraint. Otherwise take one from the weight of every satisfied
   constraint that is above one.
3. Then leave the optimum by the best of the escape moves the problem
   offers. While a constraint is violated these are the moves that satisfy
   one violated constraint drawn uniformly. Once nothing is violated they are
   the moves that work toward a better objective, a vertex leaving a cover or
   leaving the smallest color class. An escape the tabu memory forbids is
   taken only when it forbids all of them.

A move that was applied may not be undone for a number of steps drawn from
`tabu_tenure`. A weight never rises above the larger of 1000 and the number of
constraints.

The best solution is kept by the problem's own order, which the weights never
enter, so what the run reports is comparable with every other heuristic. The
weights belong to the search and start again at one on every `run`.

A run ends on its stop condition. It also ends at a weighted local optimum
with no escape, which is a solution that violates nothing and that no move
takes toward a better objective. On MaxSAT that is an assignment satisfying
every clause.

## Constructor

```rust
WeightedLocalSearch::<P>::new(stop_condition: StopCondition) -> Self
```

`with_tabu_tenure((min, max))` sets the tenure range, `(3, 12)` unless set, and
panics on an empty range. `with_smoothing(p)` sets the smoothing probability,
`0.0003` unless set, and panics outside `[0, 1]`. `weights()` returns the
`ConstraintWeights` of the run in progress or of the last one, which give the
weight of each constraint, the violated ones and the problem's prices.

## Adding it to a problem

A problem opts in by implementing `WeightedConstraints`. It names the move the
search makes and a `Prices` type, whatever the problem caches to price a move
under the weights. The search keeps the weights, the set of violated
constraints and the rule that raises and lowers them, and calls the problem
back when a weight changes.

| Method | What it does |
|---|---|
| `unit_prices` | Builds the prices with every weight at one, and the set of the constraints a solution violates. The size of that set is the number of constraints. |
| `weighted_gain` | Prices a move, positive when it lowers the weighted cost. |
| `apply_weighted` | Applies a move to the solution, reprices the prices and updates the violated set for the result. |
| `reweight` | Reprices after the weight of one constraint changed. |
| `repairs` | Fills the moves that satisfy one violated constraint. |
| `objective_escapes` | Fills the moves that work toward a better objective once nothing is violated. Optional, with none by default, which ends the search there as on MaxSAT. |

The search scans every move each step, so `weighted_gain` should read a cached
price. `Sat` keeps a weighted flip gain per variable, `VertexCover` keeps,
per vertex, the weight of its edges whose other end is out of the set, the
edges it alone covers when it is in the set and the uncovered ones it would
cover by entering, and `GraphColoring` keeps, per vertex and color, the weight of the edges to
neighbors of that color. Each updates its cache in the same pass that applies
the move.

## Benchmark config

```toml
[[heuristics]]
kind = "WeightedLocalSearch"
tabu_tenure = [3, 12]     # optional (default shown)
smoothing = 0.0003        # optional (default shown)
[heuristics.stop_condition]
max_duration_secs = 30.0
```

It takes no `neighbor`, since the problem names the move.

## References

- Thornton, J., Pham, D. N., Bain, S. and Ferreira Jr., V. "Additive versus
  Multiplicative Clause Weighting for SAT." *Proc. AAAI-04*, 191-196, 2004.
- Lin, P., Cai, S., Zou, M. and Lin, J. "New Characterizations and Efficient
  Local Search for General Integer Linear Programming." 2023.
  [arXiv:2305.00188](https://arxiv.org/abs/2305.00188)
