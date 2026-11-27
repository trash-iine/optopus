# WeightedLocalSearch

**API:** [`WeightedLocalSearch`](../api/optopus/heuristic/struct.WeightedLocalSearch.html)

Weighted Local Search gives every constraint of a problem a weight and raises
the weights of the constraints it finds violated each time it is stuck. A
constraint the search keeps failing comes to outweigh the others, so the
search is pushed to satisfy it without being told which constraints are the
hard ones. Once it holds a solution that violates nothing, it alternates
between bettering the objective of that solution and repairing what the
betterment broke.

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

WeightedLocalSearch::<VertexCover>::new(
    StopCondition::duration(std::time::Duration::from_secs(10)),
    (3, 12),
    0.0003,
)
.run(&mut state)?;
println!("cover size = {}", state.best_solution.cover_size);
```

## Algorithm sketch

Every weight is an integer and starts at one, the weight of each constraint and
the weight of the objective. The objective here is what a solution costs apart
from its constraints, such as the size of a cover or the number of colors used.
A penalty the problem's own objective adds for a violated constraint is not
part of it.

The tight moves of a constraint are the moves of its variables that leave it
as tight as they can. For a violated constraint they are the moves that
satisfy it. For a satisfied one they keep it satisfied and take up the slack
it has, a vertex leaving an edge whose two ends are both in a cover, or a true
literal turning false in a clause with another true literal.

Each `run_once` makes one step, in one of three modes by the solution in hand.

1. Until a solution that violates nothing has been found, the search looks for
   one. It draws three violated constraints, prices their tight moves by the
   weight of the constraints a move satisfies less the weight of those it
   violates, and applies the best move with a positive price that the tabu
   memory allows.
2. While the solution violates nothing, the search betters the objective. It
   applies the move that violates nothing and lowers the objective the most.
   When there is none, it applies the best priced of the problem's objective
   escapes, a vertex leaving the cover or leaving the smallest color class,
   which violates something. They are priced as in the third mode.
3. Otherwise the search repairs toward a solution better than the best found.
   A move's price adds the weight of the objective if it leads below the best
   objective found and takes it away if not. The search draws three violated
   constraints as in the first mode, and if none of their tight moves has a
   positive price it draws thirty satisfied ones and tries their tight moves.

When no drawn move in the first or third mode has a positive price, the
search changes the weights. With probability `1 − smoothing` every violated
constraint gains one, and once a solution that violates nothing has been found
the objective gains one too if the solution is no better than the best.
Otherwise every satisfied constraint above one loses one, and the objective
loses one if the solution is better than the best. The search then applies the
best priced tight move of one violated constraint drawn uniformly. Here and
among the objective escapes, a move the tabu memory forbids is taken only when
it forbids all of them.

When the drawn constraints have more tight moves than a step prices, 2000 for
the violated ones, 350 for the satisfied ones and 150 at a weight change, it
prices that many drawn uniformly. A step costs about the same however large
the problem is, except in the second mode, which looks at every move.

A move that was applied may not be undone for a number of steps drawn from
`tabu_tenure`. A constraint's weight never rises above the larger of 1000 and
the number of constraints, and the objective's above a tenth of that. After
1,500,000 steps without a better best solution the search restarts. It takes
each variable from the best solution or from a fresh one with even odds, and
every weight starts again at one.

The best solution is kept by the problem's own order, which the weights never
enter, so what the run reports is comparable with every other heuristic. The
weights and the tabu memory belong to the search and start again on every
`run`.

A run ends on its stop condition. It also ends on a solution that violates
nothing and that no move takes toward a better objective. On MaxSAT that is an
assignment satisfying every clause.

## Constructor

```rust
WeightedLocalSearch::<P>::new(
    stop_condition: StopCondition,
    tabu_tenure: (u64, u64),
    smoothing: f64,
) -> Self
```

The tuned values are `(3, 12)` and `0.0003`. It panics on an empty tenure
range and on a smoothing outside `[0, 1]`. `with_violated_samples(3, 2000)`,
`with_satisfied_samples(30, 350)` and `with_random_constraint_samples(150)` set
how many constraints a step draws and how many of their moves it prices, the
values shown being the defaults. `weights()` returns the `ConstraintWeights` of
the run in progress or of the last one, which give the weight of each
constraint, the violated ones and the problem's prices, and
`objective_weight()` the weight of the objective.

## Adding it to a problem

A problem opts in by implementing `WeightedConstraints`. It names the move the
search makes and a `Prices` type, whatever the problem caches to price a move
under the weights. The search keeps the weights, the set of violated
constraints and the rule that raises and lowers them, and calls the problem
back when a weight changes.

| Method | What it does |
|---|---|
| `unit_prices` | Builds the prices with every weight at one, and the set of the constraints a solution violates. Its bound is the number of constraints. |
| `weight_gain` | Prices a move, the weight of the constraints it satisfies less the weight of those it violates. |
| `apply_weighted` | Applies a move to the solution, reprices the prices and updates the violated set for the result. |
| `reweight` | Reprices after the weight of one constraint changed. |
| `tight_moves` | Fills the tight moves of one constraint. |
| `objective_escapes` | Fills the moves that take one step toward a better objective whatever they violate. Optional, with none by default, which ends the search on a solution that violates nothing, as on MaxSAT. |
| `objective` / `objective_change` | The objective apart from the constraints, and how a move changes it. Optional, zero by default for a problem whose objective is the constraints alone. |
| `restart_from` | The solution a restart starts from, given the best. Optional, a fresh solution by default. The built-in problems mix the best with a fresh solution variable by variable. |

The second mode looks at every move, so `weight_gain` should read a cached
price. `Sat` keeps a weighted flip gain per variable. `VertexCover` keeps,
per vertex, the weight of its edges whose other end is out of the set. When the
vertex is in the set these are the edges it alone covers, and when it is out
they are the uncovered edges it would cover by entering. `GraphColoring` keeps,
per vertex and color, the weight of the edges to neighbors of that color. Each
updates its cache in the same pass that applies the move.

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
