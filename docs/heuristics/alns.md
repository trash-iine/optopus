# AdaptiveLargeNeighborhoodSearch

**API:** [`AdaptiveLargeNeighborhoodSearch`](../api/optopus/heuristic/struct.AdaptiveLargeNeighborhoodSearch.html)

Adaptive Large Neighborhood Search ruins part of the incumbent and recreates
it, choosing the operator pair by a roulette wheel whose weights track recent
performance.

Runs on any problem implementing [`Ruinable`](../traits.md) whose solution
implements [`Evaluate`](../traits.md), which is [CVRP](../problems/vrp.md) and
[TSP](../problems/tsp.md). `Ruinable` describes the shape a problem needs,
elements assigned to containers that compete for a finite resource, and names
the family of problems that already has it. A tour is the single-container
case, and what that costs is explained under [Operators](#operators).

## Example

```rust
use optopus::prelude::*;

let vrp = Vrp::load_file("data/instances/vrp/demo16.vrp")?;
let mut state = SearchState::new(&vrp);

let mut alns = alns_for_vrp(
    StopCondition::iterations(10_000),
    /* removal_fraction = */ 0.15,
    /* cooling_rate     = */ 0.9995,
);
alns.run(&mut state)?;

let sol = &state.best_solution;
println!("total distance = {}", sol.distance);
for (vehicle, route) in sol.routes.iter().enumerate() {
    println!("vehicle {vehicle}: depot -> {route:?} -> depot");
}
```

Takes no `neighbor` type parameter, since it owns its move set. `demo16.vrp` is
the committed 15-customer fixture.

`alns_for_vrp` is the constructor plus the anchored route descent, which is
what makes ruin-and-recreate pay on routes. The same pair written out, with one
of the builders chained on:

```rust
use optopus::prelude::*;
use optopus::problem::vrp::AnchoredRouteDescent;

let mut alns = AdaptiveLargeNeighborhoodSearch::<Vrp>::new(
    StopCondition::iterations(10_000),
    /* removal_fraction = */ 0.15,
    /* cooling_rate     = */ 0.9995,
)
.with_local_repair(Box::new(AnchoredRouteDescent::new()))
.with_max_removal(20);
```

Reach for it to chain the [builders](#constructor), as the last line does, or
to hand VRP a [`LocalRepair`](../traits.md) of your own. A problem without a
wiring function uses this form too, supplying its own repair or leaving
`with_local_repair` off for plain ruin-and-recreate.

On TSP the wiring is `alns_for_tsp`, which pairs the search with
`AnchoredTourDescent`, an Or-opt and 2-opt descent anchored at the
re-inserted cities:

```rust
use optopus::prelude::*;

let tsp = Tsp::load_file("data/instances/tsp/berlin52.tsp")?;
let mut state = SearchState::new(&tsp);

alns_for_tsp(StopCondition::iterations(10_000), 0.15, 0.9995).run(&mut state)?;
println!("tour length = {}", state.best_solution.objective);
```

## Algorithm sketch

Each `run_once` produces one candidate:

1. Select operators, one destroy and one repair operator, by roulette wheel
   over the adaptive weights.
2. Destroy, remove `removal_fraction · n` elements from the incumbent.
3. Repair, re-insert all of them.
4. Descend, run the problem's `LocalRepair` over the recreated solution,
   anchored at the re-inserted elements (see below).
5. Accept, simulated-annealing criterion on the problem's `partial_energy`,
   the objective with CVRP's capacity penalty folded in. The temperature is
   initialized so that a solution 5% worse is
   accepted with probability ≈ 0.5, then cooled by `cooling_rate` each
   iteration.
6. Score, reward the operator pair: `4` for a new global best, `2` for
   better than current, `1` for an accepted worse solution, `0` otherwise.
   Every 100 iterations the segment's average scores are blended into the
   weights with reaction factor `0.1`.

A destroy+repair step is not a single
`MoveToNeighbor`, so the heuristic operates directly on `state.solution` rather
than through `state.apply`.

### Operators

| Destroy | Removes |
|---|---|
| Random | `k` elements drawn uniformly. |
| Worst | the `k` elements with the largest removal gain, those whose detour costs the most. |
| Shaw | the `k` elements most related to a random seed element. |

| Repair | Inserts |
|---|---|
| Greedy | each removed element at its cheapest placement, in random order. |
| Regret-2 | the element with the largest regret first, the gap between its cheapest placement and its cheapest placement in a different container. |

Regret is measured across containers, not across positions. The
second-cheapest slot is almost always the one next door in the same container,
a gap of nearly zero for every element, which would make regret-2
indistinguishable from greedy.

What each problem reads into that:

| | CVRP | TSP |
|---|---|---|
| Element, container | customer, vehicle | city, the tour |
| Relatedness | `distance(a, b) + \|demand(a) − demand(b)\|` | `distance(a, b)` |
| Insertion cost | detour plus the capacity penalty, so a placement is always available even when every route is full | detour |
| Regret-2 | across routes | undefined with one container, so it inserts the pool in a fixed order that ranks nothing, and the destroys and greedy carry the search |
| `LocalRepair` | `AnchoredRouteDescent`, the granular route descent | `AnchoredTourDescent`, Or-opt and 2-opt over the nearest neighbours |

## Constructor

```rust
AdaptiveLargeNeighborhoodSearch::<P>::new(
    stop_condition: StopCondition,
    removal_fraction: f64,   // fraction of elements ruined per iteration
    cooling_rate: f64,       // geometric cooling factor per iteration
) -> Self
```

Panics if `removal_fraction` or `cooling_rate` is outside `(0, 1]`.

Everything else is a builder with a published default, so `new` stays at three
arguments.

| Builder | Sets | Default |
|---|---|---|
| `with_local_repair(Box<dyn LocalRepair<P>>)` | the anchored post-repair local search | none, and set by `alns_for_vrp` and `alns_for_tsp` |
| `with_scoring(best, better, accept)` | the rewards an operator pair earns | `4.0 / 2.0 / 1.0` |
| `with_adaptation(segment_len, reaction)` | iterations per scoring segment, and how much of its average blends into the weights | `100` / `0.1` |
| `with_max_removal(n)` | ceiling on the per-iteration removal count | `50` |

Only the ratios between the three rewards matter, since the weights are a
convex blend of segment averages.

The two local repairs carry their own builders, with the same shape.
`AnchoredRouteDescent::new()` and `AnchoredTourDescent::new()` give the
published defaults, and each chains any of

| Builder | Sets | Default |
|---|---|---|
| `with_granularity(k)` | nearest partners each move considers | `20` on routes, `10` on a tour |
| `with_ring(r)` | partners of each anchor swept along with it, `0` for the anchors alone | `5` |
| `with_max_passes(p)` | passes over the sweep list per repair | `4` |

The defaults are associated constants on each descent
(`AnchoredTourDescent::DEFAULT_GRANULARITY` and so on, the ring's being
`AnchoredSweep::DEFAULT_RING`). A tuned descent is handed to
`with_local_repair` in place of the one `alns_for_vrp` or `alns_for_tsp`
would have built.

`clear()` resets the operator weights and the temperature but keeps what the
builders set. A `LocalRepair`'s own instance-derived caches, the candidate
lists both descents hold, are its own to keep or drop.

## Benchmark config

```toml
[[heuristics]]
kind = "AdaptiveLargeNeighborhoodSearch"
removal_fraction = 0.15    # optional (default shown)
cooling_rate = 0.9995      # optional (default shown)
[heuristics.stop_condition]
max_duration_secs = 30.0
```

Those two keys are the whole config surface. The builders above, the
search's and the descents' alike, are reachable from Rust and not from a TOML,
so a config naming one of them is accepted and then ignored.

## References

- Ropke, S. and Pisinger, D. "An Adaptive Large Neighborhood Search Heuristic
  for the Pickup and Delivery Problem with Time Windows." *Transportation
  Science*, 40(4), 455-472, 2006.
- Shaw, P. "Using Constraint Programming and Local Search Methods to Solve
  Vehicle Routing Problems." In CP 1998, pp. 417-431. Springer, 1998.
  (Shaw removal.)
