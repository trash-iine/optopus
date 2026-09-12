# Alns

**API:** [`Alns`](../api/optopus/heuristic/struct.Alns.html)

Adaptive Large Neighborhood Search (Ropke & Pisinger) ruins part of the
incumbent and recreates it, choosing the operator pair by a roulette wheel whose
weights track recent performance.

Runs on any problem implementing [`Ruinable`](../traits.md) whose solution
implements [`Evaluate`](../traits.md), which is [CVRP](../problems/vrp.md) so
far. See
`Ruinable` for the shape a problem needs, elements assigned to containers that
compete for a finite resource, and for the family of problems that already has
it.

## Example

```rust
use optopus::prelude::*;

let vrp = Vrp::load_file("data/instances/vrp/demo16.vrp")?;
let mut state = SearchState::new(&vrp);

let mut alns = Alns::<Vrp>::new(
    StopCondition::iterations(10_000),
    /* removal_fraction = */ 0.15,
    /* cooling_rate     = */ 0.9995,
)
.with_local_repair(Box::new(AnchoredRouteDescent::new()));
alns.run(&mut state)?;

let sol = &state.best_solution;
println!("total distance = {}", sol.distance);
for (vehicle, route) in sol.routes.iter().enumerate() {
    println!("vehicle {vehicle}: depot -> {route:?} -> depot");
}
```

Takes no `neighbor` type parameter, since it owns its move set. `demo16.vrp` is the
committed 15-customer fixture; the measurements below are on CVRPLIB X
instances.

## Algorithm sketch

Each `run_once` produces one candidate:

1. Select operators, one destroy and one repair operator, by roulette wheel
   over the adaptive weights.
2. Destroy, remove `removal_fraction · n` customers from the incumbent.
3. Repair, re-insert all of them.
4. Descend, run the shared granular descent over the recreated routes,
   anchored at the re-inserted customers (see below).
5. Accept, simulated-annealing criterion on the penalty-augmented
   objective; the temperature is initialized so that a solution 5% worse is
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
| Random | `k` customers drawn uniformly. |
| Worst | the `k` customers with the largest removal gain, those whose detour costs the most. |
| Shaw | the `k` customers most related to a random seed customer, relatedness being `distance(seed, c) + \|demand(seed) − demand(c)\|`. |

| Repair | Inserts |
|---|---|
| Greedy | each removed customer at its cheapest insertion point, cheapest customer first. |
| Regret-2 | the customer with the largest regret first, the gap between its cheapest insertion and its cheapest insertion into a different route. |

Regret is measured across routes, not across positions: the second-cheapest
slot is almost always the one next door in the same route, a gap of nearly
zero for every customer, which would make regret-2 indistinguishable from
greedy. Insertion costs are augmented with the capacity penalty, so an insertion
is always available even when every route is full.

## Constructor

```rust
Alns::<P>::new(
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
| `with_local_repair(Box<dyn LocalRepair<P>>)` | the anchored post-repair local search | none |
| `with_scoring(best, better, accept)` | the rewards an operator pair earns | `4.0 / 2.0 / 1.0` |
| `with_adaptation(segment_len, reaction)` | iterations per scoring segment, and how much of its average blends into the weights | `100` / `0.1` |
| `with_max_removal(n)` | ceiling on the per-iteration removal count | `50` |

Only the ratios between the three rewards matter, since the weights are a
convex blend of segment averages.

`clear()` resets the operator weights and the temperature but keeps what the
builders set. A `LocalRepair`'s own instance-derived caches, such as VRP's
candidate lists, are its own to keep or drop.

## Benchmark config

```toml
[[heuristics]]
kind = "AdaptiveLargeNeighborhoodSearch"
removal_fraction = 0.15    # optional (default shown)
cooling_rate = 0.9995      # optional (default shown)
[heuristics.stop_condition]
max_duration_secs = 30.0
```

## References

- Ropke, S. and Pisinger, D. "An Adaptive Large Neighborhood Search Heuristic
  for the Pickup and Delivery Problem with Time Windows." *Transportation
  Science*, 40(4), 455-472, 2006.
- Shaw, P. "Using Constraint Programming and Local Search Methods to Solve
  Vehicle Routing Problems." In CP 1998, pp. 417-431. Springer, 1998.
  (Shaw removal.)
