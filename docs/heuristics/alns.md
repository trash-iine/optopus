# AdaptiveLargeNeighborhoodSearchForVrp

**API:** [`AdaptiveLargeNeighborhoodSearchForVrp`](../api/optopus/heuristic/struct.AdaptiveLargeNeighborhoodSearchForVrp.html)

Problem-specific heuristic for [CVRP](../problems/vrp.md). Adaptive Large
Neighborhood Search (Ropke & Pisinger) *ruins* part of the incumbent and
*recreates* it, choosing the operator pair by a roulette wheel whose weights
track recent performance.

## Algorithm sketch

Each `run_once` produces one candidate:

1. **Select operators** — one destroy and one repair operator, by roulette wheel
   over the adaptive weights.
2. **Destroy** — remove `removal_fraction · n` customers from the incumbent.
3. **Repair** — re-insert all of them.
4. **Descend** — run the shared granular descent over the recreated routes,
   **anchored at the re-inserted customers** (see below).
5. **Accept** — simulated-annealing criterion on the penalty-augmented
   objective; the temperature is initialized so that a solution 5% worse is
   accepted with probability ≈ 0.5, then cooled by `cooling_rate` each
   iteration.
6. **Score** — reward the operator pair: `4` for a new global best, `2` for
   better than current, `1` for an accepted worse solution, `0` otherwise.
   Every 100 iterations the segment's average scores are blended into the
   weights with reaction factor `0.1`.

Like [LKH](lkh.md), a destroy+repair step is not a single
`MoveToNeighbor`, so the heuristic operates directly on `state.solution` rather
than through `state.apply`.

### Operators

| Destroy | Removes |
|---|---|
| Random | `k` customers drawn uniformly. |
| Worst | the `k` customers with the largest removal gain — those whose detour costs the most. |
| Shaw | the `k` customers most *related* to a random seed customer, relatedness being `distance(seed, c) + \|demand(seed) − demand(c)\|`. |

| Repair | Inserts |
|---|---|
| Greedy | each removed customer at its cheapest insertion point, cheapest customer first. |
| Regret-2 | the customer with the largest regret first — the gap between its cheapest insertion and its cheapest insertion into a *different route*. |

Regret is measured **across routes, not across positions**: the second-cheapest
*slot* is almost always the one next door in the same route, a gap of nearly
zero for every customer, which would make regret-2 indistinguishable from
greedy. Insertion costs are augmented with the capacity penalty, so an insertion
is always available even when every route is full.

### Why the anchored descent

Recreation leaves the disturbed routes locally poor: a customer is inserted
where it is cheapest *now*, with no chance to fix the edges that choice spoils.
Each recreated solution is therefore handed to `ops::Descent::run_around` —
the same granular descent [HGS](hgs.md) uses, at Γ = 20 and up to 4 passes,
under `Vrp::penalty_weight()`.

Anchoring is what makes it pay. Measured at 30 s × 5 runs on ten CVRPLIB X
instances, the anchored descent is **−0.38% mean objective with 8/10 instances
improved**, and −0.97% on X-n701 at 60 s. A *full* sweep per iteration was
+0.07% — a wash, and −2% on X-n459 — because on a mid-sized instance the ruin's
anchors, widened by a full Γ = 20 candidate list, already cover everything.

## Constructor

```rust
AdaptiveLargeNeighborhoodSearchForVrp::new(
    stop_condition: StopCondition,
    removal_fraction: f64,   // fraction of customers ruined per iteration
    cooling_rate: f64,       // geometric cooling factor per iteration
) -> Self
```

**Panics** if `removal_fraction` or `cooling_rate` is outside `(0, 1]`.

`clear()` resets the operator weights and the temperature; the descent's
instance-derived candidate lists survive it, since they depend on nothing else.

## Benchmark config

```toml
[[heuristics]]
kind = "AdaptiveLargeNeighborhoodSearch"
removal_fraction = 0.15    # optional (default shown)
cooling_rate = 0.9995      # optional (default shown)
[heuristics.stop_condition]
max_duration_secs = 30.0
```

## Relation to HGS

The two CVRP heuristics share their route machinery
(`src/heuristic/specific/vrp/ops/`) — the same granular descent, candidate
lists and route arithmetic — and differ in what drives it. At 30 s on CVRPLIB X
they are a wash; see [hgs.md](hgs.md#measured-quality) for the table.

## References

- Ropke, S. and Pisinger, D. "An Adaptive Large Neighborhood Search Heuristic
  for the Pickup and Delivery Problem with Time Windows." *Transportation
  Science*, 40(4), 455-472, 2006.
- Shaw, P. "Using Constraint Programming and Local Search Methods to Solve
  Vehicle Routing Problems." In *CP 1998*, pp. 417-431. Springer, 1998.
  (Shaw removal.)
