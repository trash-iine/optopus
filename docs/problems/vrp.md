# CVRP

Capacitated Vehicle Routing Problem. A depot and `n` customers with 2D
coordinates and integer demands are served by a homogeneous fleet.
**Minimize** total travel distance such that every customer is visited exactly
once and no vehicle exceeds its capacity.

## Solution

```rust
pub struct VrpSolution {
    pub routes: Vec<Vec<usize>>,   // one per vehicle; customer indices, depot implicit
    pub route_loads: Vec<i64>,     // cached total demand of each route
    pub distance: f64,             // true total travel distance
    pub overload: i64,             // Σ max(0, load_r − capacity)
    pub objective: f64,            // distance + penalty_weight * overload
}
```

`routes.len()` always equals `Vrp::num_vehicles`; an **idle vehicle is an empty
route** (distance `0`). Each route lists only the customers (`1..=n`) it
visits — the depot (index `0`) is implicit at both ends.

Capacity is a **soft** constraint, handled with a penalty exactly like
[Vertex Cover](vertex_cover.md): `Vrp::penalty_weight()` is chosen larger than
any possible tour length, so whenever a feasible solution exists, every optimum
of `objective` is feasible. `Rankable::is_better_than` compares `objective`.

## Neighbors

| Type | Move | Scope |
|---|---|---|
| `VrpRelocateNeighbor` | Move one customer to a position in another route. | inter-route only |
| `VrpSwapNeighbor` | Exchange two customers between two routes. | inter-route only |
| `VrpTwoOptNeighbor` | Reverse a segment within one route. | intra-route only |

All three implement `EnabledTabu`, so `TabuSearch` works out of the box. Each
has an inherent `new(prob, sol, …)` that computes the cached `gain` and
`overload_delta`; building a move any other way is a contract violation
(`apply_to_solution` trusts those caches). Relocate and swap **panic** if given
two positions in the same route, and 2-opt panics unless `p < q`.

Note that these moves bake `penalty_weight` into their gains. A heuristic that
needs to tune the capacity penalty at runtime — as
[HybridGeneticSearchForVrp](../heuristics/hgs.md) does — cannot use them and
supplies its own move evaluation.

## Crossover

- `VrpOrderCrossover` — flattens both parents into giant tours, applies Order
  Crossover (`common::order_crossover`), then splits the child greedily back
  into `num_vehicles` routes.

## Split: decoding a giant tour

```rust
pub fn split_giant_tour(prob: &Vrp, giant: &[usize], penalty: f64) -> Vec<Vec<usize>>
```

Prins' Split: given a customer permutation, dynamic programming finds the
**distance-optimal** way to cut it into at most `num_vehicles` routes, in
O(fleet · n · route length). This is what makes the giant-tour encoding usable
as a genetic representation — the decoder is exact, so the search only has to
get the customer *order* right.

Capacity is soft here too: overloaded routes are allowed at `penalty` per unit.
That matters because CVRPLIB fleets are sized for the *best* tour, so an
arbitrary permutation often admits no feasible split at all.

## Construction

```rust
use optopus::prelude::*;

// In-memory. `num_vehicles = 0` picks a fleet that can actually serve everyone.
let vrp = Vrp::new(
    "demo",
    vec![(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)],  // [0] is the depot
    vec![0, 1, 1],                             // demands; [0] is ignored
    2,                                         // capacity
    2,                                         // num_vehicles (0 = auto)
);

// Nearest-integer EUC_2D distances (the CVRPLIB convention):
let vrp = Vrp::with_rounding("demo", vec![(0.0, 0.0), (1.0, 0.0)], vec![0, 1], 2, 1);

// Load from a CVRPLIB file:
let vrp = Vrp::load_file("data/instances/vrp/X-n101-k25.vrp")?;
# Ok::<(), optopus::error::OptError>(())
```

### Fleet size

Passing `num_vehicles = 0` sizes the fleet by **first-fit-decreasing plus a 10%
margin**, not by `ceil(total demand / capacity)`. The latter is only the
bin-packing *lower* bound and is frequently unachievable: three demand-2
customers do not fit into two capacity-3 vehicles, and CVRPLIB's `X-n101-k25`
needs 26 vehicles despite its name. The margin is there because the
distance-optimal solution routinely uses a few more vehicles than the
minimum — splitting a remote customer onto its own route can be cheaper than
detouring to it. Idle vehicles cost nothing, an undersized fleet costs the
optimum.

## File format (CVRPLIB)

```text
NAME : <name>
COMMENT : (... Min no of trucks: K ...)
TYPE : CVRP
DIMENSION : N
EDGE_WEIGHT_TYPE : EUC_2D
CAPACITY : Q
NODE_COORD_SECTION
1 x1 y1
...
DEMAND_SECTION
1 0
...
DEPOT_SECTION
1
-1
EOF
```

Only `EUC_2D` is supported, always with nearest-integer rounding. `No of
trucks: K` is read from `COMMENT` when present; otherwise the fleet is sized as
described above. The node named by `DEPOT_SECTION` is re-indexed to `0`.

## Optional traits

- `Distance` — broken pairs: how many of the two solutions' customer
  adjacencies the other one does not have. It counts *trips*, not vehicle
  labels, so permuting the routes or driving one backwards is a distance of `0`,
  and two solutions are at `0` exactly when they describe the same set of trips.
  The underlying count is directional (a solution using more routes has more
  depot departures to lose), so the trait takes the larger of the two
  directions; HGS ranks diversity on the directional count itself, which is what
  Vidal's biased fitness is defined on.
- `Evaluate<f64>` — on the neighbor types, not the solution.
- `SubProblemExtractable` is **not** implemented, so
  `crossover_kind = "SubProblem"` is unavailable for VRP.

## Heuristics

Any neighborhood-based heuristic works via the three moves above. Two
CVRP-specific heuristics are also available:

- [HybridGeneticSearchForVrp](../heuristics/hgs.md) — giant-tour GA over
  feasible and infeasible sub-populations.
- [AdaptiveLargeNeighborhoodSearchForVrp](../heuristics/alns.md) —
  ruin-and-recreate with adaptive operator weights, descending around the
  customers each repair re-inserted.

The two share their route machinery (`src/heuristic/specific/vrp/ops/`): the same
granular descent, candidate lists and route arithmetic, differing in what drives
them. At 30 s on CVRPLIB X they are comparable — see
[hgs.md](../heuristics/hgs.md#measured-quality).

## References

- Uchoa, E., Pecin, D., Pessoa, A., Poggi, M., Vidal, T., and Subramanian, A.
  "New Benchmark Instances for the Capacitated Vehicle Routing Problem."
  *European Journal of Operational Research*, 257(3), 845-858, 2017.
  (The CVRPLIB "X" set.)
- Prins, C. "A Simple and Effective Evolutionary Algorithm for the Vehicle
  Routing Problem." *Computers & Operations Research*, 31(12), 1985-2002, 2004.
  (Split.)
- See [`data/instances/README.md`](../../data/instances/README.md) for
  instance sources and download instructions.
