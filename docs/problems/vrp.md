# CVRP

**API:** [`Vrp`](../api/optopus/problem/vrp/struct.Vrp.html)

Capacitated Vehicle Routing Problem (CVRP): a depot (customer `0`) and `n`
customers `1, ..., n`, each with 2D coordinates and an integer demand `q_i`,
are served by a homogeneous fleet of `K` vehicles with shared capacity `Q`,
each starting and ending at the depot. Partition the customers into at most
`K` routes `R_1, ..., R_K` — each a sequence of customers visited by one
vehicle — so that every customer is served exactly once and no route's total
demand exceeds `Q`. **Minimize** total travel distance:

```text
minimize  Σ_{k=1}^{K} distance(R_k)
subject to  R_1, …, R_K partition {1, …, n},  Σ_{i∈R_k} q_i ≤ Q for every route
```

CVRP generalizes TSP — a single vehicle with unlimited capacity recovers it
exactly — and is the base case of the routing-problem family used throughout
logistics and last-mile delivery planning.

Capacity is a **soft** constraint, handled with a penalty exactly like
[Vertex Cover](vertex_cover.md): `penalty_weight` is chosen larger than any
possible tour length, so whenever a feasible solution exists, every optimum of
the penalty-augmented objective is feasible:

```text
objective = distance + penalty_weight · Σ_k max(0, load(R_k) − Q)
```

## Example

Running a search and reading back each vehicle's route:

```rust
use optopus::prelude::*;

let vrp = Vrp::new(
    "demo",
    vec![(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)],  // [0] is the depot
    vec![0, 1, 1, 1],                                      // demands; [0] is ignored
    2,                                                      // capacity
    2,                                                      // num_vehicles (0 = auto)
);
let mut state = SearchState::new(&vrp);
LocalSearch::<VrpRelocateNeighbor>::new(StopCondition::iterations(10_000))
    .run(&mut state)
    .unwrap();

let sol = &state.best_solution;
println!("total distance = {}", sol.distance);
for (vehicle, route) in sol.routes.iter().enumerate() {
    println!("vehicle {vehicle}: depot -> {route:?} -> depot"); // customer indices only; the depot is implicit
}
```

Pass nearest-integer `EUC_2D` distances (the CVRPLIB convention) instead via
`Vrp::with_rounding`, with the same arguments.

### Fleet size

Passing `num_vehicles = 0` sizes the fleet by **first-fit-decreasing plus a 10%
margin**. The margin is there because the
distance-optimal solution routinely uses a few more vehicles than the
minimum — splitting a remote customer onto its own route can be cheaper than
detouring to it. Idle vehicles cost nothing, an undersized fleet costs the
optimum.

## Solution

[`VrpSolution`](../api/optopus/problem/vrp/struct.VrpSolution.html) represents
the partition from the definition above: `routes` is `R_1, ..., R_K`, the
cached `route_loads` is `Σ_{i∈R_k} q_i` per route, `distance` is
`Σ_k distance(R_k)`, `overload` is `Σ_k max(0, load(R_k) − Q)`, and
`objective` is the penalty-augmented form defined above. 
An **idle vehicle is an empty route** (distance `0`), and each route lists only
the customers (`1..=n`) it visits — the depot (index `0`) is implicit at both
ends. 

## Neighbors

| Type | Move | Scope |
|---|---|---|
| `VrpRelocateNeighbor` | Move one customer to a position in another route. | inter-route only |
| `VrpSwapNeighbor` | Exchange two customers between two routes. | inter-route only |
| `VrpTwoOptNeighbor` | Reverse a segment within one route. | intra-route only |

Note that these moves bake `penalty_weight` into their gains. A heuristic that
needs to tune the capacity penalty at runtime — as
[HybridGeneticSearchForVrp](../heuristics/hgs.md) does — cannot use them and
supplies its own move evaluation.

## Crossover

- `VrpOrderCrossover` — flattens both parents into giant tours, applies Order
  Crossover (`common::order_crossover`), then splits the child greedily back
  into `num_vehicles` routes.

A second, DP-optimal decoder for the same giant-tour encoding,
[`split_giant_tour`](../api/optopus/problem/vrp/fn.split_giant_tour.html)
(Prins' Split), is not part of this crossover — it is what
[HybridGeneticSearchForVrp](../heuristics/hgs.md) decodes its own giant-tour
offspring with.

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

```rust
use optopus::prelude::*;

let vrp = Vrp::load_file("data/instances/vrp/X-n101-k25.vrp")?;
# Ok::<(), optopus::error::OptError>(())
```

## References

- Uchoa, E., Pecin, D., Pessoa, A., Poggi, M., Vidal, T., and Subramanian, A.
  "New Benchmark Instances for the Capacitated Vehicle Routing Problem."
  *European Journal of Operational Research*, 257(3), 845-858, 2017.
  (The CVRPLIB "X" set.)
- Prins, C. "A Simple and Effective Evolutionary Algorithm for the Vehicle
  Routing Problem." *Computers & Operations Research*, 31(12), 1985-2002, 2004.
  (Split.)

