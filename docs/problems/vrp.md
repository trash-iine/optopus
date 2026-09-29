# VRP

**API:** [`Vrp`](../api/optopus/problem/vrp/struct.Vrp.html)

A depot (node `0`) and `n` customers `1, ..., n`, each with an integer demand
`q_i`, a service time and pairwise distances `d(i, j)`, are served by a fleet
of one or more [vehicle types](../api/optopus/problem/vrp/struct.VehicleType.html),
each with its own capacity, speed, fixed and per-distance cost, minimum usage
count and maximum route duration. Every type contributes `max_count` vehicle
*slots*; partition the customers into routes, one per slot, so that every
customer is served exactly once. Minimize a time term, either the total of
every route's duration or the longest single route, plus the fleet's cost,
with capacity, route-time and minimum-usage violations penalized:

```text
route_time(slot) = route_distance(slot) / speed(type of slot) + Σ_{c in route} service_time(c)
time_component   = TotalTime => Σ_slots route_time(slot)  |  Makespan => max_slots route_time(slot)
objective = time_component
          + cost_weight · (Σ fixed_cost(used slots) + Σ variable_cost_per_distance(slot) · route_distance(slot))
          + penalty_weight · (overload + time_excess + min_count_shortfall)
```

The Capacitated VRP (CVRP) is the one-type instance with speed `1`, no
service times, no costs and no route-time limit, which every plain
constructor builds. Its objective then reads as the familiar

```text
objective = distance + penalty_weight · Σ_k max(0, load(R_k) − Q)
```

CVRP generalizes TSP, a single vehicle with unlimited capacity recovers it
exactly, and is the base case of the routing-problem family used throughout
logistics and last-mile delivery planning.

Every constraint is soft, handled with a penalty exactly like
[Vertex Cover](vertex_cover.md): `penalty_weight` is an upper bound on the
whole non-penalty part of the objective plus one, so whenever a feasible
solution exists, every optimum of the penalty-augmented objective is
feasible. The bound is deliberately loose, only strict dominance matters.
Overload and shortfall are integers, so any violation of them costs the full
weight; route-time excess is continuous, so a violation smaller than one time
unit is scaled down accordingly.

## Slots are bound to vehicle types in fixed contiguous blocks

Every solution has exactly one route per slot, and `num_slots = Σ max_count`
over the fleet. Each vehicle type owns a contiguous block of slots, type `0`
gets slots `0..max_count[0]`, type `1` the next block, and so on, computed
once at construction and never changed after that. A slot's type is therefore
part of the instance, never of the solution, and moving or swapping a
customer between slots of different types is exactly a change of vehicle
type for that customer.

Idle slots of one type are interchangeable, so a move that opens a new route
of a type considers only the first empty slot of that type
(`Vrp::idle_representatives`); the others would offer the same route under
another label.

## Example

Running a search on a CVRP instance and reading back each vehicle's route:

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
println!("total distance = {}", sol.total_distance());
for (vehicle, route) in sol.routes.iter().enumerate() {
    println!("vehicle {vehicle}: depot -> {route:?} -> depot"); // customer indices only; the depot is implicit
}
```

Pass nearest-integer `EUC_2D` distances (the CVRPLIB convention) instead via
`Vrp::with_rounding`, with the same arguments.

A heterogeneous fleet is built with `Vrp::with_fleet`:

```rust
use optopus::prelude::*;

let fleet = Vrp::with_fleet(
    "demo",
    vec![(0.0, 0.0), (2.0, 0.0), (0.0, 3.0), (-2.0, -1.0)], // [0] is the depot
    vec![0, 4, 3, 2],                                        // demands; [0] is ignored
    vec![0.0, 1.0, 1.0, 0.5],                                // service times; [0] is ignored
    vec![
        VehicleType::new("truck", 6, 1.0, 2)  // capacity, speed, max_count
            .with_costs(10.0, 0.1)            // fixed_cost, variable_cost_per_distance
            .with_min_count(1)
            .with_max_route_time(20.0),
        VehicleType::new("van", 3, 2.0, 1).with_costs(2.0, 0.4),
    ],
    ObjectiveMode::TotalTime,
    1.0,     // cost_weight
    false,   // rounded (EUC_2D nearest-integer distances)
);

let mut state = SearchState::new(&fleet);
LocalSearch::<VrpRelocateNeighbor>::new(StopCondition::iterations(10_000))
    .run(&mut state)
    .unwrap();

let sol = &state.best_solution;
println!("objective = {}", sol.objective);
for (slot, route) in sol.routes.iter().enumerate() {
    let vt = fleet.vehicle_type_of_slot(slot);
    println!("slot {slot} ({}): depot -> {route:?} -> depot", vt.name);
}
```

`vehicle_types` lays out the slots in declaration order: `num_slots()` is `3`
here (2 truck slots, then 1 van slot); `type_of_slot(2) == 1` (the van).
`VehicleType::new` takes `(name, capacity, speed, max_count)` with no cost, no
minimum count and no route-time limit; `with_costs` / `with_min_count` /
`with_max_route_time` opt into the rest.

### Distance storage

The distances live in a
[`DistanceStore`](../api/optopus/common/distance_store/struct.DistanceStore.html), the
same store `Tsp` uses.

| Constructor | Keeps | Use it when |
|---|---|---|
| `Vrp::new(...)`, `Vrp::with_rounding(...)`, `Vrp::with_fleet(...)` | the full `nodes × nodes` matrix up to `Vrp::DIST_MATRIX_MAX_N` nodes (2000), `Vrp::NEAREST_NEIGHBORS_ABOVE_CAP` neighbours per node (20) above that | the usual case |
| `Vrp::with_nearest_neighbors(name, coords, demands, capacity, num_vehicles, rounded, k)` | the `k` nearest neighbours of every node, `nodes × k` distances | the matrix would not fit |
| `Vrp::from_distance_matrix(name, matrix, demands, capacity, num_vehicles)` | the matrix as given | the distances are not Euclidean or the coordinates are not available |

A nearest-neighbour instance still answers `distance(i, j)` for every pair.
A pair outside the lists is computed from the coordinates, so the values are
the same as with the full matrix and only the far pairs cost more. The
granular candidate lists the descents build read `distance` and so work on
every store.

`Vrp::from_distance_matrix` takes `Vec<Vec<f64>>` with node `0` the depot and
rejects a matrix that is not square. Give it a symmetric matrix, since the
2-opt gains price a segment reversal from the four exchanged edges alone.
Such an instance has no coordinates, so `coordinates()` and `rounded()`
return `None` on it.

### Fleet size

Passing `num_vehicles = 0` to a CVRP constructor sizes the fleet by
first-fit-decreasing plus a 10% margin. The margin is there because the
distance-optimal solution routinely uses a few more vehicles than the
minimum, splitting a remote customer onto its own route can be cheaper than
detouring to it. Idle vehicles cost nothing, an undersized fleet costs the
optimum.

## Objective mode

[`ObjectiveMode`](../api/optopus/problem/vrp/enum.ObjectiveMode.html) selects
which aggregation of the per-slot `route_time` the time term charges:

- `TotalTime`, the additive sum over every slot, the usual routing objective.
- `Makespan`, the longest single route's duration, a min-max "when is the
  last vehicle back" objective.

`Makespan` is not additive, so a search under it cannot price a move by
adding up what the move changes. Every heuristic still works on it: the
incremental update resolves the new maximum while the move is priced, at a
cost bounded by the fleet size rather than the customer count.

## Solution

[`VrpSolution`](../api/optopus/problem/vrp/struct.VrpSolution.html) has one
entry per slot in `routes` (`routes.len() == num_slots()`; an idle vehicle is
an empty route), each route listing only the customers (`1..=n`) it visits
with the depot implicit at both ends, plus every quantity in the objective
cached per slot or in total (the API page lists them; `total_distance()` sums
the route distances). `Vrp::solution_from_routes` recomputes every cached
field from a plain `Vec<Vec<usize>>` partition and is the reference every
incremental move update is tested against; `Vrp::validate_routes` checks
that a partition visits every customer `1..=n` exactly once, over exactly
`num_slots()` routes.

`Distance` (used for GA diversity) is the broken-pairs adjacency count: it is
blind to which vehicle drives a trip, so two solutions that drive the same
trips with different vehicle types are at distance `0`, diversity here
meaning a different set of trips.

## Neighbors

| Type | Move | Scope |
|---|---|---|
| `VrpRelocateNeighbor` | Move one customer to a position in another slot. | inter-slot only |
| `VrpSwapNeighbor` | Exchange two customers between two slots. | inter-slot only |
| `VrpTwoOptNeighbor` | Reverse a segment within one slot's route. | intra-route only |

Relocating or swapping a customer across slots of different types reprices,
retimes and possibly recapacitates that customer under the destination type.
Relocate is the only move that can change which slots are used, so it is the
only one that moves the fixed cost, `used_count` and `min_count_shortfall`.

These moves bake `penalty_weight` into their gains. A heuristic that needs to
tune the penalty at runtime, as [HybridGeneticSearchForVrp](../heuristics/hgs.md)
does, prices the same edits under its own penalty through the route machinery
the problem shares with it.

## Crossover

- `VrpOrderCrossover`, flattens both parents into giant tours, applies Order
  Crossover (`common::order_crossover`), then decodes the child back into
  `num_slots()` routes with
  [`split_giant_tour`](../api/optopus/problem/vrp/fn.split_giant_tour.html)
  (Prins' Split): a dynamic program that, for the customer order OX produced,
  chooses the cut positions optimally, each route priced with the vehicle
  type of the slot that drives it. The child is therefore never worse than
  any other way of cutting the same order, including the partition a parent
  itself carried.

Split is decoded here under the fixed `penalty_weight` above, so what the DP
minimizes is exactly the offspring's `objective`, with two documented
exceptions: the minimum-count shortfall is a property of the whole partition
and is not priced per cut, and under `Makespan` the DP decodes with the
additive total-time objective as a proxy, since a maximum over routes added
to sums over routes is not a scalar shortest path. That penalty is the only
thing separating this operator from the recombination step of
[HybridGeneticSearchForVrp](../heuristics/hgs.md), which drives the same decoder
with a penalty it retunes as the search runs (and follows it with a granular
local descent).

## Ruin and recreate

`Vrp` implements `Ruinable` with customers as the elements and vehicle slots
as the containers, so [AdaptiveLargeNeighborhoodSearch](../heuristics/alns.md)
runs on it, on any fleet. `alns_for_vrp` pairs the search with
`AnchoredRouteDescent`, the granular route descent run around the customers a
ruin just re-inserted, whose granularity, ring and pass count are builders
described on that page.

## File formats

`Vrp::load_file` reads the TOML fleet format when the path ends in `.toml`
and CVRPLIB otherwise.

### CVRPLIB

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
described above. The node named by `DEPOT_SECTION` is re-indexed to `0`. The
result is the single-type instance `Vrp::new` builds.

```rust
use optopus::prelude::*;

let vrp = Vrp::load_file("data/instances/vrp/X-n101-k25.vrp")?;
# Ok::<(), optopus::error::OptError>(())
```

### TOML

A schema of its own, with unknown keys rejected on every table, so a
misspelled optional key is a parse error, not a silently ignored default.

```toml
name = "demo_fleet"                 # optional; defaults to the file stem
objective_mode = "TotalTime"        # required: "TotalTime" | "Makespan"
cost_weight = 1.0                   # optional, default 1.0
# rounded omitted -> false (plain Euclidean distances, not EUC_2D rounding)

[depot]
x = 0.0
y = 0.0

[[vehicle_types]]
name = "truck"
capacity = 6
speed = 1.0
fixed_cost = 10.0                   # optional, default 0.0
variable_cost_per_distance = 0.1    # optional, default 0.0
min_count = 1                       # optional, default 0
max_count = 2                       # required: this type contributes 2 slots
max_route_time = 40.0               # optional, default: unconstrained (omit the key)

[[vehicle_types]]
name = "van"
capacity = 3
speed = 2.0
fixed_cost = 2.0
variable_cost_per_distance = 0.4
max_count = 2                       # required: this type contributes 2 more slots
# min_count and max_route_time omitted -> 0 and unconstrained

[[customers]]
id = 1                              # ids must be exactly 1..=n: no gaps, no duplicates
x = 4.0
y = 1.0
demand = 2
service_time = 1.0                  # optional, default 0.0

[[customers]]
id = 2
x = 5.0
y = -2.0
demand = 3
# service_time omitted -> 0.0
```

`vehicle_types` is laid out in declaration order: this fleet has 2 truck
slots (`0`, `1`) then 2 van slots (`2`, `3`), `num_slots() == 4`. The
committed `data/instances/vrp/demo_fleet.toml` is this fleet over six
customers.

```rust
use optopus::prelude::*;

let fleet = Vrp::load_file("data/instances/vrp/demo_fleet.toml")?;
# Ok::<(), optopus::error::OptError>(())
```

In a benchmark config both formats are `problem = "Vrp"`:

```toml
[[instances]]
path = "data/instances/vrp/demo_fleet.toml"
problem = "Vrp"

[[heuristics]]
kind = "LocalSearch"
neighbor = "Relocate"          # Relocate | Swap | TwoOpt
[heuristics.stop_condition]
max_iteration = 2000
```

## Heuristics

Every heuristic that runs on the CVRP runs on a heterogeneous fleet: the
generic ones over the three neighbors, `GeneticAlgorithm` with
`VrpOrderCrossover`, and the two problem-specific ones,
[AdaptiveLargeNeighborhoodSearch](../heuristics/alns.md) and
[HybridGeneticSearch](../heuristics/hgs.md), whose route machinery prices
every edit with the vehicle type of the slot it touches.

## References

- Uchoa, E., Pecin, D., Pessoa, A., Poggi, M., Vidal, T., and Subramanian, A.
  "New Benchmark Instances for the Capacitated Vehicle Routing Problem."
  European Journal of Operational Research, 257(3), 845-858, 2017.
  (The CVRPLIB "X" set.)
- Prins, C. "A Simple and Effective Evolutionary Algorithm for the Vehicle
  Routing Problem." Computers & Operations Research, 31(12), 1985-2002, 2004.
  (Split.)
