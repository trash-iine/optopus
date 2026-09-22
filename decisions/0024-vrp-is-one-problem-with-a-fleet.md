# The VRP is one problem, homogeneous and heterogeneous fleet alike

- Status: adopted
- Area: vrp
- Date: 2026-09-21
- Code: src/problem/vrp/, src/problem/vrp/ops/pricing.rs

## Decision

`Vrp` carries a fleet of `VehicleType`s (capacity, speed, fixed and variable
cost, minimum and maximum count, route-time limit), service times and an
objective mode. The homogeneous CVRP is the one-type instance the plain
constructors build. There is no second problem type: the heterogeneous-fleet
variant was first written as `FleetVrp`, a separate type with its own
solution, moves and crossover, and was folded in before it was merged.

The reason given for the separate type was that everything reading a CVRP
route is typed on `&Vrp` and one `capacity: i64`, so that generalizing in
place would tax every CVRP heuristic with per-slot lookups. An inventory
showed the premise was oversized. What the route machinery reads of the
problem is eight things (`distance`, `demands`, `capacity`, `get_n`,
`num_vehicles`, `route_distance`, `solution_from_routes`, `penalty_weight`);
every penalty is an explicit argument; and the single-capacity assumption sits
in five local places. The cost turned out to be one of layout, not of
generality, and was recovered.

What the merge changed, beyond what the rustdoc of `ops/pricing.rs`,
`RouteState`, `split_giant_tour` and `Vrp::load_file` already says:

- HGS adapts one penalty over the whole violation (overload, time excess,
  shortfall) rather than over capacity alone.
- Idle slots of one vehicle type are label-equivalent, so the generic
  relocate, `try_relocate_to_idle` and the ruin's bucket enumeration consider
  one representative per type.
- `ProblemKind::FleetVrp` is gone; both instance formats are `problem = "Vrp"`.

## Measurement

Fixed budget on X-n195-k51, seed 1, one thread, main (`70529a3`) against this
change. The first unified build was what the premise predicted, and the
final one is not.

| | main | first unified | final |
|---|---|---|---|
| ALNS, 3000 iterations | 1.98 s | 5.97 s | 2.27 s |
| HGS, 1500 generations | 3.87 s | 8.63 s | 3.59 s |
| TabuSearch Relocate, iterations in 10 s (X-n101) | 41 738 | 15 385 | 76 518 |
| TabuSearch Swap, iterations in 10 s | 83 581 | 31 076 | 122 016 |
| SimulatedAnnealing Relocate, iterations in 10 s | 208.6 M | 139.4 M | 340.8 M |

At 30 s per run, 3 runs, seed 42, rayon-parallel as the benchmark runs
(main against final, mean objective):

| Instance | ALNS main | ALNS final | HGS main | HGS final |
|---|---|---|---|---|
| X-n101-k25 | 27597.0 | 27633.3 | 27639.0 | 27632.7 |
| X-n195-k51 | 44555.3 | 44551.0 | 44558.0 | 44549.3 |
| X-n459-k26 | 25554.3 | 25499.0 | 25191.3 | 25114.7 |

Every difference is inside one standard deviation of the three runs.

What closed the gap, in order of effect:

1. Per-slot pricing terms (`Vrp::slot_terms`). Reading the type through
   `slot_type[s]` then `vehicle_types[t]` put a `String`-headed 80-byte
   struct and a second dependent load on every candidate; a contiguous
   per-slot copy of the six numbers halved the ALNS time on its own.
2. `price` and `apply` are const-generic over the edit count, so the
   per-slot loop unrolls and the slice checks go.
3. `#[inline(always)]` on `price` and the three `RouteState` readers the
   descent calls per candidate. With `#[inline]` LLVM left `price` a call
   returning a 64-byte struct in the hottest loop of the crate.
4. The generic moves carry indices and a gain only, and re-price on apply;
   the first version cached the whole price in every candidate and copied it
   through `max_by`.
5. Cumulative route distances, which only the 2-opt* tail exchange reads,
   are kept current by the descent alone; the ruin operators leave routes
   stale. `to_partial` clones the solution's caches instead of recomputing
   them.
6. Under `TotalTime` the makespan is refreshed on apply, never per
   candidate.
7. The neighborhood scans hoist what does not vary per candidate: the lifted
   customer of a relocate, one side of a swap, and the penalty weight, whose
   `OnceLock` load cannot be hoisted by the compiler.

The trajectory pins of `tests/alns_generalization.rs` and
`tests/hgs_generalization.rs` (the latter captured on main before the change)
hold bit for bit: with speed 1, no service time and no cost the fleet
objective reduces to `distance + penalty * overload` exactly, and demo16 did
not expose the association changes in the two-route moves.

## Do not retry

Do not reintroduce a homogeneous fast path or a second problem type on the
grounds of speed. The tax was layout, and the table says so. If a fleet term
ever needs a shortcut, it belongs in `price` as a per-term branch on the
type's own numbers, the way an infinite route-time limit and a zero cost are
skipped now.
