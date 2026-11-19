# AdaptiveIteratedLocalSearchForVrp

**API:** [`AdaptiveIteratedLocalSearchForVrp`](../api/optopus/heuristic/struct.AdaptiveIteratedLocalSearchForVrp.html)

Problem-specific heuristic for the [VRP](../problems/vrp.md). Adaptive
Iterated Local Search, in its variant for large instances, perturbs a
reference solution, descends, and decides whether the result becomes the new
reference. How many customers a perturbation removes and how good a result has
to be both adapt during the run, and both tighten toward its end.

## Example

```rust
use optopus::prelude::*;

let vrp = Vrp::load_file("data/instances/vrp/demo16.vrp")?;
let mut state = SearchState::new(&vrp);

let mut ails = AdaptiveIteratedLocalSearchForVrp::new(StopCondition::iterations(20_000));
ails.run(&mut state)?;

let sol = &state.best_solution;
println!("total distance = {}", sol.total_distance());
for (vehicle, route) in sol.routes.iter().enumerate() {
    println!("vehicle {vehicle}: depot -> {route:?} -> depot");
}
```

Takes no `neighbor` type parameter, since it owns its move set.

## Algorithm sketch

The first `run_once` takes the incumbent to a local optimum, which becomes the
reference. Each later one is an iteration.

1. Draw a removal and an insertion, each from two.
2. Remove customers from the reference, as many as the removal's degree says.
3. Put them back in a random order, by the insertion.
4. Descend with the granular route descent, anchored at the customers that
   were moved, until a pass changes nothing.
5. Measure the distance between the result and the reference, and adjust the
   removal's degree with it.
6. Accept the result as the new reference if it is under the threshold.

### Perturbation

| Removal | Takes |
|---|---|
| Concentric | the customers nearest one drawn customer |
| Sequential | runs of consecutive customers, each following one drawn customer around its route |

| Insertion | Puts a customer |
|---|---|
| By distance | beside its nearest placed customer |
| By cost | beside whichever of its `neighbors` nearest placed customers makes the smallest detour |

The depot takes its turn among a customer's nearest, standing for the start
of every route, and "beside" is before or after, whichever detour is smaller.
Neither insertion looks at capacity. The descent that follows prices an
overload at the penalty the problem charges, which outweighs any distance, so
it restores feasibility first and shortens the routes after.

### Perturbation degree

Each removal keeps its own degree, the number of customers it removes,
starting at `distance_max`. The distance between a result and the reference
is the number of the result's edges the reference does not have. Every
`gamma` uses of a removal, its degree is multiplied by

```text
ideal distance / average distance its results ended at
```

and kept between `1` and `n − 2`. A removal whose results fall back onto the
reference removes more the next time, and one whose results land far away
removes less. The ideal distance falls geometrically from `distance_max` to
`distance_min` over the run.

### Acceptance

A feasible result of cost `f` becomes the reference when

```text
f ≤ low + eta · (mean − low)
```

`mean` is a moving average of the costs the descent has been ending at, and
`low` is the best of them over the last `gamma` to `2 · gamma` iterations.
`eta` falls geometrically from `1` to `eta_min` over the run. At the start an
average result is accepted, and at the end nearly only one that matches a
recent best. A result that breaks a constraint stays out of the averages and
replaces only a reference that breaks one worse.

### The length of the run

Both schedules are laid out over the number of iterations the run will make.
That number is, in this order,

1. the one given to `with_expected_iterations`,
2. the stop condition's iteration limit,
3. an estimate from the share of the stop condition's time limit that has
   passed since the first iteration, renewed every `gamma` iterations. The
   descent that makes the first reference is not counted, so a long one does
   not shorten the schedules.

A run with none of the three is an error. A stop on failed updates alone does
not say how long the run is.

## Constructor

```rust
AdaptiveIteratedLocalSearchForVrp::new(stop_condition: StopCondition) -> Self
```

`stop_condition` decides when the run ends, see [Stop
conditions](../guide/stop_conditions.md).

Everything else is a builder with a default, each an associated constant of
the type.

| Builder | Sets | Default |
|---|---|---|
| `with_gamma(gamma)` | iterations between two adjustments of a removal's degree, and the window of the two averages | `30` |
| `with_distances(min, max)` | the ideal distance at the end and at the start of the run | `15.0` / `30.0` |
| `with_eta_min(eta_min)` | what `eta` falls to | `0.01` |
| `with_neighbors(neighbors)` | nearest customers an insertion by cost and a move of the descent consider | `40` |
| `with_expected_iterations(n)` | the length the schedules are laid out over | from the stop condition |

`clear()` puts the schedules back to the start of a run.

## Benchmark config

```toml
[[heuristics]]
kind = "AdaptiveIteratedLocalSearch"
gamma = 30               # optional (default shown)
distance_min = 15.0      # optional (default shown)
distance_max = 30.0      # optional (default shown)
eta_min = 0.01           # optional (default shown)
neighbors = 40           # optional (default shown)
# expected_iterations = 1000000   # optional, see below
[heuristics.stop_condition]
max_duration_secs = 60.0
```

`expected_iterations` is needed only when the stop condition sets neither
`max_iteration` nor `max_duration_secs`. Under a time limit the schedules
follow the clock, so two runs with one seed differ, as any run stopped by time
does.

## What differs from the published search

The perturbation, the degree and the acceptance follow the published search.
The descent does not. The published search has a feasibility phase that ranks
moves by cost per unit of overload removed, and a local search with the moves
SHIFT, SWAP, 2-opt, SWAP\* and 2-opt\*. Here one descent does both jobs, the
granular route descent that [Hybrid Genetic Search](hgs.md) also runs, under
the problem's penalty. It has no SWAP\*.

## References

- Máximo, V. R., Cordeau, J.-F. and Nascimento, M. C. V. "AILS-II: An
  Adaptive Iterated Local Search Heuristic for the Large-Scale Capacitated
  Vehicle Routing Problem." *INFORMS Journal on Computing*, 36(4), 974-986,
  2024.
