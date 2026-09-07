# HybridGeneticSearchForVrp

**API:** [`HybridGeneticSearchForVrp`](../api/optopus/heuristic/struct.HybridGeneticSearchForVrp.html)

Problem-specific heuristic for [CVRP](../problems/vrp.md). Hybrid Genetic
Search (Vidal et al.) is the strongest known general-purpose CVRP metaheuristic.

## Example

```rust
use optopus::prelude::*;

let vrp = Vrp::load_file("data/instances/vrp/demo16.vrp")?;
let mut state = SearchState::new(&vrp);

let mut hgs = HybridGeneticSearchForVrp::new(
    StopCondition::iterations(10_000),
    /* min_population_size = */ 25,   // μ
    /* generation_size     = */ 40,   // λ
    /* granularity         = */ 20,   // Γ, clamped to n - 1
    /* target_feasible     = */ 0.2,
    /* restart_generations = */ Some(20_000),
);
hgs.run(&mut state)?;

let sol = &state.best_solution;
println!("total distance = {}", sol.distance);
for (vehicle, route) in sol.routes.iter().enumerate() {
    println!("vehicle {vehicle}: depot -> {route:?} -> depot");
}
```

Takes no `neighbor` type parameter, since it owns its move set.

## Algorithm sketch

The representation is the giant tour, so an individual is a customer
permutation, decoded into routes by `split_giant_tour`, a dynamic program that
finds the distance-optimal cut positions for that permutation. Because the
decoder is exact, the genetic operator only has to get the customer order
right.

Each `run_once` produces one offspring:

1. Selection, binary tournament on biased fitness, over the union of both
   sub-populations.
2. Crossover, Order Crossover (OX) on the two parents' giant tours.
3. Decode, `split_giant_tour` under the current capacity penalty.
4. Local search, granular descent to a local optimum.
5. Repair, an infeasible child gets a 50% chance of a second descent at
   10× then 100× the penalty; if that succeeds, the feasible copy is inserted
   as well as the original.
6. Survival, the child joins the feasible or infeasible sub-population.
   Each grows to `min_population_size + generation_size` and is then culled back
   to `min_population_size`.

### Biased fitness

Selecting on cost alone collapses the population onto one basin within a few
hundred generations. Instead each individual is ranked by

```text
fitness = rank_cost / (N−1) + (1 − N_ELITE/N) · rank_diversity / (N−1)
```

where `rank_diversity` orders individuals by decreasing contribution, the
mean broken-pairs distance to their 5 nearest neighbors in the sub-population.
A solution therefore earns its place either by being cheap or by being unlike
the rest. Clones (distance `0` from another member) are always evicted first.

Broken-pairs distance is the fraction of customers whose route neighbors differ,
invariant to relabeling and reversing routes. The
[`Distance`](../api/optopus/trait_defs/trait.Distance.html) impl on
`VrpSolution` uses the same count but symmetrizes it, while biased fitness here
ranks on the directional count, which is the form Vidal defines.

### Two sub-populations and the adaptive penalty

Feasible and infeasible individuals are kept in separate sub-populations, and
the capacity penalty is retuned every 100 offspring to hold the feasible share
near `target_feasible` (default 0.2): too few feasible offspring raise it, too
many lower it.

Searching at a deliberately low feasible rate is the point, since the shortest
path between two good feasible solutions usually crosses infeasible ground. The
penalty starts at the instance's average distance per unit of demand, which
makes it scale-free, and is clamped to a few decades either side.

HGS therefore keeps its own individuals rather than `VrpSolution`s, since
`Vrp::penalty_weight()` is a fixed constant large enough to make any optimum
feasible. `Vrp::solution_from_routes` converts back when writing to the search
state, so reported objectives stay comparable with every other heuristic.

### Granular local search

For each customer `u`, only its `granularity` nearest customers are considered
as move partners. The move set:

| Move | Scope |
|---|---|
| relocate a segment of 1–2 customers, optionally reversed | inter- and intra-route |
| swap segments of 1–2 customers | inter-route |
| 2-opt (reverse a sub-path) | intra-route |
| 2-opt\* (exchange route tails) | inter-route |
| relocate a segment onto an idle vehicle | |
Every move is evaluated in O(1) from the distances at its endpoints and the
first improving one is applied. The descent runs over `distance + penalty ·
overload` with the penalty supplied by the driver, which is why it does not
reuse the `VrpRelocateNeighbor` family.

## Constructor

```rust
HybridGeneticSearchForVrp::new(
    stop_condition: StopCondition,
    min_population_size: usize,        // μ
    generation_size: usize,            // λ
    granularity: usize,                // Γ
    target_feasible: f64,
    restart_generations: Option<u64>,
) -> Self
```

Reasonable defaults: `μ = 25`, `λ = 40`, `Γ = 20`, `target_feasible = 0.2`,
`restart_generations = Some(20_000)`.

The first individual is seeded from `state.solution`, so composing HGS inside
`Sequential`, `Iterated`, or `Restart` carries the incumbent forward. A restart
(after `restart_generations` without improvement) reseeds from scratch instead,
so it does not land back in the basin it just failed to escape;
`state.best_solution` is preserved across it.

`clear()` empties both sub-populations, resets the feasible-share and
stagnation counters, and returns the capacity penalty to its initial value, so
a fresh episode re-derives the penalty rather than inheriting a tuned one.

## Benchmark config

```toml
[[heuristics]]
kind = "HybridGeneticSearch"
min_population_size = 25
generation_size = 40
granularity = 20
target_feasible = 0.2
restart_generations = 20000
[heuristics.stop_condition]
max_duration_secs = 30.0
```

All fields are optional. The acceptance counters in the report carry the
feasible share of offspring, which is what the adaptive penalty steers.

## Measured quality

30 s per run, 3 runs, seed 42, `μ=25 λ=40 Γ=20`, against the CVRPLIB best-known
solutions (`data/instances/scripts/fetch_cvrp.sh`). ALNS is
`AdaptiveLargeNeighborhoodSearch` at the same budget.

| Instance | BKS | ALNS best | HGS best | ALNS gap | HGS gap |
|---|---|---|---|---|---|
| X-n101-k25 | 27591 | 27597 | 27597 | +0.02% | +0.02% |
| X-n195-k51 | 44225 | 44334 | 44506 | +0.25% | +0.64% |
| X-n502-k39 | 69226 | 69872 | 70025 | +0.93% | +1.15% |

HGS and [ALNS](alns.md) are level at this budget, with every difference under
0.7% over ten X instances, so either is a reasonable default for CVRP.

Reproduce with `data/benchmarks/vrp/hgs_{small,medium,large}.toml`.

## References

- Vidal, T., Crainic, T. G., Gendreau, M., Lahrichi, N., and Rei, W. "A Hybrid
  Genetic Algorithm for Multidepot and Periodic Vehicle Routing Problems."
  Operations Research, 60(3), 611-624, 2012.
- Vidal, T. "Hybrid Genetic Search for the CVRP: Open-Source Implementation and
  SWAP\* Neighborhood." Computers & Operations Research, 140, 105643, 2022.
- Prins, C. "A Simple and Effective Evolutionary Algorithm for the Vehicle
  Routing Problem." Computers & Operations Research, 31(12), 1985-2002, 2004.
