# HybridGeneticSearchForVrp

Problem-specific heuristic for [CVRP](../problems/vrp.md). Hybrid Genetic
Search (Vidal et al.) is the strongest known general-purpose CVRP metaheuristic.

## Algorithm sketch

The representation is the **giant tour**: an individual is a customer
permutation, decoded into routes by `split_giant_tour` — a dynamic program that
finds the distance-optimal cut positions for that permutation. Because the
decoder is exact, the genetic operator only has to get the customer *order*
right.

Each `run_once` produces one offspring:

1. **Selection** — binary tournament on biased fitness, over the union of both
   sub-populations.
2. **Crossover** — Order Crossover (OX) on the two parents' giant tours.
3. **Decode** — `split_giant_tour` under the current capacity penalty.
4. **Local search** — granular descent to a local optimum.
5. **Repair** — an infeasible child gets a 50% chance of a second descent at
   10× then 100× the penalty; if that succeeds, the feasible copy is inserted
   *as well as* the original.
6. **Survival** — the child joins the feasible or infeasible sub-population.
   Each grows to `min_population_size + generation_size` and is then culled back
   to `min_population_size`.

### Biased fitness

Selecting on cost alone collapses the population onto one basin within a few
hundred generations. Instead each individual is ranked by

```text
fitness = rank_cost / (N−1) + (1 − N_ELITE/N) · rank_diversity / (N−1)
```

where `rank_diversity` orders individuals by *decreasing* contribution — the
mean broken-pairs distance to their 5 nearest neighbors in the sub-population.
A solution therefore earns its place either by being cheap or by being unlike
the rest. Clones (distance `0` from another member) are always evicted first.

Broken-pairs distance is the fraction of customers whose route neighbors differ.
It is invariant to relabeling and reversing routes, unlike `VrpSolution`'s own
`Distance` impl, which counts route *indices*.

### Two sub-populations and the adaptive penalty

Feasible and infeasible individuals are kept in separate sub-populations, and
the capacity penalty is retuned every 100 offspring to hold the feasible share
near `target_feasible` (default 0.2): too few feasible offspring raise it, too
many lower it.

Searching at a deliberately *low* feasible rate is the point — the shortest
path in solution space between two good feasible solutions usually crosses
infeasible ground. The penalty starts at the instance's average distance per
unit of demand, so it is scale-free, and is clamped to ±3/+4 decades of that.

This is also why HGS keeps its own individuals instead of `VrpSolution`s:
`Vrp::penalty_weight()` is a fixed, deliberately enormous constant chosen so
that any optimum is feasible, which is the opposite of what the search needs.
Solutions are converted back with `Vrp::solution_from_routes` only when writing
to the search state, so reported objectives stay comparable with every other
heuristic.

### Granular local search

For each customer `u`, only its `granularity` nearest customers are considered
as move partners. The move set:

| Move | Scope |
|---|---|
| relocate a segment of 1–2 customers, optionally reversed | inter- and intra-route |
| swap segments of 1–2 customers | inter-route |
| 2-opt (reverse a sub-path) | intra-route |
| 2-opt\* (exchange route tails) | inter-route |
| relocate a segment onto an idle vehicle | — |

Every move is evaluated in O(1) from the distances at its endpoints, and the
first improving one is applied. The descent is over `distance + penalty ·
overload`, with the penalty supplied by the driver — which is why it cannot
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
| X-n195-k51 | 44225 | **44334** | 44506 | **+0.25%** | +0.64% |
| X-n502-k39 | 69226 | **69872** | 70025 | **+0.93%** | +1.15% |

HGS used to win this table by 0.6-2.5pp, and no longer does: ALNS now runs the
same granular descent after each repair (anchored at the customers it
re-inserted), which was the whole of its disadvantage. Over ten X instances of
101-459 customers at 30 s × 5 runs the two are a wash on average objective —
ALNS ahead on the four largest, HGS on the mid-sized ones, every difference
under 0.7%. Which of the two to reach for at a *long* budget is not settled by
this measurement — the 600 s band has not been re-run since the change.

Reproduce with `data/benchmarks/vrp/hgs_{small,medium,large}.toml`.

## Not implemented

- **SWAP\*** (Vidal 2022) — exchanging two customers between routes without
  insisting on their current positions. The main remaining quality lever.
- Multi-layered populations and the parallel variants.

## References

- Vidal, T., Crainic, T. G., Gendreau, M., Lahrichi, N., and Rei, W. "A Hybrid
  Genetic Algorithm for Multidepot and Periodic Vehicle Routing Problems."
  *Operations Research*, 60(3), 611-624, 2012.
- Vidal, T. "Hybrid Genetic Search for the CVRP: Open-Source Implementation and
  SWAP\* Neighborhood." *Computers & Operations Research*, 140, 105643, 2022.
- Prins, C. "A Simple and Effective Evolutionary Algorithm for the Vehicle
  Routing Problem." *Computers & Operations Research*, 31(12), 1985-2002, 2004.
