# Ruin-and-recreate needs an anchored local repair on routes and on tours, most of all on large instances

- Status: adopted
- Area: vrp, tsp
- Date: 2026-09-20
- Code: src/heuristic/specific/vrp/mod.rs (`alns_for_vrp`), src/heuristic/specific/tsp/mod.rs (`alns_for_tsp`), src/problem/tsp/ruin.rs (`AnchoredTourDescent`)

## Decision

`AdaptiveLargeNeighborhoodSearch` takes its `LocalRepair` as an option, since a
problem with no anchored local search still runs plain ruin-and-recreate. On
CVRP and on the TSP the option is not one, so `alns_for_vrp` and `alns_for_tsp`
name the pair and the benchmark registrations build through them.

The repair is anchored. Each recreated solution descends around the re-inserted
elements plus their five nearest partners, not over every element.

Two measurements back this, and they answer different questions. Anchored
against a full sweep, both arms with a descent, the anchoring wins. With the
anchored descent against none, the descent wins everywhere and by more as the
instance grows. Greedy re-insertion does not repair the edges its own choices
spoil. On a small instance the next ruin covers the damage, and on a large one
the same anchors reach a smaller share of the routes or the tour, so what
greedy spoils stays spoiled. The arm without the descent is also far less
repeatable.

On the tour, regret-2 is undefined with a single container and was left as it
is rather than redefined over positions.

## Do not retry

Do not drop the `LocalRepair` from the VRP or TSP registration for simplicity,
and do not replace the anchored descent with a full sweep. Do not quote this as
a general result for `Ruinable`. It is measured on routes and tours only, and a
problem whose repair does not spoil its neighbours may not need one, which is
why the trait keeps it optional.

Do not read the TSP rows as a case for or against regret over positions. Both
arms ran the same degenerate regret, and a position-based one is unmeasured.

## Appendix

Anchored against a full sweep, VRP, ten X instances at 30 s by five runs. The
anchored descent is −0.38% mean objective with 8 of 10 improved, and −0.97% on
X-n701 at 60 s. A full sweep per iteration was +0.07%, and −2% on X-n459.

Descent against none, through a throwaway patch, the same config for both arms,
30 s by five runs, seed 42. On ten X instances the descent wins 10 of 10 by
2.18% mean. Up to about 560 customers the difference is 0.01 to 1.17%, from
700 up 4.43%, 6.05% and 8.17%. On X-n801 the standard deviation is 201 with the
descent and 1424 without.

| TSP instance | Optimum | With descent, avg (std) | Without, avg (std) | Gap |
|---|---|---|---|---|
| eil101 | 629 | 631.2 (1.9) | 632.6 (2.3) | 0.22% |
| ch150 | 6528 | 6549.4 (3.7) | 6556.4 (5.3) | 0.11% |
| dsj1000 | 18659688 | 19428404 (27962) | 19938092 (216080) | 2.6% |

The TSP descent arm also finished ahead of `LinKernighanHelsgaun` and 2-opt
`LocalSearch`, but both stop at their first local optimum well inside the
budget, so that is not a comparison at equal effort.
