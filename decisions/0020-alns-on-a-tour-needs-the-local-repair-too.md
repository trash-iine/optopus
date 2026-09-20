# Ruin-and-recreate on a tour needs the local repair too, and for the same reason

- Status: reference
- Area: tsp
- Date: 2026-09-20
- Code: src/problem/tsp_2d/ruin.rs (`AnchoredTourDescent`), src/heuristic/specific/tsp/mod.rs (`alns_for_tsp`)

## Decision

`TspWithCoordinates` is the second `Ruinable`, with the tour as its one
container. Regret-2 is undefined there and was left as it is rather than
redefined over positions. What was added is what VRP needed, a local repair
anchored at the re-inserted cities, and `alns_for_tsp` names the pair the way
`alns_for_vrp` does.

## Measurement

Three TSPLIB instances at 30s by five runs, seed 42, the same config for both
arms and the descent switched off through a throwaway patch.

| Instance | Optimum | With descent, avg (std) | Without, avg (std) | Gap |
|---|---|---|---|---|
| eil101 | 629 | 631.2 (1.9) | 632.6 (2.3) | 0.22% |
| ch150 | 6528 | 6549.4 (3.7) | 6556.4 (5.3) | 0.11% |
| dsj1000 | 18659688 | 19428404 (27962) | 19938092 (216080) | 2.6% |

The shape is 0018's. On a hundred cities the descent is worth a fraction of a
percent, and the next ruin covers what greedy re-insertion left behind. On a
thousand it is worth 2.6% and the spread is eight times narrower with it. The
anchors reach a smaller share of a larger tour, so what greedy spoils stays
spoiled.

The descent arm also finished ahead of `LinKernighanHelsgaun` and 2-opt
`LocalSearch` on all three, but both of those stop at their first local
optimum well inside the budget, so that is not a comparison at equal effort.
Another benchmark shared the machine during both arms, equally.

## Do not retry

Do not drop the `LocalRepair` from the TSP registration for simplicity. It
costs a few percent on the largest instance, the same place it does on routes.

Do not read this as a case for or against regret over positions. Both arms
ran the same degenerate regret that ranks nothing, and a position-based one is
unmeasured.
