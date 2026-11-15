# A beam over insertions in a drawn order is behind ALNS on every instance measured

- Status: adopted
- Area: heuristic, tsp, flow_shop
- Date: 2026-11-15
- Code: src/heuristic/iterative_beam_search.rs

## Decision

`IterativeBeamSearch` builds a solution by insertion over any `Ruinable`
problem whose partial is `Clone`, keeps the best `D` partial solutions per
element, and restarts at twice the width. The widening restart is the one the
iterative beam search published for the flow shop uses (the second reference
of the heuristic's page), read in its text. Its
branching is not reproduced. That search appends any unscheduled job to an end
of the sequence and ranks nodes by a bound with the idle time. The trait has
neither an end to append to nor a bound, so here the next element is fixed by
an order drawn per pass, a child inserts it anywhere, and the rank is
`partial_energy`.

In that form it is a construction and not a competitor. On the six instances
below it is behind both ALNS and the GRASP composition on all of them. It is
kept as the one constructive search that needs nothing of a
problem beyond `Ruinable`, with the numbers below stated on its page's terms:
a wider pass is better than a narrow one on average, and no pass is a search
of the neighborhood of a good solution.

`TspPartial`, `FlowShopPartial` and `VrpPartial` gained `Clone` for it.

## Do not retry

Widening further does not close the gap. The runs below reached widths of
thousands and their best solutions came within the first 8 of the 20 seconds.

What the published search has and this one lacks is a guide that says how
good a partial solution can still become. Ranking by `partial_energy` prefers
the partial that is cheap so far. A second attempt should start from a bound
on the problem side, a new trait method with the flow shop as its first
implementer, and should not be made generic until a second problem has one.

## Appendix

`data/benchmarks/flow_shop/beam_taillard.toml` and
`data/benchmarks/tsp/beam_tsplib.toml`, five runs, seed 42, 20 s, the five
runs in parallel on an Apple M4. Average objective, lower is better. GRASP is
`Iterated` over a descent and `GreedyRandomizedConstruction` at 0.2.

| Instance | IterativeBeamSearch | AdaptiveLargeNeighborhoodSearch | GRASP |
|---|---|---|---|
| Flow shop ta031 | 2729.0 | 2724.0 | 2724.0 |
| Flow shop ta051 | 3933.2 | 3885.8 | 3924.6 |
| Flow shop ta081 | 6391.6 | 6296.2 | 6389.6 |
| TSP eil101 | 647.2 | 630.2 | 643.6 |
| TSP ch150 | 6757.4 | 6557.6 | 6672.2 |
| TSP dsj1000 | 20560235.6 | 19536092.8 | 20126607.2 |

The beam search made 11 to 16 passes per run, so its last pass ran at a width
of one to thirty thousand. The VRP and bin packing were not measured.
