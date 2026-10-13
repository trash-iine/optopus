# The ALNS greedy repair is the original paper's basic greedy, at a cost in time and no gain per iteration

- Status: adopted
- Area: heuristic, vrp, tsp, flow_shop
- Date: 2026-10-13
- Code: src/building_blocks/search/ruin_recreate.rs, src/heuristic/alns.rs

## Decision

The ALNS greedy repair put the removed elements back in random order, each at
its cheapest placement. The original paper's basic greedy (Ropke and Pisinger
2006, §2.2.1) reprices every unplanned element at each step and inserts the
cheapest. ALNS now runs that, `cheapest_first_insertion`, chosen for fidelity
with the measurement below in hand. The random-order repair stays as
`random_order_insertion` for Iterated Greedy. No function keeps the name
`greedy_insertion`, so a caller of the old one fails to compile.

## Measurement

Old against new, 4 rayon threads, seed 42, the arms run one after the other.
Average objective of the new arm against the old, per instance.

| | Budget | New against old | Iterations, new / old |
|---|---|---|---|
| Flow shop ta051 to ta060 | 30 s × 3 | −0.24% to +0.20%, better on 7 of 10 | 0.99 to 1.07 |
| VRP X-n101, n153, n214 | 30 s × 5 | −0.48% to +0.02% | 0.91 to 1.00 |
| VRP X-n303, n401, n502 | 30 s × 5 | +0.15% to +0.49% | 0.69 to 0.78 |
| TSP eil101, ch150, dsj1000 | 30 s × 5 | +0.10% to +2.53% | 0.55 to 0.70 |

At a fixed iteration count (VRP n303 to n502 at 6000, TSP ch150 and dsj1000 at
10000) the two are within 0.25% everywhere, in both directions. The time lost
is the k² repricing, and the quality per iteration is the same. Decision 0026
was measured with the old repair.

## Do not retry

Going back to the random order for speed trades fidelity for the numbers above
and is a choice to make with them, not a fix. Repricing only the container the
last insertion changed, as the paper does, needs `Ruinable` to promise that the
other containers' costs stay put, and gains nothing on the tour or the flow
shop, which have a single container.
