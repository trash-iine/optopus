# AILS keeps its diversity control and borrows the granular descent

- Status: adopted
- Area: heuristic, vrp
- Date: 2026-11-17
- Code: src/heuristic/specific/vrp/ails.rs

## Decision

`AdaptiveIteratedLocalSearch` is the variant of AILS for large instances, as
a VRP heuristic. Its loop was read in the paper and its details in the code
the authors publish under the MIT license, and written again in this crate's
structures. Taken from there are the two removals, the two insertions with
the depot's turn among a customer's nearest, the degree of each removal
multiplied every `gamma` uses by the ideal distance over the average distance
reached, the distance as the result's edges the reference lacks, the
threshold `low + eta · (mean − low)` with its two windows, and the two
geometric schedules with their estimate of the run's length under a time
limit.

The descent is not the published one. That search has a feasibility phase
that ranks moves by cost per unit of overload removed, and a local search with
SHIFT, SWAP, 2-opt, SWAP\* and 2-opt\*, both limited to the routes a
perturbation touched. Here `AnchoredRouteDescent` does both, the granular
descent anchored at the customers that were moved, under the problem's
penalty, which outweighs any distance and so repairs before it shortens.
Writing a second move set and a second way to price an overload for one
heuristic was the alternative, and the measurement below did not ask for it.

It is not built on `BreakoutLocalSearch`. That loop always continues from
the local optimum it reached, and this one returns to a reference unless the
result passes a threshold, which is the step the two differ in.

## Do not retry

A separate feasibility phase is worth writing only if runs end on solutions
that break capacity. Every run below ended feasible.

SWAP\* is the one published move the descent lacks. It belongs in the descent.

## Appendix

`data/benchmarks/vrp/ails_x.toml`, five runs, seed 42, 60 s, the five runs in
parallel on an Apple M4. Average objective, every run feasible.

| Instance | AdaptiveIteratedLocalSearch | AdaptiveLargeNeighborhoodSearch | HybridGeneticSearch |
|---|---|---|---|
| X-n214-k11 | 10971.8 | 10967.4 | 11033.8 |
| X-n303-k21 | 21864.8 | 22020.8 | 22185.2 |
| X-n401-k29 | 66580.6 | 67868.4 | 67541.0 |
| X-n1001-k43 | 73443.0 | 81213.8 | 77436.2 |

On X-n214-k11 it is level with ALNS, 4 apart with standard deviations of 18
and 41. It leads from X-n303-k21 up, and by 5% over the next best on
X-n1001-k43. It made 78 to 101 thousand iterations per run on every instance,
so the cost of an iteration did not grow with the instance.

One run of 20 s each with 20 neighbors instead of 40 gave 10979 against 10990
on X-n214-k11 and 73876 against 73822 on X-n1001-k43, with twice the
iterations. That does not separate the two, and the published 40 stays.
