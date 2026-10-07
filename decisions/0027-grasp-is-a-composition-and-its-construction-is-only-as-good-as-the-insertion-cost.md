# GRASP is a composition, and its construction is only as good as the problem's insertion cost

- Status: reference
- Area: heuristic, flow_shop, bin_packing
- Date: 2026-10-07
- Code: src/heuristic/greedy_randomized_construction.rs, src/building_blocks/search/ruin_recreate.rs (`randomized_greedy_insertion`)

## Decision

No GRASP type was added. The construction phase is one step,
`GreedyRandomizedConstruction<P: Ruinable>`, which builds a single solution
from an empty partial by the restricted candidate list over each element's
cheapest `insertion_cost`. GRASP is `Iterated { search, construction }`, which
already alternates the two and keeps the best of every cycle. The step reads
only `Ruinable`, so it runs on TSP, VRP, the flow shop and bin packing with no
problem-specific construction.

The price of that genericity is that the greedy rule is "cheapest insertion
first", whatever that means on the problem.

## Measurement

Same time budget per arm, seed 42, `alpha = 0.2`, the descent a `LocalSearch`
over `Relocate`, against multi-start descent (`Restart` with
`failed_updates(0)`) from the problem's random solutions.

| Problem | Instances | Budget × runs | GRASP | Multi-start | ALNS |
|---|---|---|---|---|---|
| Flow shop (makespan) | ta051 to ta060 | 5 s × 3 | 3794.4, better on 10 of 10 | 3821.4 | |
| Bin packing (bins) | t249_00 to t249_04 | 3 s × 3 | about 92 | about 90 | 84 |
| Bin packing (bins) | u250_00 to u250_04 | 3 s × 3 | 102 to 105 | 101.7 to 104 | 99 to 102 |

On a 150 city random tour with 2-opt, a single 2 s run, GRASP gave 9459.8 to
9881.4 over `alpha` from 0 to 1 against 9555.5 for multi-start.

The ALNS column on bin packing ran without a `LocalRepair`, since bin packing
has none yet, so it is ruin and recreate alone.

On the flow shop the cheapest insertion is the NEH rule, a good one, and GRASP
pays. On bin packing the cheapest element to insert is one that fits an open
bin, so the large items are left for last and each opens a bin, the opposite of
First Fit Decreasing, and GRASP loses even to random restarts. ALNS on the same
`Ruinable` is far ahead there.

## Do not retry

Do not read the bin packing row as GRASP being weak there. A size-ordered
construction would be a different greedy rule, which `Ruinable` does not
express, and adding one would need a second problem that wants it.

Do not add a `Grasp` type to make the composition shorter.
