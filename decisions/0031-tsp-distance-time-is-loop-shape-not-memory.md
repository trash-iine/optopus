# The TSP's distance time was the shape of the scan loop, not the matrix's memory

- Status: reference
- Area: tsp
- Date: 2026-10-16
- Code: src/problem/tsp/neighbor.rs, src/problem/tsp/ruin.rs, src/building_blocks/instance/distance_store.rs

## Decision

After decisions/0030, `DistanceStore::distance` was still 10 to 36% of the TSP's local search and ALNS. The suspicion was that each lookup was waiting on memory, so locality was tried first. That gave little. What paid was rewriting the 2-opt scan as an iterator whose `fold` runs two plain loops. A symmetric matrix also lets the insertion scans read along the inserted city's row, which is a small gain and is kept.

## Measurement

The protocol was decisions/0003: each arm built as is and with `align-all-functions=6`, `num_runs = 1`, minimum of three runs. Every change is bit-identical. Ratios are normal / aligned.

| Change | Workload | Ratio |
|---|---|---|
| `distance_to` reads `c`'s row when symmetric | ALNS dsj1000 | 0.953x / 0.955x |
| `TwoOptScan` with a two-loop `fold` | LS TwoOpt dsj1000 | 0.701x / 0.693x |
| same | Tabu TwoOpt ch150 | 0.575x / 0.579x |

All other TSP workloads measured 0.98 to 1.01x.

Four probes support reading the gain as loop shape rather than memory:

- Reading `d(prev, next)` was dropped from the ALNS repair entirely, and the repair got no faster.
- The fold was kept and the tour edge read from the matrix instead of the array: 0.708x instead of 0.703x.
- Keeping the `flat_map` adapters and adding the edge array (one `Arc` per row) made ch150 1.37x slower.
- Moving the loop body into a helper method lost a third of the gain (0.81x and 0.64x).

On random EUC_2D tours, growing the matrix from 7.6 MB (n = 1000) to 30.5 MB (n = 2000, the largest that keeps a matrix) raised the cost per 2-opt candidate only from 5.5 to 6.2 ns for LS and from 3.2 to 3.8 ns for Tabu.

## Not attempted

Storing the matrix as `f32` where every value is exact would recover at most that 14 to 21%, and only for n between about 1400 and 2000. It would also mean two matrix types behind every lookup.

## Do not retry

Do not try more locality work on matrices that fit in L2. The ALNS repair's remaining lookups come from its per-place granularity, which decisions/0015 already prices. The other two-level neighborhoods (VRP, bin packing, QAP, the binary swaps) are still nested adapters, and they are where the 2-opt result is worth repeating, each measured on its own.
