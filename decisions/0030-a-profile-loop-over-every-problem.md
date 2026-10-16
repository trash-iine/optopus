# A profile loop over every problem found the cost in the problems' own pricing, and each fix had to beat the alignment noise

- Status: reference
- Area: all
- Date: 2026-10-16
- Code: src/problem/{bin_packing,max_cut,qubo,vertex_cover,sat,qap,job_shop_scheduling,flow_shop}/, src/building_blocks/representation/binary/neighbor.rs

## Decision

57 fixed-iteration workloads covering every problem and heuristic were profiled, and the top of each profile was fixed wherever the cost lay in how a move is priced or applied. No heuristic and no trait bound changed. Each fix keeps the neighborhood and the move distribution. Two of them change the RNG stream, and the rest are bit-identical.

## Measurement

Each arm was built as is and with `-C llvm-args=-align-all-functions=6`, and run with `num_runs = 1`, `RAYON_NUM_THREADS=1`, arms interleaved, taking the minimum of three runs. A fix was kept when the aligned build showed it and the normal build stayed under the 1.06 noise floor. When the two builds disagreed, the instruction streams were diffed first. Ratios below are origin/main against the final commit, normal / aligned.

| Workload | Ratio |
|---|---|
| SA Relocate, bin packing u250_00 (RNG changes) | 0.129x / 0.130x |
| SA Swap, MaxCut G22 (RNG changes) | 0.025x / 0.026x |
| SA Flip, MaxSAT station_repacking_49 | 0.085x / 0.087x |
| Tabu Flip, Tabu Swap, MaxSAT uf200-01 | 0.53x, 0.37x / 0.53x, 0.37x |
| Tabu Swap, QAP tai35a | 0.270x / 0.270x |
| Tabu Swap swv11, Tabu Relocate la21, job shop | 0.29x, 0.60x / 0.30x, 0.61x |
| IG, ALNS, Tabu Relocate, flow shop ta051 | 0.74x, 0.76x, 0.77x / 0.75x, 0.77x, 0.82x |
| Tabu Swap, MaxCut G1, QUBO bqp250_1, Vertex Cover G22 | 0.59x, 0.68x, 0.64x / 0.56x, 0.70x, 0.64x |

VNS with a random swap shake moved to 0.86x and a different solution. The other 42 workloads, BLS, PA, PT, GA, LKH, HGS and ALNS included, measure 0.98x to 1.04x in the normal build and 0.98x to 1.05x in the aligned one, with the same solutions as before. At the same budget across ten seeds, the two RNG-changing fixes come out level: 104.6 against 104.2 bins, and 12618.7 against 12615.4 cut.

## Tried and not kept

- Rewriting `Qap::swap_delta`, either with transposed matrices or by summing every `k` and then removing `r` and `s`, ran 1.2 to 1.4x slower than what the compiler makes of the plain formula.
- Stopping a job shop relocate candidate once its schedule rejoins the solution's made la21 1.6x slower, because a relocation rarely rejoins before the end. Only the swap scan stops early.
- Reading the edge weight before the two gains in `MaxCutSwapNeighbor::new` changed `BestSwap::run_once`, which inlines it. BLS moved +8% and +14% in the normal build and 0% in the aligned one. With the read order kept, the instructions are identical.

## Not attempted

- A dense table for pair tabu keys would take 15 to 18% off the QAP, VRP, flow shop and job shop tabu scans. It would also make `TabuMemory::clear`, which runs once per phase, zero up to n² entries.
- `BeamSearch` copies every candidate. Ranking moves before copying would require the solution to implement `Evaluate`, which the search does not require today.
- In the ALNS repair on a tour (0029), LKH's depth-first search and RL's scoring of every move, the cost is the algorithm itself.

## Do not retry

Do not retry the rewrites listed above. A timing difference between the two builds of the same code is not evidence either way: the old code alone measured 0.94x to 1.05x between them on these workloads, TSP, VRP and SAT included, which is the measurement 0003 asked for before the flag goes into the build.
