# A profile loop over every problem found the cost in the problems' pricing and in the shape of their scans

- Status: reference
- Area: all
- Date: 2026-10-16
- Code: src/problem/*/neighbor.rs, src/building_blocks/representation/binary/neighbor.rs, src/building_blocks/search/pair_scan.rs, src/building_blocks/instance/distance_store.rs

## Decision

57 fixed-iteration workloads covering every problem and heuristic were profiled
in three rounds, by the protocol of 0002. No heuristic and no trait bound
changed, and each fix keeps the neighborhood and the move distribution.

The first round fixed how moves are priced or applied, in bin packing, the
binary swaps (`BinaryProblem::random_variable`), MaxSAT, QAP, the job shop and
the flow shop. Two of the fixes change the RNG stream, the rest are
bit-identical. The TSP's distance lookups looked like memory waits and were
not. What paid was the 2-opt scan as its own iterator whose `fold`, which is
how `max_by` and `filter` drive a scan, runs two plain loops. The same rewrite
paid where a scan collected each row into a `Vec` or nested three adapters
deep, and `building_blocks::search::pair_scan` lists QAP, bin packing relocate
and VRP swap and 2-opt that way.

A hand-written scan has to be fast in `next` too. `max_by` takes its first
element through `filter`'s `next`, one call per pair, and where most moves are
rejected most of the scan runs there. `PairScan` walks the rows in locals.

## Do not retry

- `Qap::swap_delta` rewritten (1.2 to 1.4x), a job shop relocate stopped once
  it rejoins (1.6x), and `MaxCutSwapNeighbor::new` reading the weight first.
- Locality work on matrices that fit in L2, and `f32` storage (at most 14 to
  21%, for n between about 1400 and 2000).
- A scan moved to `pair_scan` without both a TabuSearch and a LocalSearch run.
  VRP relocate went to 0.86x on one and 1.04x on the other. Overriding `find`
  made bin packing 3.19x slower.
- A dense pair tabu table (`clear` would zero n² entries), ranking `BeamSearch`
  candidates before copying (it would need `Evaluate`), and the ALNS repair on
  a tour (0019), LKH and RL's move scoring, where the cost is the algorithm.

## Appendix

Ratios are normal / aligned build, origin/main before the round against its
final commit.

| Round 1 workload | Ratio |
|---|---|
| SA Relocate, bin packing u250_00 (RNG changes) | 0.129x / 0.130x |
| SA Swap, MaxCut G22 (RNG changes) | 0.025x / 0.026x |
| SA Flip, MaxSAT station_repacking_49 | 0.085x / 0.087x |
| Tabu Flip, Tabu Swap, MaxSAT uf200-01 | 0.53x, 0.37x / 0.53x, 0.37x |
| Tabu Swap, QAP tai35a | 0.270x / 0.270x |
| Tabu Swap swv11, Tabu Relocate la21, job shop | 0.29x, 0.60x / 0.30x, 0.61x |
| IG, ALNS, Tabu Relocate, flow shop ta051 | 0.74x, 0.76x, 0.77x / 0.75x, 0.77x, 0.82x |
| Tabu Swap, MaxCut G1, QUBO bqp250_1, Vertex Cover G22 | 0.59x, 0.68x, 0.64x / 0.56x, 0.70x, 0.64x |

VNS with a random swap shake went to 0.86x with a different solution. The other
42 measured 0.98x to 1.05x with the same solutions. Across ten seeds the two
RNG-changing fixes come out level, 104.6 against 104.2 bins and 12618.7
against 12615.4 cut. With the read order kept in `MaxCutSwapNeighbor::new`, the
instructions are identical, where the reordered read moved BLS +8% and +14% in
the normal build only.

| Rounds 2 and 3 workload | Change | Ratio |
|---|---|---|
| LS TwoOpt, TSP dsj1000 | 2-opt scan's `fold` | 0.701x / 0.693x |
| Tabu TwoOpt, TSP ch150 | same | 0.575x / 0.579x |
| ALNS, TSP dsj1000 | symmetric matrix read along a row | 0.953x / 0.955x |
| Tabu Swap, MaxCut G1, QUBO bqp250_1, VC G22 | binary swap row kept in scratch | 0.293x, 0.435x, 0.523x / 0.289x, 0.423x, 0.517x |
| Tabu TwoOpt, Tabu Swap, VRP X-n101-k25 | `pair_scan` | 0.658x, 0.910x / 0.657x, 0.888x |
| Tabu Swap, QAP tai35a | `pair_scan` | 0.787x / 0.796x |
| Tabu Relocate, bin packing u250_00 | `pair_scan` | 0.887x / 0.955x |
| Tabu Swap, MaxSAT uf200-01 | swap run priced in place | 0.899x / 0.900x |

On the binary swaps, dropping the `Vec` with `next` only gave 0.47x, 0.48x and
0.62x, the two-loop `fold` the rest. On the 2-opt scan the edge array was
0.703x against 0.708x without it, the array behind `flat_map` (an `Arc` per row)
was 1.37x slower, and moving the loop body into a helper lost a third of the
gain. Dropping `d(prev, next)` from the ALNS repair entirely made it no faster,
and growing the matrix from 7.6 MB to 30.5 MB raised a 2-opt candidate only
from 5.5 to 6.2 ns. In round 3 the other 53 workloads measured 0.98x to 1.07x
aligned. MaxCut VNS, ILS and BLS ran 1.05x to 1.36x slower in the normal build
only, their `LocalSearch::run_once` the same 164 instructions at another
address.

On bin packing relocate, where most moves do not fit, a `PairScan` keeping the
row's range in `self` was 1.70x slower than `flat_map`, the range stored and the
item's bin and size reloaded on every pair. Walking it in locals gave 1.23x,
and walking the current row apart from the rows after it 0.89x.
