# Nested neighborhood scans pay as plain loops where the adapters collected rows or nested three deep

- Status: reference
- Area: max_cut, qubo, vertex_cover, sat, qap, bin_packing, vrp
- Date: 2026-10-16
- Code: src/building_blocks/representation/binary/neighbor.rs, src/problem/sat/neighbor.rs, src/building_blocks/search/pair_scan.rs, src/problem/{qap,bin_packing,vrp}/neighbor.rs

## Decision

decisions/0031 found the tour's 2-opt scan was slow for the shape of its loop. The same rewrite was tried on every other neighborhood still written as nested `flat_map` adapters. The binary swap scan and the MaxSAT swap scan collected each row into a fresh `Vec`, and now keep the row in their scratch while `next` lists it. The QAP swaps, the bin packing relocations and the VRP swap and 2-opt moves are listed by `building_blocks::search::pair_scan`, which runs two index ranges as plain loops and hands each row what its pairs share. Everything else measured stayed on `flat_map`.

## Measurement

The protocol of decisions/0030, against its final commit. Every change is bit-identical. Ratios are normal / aligned.

| Workload | Change | Ratio |
|---|---|---|
| TabuSearch Swap, MaxCut G1 | binary swap row kept in the scratch | 0.293x / 0.289x |
| TabuSearch Swap, QUBO bqp250_1 | same | 0.435x / 0.423x |
| TabuSearch Swap, Vertex Cover G22 | same | 0.523x / 0.517x |
| TabuSearch TwoOpt, VRP X-n101-k25 | `pair_scan` | 0.658x / 0.657x |
| TabuSearch Swap, QAP tai35a | `pair_scan` | 0.787x / 0.796x |
| TabuSearch Relocate, bin packing u250_00 | `pair_scan` | 0.887x / 0.955x |
| TabuSearch Swap, MaxSAT uf200-01 | MaxSAT swap run priced in place | 0.899x / 0.900x |
| TabuSearch Swap, VRP X-n101-k25 | `pair_scan` | 0.910x / 0.888x |

On the binary swap scans, dropping the `Vec` alone, with `next` only, gave 0.47x, 0.48x and 0.62x, and the two-loop `fold` the rest.

The other 53 workloads, every heuristic and problem of decisions/0030 plus a TabuSearch Swap on bin packing and GRASP with LS Swap and TwoOpt on VRP, measure 0.98x to 1.07x in the aligned build with the same solutions. In the normal build five MaxCut workloads moved: VNS 1.36x, ILS 1.35x, BLS on G22 1.15x, BLS on G55 1.05x and BeamSearch 1.06x, against 0.99x to 1.04x aligned. VNS, ILS and BLS spend their time in `LocalSearch::run_once` over the flip scan, whose 164 instructions are the same in both commits at a different address, and BeamSearch in copying the solution. This change touches neither, so that is the layout effect of decisions/0003.

`next` matters as much as `fold`. `TabuSearch` and `LocalSearch` run `filter(..).max_by(..)`, and `max_by` takes its first element through `filter`'s `next`, which asks the scan's `find`, which a hand-written iterator answers by calling `next` once per pair. On u250_00 most relocations do not fit, so most of the scan ran there:

- A `PairScan` that kept the row's range in `self` made TabuSearch Relocate on u250_00 1.70x slower than `flat_map`. The profile put the time on the range's increment and on reloading the item's bin and size, which do not change along the row, on every pair.
- Walking the row in locals brought it to 1.23x. The loop still ran 12 or 13 instructions per pair to the 10 of `flat_map`, a bounds check on the load among them, and building each row's range from `inner(i)` or testing `j < end` by hand did not remove it.
- Walking the rest of the current row apart from the rows after it, so those start from `inner(i)` as plain loops do, brought it to 0.89x.

## Tried and not kept

- The VRP relocate scan through `pair_scan`, the lifted customer shared along a row of every place. TabuSearch Relocate on X-n101 went to 0.86x, and GRASP with LS Relocate on X-n214 to 1.04x in both builds.
- Overriding `find` on `PairScan`, so a descent's predicate runs inside the loop. GRASP stayed at 1.04x, and TabuSearch Relocate on u250_00 went to 3.19x.
- `pair_scan` on the bin packing swap scan (TabuSearch Swap 1.00x) and both graph coloring scans (TabuSearch Flip 1.016x, SA Swap 1.01x).

## Not attempted

The flow shop scans collect a row at a time too, but each move is priced in O(m·n) by `makespan_replacing`, which is 64% of TabuSearch Swap on ta051, so the loop is not where its time goes. The job shop scans are already their own loops (decisions/0030).

## Do not retry

Do not move a scan to `pair_scan`, or write it as its own iterator, without measuring both a TabuSearch and a LocalSearch run of it. The two drive the scan through different methods, and the same rewrite went to 0.86x on one and 1.04x on the other.
