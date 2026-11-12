# A scan prices each candidate at one call site, since the pricing can inline into a wrapper that then stays out of line

- Status: adopted
- Area: all
- Date: 2026-11-13
- Code: src/problem/integer/neighbor.rs (`Changes::fold`, `IntSwapNeighbor::iter`, `IntReverseNeighbor::iter`), src/building_blocks/search/pair_scan.rs

## Decision

A scan's `fold`, which `max_by` and `filter` drive (0020), prices each
candidate at one place in its loops, with no closure or helper in between.
LLVM inlines bottom up against one threshold, so a wrapper called from several
loops can take in the delta, grow too big to inline into any of them, and cost
a call per candidate, with no source change. `Changes::fold` called an `emit`
closure from three loops. GraphColoring's TabuSearch Flip ran as fast as now
at the default threshold and at `-inline-threshold=1000`, 3.97x slower at 400,
and 3.97x by default with `#[inline(always)]` on the delta and `slot_delta`.
Now the pricing is written once, in one loop over the variables and one over
their values, and the 400 build runs 1.08x.

The generic swap and reversal scans list their pairs through `pair_scan`
rather than `flat_map`, `filter` and `map`, whose `FlattenCompat::fold` had
left its closure out of line on GraphColoring. `pair_scan`'s `fold` keeps its
two loops (0020), so `build` has two call sites there.

To see it, compare instructions retired and cycles (`/usr/bin/time -l`), and
`sample` the run for a function that should have inlined. Other thresholds
(`-C llvm-args=-inline-threshold=N`) show how near a scan sits to the edge.

## Do not retry

- A helper for the per-candidate step, even `#[inline(always)]`. At the 400
  threshold GraphColoring's Flip stayed at 1.25x, against 1.08x at one site.
- `pair_scan`'s `fold` walking the current row in the loop of the rows after
  it, for one call site. Bin packing relocate went to 1.10x (0020 has why).
- Writing a problem's pricing helper into the delta the scan calls, where the
  delta then outgrows what the scan inlines. GraphColoring's
  `swap_conflicts_delta` inside `assign_swap_delta` took Swap 0.79x to 0.94x.

## Appendix

rustc 1.96, aarch64 macOS, fat LTO, `RAYON_NUM_THREADS=1`, min of 3. Ratios
are against the scan before this record at the default threshold, normal build,
with aligned builds where given. GraphColoring G1, TabuSearch Flip 10000
iterations, 3 runs, every arm with the same solutions.

| Flip arm | Instructions | Cycles |
|---|---|---|
| old `fold`, threshold 50 / 100 / 150 | 1.14 / 1.00 / 1.00 | 1.08 / 0.99 / 1.05 |
| old `fold`, threshold 400 | 2.13 | 3.97 |
| old `fold`, threshold 1000 | 1.00 | 1.00 |
| old `fold`, `#[inline(always)]` on `assign_delta` and `slot_delta` | 2.13 | 3.97 |
| old `fold`, `#[inline(always)]` on `assign_delta` only | 1.00 | 1.00 |
| old `fold`, no `#[inline]` on any GraphColoring method | 1.00 | 1.01 |
| old `fold`, the delta's arithmetic in an `#[inline(never)]` helper | 1.66 | 3.08 |
| old `fold`, `emit` as an `#[inline(always)]` fn, threshold 400 | 1.14 | 1.25 |
| one call site | 1.00 | 0.96 / 0.99 |
| one call site, threshold 400 | 1.14 | 1.08 |

GraphColoring's Flip loop with no call runs 6.3 instructions a cycle, and a
call per candidate halves that. At 400 and with both forced inline, 96% of the samples sat in
`Changes::fold::{{closure}}`, the same symbol in both. With one call site they
all sit in `TabuSearch::run_once`.

| Swap arm, G1 | Instructions | Cycles |
|---|---|---|
| `pair_scan`, TabuSearch 300 iterations | 0.72 | 0.79 / 0.80 |
| same, threshold 400 | 0.72 | 0.80 |
| same, `pair_scan`'s `fold` as one loop | 0.74 | 0.81 |
| same, `swap_conflicts_delta` inside `assign_swap_delta` | 0.87 | 0.94 / 0.94 |
| `pair_scan`, LocalSearch on G1 and G43, 20 runs | 0.79 | 0.88 / 0.88 |

The `flat_map` scan's top of stack was the `FlattenCompat` closure,
`HashMap::get` for the pair tabu and `swap_conflicts_delta`, and the
`pair_scan` one's only `TabuSearch::run_once`. With the helper written in,
`assign_swap_delta` is the one out of line. Out of line, a function returning
`KeyDelta<2>` hands it back through memory, where an `i64` or an `(f64, f64)`
comes back in registers. On the `flat_map` scan that alone measured 1.06x.

With `pair_scan`'s `fold` as one loop, TabuSearch Relocate on bin packing
t120_00 for 60000 iterations ran 1.10 the instructions and 1.12 / 1.09 the
cycles, and QAP tai100a and VRP X-n101-k25 measured 0.99 to 1.02. With `pair_scan` unchanged all three run the
same instructions as before. The `IntegerProblem` examples measured 0.90 to
1.03 with the same output. GraphColoring's LocalSearch Flip on G1 and G43
measured 1.00 / 1.07 with the same instructions, and 1.00 with
`-align-loops=64` added. GraphColoring's 81 pinned runs are bit-identical.
