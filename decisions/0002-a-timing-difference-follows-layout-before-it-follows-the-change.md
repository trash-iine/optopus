# A timing difference follows code and heap layout before it follows the change, and how to A/B here

- Status: reference
- Area: all
- Date: 2026-09-07
- Code: Cargo.toml (`lto = "fat"`), src/building_blocks/search/tabu.rs

## Decision

A hot loop's time moves with where its code and heap buffers land,
independently of what the change did. Treat a timing difference as
unattributed until the instruction streams have been compared.

Code layout. A semantically identical change, with the same instruction stream
and heap layout, moved Breakout Local Search by 7 to 10%. Two spellings of the
same stop condition differ by 7%, which is why `descend` hands `LocalSearch` an
unreachable iteration cap rather than an empty condition.

Heap layout. A scan walks the graph's vertex list, the solution's gains and
the dense tabu array in lockstep, and where the allocator puts them decides
whether they collide in cache. Under thin LTO an inline `TabuMemory` map drew
such an arrangement. Fat LTO, now in `Cargo.toml`, removes it.

## How to A/B here

Interleave the arms, `num_runs = 1`, `RAYON_NUM_THREADS=1` (the cores differ),
a fixed iteration budget, minimum of three. The noise floor is 1.01 to 1.06.
Compare solution vectors, not objectives. Build each arm as is and with
`-C llvm-args=-align-all-functions=6`, and keep a change when the aligned build
shows it and the normal one stays under the floor. When they disagree, diff
the hot symbol (`nm -n`, `otool -tV`) first. Re-measure the integrated change,
since a patched-in-place arm has disagreed with it by 6%.

## Do not retry

The alignment flag is for measuring, not for the build. It is an unstable LLVM
flag, and the old code alone measured 0.94x to 1.05x between the two builds
across MaxCut, TSP, VRP and SAT (0020). Do not box the tabu map again either.
What looked like its cost was the heap cliff.

## Appendix

Deleting `positive_gain` shrank `MaxCutSolution` from 168 to 112 bytes and cost
8%. Padding it back by 8 bytes restored the speed exactly, and so did every
larger padding, a cliff at one size rather than a gradient. Padding changes no
allocation, so the hot arrays sat at identical addresses.
`MaxCutFlipNeighbor::apply_to_solution` was the same 396 bytes either way, but
for a 4-byte relocation. The alignment flag erased the gap, 1.098 to 0.999.

Under `lto = "thin"` the inline map cost `TabuSearch` 47% on G32 and 35% on G1
at identical work, with bit-identical objectives, move counts and seeds. Giving
the dense array 4096 slots of slack also fixed it, and boxing the map with 512
slots put it back. Under `lto = "fat"` the inline arrangement measures −2.0%
and +0.2%.

The heap cliff was not the struct's size, the cache lines its fields
occupy, inlining of `is_enabled`, the `TabuKey` construction or the closure
that reads the state. Instruction counts were unchanged. The cost was stalls.
