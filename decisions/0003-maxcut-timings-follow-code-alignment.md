# A timing difference on MaxCut is not on its own evidence

- Status: reference
- Area: max_cut
- Date: 2026-09-07

## Decision

The MaxCut hot loop is sensitive to where its code lands. A semantically
identical change, with the same instruction stream and the same heap layout, can
move Breakout Local Search by 7 to 10 percent. Treat any timing difference as
unattributed until the instruction streams have been compared.

Two spellings of the same stop condition differ by 7%, which is why `descend`
hands `LocalSearch` an unreachable iteration cap rather than an empty condition.

## Measurement

Deleting `positive_gain` shrank `MaxCutSolution` from 168 to 112 bytes and cost
8%. Padding the struct back by 8 bytes restored the speed exactly, and so did
every larger padding, so it was a cliff at one size rather than a gradient.
Padding changes no allocation, so the hot arrays sat at identical addresses.
`MaxCutFlipNeighbor::apply_to_solution` was the same 396 bytes of instructions
either way and differed only by a 4-byte relocation. Building with
`-C llvm-args=-align-all-functions=6` erased the gap, 1.098 to 0.999.

## How to A/B here

Run arms interleaved with `num_runs = 1` sequential, since parallel runs land on
performance and efficiency cores unpredictably. Take the minimum of three
repetitions. The machine noise floor is about 1.01 to 1.06. Compare solution
vectors and not just objectives, because a refactor meant to preserve behaviour
must be bit-identical. Prefer a fixed iteration budget over wall clock. Check
whether the instruction streams differ at all before attributing anything, using
`nm -n` for the symbol and `otool -tV` to diff it. Re-measure the integrated
change rather than a patched-in-place arm, because those have disagreed by 6%.

## Do not retry

The alignment flag looks attractive, about 0.2% cost to remove a 10% landmine,
but it is an unstable LLVM flag measured only on four MaxCut instances. It needs
TSP, VRP and SAT numbers before it belongs in the build configuration.
