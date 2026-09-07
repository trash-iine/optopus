# Exact reduction is worth crossing to on sparse graphs and nowhere else

- Status: adopted
- Area: max_cut
- Date: 2026-08
- Code: src/problem/max_cut/kernel.rs

## Decision

`MaxCutKernel::new` produces an exactly reduced instance, and any
`Heuristic<MaxCut>` searches it through `open_reduction` and `close_reduction`.
There is no wrapper heuristic. A wrapper existed and was deleted once
`ProblemReduction` existed, because it held nothing else;
`tests/reduction_crossing.rs` reproduces its trajectory.

Whether reduction pays is a property of the instance and not of tuning, so
`is_trivial()` answers it in one comparison and the crossing can be skipped.

## Measurement

Sparse graphs shrink. G70 goes from 8646 to 2164 vertices, and a tree reduces to
zero vertices, which solves it exactly without any search. Regular and dense
graphs do not shrink at all.

Searching the kernel beats searching the original on 9 of 9 sparse instances,
worth 5673 cut points in total under Breakout Local Search.

## Do not retry

The reduction rules are validated by exhaustive brute force on instances of nine
vertices or fewer, not transcribed from the paper. Keep it that way.
