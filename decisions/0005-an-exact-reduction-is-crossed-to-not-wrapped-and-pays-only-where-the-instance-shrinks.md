# An exact reduction is crossed to by any heuristic rather than wrapped, and pays only where the instance shrinks

- Status: adopted
- Area: heuristic, max_cut
- Date: 2026-08
- Code: src/trait_defs/reduction.rs, src/search_state/mod.rs (`open_reduction`, `close_reduction`), src/problem/max_cut/kernel.rs

## Decision

A `ProblemReduction` is searched by crossing to it. `open_reduction` gives any
heuristic of the target a state of the reduced instance, and `close_reduction`
lifts what it found back. There is no wrapper heuristic. One existed for the
MaxCut kernel and was deleted once `ProblemReduction` existed, because it held
nothing else, and `tests/reduction_crossing.rs` reproduces its trajectory.

Whether an exact reduction pays is a property of the instance, not of tuning. It
pays where the instance shrinks, and where it does not the crossing buys
nothing. So a reduction is worth a cheap test of whether it shrank, and a caller
skips the crossing when it did not. `MaxCutKernel::is_trivial()` is that test
for the kernel, one comparison. It is the kernel's own method, not part of the
trait.

The rules of an exact reduction are validated against exhaustive enumeration
on small instances rather than transcribed from a paper. The kernel's rules
were checked that way on instances of nine vertices or fewer.

The evidence is MaxCut's kernel only, where sparse graphs shrink and regular
and dense graphs do not shrink at all.

## Do not retry

Do not add a wrapper heuristic around a reduction. The crossing is the whole
of what it would hold.

Do not take a reduction's rules from a paper without checking them against
enumeration.

## Appendix

MaxCut kernel. G70 goes from 8646 to 2164 vertices, and a tree reduces to zero
vertices, which solves it exactly without any search. Regular and dense graphs
do not shrink.

Searching the kernel beats searching the original on 9 of 9 sparse instances,
worth 5673 cut points in total under Breakout Local Search.
