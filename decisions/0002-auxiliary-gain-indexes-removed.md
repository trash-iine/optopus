# The optional improving-move and plateau indexes were all removed

- Status: adopted
- Area: max_cut, qubo
- Date: 2026-09-07
- Code: src/problem/max_cut/problem.rs

## Decision

`MaxCutSolution` carried `positive_gain` and `zero_gain`, `QuboSolution`
carried `negative_gain`, and all three were incrementally maintained sets built
on a shared `GainIndex`. All of them are gone, and so is `GainIndex`.
`MaxCutSolution` is now `x`, `gain` and `objective`.

`positive_gain` lost its last reader when the descent became a `LocalSearch`,
and QUBO's `negative_gain` never had one. Neither could be read from outside the
crate anyway, since the fields were private and only the write-side
`enable_*_index` methods were public.

`zero_gain` did have a reader, Population Annealing's cluster move, and was
removed because the index cost that reader more than it saved. The plateau is
read once per step, while the index charged a membership update on every
accepted Metropolis flip and rode along in every replica clone `resample` makes.

## Measurement

Dropping `zero_gain` and scanning `gain` instead ran Population Annealing at
0.92 to 0.94 of the time at a fixed iteration budget, with the objective
unchanged within noise. Every heuristic that never enabled an index stayed
bit-identical.

Dropping `positive_gain` also made Breakout Local Search 8% slower, which was
code alignment rather than the deletion; see 0003.

## Do not retry

Reintroducing an index needs a consumer that reads it more often than once per
sweep, and a public way to read it.
