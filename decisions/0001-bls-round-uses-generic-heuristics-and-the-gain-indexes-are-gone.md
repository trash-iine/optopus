# Three quarters of a Breakout Local Search round are generic heuristics, and the gain indexes they made idle are gone

- Status: adopted
- Area: max_cut, qubo
- Date: 2026-09-07
- Code: src/heuristic/specific/max_cut/mod.rs, src/problem/max_cut/problem.rs

## Decision

The descent is a `LocalSearch`, the strong perturbation a `RandomWalk`, and the
weak flip a `TabuSearch`. Only the weak swap is still a hand-written operator,
in `best_swap.rs`, because `M2` moves one vertex per partition side in a single
move and a pair of independent one-step searches cannot express that. This
works because recording is a mode on the `SearchState`, so a generic
heuristic's `apply` writes the same prohibitions a hand-written operator would.
Benlic and Hao put the tabu list update inside the descent loop, and that still
holds.

The descent was the last reader of `MaxCutSolution::positive_gain`. With it,
`zero_gain` and QUBO's `negative_gain`, which never had a reader, went too, and
so did the `GainIndex` the three sets were built on. `MaxCutSolution` is `x`,
`gain` and `objective`. `zero_gain` was read by Population Annealing's cluster
move once per step, while the index charged a membership update on every
accepted Metropolis flip and rode along in every replica clone, so it cost its
reader more than it saved. That reader is gone too (0010).

## Do not retry

Reverting any of the three substitutions is a speed-only change. The descent
was priced and accepted, so do not put it back as a bug fix.

Reintroducing an index needs a consumer that reads it more often than once per
sweep, and a public way to read it. The old fields were private and only the
write-side `enable_*_index` methods were public.

## Appendix

Ten G-set instances, 30 s by five runs for quality, a fixed iteration budget for
time.

| Substitution | Time at equal iterations | Quality |
|---|---|---|
| descent to `LocalSearch` | 1.15x | −41.8 total, 3 better and 5 worse |
| strong kick to `RandomWalk` | 1.01x | bit-identical solutions |
| weak flip to `TabuSearch` | 0.54x | +321.4 total, 7 better and 0 worse |

The descent loses because it gave up the index of improving flips for a scan of
all `n`. The weak flip speedup is not explained (0002).

Dropping `zero_gain` and scanning `gain` ran Population Annealing at 0.92 to
0.94 of the time at a fixed budget, the objective unchanged within noise. Every
heuristic that never enabled an index stayed bit-identical. Dropping
`positive_gain` made BLS 8% slower, which was code alignment (0002).
