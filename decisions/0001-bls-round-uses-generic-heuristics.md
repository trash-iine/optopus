# Three quarters of a Breakout Local Search round are generic heuristics

- Status: adopted
- Area: max_cut
- Date: 2026-09-07
- Code: src/heuristic/specific/max_cut/bls.rs

## Decision

The descent is a `LocalSearch`, the strong perturbation a `RandomWalk`, and the
weak flip a `TabuSearch`. Only the weak swap is still a hand-written operator,
in `best_swap.rs`, because `M2` moves one vertex per partition side in a single
move and a pair of independent one-step searches cannot express that.

This works because recording is a mode on the `SearchState`, so a generic
heuristic's `apply` writes the same prohibitions a hand-written operator would.
Benlic and Hao put the tabu list update inside the descent loop, and that still
holds.

## Measurement

Ten G-set instances, 30s by five runs for quality, a fixed iteration budget for
time.

| Substitution | Time at equal iterations | Quality |
|---|---|---|
| descent to `LocalSearch` | 1.15x | −41.8 total, 3 better and 5 worse |
| strong kick to `RandomWalk` | 1.01x | bit-identical solutions |
| weak flip to `TabuSearch` | 0.54x | +321.4 total, 7 better and 0 worse |

The descent loses because it gave up an index of the improving flips for a scan
of all `n`. The weak flip speedup is not explained; see 0003.

## Do not retry

Reverting any of the three is a speed-only change. The descent substitution was
priced and accepted, so do not re-propose putting it back as a bug fix.
