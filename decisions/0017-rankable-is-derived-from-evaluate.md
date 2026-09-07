# Rankable is derived from Evaluate, and the derivation is free

- Status: adopted
- Area: heuristic, all
- Date: 2026-09-08
- Code: src/trait_defs/rankable.rs

## Decision

`Rankable` is no longer implemented by hand anywhere in the crate. A blanket
impl derives it from `Evaluate`, comparing the two `minimized` values, which is
the objective with its direction already applied.

Both traits were stating the same fact. `MaxCutSolution` reported
`Evaluable::Maximize(objective)` from one and `self.objective > other.objective`
from the other, and all thirty pairs in the crate were like that. Two statements
of one direction is one that can be changed alone, leaving a search ranking
against an objective the problem no longer optimizes, and nothing would fail.

`Evaluable::worsening_amount` went at the same time and for the same reason. It
had been a name for `minimized` since population annealing needed the second
reading, and callers had to choose between two names for one number.

A type that does not implement `Evaluate` may still implement `Rankable`
directly, so an order that is not a comparison of one number, a lexicographic
objective for example, is still expressible. It gives up `Evaluate` and every
heuristic that computes with the objective.

## Measurement

`LocalSearch` and `TabuSearch` select with `max_by(rank_cmp)` over a whole
neighborhood, so `is_better_than` is the innermost comparison the library runs.
On MaxCut a single `f32` comparison became two widenings, two negations and an
`f64` comparison, which had to be measured rather than assumed.

Fixed iteration budget, `num_runs = 1`, sequential, arms alternated, minimum of
three.

| Case | Derived / hand-written |
|---|---|
| Restart{LocalSearch} G1, n=800 dense | 1.018x |
| TabuSearch G1, n=800 dense | 1.007x |
| Restart{LocalSearch} G55, n=5000 | 1.000x |
| TabuSearch G55, n=5000 | 1.017x |
| TabuSearch G70, n=10000 | 1.006x |

Inside the noise floor for this machine, 0003. Every run produced the same
objective as the arm it was paired with, and the trajectory pins in
`tests/pa_trajectory.rs`, `tests/reduction_crossing.rs` and the bit-identity
tests in `tests/benchmark_e2e.rs` all pass on constants captured before the
change, which is what says no comparison changed its answer.

## Do not retry

Do not hand-write `Rankable` back on the hot move types for speed. The
comparison was measured at the two call sites that run it innermost and there is
nothing to win.
