# An abstraction costs nothing, except where a trait asks at a finer granularity than its answer changes at

- Status: reference
- Area: heuristic, all
- Date: 2026-09-08
- Code: src/trait_defs/rankable.rs, src/heuristic/{alns,population_annealing,bls}.rs, src/problem/vrp/ruin.rs

## Decision

Breakout Local Search, Population Annealing and ALNS became generic over the
problem, each keeping its trajectory. `Rankable` is no longer implemented by
hand anywhere. A blanket impl derives it from `Evaluate` by comparing the two
`minimized` values, since all thirty hand-written pairs stated the direction a
second time, and a second statement can be changed alone with nothing failing.
`Evaluable::worsening_amount` went for the same reason, a second name for
`minimized`. A type without `Evaluate` may still implement `Rankable`
directly, so a lexicographic order stays expressible.

Every one of these measured free but ALNS on the VRP. The VRP-only code read
`Vrp::penalty_weight()` once per repair and folded the capacity penalty in per
route. `Ruinable::insertion_cost` is per candidate place, so the read, behind a
`OnceLock`, moved inside a loop regret-2 runs O(k² · places) times. Caching the
weight on the partial recovered most of it. The remaining 7 to 10% is the same
shape. `excess_delta_insert` depends on the bucket and not the place, and the
trait has nowhere to hoist it. Closing it needs a per-bucket step in
`Ruinable`, a design change that was not made.

## For the next trait

A trait boundary is free when it asks the problem for a decision. It is not
free when it asks at a finer granularity than the answer changes at, because
everything the coarser level had hoisted moves back into the inner loop,
silently. Check what the specific code computed once per container before
assuming a per-item method is equivalent.

## Do not retry

Do not hand-write `Rankable` back on the hot move types for speed, and do not
give `Ruinable` a refill method. Both were measured and gain nothing.

## Appendix

Fixed iteration budget, `num_runs = 1`, sequential, arms alternated, minimum of
three. Every arm gave the same objective as the code it replaced, and the pins
in `tests/pa_trajectory.rs`, `tests/reduction_crossing.rs` and
`tests/benchmark_e2e.rs` pass on constants captured before.

| Change | Case | New / old |
|---|---|---|
| generic BLS | MaxCut n=2000, degree 24, 2M iterations | 1.019x |
| generic BLS | MaxCut n=5000, degree 20, 2M iterations | 1.003x |
| generic PA | MaxCut n=2000, degree 24, 20k iterations | 0.991x |
| generic ALNS | X-n214-k11, 20k iterations | 1.129x, 1.073x cached |
| generic ALNS | X-n1001-k43, 5k iterations | 1.203x, 1.103x cached |
| derived `Rankable` | Restart{LocalSearch} G1 / G55 | 1.018x / 1.000x |
| derived `Rankable` | TabuSearch G1 / G55 / G70 | 1.007x / 1.017x / 1.006x |

On MaxCut the derived comparison turns one `f32` comparison into two widenings,
two negations and an `f64` comparison. Everything but ALNS is inside the noise
floor (0002).

Removing three ALNS suspects moved nothing. They were the position index
rebuilt after a repair, the element listing done twice per iteration and the
working representation reallocated per iteration. So the partial is built once
per iteration.
