# Generalizing a heuristic costs nothing, except where the trait's granularity repeats work

- Status: reference
- Area: heuristic, vrp
- Date: 2026-09-07
- Code: src/heuristic/{alns,population_annealing,breakout_local_search}.rs, src/problem/vrp/ruin.rs

## Decision

Breakout Local Search, Population Annealing and ALNS became generic over the
problem. Each keeps the trajectory it had, so the only question is what the
abstraction costs at run time. Three of the four measured as free. The fourth
was not, and the reason is worth knowing before writing the next such trait.

## Measurement

Fixed iteration budget, `num_runs = 1` so the runner stays sequential,
alternating arms, minimum of three repeats. Every arm produced the same
objective as the implementation it replaced.

| Search | Instance | Budget | Generic / specific |
|---|---|---|---|
| BLS | MaxCut n=2000, degree 24 | 2M iterations | 1.019x |
| BLS | MaxCut n=5000, degree 20 | 2M iterations | 1.003x |
| Population Annealing | MaxCut n=2000, degree 24 | 20k iterations | 0.991x |
| ALNS | CVRP X-n214-k11 | 20k iterations | 1.073x |
| ALNS | CVRP X-n1001-k43 | 5k iterations | 1.103x |

The BLS and PA numbers are inside the noise floor for this machine (see 0003).

ALNS started at 1.129x and 1.203x. Three suspects were measured and cleared:
the position index rebuilt after a repair, the element listing done twice per
iteration, and the working representation reallocated per iteration. Removing
all three moved nothing.

What moved it was `Vrp::penalty_weight()` behind a `OnceLock`. The VRP-only
implementation read it once per repair and folded the capacity penalty in per
route. `Ruinable::insertion_cost` is per candidate place, so the read moved
inside a loop that regret-2 runs O(k² · places) times per iteration. Caching
the weight on the partial took 1.129x to 1.073x and 1.203x to 1.103x.

## What is left

The remaining 7 to 10% is the same shape. `excess_delta_insert` depends on the
bucket and not the place, and the trait has nowhere to hoist it to. Closing it
needs a per-bucket step in `Ruinable`, which is a design change and was not
made.

## For the next trait

A trait boundary is free when it asks the problem for a decision. It is not
free when the granularity it asks at is finer than the granularity the answer
changes at: everything the coarser level had hoisted moves back into the inner
loop, silently. Check what the specific implementation computed once per
container before assuming a per-item method is equivalent.
