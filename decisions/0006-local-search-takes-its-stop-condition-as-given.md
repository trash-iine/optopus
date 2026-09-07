# LocalSearch::new no longer forces max_failed_update

- Status: adopted
- Area: heuristic
- Date: 2026-09-07
- Code: src/heuristic/local_search.rs

## Decision

`LocalSearch::new` used to force an unset `max_failed_update` to `Some(1)`. It
now takes the condition as given.

The forcing was redundant on a cold start. A descent applies only strictly
improving moves, so each one also improves the best and leaves
`iteration - best_iteration` at zero, while reaching a local optimum already
ends the run through `is_done`.

It was wrong on a warm start. A state whose `best_solution` already beat its
`solution` returned without taking a single move. Breakout Local Search descends
from exactly there, right after a perturbation.

## Measurement

Every in-tree caller runs `LocalSearch` on a `ClearBest` clone or a fresh state,
so no existing behaviour changes. `tests/reduction_crossing.rs` pins
`LocalSearch` trajectories and passes unchanged.

## Do not retry

Do not reintroduce the forcing to make the constructor "safe". The local-optimum
stop is already guaranteed elsewhere, and the forcing silently disables warm
starts.
