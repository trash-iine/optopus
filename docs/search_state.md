# `SearchState` API Reference

**API:** [`SearchState`](api/optopus/search_state/struct.SearchState.html) · [`SearchStateCloneType`](api/optopus/search_state/enum.SearchStateCloneType.html)

Companion reference for [Concepts](concepts.md). This page documents the
struct, its methods, and the sub-run clone/merge variants in full.

## Struct

```rust
pub struct SearchState<'a, P: ProblemTrait> {
    pub instance: &'a P,
    pub solution: P::Solution,           // current
    pub best_solution: P::Solution,      // global best
    pub initial_solution: P::Solution,   // what this sub-run started from
    pub iteration: u64,
    pub best_iteration: u64,
    pub best_time: Instant,
    pub n_accepted: u64,                 // moves applied
    pub n_rejected: u64,                 // iterations advanced without a move
    pub n_best_updates: u64,             // times best_solution was replaced
    pub rng: SmallRng,                   // ALL randomness flows through this
    pub trajectory: Vec<TrajectoryPoint>,// anytime curve; empty without a probe
    // pub(crate) start_* — sub-run merge anchors, hands off
}
```

Everything `pub` is live state that heuristics read *and* write. The
`pub(crate) start_*` anchors exist only so `clone_for_new_run` /
`update_state` can merge a sub-run's deltas back; writing them from outside
would corrupt that accounting.

## Methods

| Method | What it does |
|---|---|
| `SearchState::new(problem)` | Random initial solution, seeded from OS entropy. |
| `SearchState::new_with_seed(problem, seed)` | Same, from a deterministic seed — bit-reproducible runs. |
| `SearchState::with_solution(problem, sol)` | Warm start from a known solution. |
| `SearchState::with_solution_and_seed(problem, sol, seed)` | Warm start with a deterministic seed. |
| `apply(neighbor)` | Apply move + advance iteration + refresh best. |
| `apply_move_only(neighbor)` | Apply move + advance iteration; do **not** refresh best (used during perturbation phases). |
| `progress_iteration()` | Advance iteration with no move applied. |
| `update_best()` | Refresh best from current solution. |
| `random_neighbor::<N>(context)` | Draw one uniformly random move, or `InvalidState` when the neighborhood is empty. What SA / LAHC / `RandomWalk` call each step. |
| `is_neighbor_better_than_current(n)` / `_best(n)` | Lookahead checks. |
| `set_objective_probe(probe)` | Install `fn(&P::Solution) -> f64`; from then on every best update appends to `trajectory`. Without it recording is off and `update_best` stays allocation-free. |
| `duration()` | Elapsed time since the current sub-run started. |
| `clone_for_new_run(kind)` + `update_state(sub)` | Sub-run isolation pattern (see below). |
| `open_reduction(&reduction)` | Opens a sub-state on a `ProblemReduction`'s target, warm-started from the current solution and seeded from this state's RNG. |
| `close_reduction(&reduction, &sub)` | Folds that sub-run back: merges its counters, then installs its lifted best solution as the current one and refreshes best. |

## `SearchStateCloneType` variants

| Variant | Solution | Best | Counters |
|---|---|---|---|
| `Simple` | current | retained | `start_iteration = iteration`; clocks unchanged |
| `ClearBest` | current | reset to current | `iteration = 0`, clocks reset |
| `StartBest` | best | retained | `iteration = 0`, clocks reset |

`update_state` panics if the sub-state references a different problem instance
and accumulates the sub-run's iteration delta into the parent counter.

Every variant **forks** the RNG: the child gets an independent stream and the
parent's advances by one fork, so a sub-run's internal RNG consumption never
leaks back and composition stays deterministic under a fixed seed.
`initial_solution` is re-anchored by `ClearBest` / `StartBest` and inherited by
`Simple`.
