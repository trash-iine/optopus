# `SearchState` API Reference

Companion reference for [Concepts](concepts.md). This page documents the
struct, its methods, and the sub-run clone/merge variants in full.

## Struct

```rust
pub struct SearchState<'a, P: ProblemTrait> {
    pub instance: &'a P,
    pub solution: P::Solution,        // current
    pub best_solution: P::Solution,   // global best
    pub iteration: u64,
    pub best_iteration: u64,
    pub best_time: Instant,
    // pub(crate) start_iteration / start_time — sub-run management
}
```

## Methods

| Method | What it does |
|---|---|
| `SearchState::new(problem)` | Random initial solution. |
| `SearchState::with_solution(problem, sol)` | Warm start from a known solution. |
| `apply(neighbor)` | Apply move + advance iteration + refresh best. |
| `apply_move_only(neighbor)` | Apply move + advance iteration; do **not** refresh best (used during perturbation phases). |
| `progress_iteration()` | Advance iteration with no move applied. |
| `update_best()` | Refresh best from current solution. |
| `is_neighbor_better_than_current(n)` / `_best(n)` | Lookahead checks. |
| `get_best_move_par_chunks(iter, chunk_size)` | Parallel best-move scan via Rayon. |
| `duration()` | Elapsed time since the current sub-run started. |
| `clone_for_new_run(kind)` + `update_state(sub)` | Sub-run isolation pattern (see below). |

## `SearchStateCloneType` variants

| Variant | Solution | Best | Counters |
|---|---|---|---|
| `Simple` | current | retained | `start_iteration = iteration`; clocks unchanged |
| `ClearBest` | current | reset to current | `iteration = 0`, clocks reset |
| `StartBest` | best | retained | `iteration = 0`, clocks reset |

`update_state` panics if the sub-state references a different problem instance
and accumulates the sub-run's iteration delta into the parent counter.
