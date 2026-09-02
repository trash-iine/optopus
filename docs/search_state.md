# `SearchState`

**API:** [`SearchState`](api/optopus/search_state/struct.SearchState.html)

`SearchState<'a, P>` is the scratch-pad every heuristic drives: the current
solution, the global best, the counters, the timers, and the one RNG. A
heuristic  reads and writes this s search state instead of touching a problem instance directly.

Signatures, argument lists and per-item notes live in the rustdoc linked above
and from each section below. This page is the part rustdoc cannot give: what to
reach for when, the three clone variants side by side, and why a couple of the
operations are shaped the way they are.

## What it holds

| Role | Fields | Who writes it |
|---|---|---|
| Search | `solution`, `best_solution`, `initial_solution` | the running heuristic |
| Progress | `iteration`, `best_iteration`, `best_time` | `apply` / `progress_iteration` / `update_best` |
| Accounting | `n_accepted`, `n_rejected`, `n_best_updates` | the same three, one counter each |
| Seed | `rng` | everything that draws (see [Reproducibility](#reproducibility)) |
| Recording | `trajectory` | `update_best`, and only once a probe is installed |
| Problem | `instance` (a `&'a P`) | just borrowed for the whole run |

## Creating a state

Two axes, four constructors: where the first solution comes from, and where the
randomness comes from.

|  | Random initial solution | Given initial solution |
|---|---|---|
| **OS entropy** | `SearchState::new(problem)` | `SearchState::with_solution(problem, sol)` |
| **Fixed seed** | `SearchState::new_with_seed(problem, seed)` | `SearchState::with_solution_and_seed(problem, sol, seed)` |

```rust
let mut state = SearchState::new_with_seed(&problem, 42);
```

The seeded pair is what the benchmark uses — it derives one seed per run so a
rerun is bit-identical. The `with_solution` pair is the warm start: the given
solution becomes the current one, the best one, *and* `initial_solution`, so the
reported improvement is measured from it.

## Advancing one step

The body of `run_once` is always the same three beats: pick a move, decide, then
either apply it or burn the iteration.

```rust
let m: N = state.random_neighbor("MyHeuristic")?;   // or scan N::iter(...)
if state.is_neighbor_better_than_current(&m) {
    state.apply(&m)?;                              // applies, counts, refreshes best
} else {
    state.progress_iteration();                    // counts a rejection, no move
}
```

`random_neighbor` draws one uniformly random move and is an `InvalidState` error
when the neighborhood is empty; the `context` string it takes is the heuristic
name, so the message says who ran dry. It is what SA, LAHC and `RandomWalk` call
every step, in place of walking the neighborhood at all.

`apply` and `apply_move_only` differ in one thing: whether the best solution is
refreshed. Use `apply_move_only` inside a perturbation, where the moves are
meant to be worsening and re-checking the best on each one is wasted work, then
call `update_best` once when the phase ends. Both count an acceptance;
`progress_iteration` counts a rejection.

## Isolating a sub-run

Every meta-heuristic runs its phases on a clone and merges the result back. The
pair is the whole mechanism; there is no wrapper around it.

```rust
let mut sub = state.clone_for_new_run(SearchStateCloneType::ClearBest);
inner_heuristic.run(&mut sub)?;
state.update_state(sub);
```

`update_state` installs the sub-run's current solution, adds each counter's
*delta over the sub-run* to the parent's, and adopts the best solution only if
it actually improves. `initial_solution` is never overwritten, so the parent
keeps its own anchor for reporting. It panics if the sub-state borrows a
different problem instance — which is exactly the case a reduction is for (see
below).

Because only deltas are merged, the global `iteration` advances monotonically
across every phase however deeply they nest.

### [`SearchStateCloneType`](api/optopus/search_state/enum.SearchStateCloneType.html) variants

| Variant | Solution | Best | Clocks and anchors |
|---|---|---|---|
| `Simple` | current | retained | `start_iteration = iteration`; clocks unchanged |
| `ClearBest` | current | reset to current | `start_iteration = best_iteration = iteration`, clocks reset |
| `StartBest` | best | retained | `start_iteration = best_iteration = iteration`, clocks reset |

All three keep the parent's iteration frame (`iteration` marks where the current
phase is, and `start_iteration` marks where the phase began). Everything
budget-shaped is measured against that anchor (`iterations_this_run()`,
`StopCondition`, `update_state`'s deltas). What a shared frame buys is that an
iteration number means the same thing on both sides of a merge — which matters
for anything that records one, such as a trajectory point.

The `n_accepted` / `n_rejected` / `n_best_updates` counters are the other kind of
value: they measure a phase rather than timestamp anything, so every variant
starts them at zero and `update_state` adds them straight into the parent. 

`ClearBest` is the usual choice: the phase gets a fresh local notion of "best"
while the parent keeps the global one. `StartBest` restarts the phase from the
incumbent instead of from wherever the last phase drifted to. `Simple` hands the
whole history down unchanged.

`initial_solution` is re-anchored to the phase's starting point by `ClearBest`
and `StartBest`, and inherited by `Simple`.

Every variant forks the RNG: the child gets an independent stream and the
parent's advances by one fork's worth of state. That is why
`clone_for_new_run` takes `&mut self`, and why a sub-run's internal draws never
leak back into the parent's sequence.

## Crossing a reduction

A [`ProblemReduction`](traits.md#problemreduction) maps one instance to another
— a kernel, say — and a sub-run on it cannot go through `update_state`, which
requires the *same* instance. This pair is the crossing:

```rust
let mut sub = state.open_reduction(&kernel);   // warm start, seed drawn from state.rng
heuristic.run(&mut sub)?;
state.close_reduction(&kernel, &sub);          // counters, then the lifted best
```

`open_reduction` projects the incumbent as the sub-run's starting solution and
takes exactly one draw from this state's RNG for its seed — which is what keeps
a seeded run reproducible through a reduction.

`close_reduction` is one method rather than two because the order inside it is
load-bearing. It merges the sub-run's counters *first*, then installs
`lift(..., &sub.best_solution)` and refreshes the best. Installing first would
record a `best_iteration` that omits the sub-run's work entirely — a mistake
invisible in the objective. Note that it is the sub-run's **best** solution that
crosses, not where it happened to stop.

The rest is the caller's loop, deliberately: `tests/reduction_crossing.rs` is
that loop with its trajectory pinned, and
[MaxCut kernelization](problems/max_cut_kernel.md) is the worked example.

## Recording the anytime curve

`trajectory` stays empty until an objective probe is installed. Recording is
opt-in so that `update_best` — called on essentially every improving step —
allocates nothing by default.

```rust
state.set_objective_probe(|sol| sol.objective as f64);
```

From then on each *actual* best update appends a `TrajectoryPoint` (absolute
instant, iteration, objective). The probe is inherited by sub-run clones, so
improvements found inside a meta-heuristic phase are recorded too, and
`update_state` remaps their iterations into the parent's frame while leaving the
instants alone. `benchmark/runner.rs` is the one caller in the library; what it
collects becomes the `trajectory` in the report and the curve the benchmark
viewer plots.

`duration()` is the elapsed time of the *current* sub-run, measured from the
`start_time` that `ClearBest` / `StartBest` reset — which is what stop
conditions compare against.

## Reproducibility

Every source of randomness in a run reaches `state.rng` and nothing else:
initial solutions, `random_neighbor`, tabu tenures (`EnabledTabu` takes the RNG
as a parameter), crossovers, BLS perturbations, and the seeds of sub-runs and
reductions. Fix the seed at construction and the whole composition — nested
meta-heuristics included — replays bit for bit.
