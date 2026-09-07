# `SearchState`

**API:** [`SearchState`](api/optopus/search_state/struct.SearchState.html)

`SearchState<'a, P>` is the scratch-pad every heuristic drives. It holds the
current solution, the global best, the counters, the timers and the one RNG. A
heuristic reads and writes the state rather than touching the problem instance
directly.

Signatures and per-item notes are in the rustdoc linked above. This page covers
what to reach for when, and how the pieces fit together.

## What it holds

| Role | Fields | Who writes it |
|---|---|---|
| Search | `solution`, `best_solution`, `initial_solution` | the running heuristic |
| Progress | `iteration`, `best_iteration`, `best_time` | `apply` / `progress_iteration` / `update_best` |
| Accounting | `n_accepted`, `n_rejected`, `n_best_updates` | the same three, one counter each |
| Seed | `rng` | everything that draws (see [Reproducibility](#reproducibility)) |
| Recording | `trajectory` | `update_best`, once a probe is installed |
| Tabu | the private tabu memory | `apply` / `apply_move_only` (see [Remembering tabu moves](#remembering-tabu-moves)) |
| Problem | `instance` (a `&'a P`) | borrowed for the whole run |

## Creating a state

Two axes give four constructors, one for where the first solution comes from and
one for where the randomness comes from.

|  | Random initial solution | Given initial solution |
|---|---|---|
| OS entropy | `SearchState::new(problem)` | `SearchState::with_solution(problem, sol)` |
| Fixed seed | `SearchState::new_with_seed(problem, seed)` | `SearchState::with_solution_and_seed(problem, sol, seed)` |

```rust
let mut state = SearchState::new_with_seed(&problem, 42);
```

The benchmark uses the seeded pair, deriving one seed per run so that a rerun is
bit-identical. The `with_solution` pair is the warm start, where the given
solution becomes the current one, the best one and `initial_solution`, so the
reported improvement is measured from it.

## Advancing one step

The body of `run_once` is always the same three beats. Pick a move, decide, then
either apply it or burn the iteration.

```rust
let m: N = state.random_neighbor("MyHeuristic")?;   // or scan N::iter(...)
if state.is_neighbor_better_than_current(&m) {
    state.apply(&m)?;                              // applies, counts, refreshes best
} else {
    state.progress_iteration();                    // counts a rejection, no move
}
```

`random_neighbor` draws one uniformly random move and returns an `InvalidState`
error when the neighborhood is empty. The `context` string it takes is the
heuristic name, so the message says who ran dry.

`apply` and `apply_move_only` differ in whether the best solution is refreshed.
Use `apply_move_only` inside a perturbation, where the moves are meant to be
worsening, and call `update_best` once when the phase ends. Both count an
acceptance; `progress_iteration` counts a rejection.

## Remembering tabu moves

Recording is a mode and it starts off, in a fresh state and in every sub-run.
Turn it on with `start_record_tabu((min, max))`, which sets the tenure and the
mode together. While it is on, `apply` and `apply_move_only` record the move
they applied, at the iteration it was made on.

```rust
state.start_record_tabu((5, 10));          // the tenure, and the mode, in one
let free = state.tabu_allows(&m);          // is this move currently forbidden?
state.apply(&m)?;                          // applies, and records
state.reset_tabu();                        // drop every prohibition
```

Arm the mode at the same rate as the tenure is decided, per iteration in
`TabuSearch::run_once` and per half-round in
`BreakoutLocalSearchForMaxCut::prepare`. Until it is armed the tenure is
`(0, 0)`, which records a move and frees it again on the next iteration.
`stop_record_tabu()` leaves the tenure in place, for a search that wants to
forbid only what it hands to `record_tabu`.

A move type opts in by implementing
[`EnabledTabu`](traits.md#core-trait-reference) and overriding
`MoveToNeighbor::tabu_policy` with `Some(self)`. **Implementing the trait
without the one-line override leaves the move with no tabu policy**, so applying
it records nothing and `TabuSearch` runs with an empty list and no complaint;
`trait_defs/tabu.rs` pins every built-in move against that. `tabu_allows` and
`record_tabu` are bounded on `EnabledTabu`, so asking about a move that has no
policy is a compile error.

What a move forbids is a `TabuKey`, either `Var(i)` for a dense index or `Pair`
and `Triple` for the rest. The shapes are separate spaces, so two move types
over the same shape share prohibitions, which is how MaxCut's flip and swap see
each other's entries, while different shapes never collide. The keys a move
reads and the keys it writes need not agree. A VRP relocate asks whether a
customer may enter its destination route and forbids the route it just left, so
the customer cannot be moved straight back. `state.reserve_tabu_vars(n)`
pre-grows the dense space when the instance size is known up front.

## Isolating a sub-run

Every meta-heuristic runs its phases on a clone and merges the result back.

```rust
let mut sub = state.clone_for_new_run(SearchStateCloneType::ClearBest);
inner_heuristic.run(&mut sub)?;
state.update_state(sub);
```

`update_state` installs the sub-run's current solution, adds each counter's
delta to the parent's, and adopts the best solution only if it improves.
`initial_solution` is never overwritten, so the parent keeps its own anchor for
reporting. It panics if the sub-state borrows a different problem instance,
which is what [crossing a reduction](#crossing-a-reduction) is for.

### [`SearchStateCloneType`](api/optopus/search_state/enum.SearchStateCloneType.html) variants

| Variant | Solution | Best | Clocks and anchors |
|---|---|---|---|
| `Simple` | current | retained | `start_iteration = iteration`; clocks unchanged |
| `ClearBest` | current | reset to current | `start_iteration = best_iteration = iteration`, clocks reset |
| `StartBest` | best | retained | `start_iteration = best_iteration = iteration`, clocks reset |

`ClearBest` is the usual choice, giving the phase a fresh local notion of "best"
while the parent keeps the global one. `StartBest` restarts the phase from the
incumbent rather than from wherever the last phase drifted to. `Simple` hands
the whole history down unchanged. `ClearBest` and `StartBest` re-anchor
`initial_solution` to the phase's starting point; `Simple` inherits it.

All three keep the parent's iteration frame. `iteration` marks where the current
phase is and `start_iteration` marks where it began, and everything
budget-shaped is measured against that anchor, so a phase still starts at zero
of its own budget while an iteration number means the same thing on both sides
of a merge. The `n_accepted` / `n_rejected` / `n_best_updates` counters measure a
phase instead, so every variant starts them at zero.

Every variant forks the RNG, giving the child an independent stream, which is
why `clone_for_new_run` takes `&mut self`. Every variant also starts the child
with an empty tabu memory, and `update_state` carries what it learned back into
the parent. A phase that should start from the parent's prohibitions asks with
`sub.inherit_tabu_from(&state)`.

## Crossing a reduction

A [`ProblemReduction`](traits.md#problemreduction) maps one instance to another,
a kernel for example. A sub-run on that instance cannot go through
`update_state`, which requires the same instance, so this pair does the
crossing.

```rust
let mut sub = state.open_reduction(&kernel);   // warm start, seed drawn from state.rng
heuristic.run(&mut sub)?;
state.close_reduction(&kernel, &sub);          // counters, then the lifted best
```

`open_reduction` projects the incumbent as the sub-run's starting solution and
takes exactly one draw from this state's RNG for its seed, which keeps a seeded
run reproducible through the reduction. `close_reduction` merges the counters
before installing the lifted solution, so `best_iteration` accounts for the
sub-run's work. It is the sub-run's best solution that crosses, not where it
happened to stop.

The loop around the pair is the caller's. `tests/reduction_crossing.rs` is that
loop with its trajectory pinned, and
[MaxCut kernelization](problems/max_cut_kernel.md) is the worked example.

## Recording the anytime curve

`trajectory` stays empty until an objective probe is installed, so that
`update_best` allocates nothing by default.

```rust
state.set_objective_probe(|sol| sol.objective as f64);
```

Each best update then appends a `TrajectoryPoint` of instant, iteration and
objective. Sub-run clones inherit the probe, and `update_state` remaps their
iterations into the parent's frame. `benchmark/runner.rs` is the one caller in
the library, and what it collects becomes the `trajectory` in the report and the
curve the benchmark viewer plots.

`duration()` is the elapsed time of the current sub-run, measured from the
`start_time` that `ClearBest` and `StartBest` reset, and is what stop conditions
compare against.

## Reproducibility

Every source of randomness reaches `state.rng` and nothing else: initial
solutions, `random_neighbor`, tabu tenures, crossovers, perturbations, and the
seeds of sub-runs and reductions. Fix the seed at construction and the whole
composition, nested meta-heuristics included, replays bit for bit.
