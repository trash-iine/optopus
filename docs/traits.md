# Core Traits Reference

Companion reference for [Concepts](concepts.md). This page lists the full
signatures of every trait a problem or move type can implement, and which
heuristic each one unlocks.

All of these traits are defined in `optopus::trait_defs` and re-exported from
`optopus::search_state` and the prelude. The first three are the minimum needed
to support every heuristic; the rest unlock specific algorithms. The last,
`ProblemReduction`, is the exception — it unlocks nothing and is implemented by
a *reduction* rather than by a problem; see the section below the table.

## Core trait reference

| Trait | Required by | Key signature |
|---|---|---|
| `ProblemTrait` | every heuristic | `type Solution: Clone + Rankable; fn new_solution(&self, rng) -> Solution` |
| `Rankable` (on `Solution`) | every heuristic | `fn is_better_than(&self, other: &Self) -> bool` |
| `MoveToNeighbor<P>` | every heuristic | `fn iter(prob, sol) -> impl Iterator<Self> + Send`<br>`fn apply_to_solution(&self, prob, sol) -> Result<()>`<br>`fn move_to_be_better_than(&self, prob, src, other) -> bool` (default: clone + apply)<br>`fn apply_to_iteration(&self, iter) -> u64` (default: `iter + 1`) |
| `Rankable` (on the neighbor) | `LocalSearch`, `BeamSearch`, `RandomWalk`, `TabuSearch` | same signature; selects the best move among candidates |
| `Evaluate<T>` (returns `Evaluable<T>`) | `SimulatedAnnealing`, `LateAcceptanceHillClimbing`, `RlSearch` | `fn evaluate(&self) -> Evaluable<T>` (default `T = f64`); `Evaluable::Maximize(T)` / `Minimize(T)` carries the optimization direction. `Evaluable<f64>::worsening_amount()` normalizes both directions to "positive = worse" (used by `boltzmann_accept`). |
| `EnabledTabu` | `TabuSearch` | `type TabuMap: Default;`<br>`fn is_move_enabled(&self, map, iter) -> bool;`<br>`fn add_to_tabu_map(&self, map, iter, tenure: (u64, u64))` |
| `Crossover<P>` | `GeneticAlgorithm` | `fn crossover(&mut self, prob, sol1, sol2) -> P::Solution` (`&mut self` lets stateful operators run a sub-heuristic) |
| `SubProblemExtractable` | `SubProblemBasedCrossover` | `fn extract_sub_problem(&self, sol1, sol2) -> Self;`<br>`fn lift_solution(&self, sol1, sol2, sub_solution) -> Self::Solution` |
| `Distance` (on `Solution`) | `GeneticAlgorithm::ParentSelection::DistantTopK` | `fn distance(&self, other: &Self) -> usize` |
| `BinaryProblem` | the shared binary machinery in `common::binary` | `type Flip;`<br>`fn variable_indices(&self) -> Range<usize>;`<br>`fn variable(sol, i) -> bool;`<br>`fn flip_move(sol, i) -> Self::Flip` |
| `ProblemReduction` | nothing — it is a facility, not a requirement | `type Source: ProblemTrait; type Target: ProblemTrait;`<br>`fn target(&self) -> &Self::Target;`<br>`fn project(&self, sol: &SourceSolution) -> TargetSolution;`<br>`fn lift(&self, source: &Self::Source, base: &SourceSolution, sol: &TargetSolution) -> SourceSolution` |

In the last row `SourceSolution` and `TargetSolution` stand for
`<Self::Source as ProblemTrait>::Solution` and
`<Self::Target as ProblemTrait>::Solution`, which is how the trait spells them
— there is no alias for either.

For QUBO, gain values are integers, so the relevant evaluators are
`Evaluate<i32>` (and `Evaluable<i32>`).

## `ProblemReduction`

Unlike everything above, this is not implemented by a problem to unlock a
heuristic. It is a **map from one problem instance to another**, with solutions
crossing in both directions: something smaller to search, a way in for a warm
start, and a way back out. `MaxCutKernel` implements it
([kernelization](problems/max_cut_kernel.md), `Source = Target = MaxCut`);
`Source` and `Target` are separate associated types because a reduction need
not stay inside its problem — a penalised objective whose penalty term is
quadratic reduces into a `Qubo`.

`lift` takes two extra arguments for reasons worth knowing before implementing
one. `base` supplies whatever the map dropped: when the target does not cover
the source's whole variable index space (a kernelization that deleted isolated
vertices), the uncovered positions keep their value from `base`. `source` is a
parameter rather than a field because a solution carries incremental caches
only the instance can rebuild, and holding a `&Self::Source` is not open to
every implementation — one stored in a heuristic outlives any single call.

**Exactness is not part of the trait.** An approximate reduction has the same
shape. Each user requires what it needs; `MaxCutKernel` guarantees
`kernel_cut(y) + offset == original_cut(lift(y))` for *every* `y`, not only
optimal ones, which is what lets a heuristic be stopped at any point and lifted.

The trait is only the map. **Running a heuristic on the target and folding the
result back is a search-state operation**, and lives there:
[`SearchState::open_reduction` and
`close_reduction`](search_state.md#methods). Doing it by hand is where copies
drift apart — silently, in `iteration` / `n_accepted` / `best_iteration` rather
than in the objective.
