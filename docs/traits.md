# Core Traits Reference

Companion reference for [Concepts](concepts.md). This page lists the full
signatures of every trait a problem or move type can implement, and which
heuristic each one unlocks.

All of these traits are defined in `optopus::trait_defs` and re-exported from
`optopus::search_state` and the prelude. The first three are the minimum needed
to support every heuristic; the rest unlock specific algorithms.

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

For QUBO, gain values are integers, so the relevant evaluators are
`Evaluate<i32>` (and `Evaluable<i32>`).
