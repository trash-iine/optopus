# Defining a Custom Problem

**API:** [`ProblemTrait`](../api/optopus/trait_defs/trait.ProblemTrait.html)

Implement three traits and the local-search family plus every meta-heuristic
works on your problem. The remaining heuristics are unlocked one optional trait
at a time, see [which heuristic needs what](#which-heuristic-needs-what) below.

The full runnable example lives at
[`examples/custom_problem.rs`](https://github.com/trash-iine/optopus/blob/main/examples/custom_problem.rs)
(`cargo run --example custom_problem`).

## Required traits

| Trait | On | Required method(s) |
|---|---|---|
| [`Evaluate`](../traits.md#core-trait-reference) | `Solution` | `evaluate(&self) -> Evaluable<f64>` |
| [`ProblemTrait`](../traits.md#core-trait-reference) | the problem struct | `type Solution`, `new_solution(rng) -> Solution` |
| [`MoveToNeighbor<P>`](../traits.md#core-trait-reference) | the neighbor type | `iter`, `apply_to_solution`, `move_to_be_better_than` |
| [`Evaluate`](../traits.md#core-trait-reference) | the neighbor type | `evaluate(&self) -> Evaluable<f64>`, a second, separate impl |

`Evaluate` really is implemented twice. On the solution it reports the
objective, wrapped in `Evaluable::Maximize` or `Evaluable::Minimize` according
to which way the problem optimizes. On the move it reports the change applying
the move would make, wrapped the same way.

Both impls also give you [`Rankable`](../traits.md#core-trait-reference), which
is how `LocalSearch`, `RandomWalk`, `BeamSearch` and `TabuSearch` pick a move
and how every heuristic decides whether a solution is an improvement. It is
derived rather than written: `is_better_than` is a comparison of the two
`evaluate` values with the direction applied, so there is no second place for
the optimization direction to be stated and get out of step.

## Skeleton

```rust
use optopus::prelude::*;
use optopus::error::OptError;

struct MyProblem { /* ... */ }

#[derive(Clone)]
struct MySolution { /* ... */ }

impl Evaluate for MySolution {
    // Maximize or Minimize, whichever way the problem goes.
    fn evaluate(&self) -> Evaluable<f64> { todo!() }
}

impl ProblemTrait for MyProblem {
    type Solution = MySolution;
    fn new_solution(&self, rng: &mut impl rand::Rng) -> Self::Solution { todo!() }
}

struct MyMove { /* coordinates of the move */ }

impl MoveToNeighbor<MyProblem> for MyMove {
    fn iter(prob: &MyProblem, sol: &MySolution) -> impl Iterator<Item = Self> + Send {
        std::iter::empty() // enumerate moves lazily
    }
    fn apply_to_solution(&self, prob: &MyProblem, sol: &mut MySolution) -> Result<(), OptError> {
        todo!()
    }
    fn move_to_be_better_than(&self, prob: &MyProblem, src: &MySolution, other: &MySolution) -> bool {
        // default impl clones src and applies; override for an O(1) gain check
        let mut cloned = src.clone();
        self.apply_to_solution(prob, &mut cloned).expect("apply ok");
        cloned.is_better_than(other)
    }
}

impl Evaluate for MyMove {
    // The change applying this move would make. Report the cached gain here.
    // `LocalSearch` and `TabuSearch` select with `max_by(rank_cmp)`, which
    // reads this through the derived `Rankable`.
    fn evaluate(&self) -> Evaluable<f64> { todo!() }
}
```

`examples/custom_problem.rs` shows the cached-gain form.

## Which heuristic needs what

Everything below is optional, implement a row only when you want that
heuristic. Full signatures are in the
[core traits reference](../traits.md#core-trait-reference).

| Heuristic | Required traits |
|---|---|
| `LocalSearch`, `RandomWalk`, `BeamSearch` | nothing |
| `Sequential`, `Iterated`, `VariableNeighborhoodSearch`, `Restart` | nothing |
| `SimulatedAnnealing`, `BangBangSimulatedAnnealing`, `LateAcceptanceHillClimbing` | [`Evaluate<f64>`](../traits.md#core-trait-reference) on the move |
| `RlSearch` | [`Evaluate<f64>`](../traits.md#core-trait-reference) + `Clone` on the move |
| `TabuSearch` | [`EnabledTabu`](../traits.md#core-trait-reference) + `Clone` on the move, plus `fn tabu_policy(&self) -> Option<&dyn EnabledTabu> { Some(self) }` in its `MoveToNeighbor` impl, that one line is what hands the policy to the [`SearchState`](../search_state.md#remembering-tabu-moves), which owns the memory |
| `GeneticAlgorithm` | [`Distance`](../traits.md#core-trait-reference) on the solution (with any parent selection, not only `DistantTopK`) plus a [`Crossover<P>`](../traits.md#core-trait-reference) impl ([`SubProblemExtractable`](../traits.md#core-trait-reference) on the problem only if you use `SubProblemBasedCrossover`) |
| the CLI benchmark (TOML config) | all of the above |

The last row is not a shortcut for "everything is nicer that way": the benchmark
factory chooses the heuristic at runtime, so it bundles the bounds
(`ConfigNeighbor = MoveToNeighbor + Rankable + Evaluate + EnabledTabu + Clone`,
and `ConfigurableProblem::Solution: Distance + Evaluate`). A problem you only drive from
Rust can stop at whichever traits its heuristics need; one registered with the
benchmark cannot register partially.

## Performance note

The default `move_to_be_better_than` clones the solution and applies the move.
Override it with an O(1) check against cached per-variable gains.
`MaxCutFlipNeighbor` and `QuboFlipNeighbor` are the reference implementations.

## Next reading

- [Core traits reference](../traits.md#core-trait-reference)
- [Custom heuristic](custom_heuristic.md)
