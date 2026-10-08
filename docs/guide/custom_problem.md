# Defining a Custom Problem

**API:** [`ProblemTrait`](../api/optopus/trait_defs/trait.ProblemTrait.html)

Implement three traits and the local-search family plus every meta-heuristic
works on your problem. The remaining heuristics are unlocked one optional trait
at a time, see [which heuristic needs what](#which-heuristic-needs-what) below.

The full runnable example lives at
[`examples/custom_problem.rs`](https://github.com/trash-iine/optopus/blob/main/examples/custom_problem.rs)
(`cargo run --example custom_problem`).

If every variable of your problem is an integer in a fixed range, or the
solution is a permutation such as a tour, build an
[`IntegerProblem`](../problems/integer.md) instead. It takes the ranges and the
objective as a closure, with nothing to implement, and comes with ready made
moves.

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

The change is the objective after the move minus the objective before it,
never the objective after. A minimizing problem's move reports
`Evaluable::Minimize(after - before)`, so a negative value improves, and a
maximizing problem's move reports `Evaluable::Maximize(after - before)`, so a
positive value does. When the objective is not a sum of independent terms, such
as a maximum or an absolute value, compute the value the objective would take
after the move and subtract the current one. Neither `evaluate` can see the
problem, so what it needs is cached when the move is built, the way
`FlipMove::new` in `examples/custom_problem.rs` caches its gain.

```rust
struct MyMove { index: usize, delta: f64 }

impl MyMove {
    fn new(prob: &MyProblem, sol: &MySolution, index: usize) -> Self {
        let after = todo!(); // the objective `sol` would have after this move
        MyMove { index, delta: after - sol.objective }
    }
}

impl Evaluate for MyMove {
    fn evaluate(&self) -> Evaluable<f64> {
        Evaluable::Minimize(self.delta) // a minimizing problem, negative improves
    }
}
```

Both impls also give you [`Rankable`](../traits.md#core-trait-reference), which
is how `LocalSearch`, `RandomWalk`, `BeamSearch` and `TabuSearch` pick a move
and how every heuristic decides whether a solution is an improvement. It is
derived rather than written. `is_better_than` compares the two `evaluate`
values with the direction applied, so there is nothing to implement.

## Skeleton

```rust
use optopus::prelude::*;
use optopus::error::OptError;
use optopus::rand; // the rand optopus is built against

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

`new_solution` takes an `impl rand::Rng`, and optopus re-exports the `rand` it
is built against. `use optopus::rand;` brings that version into scope, so the
crate defining the problem needs no `rand` dependency of its own.

## Which heuristic needs what

Everything below is optional, implement a row only when you want that
heuristic. Full signatures are in the
[core traits reference](../traits.md#core-trait-reference).

| Heuristic | Required traits |
|---|---|
| `LocalSearch`, `RandomWalk`, `BeamSearch` | nothing |
| `Sequential`, `Iterated`, `VariableNeighborhoodSearch`, `Restart` | nothing |
| `SimulatedAnnealing`, `BangBangSimulatedAnnealing`, `LateAcceptanceHillClimbing` | [`Evaluate<f64>`](../traits.md#core-trait-reference) on the move |
| `ReinforcementLearningSearch` | [`Evaluate<f64>`](../traits.md#core-trait-reference) + `Clone` on the move |
| `TabuSearch` | [`EnabledTabu`](../traits.md#core-trait-reference) + `Clone` on the move, plus `fn tabu_policy(&self) -> Option<&dyn EnabledTabu> { Some(self) }` in its `MoveToNeighbor` impl, that one line is what hands the policy to the [`SearchState`](../search_state.md#remembering-tabu-moves), which owns the memory, see [Adding tabu](#adding-tabu) |
| `GeneticAlgorithm` | [`Distance`](../traits.md#core-trait-reference) on the solution (with any parent selection, not only `DistantTopK`) plus a [`Crossover<P>`](../traits.md#core-trait-reference) impl ([`SubProblemExtractable`](../traits.md#core-trait-reference) on the problem only if you use `SubProblemBasedCrossover`) |
| the CLI benchmark (TOML config) | all of the above |

The last row is not a shortcut for "everything is nicer that way": the benchmark
factory chooses the heuristic at runtime, so it bundles the bounds
(`ConfigNeighbor = MoveToNeighbor + Rankable + Evaluate + EnabledTabu + Clone`,
and `ConfigurableProblem::Solution: Distance + Evaluate`). A problem you only drive from
Rust can stop at whichever traits its heuristics need; one registered with the
benchmark cannot register partially.

## Adding tabu

`TabuSearch` keeps a tabu list for a move only when the move says what applying
it forbids. That takes three additions to the move of the skeleton, `Clone`, an
`EnabledTabu` impl, and one method in its `MoveToNeighbor` impl. `TabuKey` and
`TabuMemory` are search machinery that knows no problem, so they are imported
from `optopus::building_blocks::search` rather than the prelude.

```rust
use optopus::building_blocks::search::{TabuKey, TabuMemory};
use optopus::rand::rngs::SmallRng;

#[derive(Clone)]
struct MyMove { index: usize }

impl EnabledTabu for MyMove {
    // Allowed once the variable it touches is no longer forbidden.
    fn is_move_enabled(&self, tabu: &TabuMemory, iteration: u64) -> bool {
        tabu.is_enabled(TabuKey::DenseVar(self.index), iteration)
    }
    // Applying it forbids that variable for a tenure drawn from the range
    // given to `TabuSearch::new`.
    fn add_to_tabu_map(&self, tabu: &mut TabuMemory, iteration: u64, rng: &mut SmallRng) {
        tabu.forbid(TabuKey::DenseVar(self.index), iteration, rng);
    }
}

impl MoveToNeighbor<MyProblem> for MyMove {
    fn tabu_policy(&self) -> Option<&dyn EnabledTabu> {
        Some(self)
    }
    // iter, apply_to_solution and move_to_be_better_than as in the skeleton
}
```

**Without `tabu_policy` the `EnabledTabu` impl still compiles, and `TabuSearch`
runs with no tabu list at all.**

`TabuKey::DenseVar(i)` suits an index bounded by the instance, such as a
variable. A move over two variables forbids a `TabuKey::Pair(i, j)`, and an
index of any size is a `TabuKey::Var(i)`. The
[`TabuKey`](../api/optopus/building_blocks/search/enum.TabuKey.html) rustdoc
lists every shape. `examples/custom_problem.rs` runs `TabuSearch` with this
policy on its OneMax problem.

## Performance note

The default `move_to_be_better_than` clones the solution and applies the move.
Override it with an O(1) check against cached per-variable gains. The one-line
form is `self.evaluate().improves_over(src.evaluate(), other.evaluate())`, which
reads the optimization direction from the `Evaluate` impls instead of restating
it as a comparison operator. `MaxCutFlipNeighbor` and `QuboFlipNeighbor` are the
reference implementations.

## Next reading

- [Core traits reference](../traits.md#core-trait-reference)
- [Custom heuristic](custom_heuristic.md)
