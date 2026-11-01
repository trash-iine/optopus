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
| [`Evaluate`](../traits.md#core-trait-reference) | `Solution` | `evaluate(&self) -> Evaluable<f64>`, or `#[derive(Evaluate)]` with `#[objective(minimize)]` on the field that holds it |
| [`ProblemTrait`](../traits.md#core-trait-reference) | the problem struct | `type Solution`, `new_solution(rng) -> Solution` |
| [`MoveToNeighbor<P>`](../traits.md#core-trait-reference) | the neighbor type | `iter`, `apply_to_solution`, `move_to_be_better_than` |
| [`Evaluate`](../traits.md#core-trait-reference) | the neighbor type | `evaluate(&self) -> Evaluable<f64>`, a second, separate impl, derived the same way |
| [`Rankable`](../traits.md#core-trait-reference) | `Solution` and the neighbor type | `is_better_than`, written by `#[derive(Rankable)]` |

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
#[derive(Evaluate, Rankable)]
struct MyMove {
    index: usize,
    #[objective(minimize)] // a minimizing problem, negative improves
    delta: f64,
}

impl MyMove {
    fn new(prob: &MyProblem, sol: &MySolution, index: usize) -> Self {
        let after = todo!(); // the objective `sol` would have after this move
        MyMove { index, delta: after - sol.objective }
    }
}
```

`#[derive(Evaluate)]` reports the marked field, wrapped in the direction the
attribute names. A number that is not one field, a sum of two of them say, is
an `impl Evaluate` written by hand instead.

`#[derive(Rankable)]` gives the move [`Rankable`](../traits.md#core-trait-reference),
and the solution derives it the same way. `Rankable` is how `LocalSearch`,
`RandomWalk`, `BeamSearch` and `TabuSearch` pick a move and how every
heuristic decides whether a solution is an improvement. With no `rank`
attribute it compares the two `evaluate` values with the direction applied.
A problem whose order is not its objective, one that compares violations
before cost for example, names its keys instead, as in
`#[rank(min = violation(), min = cost)]`, and keeps `Evaluate` as the number
the annealing heuristics read. The [`Rankable` rustdoc](../api/optopus/trait_defs/trait.Rankable.html)
has the forms.

## Skeleton

```rust
use optopus::prelude::*;
use optopus::error::OptError;
use optopus::rand; // the rand optopus is built against

struct MyProblem { /* ... */ }

#[derive(Clone, Rankable)]
struct MySolution { /* ... */ }

impl Evaluate for MySolution {
    // Maximize or Minimize, whichever way the problem goes.
    fn evaluate(&self) -> Evaluable<f64> { todo!() }
}

impl ProblemTrait for MyProblem {
    type Solution = MySolution;
    fn new_solution(&self, rng: &mut impl rand::Rng) -> Self::Solution { todo!() }
}

#[derive(Rankable)]
struct MyMove { /* coordinates of the move */ }

impl MoveToNeighbor<MyProblem> for MyMove {
    fn iter(prob: &MyProblem, sol: &MySolution) -> impl Iterator<Item = Self> + Send {
        std::iter::empty() // enumerate moves lazily
    }
    fn tabu_policy(&self) -> Option<&dyn EnabledTabu> {
        None // no tabu policy yet, see Adding tabu
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
    // reads this through `Rankable`.
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
| `TabuSearch` | [`EnabledTabu`](../traits.md#core-trait-reference) + `Clone` on the move, and `tabu_policy` in its `MoveToNeighbor` impl returning `Some(self)` rather than `None`, which is what hands the policy to the [`SearchState`](../search_state.md#remembering-tabu-moves) that owns the memory, see [Adding tabu](#adding-tabu) |
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
it forbids. That takes `Clone`, the keys the move touches in `#[tabu(..)]`, and
`Some(self)` in place of the skeleton's `None`.

```rust
#[derive(Clone, Evaluate, Rankable, EnabledTabu)]
#[tabu(dense(index))] // allowed once `index` is free, and forbids it when applied
struct MyMove {
    index: usize,
    #[objective(minimize)]
    delta: f64,
}

impl MoveToNeighbor<MyProblem> for MyMove {
    fn tabu_policy(&self) -> Option<&dyn EnabledTabu> {
        Some(self)
    }
    // iter, apply_to_solution and move_to_be_better_than as in the skeleton
}
```

The move is allowed while every key it names is free, and applying it forbids
every one of them for a tenure drawn from the range given to `TabuSearch::new`.
`dense(i)` suits an index bounded by the instance, such as a variable, and
`var(i)` is an index of any size. A move over two indices names both,
`dense(i), dense(j)`, or forbids the pair with `pair(i, j)`, or with
`unordered(i, j)` when `(i, j)` and `(j, i)` are the same move. `triple(a, b, c)`
keys three. The [`EnabledTabu`](../api/optopus/trait_defs/trait.EnabledTabu.html)
rustdoc lists the forms. A move that checks other keys than it forbids, as the
VRP relocate does, implements `EnabledTabu` by hand with
[`TabuKey`](../api/optopus/building_blocks/search/enum.TabuKey.html) and
`TabuMemory` from `optopus::building_blocks::search`.

`tabu_policy` has no default, so a move always says whether it has a policy.
`examples/custom_problem.rs` runs `TabuSearch` with this policy on its OneMax
problem.

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
