# Defining a Custom Problem

Implement three traits and every built-in heuristic works on your problem.

The full runnable example lives at
[`examples/custom_problem.rs`](../../examples/custom_problem.rs)
(`cargo run --example custom_problem`).

## Required: three traits

| Trait | On | Required method(s) |
|---|---|---|
| [`Rankable`] | `Solution` | `is_better_than(&self, other) -> bool` |
| [`ProblemTrait`] | the problem struct | `type Solution`, `new_solution(rng) -> Solution` |
| [`MoveToNeighbor<P>`] | one or more neighbor types | `iter`, `apply_to_solution`, `move_to_be_better_than` |

The optimization direction is encoded in `Rankable::is_better_than`: a
maximization problem returns `self.score > other.score`; a minimization problem
returns `<`.

## Skeleton

```rust
use optopus::prelude::*;
use optopus::error::OptError;

struct MyProblem { /* ... */ }

#[derive(Clone)]
struct MySolution { /* ... */ }

impl Rankable for MySolution {
    fn is_better_than(&self, other: &Self) -> bool { /* > or < */ todo!() }
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
        // default impl clones src + applies — override for an O(1) gain check
        let mut cloned = src.clone();
        self.apply_to_solution(prob, &mut cloned).expect("apply ok");
        cloned.is_better_than(other)
    }
}

impl Rankable for MyMove {
    fn is_better_than(&self, _other: &Self) -> bool { false }
}
```

`MyMove: Rankable` is required by the trait bound but only matters if you sort
moves directly; returning `false` is fine when heuristics decide via solution
comparison.

## Optional traits — what each heuristic needs

| Trait on solution / move | Required by | Without it |
|---|---|---|
| `Evaluate<f64>` (move) | `SimulatedAnnealing`, `BangBangSimulatedAnnealing`, `LateAcceptanceHillClimbing`, `RLSearch` | Those heuristics won't compile for your move type. |
| `Evaluate<i32>` (move) | Optional integer-valued objective deltas (used by QUBO). | Just don't impl. |
| `EnabledTabu` (move) | `TabuSearch` | Same. |
| `Distance` (solution) | `GeneticAlgorithm` (any selection), `ParentSelection::HammingTopK` | GA won't compile. |
| `SubProblemExtractable` (problem) | `SubProblemBasedCrossover` | Use the problem's uniform crossover instead. |
| `CdclEncodable` (problem) | `CdclSolver` | Skip CDCL. |

`LocalSearch`, `RandomWalk`, `BeamSearch`, `Sequential`, `Iterated`, `Restart`,
and `GeneticAlgorithm` (with a problem-specific crossover that doesn't need
`SubProblemExtractable`) only need the three required traits + `Distance` for GA.

## Performance note

The default `move_to_be_better_than` clones the solution and applies the move.
For non-trivial problems, override it with an O(1) **gain-based** check that
inspects cached per-variable gains — see `MaxCutFlipNeighbor` or
`QuboFlipNeighbor` in `src/problem/` for reference implementations.

## Next reading

- [Concepts → Core traits table](../concepts.md#core-trait-reference)
- [Custom heuristic](custom_heuristic.md)
