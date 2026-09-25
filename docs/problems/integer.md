# Integer variables

**API:** [`IntegerProblem`](../api/optopus/problem/integer/trait.IntegerProblem.html)

`IntegerProblem` is for a problem you would rather not write a move for. Its
variables are integers, each ranging over `lower..=upper`, and implementing the
trait means stating those ranges and the objective. Everything else follows.

- `ProblemTrait` is implemented for every `IntegerProblem`, with `IntSolution`
  as the solution and each variable drawn uniformly from its range as the
  initial one.
- `IntChangeNeighbor` sets one variable to any other value in its range. On a
  variable ranging over `0..=1` that is a flip. It implements everything
  `LocalSearch`, `SimulatedAnnealing`, `LateAcceptanceHillClimbing`,
  `TabuSearch`, `RandomWalk`, `BeamSearch` and `ReinforcementLearningSearch`
  ask of a move, and every heuristic that composes them.

The neighborhood of a solution holds every value of every variable, so a
variable ranging over a million values adds a million candidate moves to each
scan of `LocalSearch` or `TabuSearch`. `SimulatedAnnealing` and the other
searches that draw one random move a step are not affected.

## Example

```rust
use optopus::prelude::*;

/// Minimize the sum of (x_i - 3)^2 over five variables in 0..=10.
struct Target { vars: IntVars }

impl IntegerProblem for Target {
    fn variables(&self) -> &IntVars {
        &self.vars
    }
    fn objective(&self, values: &[i64]) -> Evaluable<f64> {
        Evaluable::Minimize(values.iter().map(|&x| ((x - 3) * (x - 3)) as f64).sum())
    }
}

let prob = Target { vars: (0..5).map(|_| IntVar::new(0, 10)).collect() };
let mut state = SearchState::new(&prob);
LocalSearch::<IntChangeNeighbor>::new(StopCondition::iterations(100))
    .run(&mut state)
    .unwrap();
println!("{:?}", state.best_solution.values());
```

The objective states the direction, by returning `Evaluable::Maximize` or
`Evaluable::Minimize`, and must return the same variant for every assignment.

[`examples/integer_problem.rs`](https://github.com/trash-iine/optopus/blob/main/examples/integer_problem.rs)
solves a bounded knapsack this way (`cargo run --example integer_problem`).

## Making it fast

Every candidate move asks `IntegerProblem::delta` how much the objective would
change. The default copies the values and evaluates the whole objective, and
warns once at runtime that it does. Override `delta` with the change computed
from the terms variable `i` appears in, as the difference of the raw objective,
new minus old.

```rust
fn delta(&self, sol: &IntSolution, i: usize, value: i64) -> f64 {
    let (old, new) = (sol.value(i), value);
    ((new - 3) * (new - 3) - (old - 3) * (old - 3)) as f64
}
```

Applying a move adds its cached change to the solution's objective rather than
evaluating it again.

## Starting from a given assignment

`IntegerProblem::solution_from` builds a solution from values you choose, and
fails if one lies outside its range. Hand it to
`SearchState::with_solution` to start a search there.

## What it does not do

There is no crossover or `Distance`, so `GeneticAlgorithm` does not run on an
`IntegerProblem`, and it cannot be registered with the CLI benchmark, which
reads problems from instance files.
