# Integer variables

**API:** [`IntegerProblem`](../api/optopus/problem/integer/trait.IntegerProblem.html)

`IntegerProblem` is for a problem you would rather not write a move for. Its
variables are integers, each ranging over `lower..=upper`, and implementing the
trait means stating those ranges and the objective. Everything else follows.

- `ProblemTrait` is implemented for every `IntegerProblem`, with `IntSolution`
  as the solution. The initial solution draws each variable uniformly from its
  range, or is a uniformly random permutation when the variables are one.
- Three moves come with it, and each implements everything `LocalSearch`,
  `SimulatedAnnealing`, `LateAcceptanceHillClimbing`, `TabuSearch`,
  `RandomWalk`, `BeamSearch` and `ReinforcementLearningSearch` ask of a move.

| Move | What it does | For |
|---|---|---|
| `IntChangeNeighbor` | sets one variable to any other value in its range, a flip on `0..=1` | independent ranges |
| `IntSwapNeighbor` | exchanges the values of two variables | assignments, permutations |
| `IntReverseNeighbor` | reverses the values of variables `i..=j` | permutations read as a tour (2-opt) |

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

## Permutations

`IntVars::permutation(n)` declares `n` variables in `0..=n-1` that take
different values. Read variable `p` as the city visited `p`th and a solution is
a tour.

```rust
struct Tour { vars: IntVars, dist: Vec<Vec<f64>> }

impl IntegerProblem for Tour {
    fn variables(&self) -> &IntVars {
        &self.vars // IntVars::permutation(n)
    }
    fn objective(&self, v: &[i64]) -> Evaluable<f64> {
        let n = v.len();
        Evaluable::Minimize((0..n).map(|p| self.dist[v[p] as usize][v[(p + 1) % n] as usize]).sum())
    }
}

LocalSearch::<IntReverseNeighbor>::new(StopCondition::iterations(1_000))
    .run(&mut state)?;
```

Changing one value of a permutation always repeats another, so
`IntChangeNeighbor` has no moves on one, and a search that uses it reports an
empty neighborhood rather than leaving the permutation. `solution_from` refuses
values that repeat.

## Making it fast

Every candidate move asks the problem what it would change. The defaults copy
the values and evaluate the whole objective, which is correct for any objective
and warns once at runtime. Override the one your move uses with the change
computed from what it touches, as the difference of the raw objective, new
minus old.

| Move | Override |
|---|---|
| `IntChangeNeighbor` | `delta(sol, i, value)` |
| `IntSwapNeighbor` | `swap_delta(sol, i, j)` |
| `IntReverseNeighbor` | `reverse_delta(sol, i, j)` |

For the tour above, four distance lookups price a reversal.

```rust
fn reverse_delta(&self, sol: &IntSolution, i: usize, j: usize) -> f64 {
    let (v, n) = (sol.values(), sol.values().len());
    if j - i + 1 >= n - 1 {
        return 0.0; // reversing all but at most one city leaves the tour as it was
    }
    let d = |a: i64, b: i64| self.dist[a as usize][b as usize];
    let (prev, next) = (v[(i + n - 1) % n], v[(j + 1) % n]);
    d(prev, v[j]) + d(v[i], next) - d(prev, v[i]) - d(v[j], next)
}
```

Applying a move adds its change to the solution's objective rather than
evaluating it again.

## A solution of your own

Some problems price a move quickly only from something the solution keeps, such
as the gain of flipping each variable of a MaxCut. `IntSolution` holds values
and the objective and nothing more, so such a problem implements
`IntAssignment` instead, with a solution type of its own.

| Method | Required | What it does |
|---|---|---|
| `domains` | yes | the variables |
| `get(sol, i)` | yes | the value of variable `i` |
| `assign(sol, i, value)` | yes | sets it, keeping the objective and anything cached up to date |
| `assign_delta(sol, i, value)` | no | what `assign` would change, read from the cache |
| `assign_swap`, `assign_reverse` and their `_delta` | no | the same for the other two moves |

Its `ProblemTrait` and the solution's `Evaluate` are written as for any
[custom problem](../guide/custom_problem.md). The three moves then work as
they do on an `IntegerProblem`, and every `IntegerProblem` is an
`IntAssignment` already.

## Starting from a given assignment

`IntegerProblem::solution_from` builds a solution from values you choose, and
fails if one lies outside its range. Hand it to `SearchState::with_solution` to
start a search there.

## What it does not do

There is no crossover or `Distance`, so `GeneticAlgorithm` does not run on an
`IntegerProblem`, and it cannot be registered with the CLI benchmark, which
reads problems from instance files.
