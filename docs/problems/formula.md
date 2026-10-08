# Formula

**API:** [`FormulaProblem`](../api/optopus/problem/integer/struct.FormulaProblem.html)

`FormulaProblem` is for an objective that is arithmetic over integer
variables. Declare the objective as an expression, choose whether to maximize
or minimize it, and add any number of penalty-weighted constraints. The
variables are the same `IntVars` an [`IntegerProblem`](integer.md) takes, so a
variable may be binary, range over any integers, or the variables may be a
permutation.

Nothing needs a delta. The expressions are compiled into monomials when the
problem is built, a change to one variable is priced from the monomials and
constraints it appears in, and every solution keeps the change of each value
each variable could take. Picking a move reads that table, and applying one
refreshes only the variables it can have affected.

The table holds one number for every value each variable could change to, so
its size is the sum of `upper - lower` over the variables. A variable ranging
over millions of values makes every solution that large, and such a variable
is better served by an [`IntegerProblem`](integer.md) with a delta written
for it. On a permutation there is no single change, so no table is kept and a
swap or a reversal is priced from the monomials directly.

```text
maximize:  objective(x) − Σ_c penalty_weight_c · violation_c(x)
minimize:  objective(x) + Σ_c penalty_weight_c · violation_c(x)
```

## Example

```rust
use optopus::prelude::*;

// maximize x[0] + 2*x[1] + 3*x[2] over binary x, with x[0] + x[1] + x[2] <= 2
let vars: IntVars = (0..3).map(|_| IntVar::binary()).collect();
let objective = Expr::Var(0) + 2.0 * Expr::Var(1) + 3.0 * Expr::Var(2);
let prob = FormulaProblem::maximize(vars, objective).with_constraint(Constraint::Comparison {
    lhs: Expr::Var(0) + Expr::Var(1) + Expr::Var(2),
    rel: ConstraintRel::Le,
    rhs: Expr::Const(2.0),
    penalty_weight: 10.0,
});

let mut state = SearchState::new(&prob);
TabuSearch::<IntChangeNeighbor>::new(StopCondition::iterations(1_000), (1, 2))
    .run(&mut state)
    .unwrap();

let values = state.best_solution.values();
println!("assignment = {values:?}");
println!("objective = {}", prob.eval_objective(values)); // the expression, before penalties
println!("penalty = {}", prob.eval_penalty(values));
```

There is no file loader. The problem is built in code, as above.

## Solution

`FormulaSolution` holds the value of every variable, read with `values()`. Its
`Evaluate` reports the penalized objective with the problem's direction,
`Evaluable::Maximize(objective − penalty)` or
`Evaluable::Minimize(objective + penalty)`, which is what the search ranks by.
`eval_objective` and `eval_penalty` give the two parts on their own.

## Expressions

The objective and both sides of a constraint are an `Expr`. It overloads the
arithmetic operators for `Expr × Expr` and `Expr × f64`, which is the normal
way to build one.

```rust
use optopus::problem::Expr;

// linear combination 2*x[0] + x[1] - 3
let linear = 2.0 * Expr::Var(0) + Expr::Var(1) - 3.0;

// a product, which on binary variables is an AND
let and_of_two = Expr::Var(0) * Expr::Var(1);

// powers are kept on integer variables
let square = Expr::Var(2) * Expr::Var(2);
```

`Expr::Var(i)` evaluates to the value of variable `i`. `Add` and `Mul` are
flattened as they are built. Division is supported only by a constant divisor.
Building a problem whose expressions read a variable outside its `IntVars`
panics.

## Constraints

A `Constraint` penalizes a violation at `violation * penalty_weight`, where
`violation` is the amount by which it is violated, `0` when it holds.

```rust
use optopus::problem::{Constraint, ConstraintRel, Expr};

// x[0] + x[1] + x[2] <= 2, penalized at weight 10.0 per unit of violation
let constraint = Constraint::Comparison {
    lhs: Expr::Var(0) + Expr::Var(1) + Expr::Var(2),
    rel: ConstraintRel::Le,
    rhs: Expr::Const(2.0),
    penalty_weight: 10.0,
};
```

`Constraint::Clamp` keeps an expression within `lo..=hi`. `Lt` and `Gt` charge
a small margin at equality, so a tie is never free under a strict relation.

## Moves

The moves are the ones every integer problem gets.

| Move | What it does |
|---|---|
| `IntChangeNeighbor` | sets one variable to another value in its range, a flip on a binary variable |
| `IntSwapNeighbor` | exchanges the values of two variables |
| `IntReverseNeighbor` | reverses the values of a range of variables |

`IntChangeNeighbor` reads its price from the solution's table. A swap is priced
from the monomials and constraints that read either variable. A reversal
evaluates a copy of the solution, since formulas rarely want one.

## Crossover and sub-problems

`IntCrossover` takes each variable from either parent, and `FormulaSolution`
implements `Distance`, so `GeneticAlgorithm` runs on a `FormulaProblem`.

`FormulaProblem` also implements `SubProblemExtractable`. The variables the
parents disagree on become a smaller `FormulaProblem`, with every other
variable replaced by its value in the expressions, so
`SubProblemBasedCrossover` runs on it too. The variables must not be a
permutation, since fixing some positions of a permutation does not leave one.
