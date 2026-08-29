# Formula

**API:** [`FormulaProblem`](../api/optopus/problem/binary_optimization/struct.FormulaProblem.html)

`FormulaProblem` is a configurable binary optimization problem for objectives
that don't match one of the named problems above: declare an arbitrary
arithmetic expression over `n` binary variables `x ∈ {0,1}^n` as the
objective, choose whether to `Maximize` or `Minimize` it, and add any number
of penalty-weighted constraints. Internally, every direction is optimized as
a maximization of a single higher-is-better `score`, so heuristics never need
to know which direction the user chose:

```text
Maximize:  score(x) = objective(x) − Σ_c penalty_weight_c · violation_c(x)
Minimize:  score(x) = −objective(x) − Σ_c penalty_weight_c · violation_c(x)
```

## Example

Running a search and reading back the assignment, its raw objective value,
and the internal `score` used for ranking:

```rust
use optopus::prelude::*;

// maximize x[0] + 2*x[1] + 3*x[2]  s.t.  x[0] + x[1] + x[2] <= 2
let objective = Expr::Var(0) + 2.0 * Expr::Var(1) + 3.0 * Expr::Var(2);
let constraint = Constraint::Comparison {
    lhs: Expr::Var(0) + Expr::Var(1) + Expr::Var(2),
    rel: ConstraintRel::Le,
    rhs: Expr::Const(2.0),
    penalty_weight: 10.0,
};
let prob = FormulaProblem::new(3, objective, OptDirection::Maximize, vec![constraint]);

let mut state = SearchState::new(&prob);
LocalSearch::<FormulaFlipNeighbor>::new(StopCondition::iterations(10_000))
    .run(&mut state)
    .unwrap();

let sol = &state.best_solution;
println!("assignment = {:?}", sol.x);
println!("objective value = {}", prob.eval_objective(&sol.x)); // the user-declared expression, before penalties
println!("score = {}", sol.score); // higher-is-better internal ranking value used by is_better_than
```

There is no file loader: build the problem programmatically from the `Expr`
AST, as above.

## Solution

[`FormulaSolution`](../api/optopus/problem/binary_optimization/struct.FormulaSolution.html)
carries the assignment `x` from the definition above (`x ∈ {0,1}^n`), the
per-variable `gain` (change in `score` if that variable were flipped), and
`score`, which is `score(x)` as defined above — **not** `objective(x)`
itself; see its rustdoc for the full field list.
`Rankable::is_better_than` returns `self.score > other.score`.

## Expressions

The objective and constraint sides are built from the [`Expr`](../api/optopus/problem/binary_optimization/enum.Expr.html)
AST; see its rustdoc for the exact variant list. `Expr` overloads the
standard arithmetic operators (`+ - * /`) for both `Expr × Expr` and
`Expr × f64`, which is the normal way to build one:

```rust
use optopus::problem::Expr;

// linear combination: 2*x[0] + x[1] - 3
let linear = 2.0 * Expr::Var(0) + Expr::Var(1) - 3.0;

// AND of two binary variables (Mul ≡ AND for {0,1} values)
let and_of_two = Expr::Var(0) * Expr::Var(1);
```

`Add` and `Mul` are flattened automatically. Division is supported only by a
constant divisor.

## Constraints

Constraints are built from [`Constraint`](../api/optopus/problem/binary_optimization/enum.Constraint.html)
(see its rustdoc for the exact variant list) and penalize violations at
`violation * penalty_weight`, where `violation` is the amount by which the
constraint is violated (`0` if satisfied):

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

`Lt` and `Gt` use a small `STRICT_EPSILON` so equality counts as a violation.

## Neighbors

| Type | Move |
|---|---|
| `FormulaFlipNeighbor` | Flip one variable. |
| `FormulaSwapNeighbor` | Swap two variables. |

Both implement `Rankable`, `Evaluate<f64>` *and* `Evaluate<i32>` (the integer
form discretizes scores; suitable when all coefficients are integer-valued),
and `EnabledTabu`; see
[`FormulaFlipNeighbor`](../api/optopus/problem/binary_optimization/struct.FormulaFlipNeighbor.html) /
[`FormulaSwapNeighbor`](../api/optopus/problem/binary_optimization/struct.FormulaSwapNeighbor.html)
rustdoc for the incremental gain-update mechanics and iteration cost.

## Crossover

- `FormulaUniformCrossover` — per-variable random parent selection.

## Optional traits

- `Distance` — Hamming distance on `x`.
- `Evaluate<f64>` and `Evaluate<i32>` — both directions of `Evaluable`.

## Notes

`CompiledPoly` and `interaction_neighbors` are private (`pub(super)`)
implementation details, so they do not appear in the published rustdoc — this
is the only public-facing place they're explained:

- A pre-compiled polynomial form (`CompiledPoly`) gives O(d) gain deltas per
  flip, where d is the number of monomials touching the flipped variable.
- `interaction_neighbors[i]` lists the variables whose gain may change when
  `i` is flipped: variables that share a monomial in the objective, plus
  variables that co-appear in any constraint expression. Gain updates skip
  every other variable.

(Contributors reading the source: both are documented in
`src/problem/binary_optimization/problem.rs`.)
