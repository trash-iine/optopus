# Formula

`FormulaProblem` is a configurable binary optimization problem defined by an
arithmetic expression over `{0, 1}` variables, optionally constrained by
penalty terms. It is the most flexible problem type: declare an objective,
choose `Maximize` or `Minimize`, and add comparison or clamp constraints.

## Solution

```rust
pub struct FormulaSolution {
    pub x: Vec<bool>,                       // variable assignment (0 = false, 1 = true)
    pub gain: Vec<f64>,                     // change in score per flip (positive = improving)
    pub score: f64,                         // combined score (always higher is better)
    // pub(crate) constraint_vals: Vec<f64> — per-constraint cached expression values
}
```

The internal `score` always follows the higher-is-better convention,
regardless of optimization direction:

- `OptDirection::Maximize`: `score =  objective − penalty`
- `OptDirection::Minimize`: `score = −objective − penalty`

`Rankable::is_better_than` returns `self.score > other.score`.

## Expressions

The objective and constraint sides are built from the `Expr` AST:

```rust
pub enum Expr {
    Const(f64),
    Var(usize),         // 0-indexed
    Neg(Box<Expr>),
    Add(Vec<Expr>),
    Mul(Vec<Expr>),     // for binary vars, Mul ≡ AND
}
```

`Expr` overloads the standard arithmetic operators (`+ - * /`) for both
`Expr × Expr` and `Expr × f64`. `Add` and `Mul` are flattened automatically.
Division is supported only by a constant divisor.

## Constraints

```rust
pub enum Constraint {
    Comparison {
        lhs: Expr,
        rel: ConstraintRel,    // Lt | Gt | Le | Ge | Eq
        rhs: Expr,
        penalty_weight: f64,
    },
    Clamp { expr: Expr, lo: f64, hi: f64, penalty_weight: f64 },
}
```

The penalty contribution is `violation * penalty_weight`, where `violation` is
the amount by which the constraint is violated (`0` if satisfied). `Lt` and
`Gt` use a small `STRICT_EPSILON` so equality counts as a violation.

## Neighbors

| Type | Move | Iteration cost |
|---|---|---|
| `FormulaFlipNeighbor` | Flip one variable; gain refresh restricted to `interaction_neighbors`. | `iter + 1` |
| `FormulaSwapNeighbor` | Swap two variables. | `iter + 2` |

Both implement `Rankable`, `Evaluate<f64>` *and* `Evaluate<i32>` (the integer
form discretizes scores; suitable when all coefficients are integer-valued),
and `EnabledTabu`.

## Crossover

- `FormulaUniformCrossover` — per-variable random parent selection.

## Construction

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
```

There is no file loader: build the problem programmatically from the `Expr`
AST.

## Optional traits

- `Distance` — Hamming distance on `x`.
- `Evaluate<f64>` and `Evaluate<i32>` — both directions of `Evaluable`.

## Notes

- A pre-compiled polynomial form (`CompiledPoly`) gives O(d) gain deltas per
  flip, where d is the number of monomials touching the flipped variable.
- `interaction_neighbors[i]` lists the variables whose gain may change when
  `i` is flipped: variables that share a monomial in the objective, plus
  variables that co-appear in any constraint expression. Gain updates skip
  every other variable.
