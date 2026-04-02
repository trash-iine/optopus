//! Formula-based binary optimization problem definition and neighborhood structures.
//!
//! [`FormulaProblem`] lets you express an optimization problem using an arithmetic AST ([`Expr`])
//! for the objective and a list of penalty-weighted constraints ([`Constraint`]).
//! Both maximization and minimization directions are supported via [`OptDirection`].

pub mod crossover;
pub mod neighbor;
pub mod problem;

pub use crossover::FormulaUniformCrossover;
pub use neighbor::{FormulaFlipNeighbor, FormulaSwapNeighbor};
pub use problem::{
    Constraint, ConstraintRel, Expr, FormulaProblem, FormulaSolution, OptDirection, Value,
};
