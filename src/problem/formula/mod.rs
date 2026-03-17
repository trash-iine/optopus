pub mod definition;
pub mod neighbor;

pub use definition::{
    Constraint, ConstraintRel, Expr, FormulaProblem, FormulaSolution, OptDirection, Value,
};
pub use neighbor::{FormulaFlipNeighbor, FormulaSwapNeighbor};