//! Problem definitions and neighborhood structures for combinatorial optimization.
//!
//! Each sub-module provides:
//! - A problem struct implementing [`crate::search_state::ProblemTrait`]
//! - A solution struct implementing [`crate::search_state::Rankable`]
//! - One or more neighborhood move types implementing [`crate::search_state::MoveToNeighbor`]
//!
//! # Available Problems
//!
//! | Module | Problem | Objective |
//! |--------|---------|-----------|
//! | [`max_cut`] | Maximum Cut | Maximize cut weight |
//! | [`qubo`] | Quadratic Unconstrained Binary Optimization | Minimize energy |
//! | [`sat`] | Maximum Satisfiability (MaxSAT) | Maximize satisfied clauses |
//! | [`tsp_2d`] | Travelling Salesman Problem | Minimize tour length |
//! | [`binary_optimization`] | Formula-based binary optimization | Configurable |

pub mod binary_optimization;
pub mod max_cut;
pub mod qubo;
pub mod sat;
pub mod tsp_2d;

pub use binary_optimization::{
    Constraint, ConstraintRel, Expr, FormulaFlipNeighbor, FormulaProblem, FormulaSolution,
    FormulaSwapNeighbor, OptDirection,
};
pub use max_cut::{MaxCut, MaxCutFlipNeighbor, MaxCutSwapNeighbor};
pub use qubo::{Qubo, QuboFlipNeighbour, QuboSolution, QuboSwapNeighbour};
pub use sat::{Sat, SatFlipNeighbor, SatSolution, SatSwapNeighbor};
pub use tsp_2d::{
    TspRelocateNeighbor, TspSolution, TspTour, TspTwoOptNeighbor, TspWithCoordinates,
};
