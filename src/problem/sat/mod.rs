//! Maximum Satisfiability (MaxSAT) problem definition and neighborhood structures.
//!
//! Given a CNF formula in DIMACS format, MaxSAT seeks an assignment of boolean variables
//! that maximizes the number of satisfied clauses.

mod neighbor;
mod problem;

pub use neighbor::{SatFlipNeighbor, SatSwapNeighbor};
pub use problem::{Sat, SatSolution};
