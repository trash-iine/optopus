//! Quadratic Unconstrained Binary Optimization (QUBO) problem definition and neighborhood structures.
//!
//! QUBO minimizes the energy `E(x) = Σ Q[i][j] * x[i] * x[j]` over binary variables `x ∈ {0,1}^n`.

mod crossover;
mod neighbor;
mod problem;

pub use crossover::QuboUniformCrossover;
pub use neighbor::{QuboFlipNeighbor, QuboSwapNeighbor};
pub use problem::{Qubo, QuboSolution};
