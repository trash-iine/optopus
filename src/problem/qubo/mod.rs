//! Quadratic Unconstrained Binary Optimization (QUBO) problem definition and
//! neighborhood structures.
//!
//! QUBO minimizes the energy `E(x) = Σ Q[i][j] * x[i] * x[j]` over binary
//! variables `x ∈ {0,1}^n`.
//!
//! ```
//! use optopus::prelude::*;
//!
//! let qubo = Qubo::from_entries([(0, 1, 1), (0, 2, 1), (1, 2, 1)]);
//!
//! let mut state = SearchState::new(&qubo);
//! let mut ls = LocalSearch::<QuboFlipNeighbor>::new(
//!     StopCondition::iterations(10_000),
//! );
//! ls.run(&mut state).unwrap();
//! println!("best energy = {}", state.best_solution.objective);
//! ```
//!
//! [`Qubo`] is the instance, built from entries or set cell by cell, and
//! [`QuboSolution`] is the assignment, carrying `x`, `gain` and `objective`.
//! Gains are `i32`, so the relevant evaluators are `Evaluate<i32>`.
//!
//! | Move | Description | Iteration cost |
//! |---|---|---|
//! | [`QuboFlipNeighbor`] | flip one variable | `+1` |
//! | [`QuboSwapNeighbor`] | exchange two variables of opposite value | `+2` |
//!
//! [`QuboUniformCrossover`] recombines two assignments per variable, and `Qubo`
//! implements `SubProblemExtractable` so `SubProblemBasedCrossover` works. The
//! [QUBO page](https://trash-iine.github.io/optopus/problems/qubo/) covers the
//! file format and worked examples.

mod crossover;
mod neighbor;
mod problem;

pub use crossover::QuboUniformCrossover;
pub use neighbor::{QuboFlipNeighbor, QuboSwapNeighbor};
pub use problem::{Qubo, QuboSolution};
