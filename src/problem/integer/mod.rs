//! Problems over bounded integer variables, with the moves built in.
//!
//! A problem that implements [`IntegerProblem`] states its variables and its
//! objective and nothing else. [`ProblemTrait`](crate::search_state::ProblemTrait)
//! comes with it, and so do three moves, so
//! [`LocalSearch`](crate::heuristic::LocalSearch),
//! [`SimulatedAnnealing`](crate::heuristic::SimulatedAnnealing),
//! [`TabuSearch`](crate::heuristic::TabuSearch) and the other move based
//! heuristics run on it directly.
//!
//! - [`IntChangeNeighbor`] sets one variable to another value in its range,
//!   which on a binary variable is a flip.
//! - [`IntSwapNeighbor`] exchanges the values of two variables.
//! - [`IntReverseNeighbor`] reverses the values of a range of variables, which
//!   on a [permutation](IntVars::permutation) read as a tour is a 2-opt move.
//!
//! The moves are written against [`IntAssignment`], which asks only to read
//! and write one value of a solution. A problem that wants a solution of its
//! own, to keep whatever makes a move cheap to price, implements that instead.

mod assignment;
mod neighbor;
mod problem;

pub use assignment::IntAssignment;
pub use neighbor::{IntChangeNeighbor, IntReverseNeighbor, IntSwapNeighbor};
pub use problem::{IntSolution, IntVar, IntVars, IntegerProblem};
