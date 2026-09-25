//! Problems over bounded integer variables, with the move built in.
//!
//! A problem that implements [`IntegerProblem`] states its variables and its
//! objective and nothing else. [`ProblemTrait`](crate::search_state::ProblemTrait)
//! and the single variable change move [`IntChangeNeighbor`] come with it, so
//! [`LocalSearch`](crate::heuristic::LocalSearch),
//! [`SimulatedAnnealing`](crate::heuristic::SimulatedAnnealing),
//! [`TabuSearch`](crate::heuristic::TabuSearch) and the other move based
//! heuristics run on it directly.

mod neighbor;
mod problem;

pub use neighbor::IntChangeNeighbor;
pub use problem::{IntSolution, IntVar, IntVars, IntegerProblem};
