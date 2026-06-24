//! Core library traits shared across the Problem, Heuristic, and SearchState layers.
//!
//! These traits form the common vocabulary that lets any heuristic work with any
//! problem. A custom problem implements [`ProblemTrait`] (plus [`Rankable`] on its
//! solution and [`MoveToNeighbor`] for its moves); optional capabilities such as
//! [`Evaluate`] (SA/LAHC), [`EnabledTabu`] (TabuSearch), [`Crossover`] /
//! [`SubProblemExtractable`] / [`Distance`] (GeneticAlgorithm) unlock additional
//! heuristics.

mod crossover;
mod evaluate;
mod neighbor;
mod problem;
mod rankable;
mod tabu;

pub use crossover::{Crossover, SubProblemExtractable};
pub use evaluate::{Evaluable, Evaluate};
pub use neighbor::MoveToNeighbor;
pub use problem::ProblemTrait;
pub use rankable::{Distance, Rankable, filter_best};
pub use tabu::EnabledTabu;
