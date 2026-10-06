//! Core library traits shared across the Problem, Heuristic, and SearchState layers.
//!
//! These traits form the common vocabulary that lets any heuristic work with any
//! problem. A custom problem implements [`ProblemTrait`] (plus [`Evaluate`] on
//! its solution and on each of its moves, and [`MoveToNeighbor`] for the moves);
//! optional capabilities such as [`EnabledTabu`] (TabuSearch), [`Crossover`] /
//! [`SubProblemExtractable`] / [`Distance`] (GeneticAlgorithm) unlock additional
//! heuristics. [`Rankable`] is derived from [`Evaluate`] rather than
//! implemented.

mod binary;
mod branch;
mod crossover;
mod evaluate;
mod int_assignment;
mod neighbor;
mod problem;
mod rankable;
mod reduction;
mod ruinable;
mod tabu;

pub use binary::{BinaryProblem, FixVariables, FixedVariables, Placed};
pub use branch::{BranchSpace, Branchable, Relaxation};
pub use crossover::{Crossover, SubProblemExtractable};
pub use evaluate::{Evaluable, Evaluate};
pub(crate) use evaluate::{raw, with_value};
pub use int_assignment::IntAssignment;
pub use neighbor::MoveToNeighbor;
pub use problem::ProblemTrait;
pub use rankable::{Distance, Rankable, filter_best, rank_cmp};
pub use reduction::ProblemReduction;
pub use ruinable::{LocalRepair, Ruinable};
pub use tabu::EnabledTabu;
