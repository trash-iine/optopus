//! Core library traits shared across the Problem, Heuristic, and SearchState layers.
//!
//! These traits form the common vocabulary that lets any heuristic work with any
//! problem. A custom problem implements [`ProblemTrait`] (plus [`Evaluate`] on
//! its solution and on each of its moves, and [`MoveToNeighbor`] for the moves);
//! optional capabilities such as [`EnabledTabu`] (TabuSearch), [`Crossover`] /
//! [`SubProblemExtractable`] / [`Distance`] (GeneticAlgorithm) unlock additional
//! heuristics. [`Rankable`] is the order every heuristic compares by, and
//! `#[derive(Rankable)]` writes it, from [`Evaluate`] where the order is the
//! objective and from named keys where it is not.
//!
//! Every trait the problem side implements and a heuristic calls is defined
//! here, including the ones of one family of problems, such as
//! [`BinaryProblem`] and [`IntAssignment`], and the operators a problem
//! supplies to one heuristic, such as [`Relaxation`] and [`LocalRepair`]. A
//! trait implemented and called within one layer stays beside its type, as
//! [`Heuristic`](crate::heuristic::Heuristic) and
//! [`PerturbationSchedule`](crate::heuristic::PerturbationSchedule) do.

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

pub(crate) use binary::variable_count;
pub use binary::{BinaryProblem, FixVariables, FixedVariables, Placed, check_assignment};
pub use branch::{BranchSpace, Branchable, Relaxation};
pub use crossover::{Crossover, SubProblemExtractable};
pub use evaluate::{Evaluable, Evaluate};
pub use int_assignment::IntAssignment;
pub use neighbor::MoveToNeighbor;
pub use optopus_derive::{Distance, Evaluate, Rankable};
pub use problem::ProblemTrait;
pub use rankable::{Distance, Rankable, better_by_objective, filter_best, rank_cmp};
pub use reduction::ProblemReduction;
pub use ruinable::{LocalRepair, Ruinable};
pub use tabu::EnabledTabu;
