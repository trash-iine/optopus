//! heuristic module provides various heuristic algorithms for combinatorial optimization problems.

mod beam_search;
mod crossover;
mod genetic_algorithm;
mod late_acceptance;
mod local_search;
mod random_walk;
pub mod reinforcement_learning;
mod restart;
mod sequential;
mod simulated_annealing;
mod specific;
mod tabu_search;

pub use beam_search::BeamSearch;
pub use crossover::SubProblemBasedCrossover;
pub use genetic_algorithm::{GeneticAlgorithm, ParentSelection};
pub use late_acceptance::LateAcceptanceHillClimbing;
pub use local_search::LocalSearch;
pub use random_walk::RandomWalk;
pub use reinforcement_learning::{RLSearch, RewardShaping};
pub use restart::Restart;
pub use sequential::{Iterated, Sequential};
pub use simulated_annealing::{BangBangSimulatedAnnealing, SimulatedAnnealing, boltzmann_accept};
pub use specific::BreakoutLocalSearchForMaxCut;
pub use specific::LinKernighanHelsgottForTsp;
pub use tabu_search::TabuSearch;

use crate::error::OptError;
use crate::search_state::{ProblemTrait, SearchState};
use serde::Serialize;

/// [`Heuristic`] trait is a common interface for heuristics.
///
/// To implement a heuristic, you need to implement [`Heuristic::is_done`] and [`Heuristic::run_once`] methods.
///
/// - [`Heuristic::is_done`] method checks if the heuristic should stop based on the current search state.
/// - [`Heuristic::run_once`] method performs one iteration of the heuristic and updates the search state accordingly
///
/// Please see [`local_search::LocalSearch`] as an example implementation of the `Heuristic` trait.
pub trait Heuristic<Problem: ProblemTrait> {
    /// Clear the internal state of the heuristic before running. This is called at the beginning of [`Heuristic::run`].
    fn clear(&mut self) {}

    /// Check if the heuristic should stop based on the current search state.
    fn is_done<'a>(&self, state: &SearchState<'a, Problem>) -> bool;

    /// Perform one iteration of the heuristic and update the search state accordingly.
    fn run_once<'a>(&mut self, state: &mut SearchState<'a, Problem>) -> Result<(), OptError>;

    /// Run the heuristic until the stopping condition is met.
    /// This method calls [`Heuristic::clear`] at the beginning and then repeatedly calls [`Heuristic::run_once`] until [`Heuristic::is_done`] returns true.
    fn run<'a>(&mut self, state: &mut SearchState<'a, Problem>) -> Result<(), OptError> {
        self.clear();
        tracing::debug!("Heuristic run started");

        while !self.is_done(state) {
            self.run_once(state)?;
        }

        tracing::debug!(
            iteration = state.iteration,
            best_iteration = state.best_iteration,
            elapsed_secs = state.duration().as_secs_f64(),
            "Heuristic run completed"
        );
        Ok(())
    }
}

/// This struct represents the stopping condition for heuristics.
///
/// It can be configured with
/// - a maximum number of iterations,
/// - a maximum duration,
/// - and a maximum number of iterations without improvement.
///
/// The [`StopCondition::is_done`] method checks if any of the conditions are met based on the current search state.
///
/// # Example
///
/// ```
/// use optopus::heuristic::StopCondition;
/// use std::time::Duration;
///
/// // only iterations
/// let sc = StopCondition::iterations(1_000_000);
///
/// // only execution time
/// let sc = StopCondition::duration(Duration::from_secs(30));
///
/// // combination of iterations and execution time
/// let sc = StopCondition::iterations(1_000_000)
///     .with_duration(Duration::from_secs(30));
/// ```
#[derive(Debug, Clone, Serialize)]
pub struct StopCondition {
    /// Maximum number of iterations to run the heuristic.
    pub max_iteration: Option<u64>,
    /// Maximum duration to run the heuristic.
    pub max_duration: Option<std::time::Duration>,
    /// Maximum number of iterations without improvement to run the heuristic.
    pub max_failed_update: Option<u64>,
}

impl StopCondition {
    /// Create a new `StopCondition` with the given parameters.
    pub fn new(
        max_iteration: Option<u64>,
        max_duration: Option<std::time::Duration>,
        max_failed_update: Option<u64>,
    ) -> Self {
        Self {
            max_iteration,
            max_duration,
            max_failed_update,
        }
    }

    /// Create a `StopCondition` with only the maximum number of iterations.
    pub fn iterations(n: u64) -> Self {
        Self {
            max_iteration: Some(n),
            max_duration: None,
            max_failed_update: None,
        }
    }

    /// Create a `StopCondition` with only the maximum duration.
    pub fn duration(d: std::time::Duration) -> Self {
        Self {
            max_iteration: None,
            max_duration: Some(d),
            max_failed_update: None,
        }
    }

    /// Create a `StopCondition` with only the maximum number of iterations without improvement.
    pub fn failed_updates(n: u64) -> Self {
        Self {
            max_iteration: None,
            max_duration: None,
            max_failed_update: Some(n),
        }
    }

    /// Add the maximum number of iterations (for chaining).
    pub fn with_iterations(mut self, n: u64) -> Self {
        self.max_iteration = Some(n);
        self
    }

    /// Add the maximum duration (for chaining).
    pub fn with_duration(mut self, d: std::time::Duration) -> Self {
        self.max_duration = Some(d);
        self
    }

    /// Add the maximum number of iterations without improvement (for chaining).
    pub fn with_failed_updates(mut self, n: u64) -> Self {
        self.max_failed_update = Some(n);
        self
    }

    /// Check if any of the stopping conditions are met based on the current search state.
    pub fn is_done<'a, Problem: ProblemTrait>(&self, state: &SearchState<'a, Problem>) -> bool {
        if let Some(max_iter) = self.max_iteration
            && state.iteration - state.start_iteration >= max_iter
        {
            return true;
        }
        if let Some(max_duration) = self.max_duration
            && state.duration() >= max_duration
        {
            return true;
        }
        if let Some(max_failed_update) = self.max_failed_update
            && state.iteration - state.best_iteration >= max_failed_update
        {
            return true;
        }
        false
    }
}
