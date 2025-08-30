mod local_search;
mod random_walk;
mod sequential;
mod simulated_annealing;
mod specific;
mod tabu_search;

pub use local_search::LocalSearch;
pub use random_walk::RandomWalk;
pub use sequential::Sequential;
pub use simulated_annealing::{BangBangSimulatedAnnealing, SimulatedAnnealing};
pub use specific::bls_for_max_cut::BreakoutLocalSearch as BreakoutLocalSearchForMaxCut;
pub use tabu_search::TabuSearch;

use crate::search_state::{ProblemTrait, SearchState};

pub trait Heuristic<Problem: ProblemTrait> {
    fn clear(&mut self) {}
    fn is_done<'a>(&self, state: &SearchState<'a, Problem>) -> bool;
    fn run_once<'a>(
        &self,
        state: &mut SearchState<'a, Problem>,
    ) -> Result<(), Box<dyn std::error::Error>>;
    fn run<'a>(
        &self,
        state: &mut SearchState<'a, Problem>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        while !self.is_done(&state) {
            self.run_once(state)?;
        }

        return Ok(());
    }
}

pub trait ParallelHeuristic<Problem: ProblemTrait>: Heuristic<Problem> {
    fn run_once_par<'a>(
        &self,
        state: &mut SearchState<'a, Problem>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        self.run_once(state)
    }
    fn run_par<'a>(
        &self,
        state: &mut SearchState<'a, Problem>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        while !self.is_done(&state) {
            self.run_once_par(state)?;
        }

        return Ok(());
    }
}

#[derive(Debug, Clone)]
pub struct StopCondition {
    pub max_iteration: Option<u64>,
    pub max_duration: Option<std::time::Duration>,
    pub max_failed_update: Option<u64>,
}

impl StopCondition {
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

    pub fn is_done<'a, Problem: ProblemTrait>(&self, state: &SearchState<'a, Problem>) -> bool {
        if let Some(max_iter) = self.max_iteration {
            if state.iteration - state.start_iteration >= max_iter {
                return true;
            }
        }
        if let Some(max_duration) = self.max_duration {
            if state.duration() >= max_duration {
                return true;
            }
        }
        if let Some(max_failed_update) = self.max_failed_update {
            if state.iteration - state.best_iteration >= max_failed_update {
                return true;
            }
        }
        return false;
    }
}
