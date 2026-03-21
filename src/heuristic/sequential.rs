use super::Heuristic;
use crate::error::OptError;
use crate::search_state::{ProblemTrait, SearchStateCloneType};

/// Sequential meta-heuristic that runs a list of heuristics one after another.
///
/// Each sub-heuristic operates on a fresh clone of the current state
/// (using [`SearchStateCloneType::ClearBest`]).
/// After each sub-heuristic completes, its results are merged back into the main state.
/// The outer stopping condition is checked between sub-heuristic runs.
pub struct Sequential<Problem: ProblemTrait> {
    pub stop_condition: super::StopCondition,
    pub heuristics: Vec<Box<dyn Heuristic<Problem>>>,
}

impl<'a, Problem: ProblemTrait> Sequential<Problem> {
    /// Create a new [`Sequential`] heuristic with the given stopping condition and list of heuristics.
    pub fn new(
        stop_condition: super::StopCondition,
        heuristics: Vec<Box<dyn Heuristic<Problem>>>,
    ) -> Self {
        Self {
            stop_condition,
            heuristics,
        }
    }

    /// Add a heuristic to the last of the sequence.
    pub fn push_heuristic(&mut self, heuristic: Box<dyn Heuristic<Problem>>) {
        self.heuristics.push(heuristic);
    }
}

impl<Problem: ProblemTrait> Heuristic<Problem> for Sequential<Problem> {
    fn is_done<'a>(&self, state: &crate::search_state::SearchState<'a, Problem>) -> bool {
        self.stop_condition.is_done(state)
    }

    fn run_once<'a>(
        &mut self,
        state: &mut crate::search_state::SearchState<'a, Problem>,
    ) -> Result<(), OptError> {
        for heuristic in self.heuristics.iter_mut() {
            let mut cloned = state.clone_for_new_run(SearchStateCloneType::ClearBest);

            heuristic.run(&mut cloned)?;

            state.update_state(cloned);

            if self.stop_condition.is_done(state) {
                return Ok(());
            }
        }

        Ok(())
    }
}
