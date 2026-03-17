use super::Heuristic;
use crate::error::OptError;
use crate::search_state::{ProblemTrait, SearchStateCloneType};

pub struct Sequential<Problem: ProblemTrait> {
    pub stop_condition: super::StopCondition,
    pub heuristics: Vec<Box<dyn Heuristic<Problem>>>,
}

impl<'a, Problem: ProblemTrait> Sequential<Problem> {
    pub fn new(
        stop_condition: super::StopCondition,
        heuristics: Vec<Box<dyn Heuristic<Problem>>>,
    ) -> Self {
        Self {
            stop_condition,
            heuristics,
        }
    }
}

impl<Problem: ProblemTrait> Heuristic<Problem> for Sequential<Problem> {
    fn is_done<'a>(&self, state: &crate::search_state::SearchState<'a, Problem>) -> bool {
        self.stop_condition.is_done(state)
    }
    fn run_once<'a>(
        &self,
        state: &mut crate::search_state::SearchState<'a, Problem>,
    ) -> Result<(), OptError> {
        for heuristic in self.heuristics.iter() {
            let mut cloned = state.clone_for_new_run(SearchStateCloneType::ClearBest);

            heuristic.run(&mut cloned)?;

            state.update(cloned);

            if self.stop_condition.is_done(state) {
                return Ok(());
            }
        }

        Ok(())
    }
}
