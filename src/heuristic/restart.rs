use super::{Heuristic, StopCondition};
use crate::error::OptError;
use crate::search_state::{ProblemTrait, SearchStateCloneType};

/// Restart meta-heuristic.
///
/// Runs an inner heuristic repeatedly. When `restart_condition` is satisfied
/// (typically `max_failed_update` — no improvement for a while), the current
/// solution is replaced with a fresh random solution while **the global best
/// solution is preserved**.
///
/// The outer `stop_condition` controls the total budget.
pub struct Restart<Problem: ProblemTrait> {
    pub stop_condition: StopCondition,
    /// The inner heuristic to run between restarts.
    pub heuristic: Box<dyn Heuristic<Problem>>,
    /// Condition that triggers a restart (evaluated against the merged state).
    pub restart_condition: StopCondition,
}

impl<Problem: ProblemTrait> Restart<Problem> {
    pub fn new(
        stop_condition: StopCondition,
        heuristic: Box<dyn Heuristic<Problem>>,
        restart_condition: StopCondition,
    ) -> Self {
        Self { stop_condition, heuristic, restart_condition }
    }
}

impl<Problem: ProblemTrait> Heuristic<Problem> for Restart<Problem> {
    fn is_done<'a>(&self, state: &crate::search_state::SearchState<'a, Problem>) -> bool {
        self.stop_condition.is_done(state)
    }

    fn run_once<'a>(
        &mut self,
        state: &mut crate::search_state::SearchState<'a, Problem>,
    ) -> Result<(), OptError> {
        let mut inner = state.clone_for_new_run(SearchStateCloneType::ClearBest);
        self.heuristic.run(&mut inner)?;
        state.update_state(inner);

        if self.restart_condition.is_done(state) {
            tracing::debug!("Restart triggered at iteration {}", state.iteration);
            let instance = state.instance;
            state.solution = instance.new_solution(&mut state.rng);
        }

        Ok(())
    }
}
