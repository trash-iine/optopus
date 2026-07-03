use super::{Heuristic, StopCondition};
use crate::error::OptError;
use crate::search_state::SearchStateCloneType;
use crate::trait_defs::ProblemTrait;

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
        Self {
            stop_condition,
            heuristic,
            restart_condition,
        }
    }
}

impl<Problem: ProblemTrait> Heuristic<Problem> for Restart<Problem> {
    fn stop_condition(&self) -> &StopCondition {
        &self.stop_condition
    }

    fn run_once<'a>(
        &mut self,
        state: &mut crate::search_state::SearchState<'a, Problem>,
    ) -> Result<(), OptError> {
        state.run_sub(self.heuristic.as_mut(), SearchStateCloneType::ClearBest)?;

        if self.restart_condition.is_done(state) {
            tracing::debug!("Restart triggered at iteration {}", state.iteration);
            let instance = state.instance;
            state.solution = instance.new_solution(&mut state.rng);
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::heuristic::LocalSearch;
    use crate::problem::{MaxCut, MaxCutFlipNeighbor};
    use crate::search_state::SearchState;
    use crate::trait_defs::Rankable;

    #[test]
    fn restart_preserves_best_solution() {
        let mc = MaxCut::from_edges([
            (0, 1, 1.0),
            (0, 2, 1.0),
            (0, 3, 1.0),
            (1, 2, 1.0),
            (2, 3, 1.0),
        ]);
        let mut state = SearchState::new_with_seed(&mc, 42);
        let initial_obj = state.best_solution.objective;

        // LocalSearch converges quickly, so `failed_updates(1)` triggers a
        // restart on nearly every outer iteration.
        let mut restart = Restart::new(
            StopCondition::iterations(200),
            Box::new(LocalSearch::<MaxCutFlipNeighbor>::new(
                StopCondition::iterations(50),
            )),
            StopCondition::failed_updates(1),
        );
        restart.run(&mut state).unwrap();

        assert!(state.iteration >= 200);
        assert!(state.best_solution.objective >= initial_obj);
        assert!(!state.solution.is_better_than(&state.best_solution));
    }
}
