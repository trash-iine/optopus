use super::{Heuristic, StopCondition};
use crate::error::OptError;
use crate::search_state::SearchStateCloneType;
use crate::trait_defs::ProblemTrait;

/// Restart meta-heuristic.
///
/// Runs an inner heuristic repeatedly. When `restart_condition` is satisfied
/// (typically `max_failed_update`, no improvement for a while), the current
/// solution is replaced with a fresh random solution while the global best
/// solution is preserved.
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
        let mut sub = state.clone_for_new_run(SearchStateCloneType::ClearBest);
        self.heuristic.run(&mut sub)?;
        state.update_state(sub);

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

    /// The restart branch is the whole of `Restart`, so a test has to observe
    /// it rather than the counters around it. The same seed is run twice and
    /// only the restart condition differs, so the two runs share every draw
    /// and every move up to the branch: the best is what survives it and the
    /// incumbent is what does not. Asserting only that the best did not get
    /// worse would pass with the branch deleted, because `update_best` never
    /// moves the best the wrong way.
    #[test]
    fn a_met_restart_condition_replaces_the_incumbent_and_keeps_the_best() {
        let mc = MaxCut::from_edges([
            (0, 1, 1.0),
            (0, 2, 1.0),
            (0, 3, 1.0),
            (1, 2, 1.0),
            (2, 3, 1.0),
        ]);

        // One outer iteration: the inner descent alone carries the budget past
        // `iterations(1)`, so `run_once` is entered exactly once.
        let run = |restart_condition| {
            let mut state = SearchState::new_with_seed(&mc, 42);
            Restart::new(
                StopCondition::iterations(1),
                Box::new(LocalSearch::<MaxCutFlipNeighbor>::new(
                    StopCondition::iterations(50),
                )),
                restart_condition,
            )
            .run(&mut state)
            .unwrap();
            state
        };

        // LocalSearch ends at a local optimum, so the last iterations improved
        // nothing and `failed_updates(1)` is met; `u64::MAX` never is.
        let restarted = run(StopCondition::failed_updates(1));
        let untouched = run(StopCondition::failed_updates(u64::MAX));

        assert_eq!(
            restarted.best_solution.x, untouched.best_solution.x,
            "the best crosses the restart untouched"
        );
        assert_ne!(
            restarted.solution.x, untouched.solution.x,
            "the incumbent is the one that is thrown away"
        );
        assert!(!restarted.solution.is_better_than(&restarted.best_solution));
    }
}
