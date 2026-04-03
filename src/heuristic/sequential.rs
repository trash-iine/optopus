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

/// Iterated meta-heuristic (ILS pattern).
///
/// Alternates between a **search** phase and a **perturbation** phase, repeating until
/// the outer stopping condition is satisfied.
///
/// Each phase runs on a [`SearchStateCloneType::ClearBest`] clone of the current state.
/// After each phase the result is merged back with [`crate::search_state::SearchState::update_state`].
///
/// # Cycle
///
/// 1. Run `search` until its inner stopping condition is met (finds a local optimum).
/// 2. Check the outer `stop_condition` — return early if done.
/// 3. Run `perturbation` to escape the local optimum.
/// 4. Merge and repeat.
pub struct Iterated<Problem: ProblemTrait> {
    pub stop_condition: super::StopCondition,
    /// Search phase (e.g. `LocalSearch` or `TabuSearch`).
    pub search: Box<dyn Heuristic<Problem>>,
    /// Perturbation phase (e.g. `RandomWalk`).
    pub perturbation: Box<dyn Heuristic<Problem>>,
}

impl<Problem: ProblemTrait> Iterated<Problem> {
    pub fn new(
        stop_condition: super::StopCondition,
        search: Box<dyn Heuristic<Problem>>,
        perturbation: Box<dyn Heuristic<Problem>>,
    ) -> Self {
        Self {
            stop_condition,
            search,
            perturbation,
        }
    }
}

impl<Problem: ProblemTrait> Heuristic<Problem> for Iterated<Problem> {
    fn is_done<'a>(&self, state: &crate::search_state::SearchState<'a, Problem>) -> bool {
        self.stop_condition.is_done(state)
    }

    fn run_once<'a>(
        &mut self,
        state: &mut crate::search_state::SearchState<'a, Problem>,
    ) -> Result<(), OptError> {
        // Search phase
        let mut search_state = state.clone_for_new_run(SearchStateCloneType::ClearBest);
        self.search.run(&mut search_state)?;
        state.update_state(search_state);

        if self.stop_condition.is_done(state) {
            return Ok(());
        }

        // Perturbation phase
        let mut perturb_state = state.clone_for_new_run(SearchStateCloneType::ClearBest);
        self.perturbation.run(&mut perturb_state)?;
        state.update_state(perturb_state);

        Ok(())
    }
}
