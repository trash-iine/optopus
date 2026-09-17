use super::{Heuristic, StopCondition};
use crate::error::OptError;
use crate::search_state::SearchStateCloneType;
use crate::trait_defs::ProblemTrait;

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

impl<Problem: ProblemTrait> Sequential<Problem> {
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
    fn stop_condition(&self) -> &StopCondition {
        &self.stop_condition
    }

    fn run_once<'a>(
        &mut self,
        state: &mut crate::search_state::SearchState<'a, Problem>,
    ) -> Result<(), OptError> {
        for heuristic in self.heuristics.iter_mut() {
            let mut sub = state.clone_for_new_run(SearchStateCloneType::ClearBest);
            heuristic.run(&mut sub)?;
            state.update_state(sub);

            if self.stop_condition.is_done(state) {
                return Ok(());
            }
        }

        Ok(())
    }
}

/// Iterated meta-heuristic (ILS pattern).
///
/// Alternates between a search phase and a perturbation phase, repeating until
/// the outer stopping condition is satisfied.
///
/// Each phase runs on a [`SearchStateCloneType::ClearBest`] clone of the current state.
/// After each phase the result is merged back with [`crate::search_state::SearchState::update_state`].
///
/// # References
///
/// - Lourenco, H. R., Martin, O. C., and Stutzle, T. "Iterated Local Search." In Glover, F.
///   and Kochenberger, G. A. (eds.), Handbook of Metaheuristics, pp. 320-353. Springer, 2003.
///   [DOI](https://doi.org/10.1007/0-306-48056-5_11)
///
/// # Cycle
///
/// 1. Run `search` until its inner stopping condition is met (finds a local optimum).
/// 2. Check the outer `stop_condition`, return early if done.
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
    fn stop_condition(&self) -> &StopCondition {
        &self.stop_condition
    }

    fn run_once<'a>(
        &mut self,
        state: &mut crate::search_state::SearchState<'a, Problem>,
    ) -> Result<(), OptError> {
        // Search phase
        let mut sub = state.clone_for_new_run(SearchStateCloneType::ClearBest);
        self.search.run(&mut sub)?;
        state.update_state(sub);

        if self.stop_condition.is_done(state) {
            return Ok(());
        }

        // Perturbation phase
        let mut sub = state.clone_for_new_run(SearchStateCloneType::ClearBest);
        self.perturbation.run(&mut sub)?;
        state.update_state(sub);

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::heuristic::{RandomWalk, StopCondition};
    use crate::problem::{MaxCut, MaxCutFlipNeighbor};
    use crate::search_state::SearchState;
    use std::cell::RefCell;
    use std::rc::Rc;

    fn small_maxcut() -> MaxCut {
        MaxCut::from_edges([
            (0, 1, 1.0),
            (0, 2, 1.0),
            (0, 3, 1.0),
            (1, 2, 1.0),
            (2, 3, 1.0),
        ])
    }

    #[test]
    fn sequential_merges_sub_run_iterations() {
        let mc = small_maxcut();
        let mut state = SearchState::new_with_seed(&mc, 42);

        // Two 10-iteration walks per cycle; the outer condition (15) is checked
        // between steps, so exactly one full cycle of 20 iterations runs.
        let mut seq = Sequential::new(
            StopCondition::iterations(15),
            vec![
                Box::new(RandomWalk::<MaxCutFlipNeighbor>::new(
                    StopCondition::iterations(10),
                )),
                Box::new(RandomWalk::<MaxCutFlipNeighbor>::new(
                    StopCondition::iterations(10),
                )),
            ],
        );
        seq.run(&mut state).unwrap();

        assert_eq!(state.iteration, 20);
        assert_eq!(state.iteration, state.n_accepted + state.n_rejected);
    }

    #[test]
    fn sequential_stops_mid_cycle_when_outer_condition_met() {
        let mc = small_maxcut();
        let mut state = SearchState::new_with_seed(&mc, 42);

        let mut seq = Sequential::new(
            StopCondition::iterations(10),
            vec![
                Box::new(RandomWalk::<MaxCutFlipNeighbor>::new(
                    StopCondition::iterations(10),
                )),
                Box::new(RandomWalk::<MaxCutFlipNeighbor>::new(
                    StopCondition::iterations(10),
                )),
            ],
        );
        seq.run(&mut state).unwrap();

        // The outer condition is satisfied right after the first step.
        assert_eq!(state.iteration, 10);
    }

    /// Records which phase ran, and advances one iteration so a budget of `n`
    /// counts `n` phases.
    struct Probe {
        id: usize,
        log: Rc<RefCell<Vec<usize>>>,
        stop_condition: StopCondition,
    }

    impl Heuristic<MaxCut> for Probe {
        fn stop_condition(&self) -> &StopCondition {
            &self.stop_condition
        }

        fn run_once<'a>(&mut self, state: &mut SearchState<'a, MaxCut>) -> Result<(), OptError> {
            self.log.borrow_mut().push(self.id);
            state.progress_iteration();
            Ok(())
        }
    }

    /// `Iterated` is search, then the outer check, then perturbation, and the
    /// order and the check are the whole of it. Two probes make the sequence
    /// readable: deleting the perturbation phase, or moving the early return
    /// past it, changes this log. Asserting that the best did not get worse
    /// would not, because `update_best` never moves the best the wrong way.
    #[test]
    fn iterated_alternates_search_and_perturbation_and_checks_between_them() {
        let mc = small_maxcut();
        let mut state = SearchState::new_with_seed(&mc, 42);

        let log = Rc::new(RefCell::new(Vec::new()));
        let probe = |id| {
            Box::new(Probe {
                id,
                log: Rc::clone(&log),
                stop_condition: StopCondition::iterations(1),
            }) as Box<dyn Heuristic<MaxCut>>
        };

        let mut ils = Iterated::new(StopCondition::iterations(5), probe(0), probe(1));
        ils.run(&mut state).unwrap();

        // Five single-iteration phases: search, perturbation, search,
        // perturbation, and a search that meets the budget, after which the
        // outer check returns before the perturbation of that cycle.
        assert_eq!(&log.borrow()[..], &[0, 1, 0, 1, 0]);
        assert_eq!(state.iteration, 5);
    }

    #[test]
    fn iterated_merges_both_phases_into_the_counters() {
        let mc = small_maxcut();
        let mut state = SearchState::new_with_seed(&mc, 42);

        let mut ils = Iterated::new(
            StopCondition::iterations(100),
            Box::new(RandomWalk::<MaxCutFlipNeighbor>::new(
                StopCondition::iterations(7),
            )),
            Box::new(RandomWalk::<MaxCutFlipNeighbor>::new(
                StopCondition::iterations(3),
            )),
        );
        ils.run(&mut state).unwrap();

        assert_eq!(state.iteration, state.n_accepted + state.n_rejected);
        assert_eq!(state.iteration % 10, 0, "a cycle is 7 + 3 iterations");
    }
}
