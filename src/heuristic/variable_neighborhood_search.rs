use super::{Heuristic, StopCondition};
use crate::error::OptError;
use crate::search_state::SearchStateCloneType;
use crate::trait_defs::{ProblemTrait, Rankable};

/// Basic Variable Neighborhood Search (VNS).
///
/// Shakes the incumbent solution in the current neighborhood `N_k`, runs the
/// local `search`, and moves to the result only if it improves the incumbent.
/// Improvement resets `k` to the first neighborhood; failure restores the
/// incumbent and advances `k` (wrapping around after the last neighborhood).
///
/// Each phase runs on a [`SearchStateCloneType::ClearBest`] clone of the
/// current state, so the global best solution is preserved by the merge even
/// when the incumbent is restored.
///
/// # References
///
/// - Mladenovic, N. and Hansen, P. "Variable neighborhood search."
///   *Computers & Operations Research*, 24(11):1097-1100, 1997.
///   [DOI](https://doi.org/10.1016/S0305-0548(97)00031-2)
///
/// # Cycle
///
/// 1. Snapshot the incumbent solution.
/// 2. Run `shakes[k]` to jump to a random point in neighborhood `N_k`.
/// 3. Check the outer `stop_condition` — restore the incumbent and return
///    early if done (never end a run on an un-searched shaken solution).
/// 4. Run `search` until its inner stopping condition is met.
/// 5. If the result improves the incumbent, keep it and reset `k = 0`;
///    otherwise restore the incumbent and advance `k` (with wrap-around).
pub struct VariableNeighborhoodSearch<Problem: ProblemTrait> {
    pub stop_condition: super::StopCondition,
    /// Local-search phase (e.g. `LocalSearch` or `TabuSearch`).
    pub search: Box<dyn Heuristic<Problem>>,
    /// Ordered shake neighborhoods `N_1..N_kmax`, typically of growing
    /// strength (e.g. `RandomWalk` with increasing iteration budgets).
    pub shakes: Vec<Box<dyn Heuristic<Problem>>>,
    /// Index of the shake neighborhood used next.
    k: usize,
}

impl<Problem: ProblemTrait> VariableNeighborhoodSearch<Problem> {
    /// # Panics
    ///
    /// Panics if `shakes` is empty.
    pub fn new(
        stop_condition: super::StopCondition,
        search: Box<dyn Heuristic<Problem>>,
        shakes: Vec<Box<dyn Heuristic<Problem>>>,
    ) -> Self {
        assert!(
            !shakes.is_empty(),
            "VariableNeighborhoodSearch requires at least one shake heuristic"
        );
        Self {
            stop_condition,
            search,
            shakes,
            k: 0,
        }
    }
}

impl<Problem: ProblemTrait> Heuristic<Problem> for VariableNeighborhoodSearch<Problem> {
    fn clear(&mut self) {
        self.k = 0;
    }

    fn stop_condition(&self) -> &StopCondition {
        &self.stop_condition
    }

    fn run_once<'a>(
        &mut self,
        state: &mut crate::search_state::SearchState<'a, Problem>,
    ) -> Result<(), OptError> {
        let incumbent = state.solution.clone();

        // Shake phase in neighborhood N_k
        state.run_sub(
            self.shakes[self.k].as_mut(),
            SearchStateCloneType::ClearBest,
        )?;

        if self.stop_condition.is_done(state) {
            state.solution = incumbent;
            return Ok(());
        }

        // Local search phase
        state.run_sub(self.search.as_mut(), SearchStateCloneType::ClearBest)?;

        // Move-or-not: improvement restarts the neighborhood sequence,
        // failure restores the incumbent and tries the next neighborhood.
        if state.solution.is_better_than(&incumbent) {
            self.k = 0;
        } else {
            state.solution = incumbent;
            self.k = (self.k + 1) % self.shakes.len();
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;
    use std::rc::Rc;

    use super::*;
    use crate::heuristic::{LocalSearch, RandomWalk, StopCondition};
    use crate::problem::{MaxCut, MaxCutFlipNeighbor, MaxCutSolution};
    use crate::search_state::SearchState;
    use crate::trait_defs::Rankable;

    fn small_maxcut() -> MaxCut {
        MaxCut::from_edges([
            (0, 1, 1.0),
            (0, 2, 1.0),
            (0, 3, 1.0),
            (1, 2, 1.0),
            (2, 3, 1.0),
        ])
    }

    fn vns_with_random_walk_shakes(
        stop_condition: StopCondition,
    ) -> VariableNeighborhoodSearch<MaxCut> {
        VariableNeighborhoodSearch::new(
            stop_condition,
            Box::new(LocalSearch::<MaxCutFlipNeighbor>::new(
                StopCondition::iterations(50),
            )),
            vec![
                Box::new(RandomWalk::<MaxCutFlipNeighbor>::new(
                    StopCondition::iterations(3),
                )),
                Box::new(RandomWalk::<MaxCutFlipNeighbor>::new(
                    StopCondition::iterations(6),
                )),
            ],
        )
    }

    #[test]
    #[should_panic(expected = "at least one shake heuristic")]
    fn new_panics_on_empty_shakes() {
        VariableNeighborhoodSearch::<MaxCut>::new(
            StopCondition::iterations(10),
            Box::new(LocalSearch::<MaxCutFlipNeighbor>::new(
                StopCondition::iterations(10),
            )),
            vec![],
        );
    }

    #[test]
    fn vns_preserves_best_and_never_ends_worse() {
        let mc = small_maxcut();
        let mut state = SearchState::new_with_seed(&mc, 42);
        let initial_obj = state.best_solution.objective;

        let mut vns = vns_with_random_walk_shakes(StopCondition::iterations(100));
        vns.run(&mut state).unwrap();

        assert!(state.iteration >= 100);
        assert!(state.best_solution.objective >= initial_obj);
        // The best solution must never be worse than the current one.
        assert!(!state.solution.is_better_than(&state.best_solution));
    }

    #[test]
    fn vns_restores_incumbent_when_no_improvement() {
        let mc = small_maxcut();
        // The optimal cut of this graph: {0, 2} vs {1, 3} cuts all 5 edges
        // except (0, 2), giving objective 4.
        let optimum = MaxCutSolution::new_from_assignment(&mc, vec![true, false, true, false]);
        assert_eq!(optimum.objective, 4.0);
        let mut state = SearchState::with_solution_and_seed(&mc, optimum, 42);

        let mut vns = vns_with_random_walk_shakes(StopCondition::iterations(100));
        vns.run(&mut state).unwrap();

        // Nothing can beat the optimum incumbent, so every cycle must have
        // restored it.
        assert_eq!(state.solution.objective, 4.0);
        assert_eq!(state.best_solution.objective, 4.0);
    }

    /// Test heuristic that records its `id` on every `run` and advances the
    /// iteration counter without touching the solution.
    struct Probe {
        id: usize,
        log: Rc<RefCell<Vec<usize>>>,
        stop_condition: StopCondition,
    }

    impl Heuristic<MaxCut> for Probe {
        fn stop_condition(&self) -> &StopCondition {
            &self.stop_condition
        }

        fn run_once<'a>(
            &mut self,
            state: &mut crate::search_state::SearchState<'a, MaxCut>,
        ) -> Result<(), OptError> {
            self.log.borrow_mut().push(self.id);
            state.progress_iteration();
            Ok(())
        }
    }

    fn probe_vns(
        stop_condition: StopCondition,
        log: &Rc<RefCell<Vec<usize>>>,
    ) -> VariableNeighborhoodSearch<MaxCut> {
        let probe = |id| {
            Box::new(Probe {
                id,
                log: Rc::clone(log),
                stop_condition: StopCondition::iterations(1),
            }) as Box<dyn Heuristic<MaxCut>>
        };
        VariableNeighborhoodSearch::new(
            stop_condition,
            Box::new(LocalSearch::<MaxCutFlipNeighbor>::new(
                StopCondition::iterations(50),
            )),
            vec![probe(0), probe(1), probe(2)],
        )
    }

    #[test]
    fn vns_cycles_shakes_and_resets_on_improvement() {
        let mc = small_maxcut();
        // Start from the all-false solution: the first local search improves
        // the incumbent (k resets to 0); the probes never move the solution,
        // so no later cycle can improve and k advances with wrap-around.
        let all_false = MaxCutSolution::new_from_assignment(&mc, vec![false; 4]);
        let mut state = SearchState::with_solution_and_seed(&mc, all_false, 42);

        let log = Rc::new(RefCell::new(Vec::new()));
        let mut vns = probe_vns(StopCondition::iterations(20), &log);
        vns.run(&mut state).unwrap();

        let recorded = log.borrow();
        assert!(recorded.len() >= 5, "log too short: {recorded:?}");
        // Cycle 1 shakes with N_0 and improves (k stays 0); the failing
        // cycles then advance k through 0, 1, 2 and wrap back to 0.
        assert_eq!(&recorded[..5], &[0, 0, 1, 2, 0], "log: {recorded:?}");
    }

    #[test]
    fn clear_resets_k_between_runs() {
        let mc = small_maxcut();
        // Start both runs at the optimum so no cycle improves and k advances
        // on every cycle.
        let optimum = MaxCutSolution::new_from_assignment(&mc, vec![true, false, true, false]);
        let mut state = SearchState::with_solution_and_seed(&mc, optimum, 42);

        let log = Rc::new(RefCell::new(Vec::new()));
        // Two cycles per run: the first run ends with k pointing at N_2.
        let mut vns = probe_vns(StopCondition::iterations(4), &log);
        vns.run(&mut state).unwrap();
        assert_eq!(*log.borrow(), vec![0, 1]);

        let mut state = SearchState::with_solution_and_seed(
            &mc,
            MaxCutSolution::new_from_assignment(&mc, vec![true, false, true, false]),
            42,
        );
        vns.run(&mut state).unwrap();

        // The second run must start again from N_0, not from N_2.
        assert_eq!(*log.borrow(), vec![0, 1, 0, 1]);
    }
}
