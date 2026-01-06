use super::{Heuristic, StopCondition};
use crate::search_state::{filter_best, MoveToNeigbor, ProblemTrait, Rankable, SearchState};
use std::cell::RefCell;

/// A local search algorithm that iteratively explores the neighborhood of the current solution.
/// This algorithm applies the best move from the neighborhood until no better moves are found.
///
/// # Example
///
/// ```
/// use optopus::algorithm::{StopCondition, Heuristic, LocalSearch};
/// use optopus::search_state::SearchState;
/// use optopus::problem::max_cut::{MaxCut, MaxCutFlipNeighbor};
///
/// let mut mc = MaxCut::new();
/// mc.add_weight(0, 1, 1.0);
/// mc.add_weight(0, 2, 1.0);
/// mc.add_weight(1, 2, 1.0);
///
/// let mut state = SearchState::new(&mc, rand::rng());
/// let sc = StopCondition::new(Some(1000), None, None);
/// let ls = LocalSearch::<MaxCutFlipNeighbor>::new(sc);
/// ls.run(&mut state);
/// ```
pub struct LocalSearch<N> {
    pub stop_condition: StopCondition,
    phantom_neighbor: std::marker::PhantomData<N>,
    no_best_move: RefCell<bool>,
}

impl<N> LocalSearch<N> {
    pub fn new(stop_condition: StopCondition) -> Self {
        let mut stop_condition = stop_condition;
        if let Some(max_failed_update) = stop_condition.max_failed_update {
            if max_failed_update != 1 {
                tracing::warn!("StopCondition.max_failed_update should be `Some(1)`.");
            }
        } else {
            stop_condition.max_failed_update = Some(1);
        }

        Self {
            stop_condition,
            phantom_neighbor: std::marker::PhantomData,
            no_best_move: RefCell::new(false),
        }
    }
}

impl<P, N> Heuristic<P> for LocalSearch<N>
where
    P: ProblemTrait,
    N: MoveToNeigbor<P> + Rankable,
{
    fn run_once<'a>(
        &self,
        state: &mut SearchState<'a, P>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let mut best_list = filter_best(
            N::iter(&state.instance, &state.solution)
                .filter(|n| state.is_neighbor_better_than_current(n)),
        );
        if let Some(best_move) = best_list.pop() {
            state.apply(&best_move);
        } else {
            *self.no_best_move.borrow_mut() = true;
        }

        Ok(())
    }

    fn is_done<'a>(&self, state: &SearchState<'a, P>) -> bool {
        self.stop_condition.is_done(state) || *self.no_best_move.borrow()
    }
}
