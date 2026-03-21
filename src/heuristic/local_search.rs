use super::{Heuristic, StopCondition};
use crate::error::OptError;
use crate::search_state::{MoveToNeigbor, ProblemTrait, Rankable, SearchState, filter_best};

/// A local search algorithm that iteratively explores the neighborhood of the current solution.
/// This algorithm applies the best move from the neighborhood until no better moves are found.
///
/// # Example
///
/// ```
/// use optopus::heuristic::{StopCondition, Heuristic, LocalSearch};
/// use optopus::search_state::SearchState;
/// use optopus::problem::{MaxCut, MaxCutFlipNeighbor};
///
/// let mut mc = MaxCut::new();
/// mc.add_weight(0, 1, 1.0);
/// mc.add_weight(0, 2, 1.0);
/// mc.add_weight(1, 2, 1.0);
///
/// let mut state = SearchState::new(&mc);
/// let mut ls = LocalSearch::<MaxCutFlipNeighbor>::new(StopCondition::iterations(1000));
/// ls.run(&mut state).unwrap();
/// ```
pub struct LocalSearch<N> {
    pub stop_condition: StopCondition,
    phantom_neighbor: std::marker::PhantomData<N>,
    no_best_move: bool,
}

impl<N> LocalSearch<N> {
    /// Create a new [`LocalSearch`] with the given stopping condition.
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
            no_best_move: false,
        }
    }
}

impl<P, N> Heuristic<P> for LocalSearch<N>
where
    P: ProblemTrait,
    N: MoveToNeigbor<P> + Rankable,
{
    fn clear(&mut self) {
        self.no_best_move = false;
    }

    fn run_once<'a>(&mut self, state: &mut SearchState<'a, P>) -> Result<(), OptError> {
        let mut best_list = filter_best(
            N::iter(&state.instance, &state.solution)
                .filter(|n| state.is_neighbor_better_than_current(n)),
        );
        if let Some(best_move) = best_list.pop() {
            state.apply(&best_move)?;
        } else {
            self.no_best_move = true;
        }

        Ok(())
    }

    fn is_done<'a>(&self, state: &SearchState<'a, P>) -> bool {
        self.stop_condition.is_done(state) || self.no_best_move
    }
}
