use super::{Heuristic, StopCondition};
use crate::error::OptError;
use crate::search_state::SearchState;
use crate::trait_defs::{MoveToNeighbor, ProblemTrait, Rankable};

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
/// let mc = MaxCut::from_edges([(0, 1, 1.0), (0, 2, 1.0), (1, 2, 1.0)]);
///
/// let mut state = SearchState::new(&mc);
/// let mut ls = LocalSearch::<MaxCutFlipNeighbor>::new(StopCondition::iterations(1000));
/// ls.run(&mut state).unwrap();
/// ```
///
/// # References
///
/// - Aarts, E. and Lenstra, J. K. (eds.) *Local Search in Combinatorial Optimization*.
///   Princeton University Press, 2003.
pub struct LocalSearch<N> {
    pub stop_condition: StopCondition,
    _neighbor: std::marker::PhantomData<N>,
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
            _neighbor: std::marker::PhantomData,
            no_best_move: false,
        }
    }
}

impl<P, N> Heuristic<P> for LocalSearch<N>
where
    P: ProblemTrait,
    N: MoveToNeighbor<P> + Rankable,
{
    fn clear(&mut self) {
        self.no_best_move = false;
    }

    fn run_once<'a>(&mut self, state: &mut SearchState<'a, P>) -> Result<(), OptError> {
        // max_by avoids the Vec allocation that filter_best() + pop() would incur;
        // tie-breaking is arbitrary, which is fine for hill-climbing.
        let best_move = N::iter(state.instance, &state.solution)
            .filter(|n| state.is_neighbor_better_than_current(n))
            .max_by(crate::trait_defs::rank_cmp);
        if let Some(best_move) = best_move {
            state.apply(&best_move)?;
        } else {
            self.no_best_move = true;
            state.progress_iteration();
        }

        Ok(())
    }

    fn stop_condition(&self) -> &StopCondition {
        &self.stop_condition
    }

    /// Done when the stop condition is met **or** the last iteration found no
    /// improving move (a local optimum was reached).
    fn is_done<'a>(&self, state: &SearchState<'a, P>) -> bool {
        self.stop_condition.is_done(state) || self.no_best_move
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::problem::{MaxCut, MaxCutFlipNeighbor};

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
    fn local_search_stops_at_local_optimum() {
        let mc = small_maxcut();
        let mut state = SearchState::new_with_seed(&mc, 42);
        let initial_obj = state.best_solution.objective;

        let mut ls = LocalSearch::<MaxCutFlipNeighbor>::new(StopCondition::iterations(1_000));
        ls.run(&mut state).unwrap();

        assert!(state.best_solution.objective >= initial_obj);
        // At a local optimum no flip has positive gain.
        assert!(state.best_solution.gain.iter().all(|&g| g <= 0.0));
        // Must stop well before the iteration budget once the optimum is reached.
        assert!(state.iteration < 1_000);
    }

    #[test]
    fn local_search_keeps_counter_invariant() {
        let mc = small_maxcut();
        let mut state = SearchState::new_with_seed(&mc, 7);
        let mut ls = LocalSearch::<MaxCutFlipNeighbor>::new(StopCondition::iterations(1_000));
        ls.run(&mut state).unwrap();
        assert_eq!(state.iteration, state.n_accepted + state.n_rejected);
    }

    #[test]
    fn local_search_clear_resets_no_best_move() {
        let mc = small_maxcut();
        let mut state = SearchState::new_with_seed(&mc, 7);
        let mut ls = LocalSearch::<MaxCutFlipNeighbor>::new(StopCondition::iterations(1_000));
        ls.run(&mut state).unwrap();
        assert!(ls.no_best_move, "run must end at a local optimum");

        ls.clear();
        assert!(!ls.no_best_move);
    }
}
