use super::{Heuristic, StopCondition};
use crate::error::OptError;
use crate::search_state::{
    EnabledTabu, MoveToNeigbor, ProblemTrait, Rankable, SearchState, filter_best,
};

/// Tabu search heuristic.
///
/// At each iteration the best non-tabu move in the neighborhood is selected and applied.
/// A move is allowed even if it is tabu when it satisfies the aspiration criterion:
/// the resulting solution would be better than the current best.
///
/// After each applied move, the move is added to the tabu map with a tenure sampled
/// uniformly from the range `tabu_tenure = (min, max)`.
///
/// To use this heuristic, the neighbor type must implement [`EnabledTabu`] which defines how to manage the tabu map and tenure.
pub struct TabuSearch<N>
where
    N: Clone + EnabledTabu,
{
    pub stop_condition: StopCondition,
    /// Tabu tenure range `(min, max)` in iterations.
    pub tabu_tenure: (u64, u64),
    tabu_map: N::TabuMap,
}

impl<N> TabuSearch<N>
where
    N: Clone + EnabledTabu,
    N::TabuMap: Default,
{
    pub fn new(
        stop_condition: StopCondition,
        tabu_tenure: (u64, u64),
        tabu_map: Option<N::TabuMap>,
    ) -> Self {
        if tabu_tenure.0 > tabu_tenure.1 {
            panic!(
                "Invalid tabu tenure range: left side should be smaller than or equal to the right side ({} <= {})",
                tabu_tenure.0, tabu_tenure.1
            );
        }
        Self {
            stop_condition,
            tabu_tenure,
            tabu_map: tabu_map.unwrap_or(N::TabuMap::default()),
        }
    }

    /// Returns a shared reference to the tabu map.
    pub fn borrow_tabu_map(&self) -> &N::TabuMap {
        &self.tabu_map
    }

    /// Returns a mutable reference to the tabu map.
    pub fn borrow_mut_tabu_map(&mut self) -> &mut N::TabuMap {
        &mut self.tabu_map
    }

    /// Takes ownership of the tabu map, replacing it with its default value.
    pub fn take_tabu_map(&mut self) -> N::TabuMap {
        std::mem::take(&mut self.tabu_map)
    }

    /// Replaces the tabu map with the given value.
    pub fn set_tabu_map(&mut self, tabu_map: N::TabuMap) {
        self.tabu_map = tabu_map;
    }
}

impl<P, N> Heuristic<P> for TabuSearch<N>
where
    P: ProblemTrait,
    N: MoveToNeigbor<P> + Clone + EnabledTabu + Rankable,
    N::TabuMap: Default,
{
    fn clear(&mut self) {
        self.tabu_map = N::TabuMap::default();
    }

    fn is_done<'a>(&self, state: &SearchState<'a, P>) -> bool {
        self.stop_condition.is_done(state)
    }

    fn run_once<'a>(&mut self, state: &mut SearchState<'a, P>) -> Result<(), OptError> {
        let mut best_list = filter_best(N::iter(&state.instance, &state.solution).filter(|n| {
            // Accept a tabu move if it satisfies the aspiration criterion
            n.is_move_enabled(&self.tabu_map, state.iteration)
                || state.is_neighbor_better_than_best(n)
        }));

        if let Some(best_move) = best_list.pop() {
            best_move.add_to_tabu_map(&mut self.tabu_map, state.iteration, self.tabu_tenure);
            state.apply(&best_move)?;
        } else {
            tracing::warn!("No best move found");
            state.progress_iteration();
        }

        Ok(())
    }
}
