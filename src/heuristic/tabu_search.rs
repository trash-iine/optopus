use super::{Heuristic, StopCondition};
use crate::error::OptError;
use crate::search_state::{
    filter_best, EnabledTabu, MoveToNeigbor, ProblemTrait, Rankable, SearchState,
};

pub struct TabuSearch<N>
where
    N: Clone + EnabledTabu,
{
    pub stop_condition: StopCondition,
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
        Self {
            stop_condition,
            tabu_tenure,
            tabu_map: tabu_map.unwrap_or(N::TabuMap::default()),
        }
    }

    pub fn borrow_tabu_map(&self) -> &N::TabuMap {
        &self.tabu_map
    }

    pub fn borrow_mut_tabu_map(&mut self) -> &mut N::TabuMap {
        &mut self.tabu_map
    }

    pub fn take_tabu_map(&mut self) -> N::TabuMap {
        std::mem::take(&mut self.tabu_map)
    }

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

    fn run_once<'a>(
        &mut self,
        state: &mut SearchState<'a, P>,
    ) -> Result<(), OptError> {
        let mut best_list = filter_best(N::iter(&state.instance, &state.solution).filter(|n| {
            n.is_move_enabled(&self.tabu_map, state.iteration)
                || state.is_neighbor_better_than_best(n)
        }));

        if let Some(best_move) = best_list.pop() {
            best_move.add_to_tabu_map(
                &mut self.tabu_map,
                state.iteration,
                self.tabu_tenure,
            );

            state.apply(&best_move)?;
        } else {
            tracing::warn!("No best move found");
            state.progress_iteration();
        }

        Ok(())
    }
}
