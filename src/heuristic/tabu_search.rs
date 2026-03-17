use super::{Heuristic, StopCondition};
use crate::error::OptError;
use crate::search_state::{
    filter_best, EnabledTabu, MoveToNeigbor, ProblemTrait, Rankable, SearchState,
};
use std::sync::RwLock;

pub struct TabuSearch<N>
where
    N: Clone + EnabledTabu,
{
    pub stop_condition: StopCondition,
    pub tabu_tenure: (u64, u64),
    tabu_map: RwLock<N::TabuMap>,
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
            tabu_map: RwLock::new(tabu_map.unwrap_or(N::TabuMap::default())),
        }
    }

    pub fn borrow_tabu_map(&self) -> std::sync::RwLockReadGuard<'_, N::TabuMap> {
        self.tabu_map.read().unwrap()
    }

    pub fn borrow_mut_tabu_map(&self) -> std::sync::RwLockWriteGuard<'_, N::TabuMap> {
        self.tabu_map.write().unwrap()
    }

    pub fn take_tabu_map(&self) -> N::TabuMap {
        let mut write_guard = self.tabu_map.write().unwrap();
        std::mem::take(&mut *write_guard)
    }

    pub fn set_tabu_map(&self, tabu_map: N::TabuMap) {
        let mut write_guard = self.tabu_map.write().unwrap();
        *write_guard = tabu_map;
    }
}

impl<P, N> Heuristic<P> for TabuSearch<N>
where
    P: ProblemTrait,
    N: MoveToNeigbor<P> + Clone + EnabledTabu + Rankable,
    N::TabuMap: Default,
{
    fn is_done<'a>(&self, state: &SearchState<'a, P>) -> bool {
        self.stop_condition.is_done(state)
    }
    fn run_once<'a>(
        &self,
        state: &mut SearchState<'a, P>,
    ) -> Result<(), OptError> {
        let mut best_list = filter_best(N::iter(&state.instance, &state.solution).filter(|n| {
            n.is_move_enabled(&self.borrow_tabu_map(), state.iteration)
                || state.is_neighbor_better_than_best(n)
        }));

        if let Some(best_move) = best_list.pop() {
            best_move.add_to_tabu_map(
                &mut self.borrow_mut_tabu_map(),
                state.iteration,
                self.tabu_tenure,
            );

            state.apply(&best_move);
        } else {
            tracing::warn!("No best move found");
            state.progress_iteration();
        }

        Ok(())
    }
}
