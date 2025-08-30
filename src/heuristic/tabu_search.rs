use super::{Heuristic, ParallelHeuristic, StopCondition};
use crate::search_state::{EnabledTabu, EnumerateMoveToNeighbor, ProblemTrait, SearchState};
use std::sync::RwLock;

pub struct TabuSearch<MoveToNeighbor>
where
    MoveToNeighbor: Clone + EnabledTabu,
{
    pub stop_condition: StopCondition,
    pub tabu_tenure: (u64, u64),
    tabu_map: RwLock<MoveToNeighbor::TabuMap>,
}

impl<MoveToNeighbor> TabuSearch<MoveToNeighbor>
where
    MoveToNeighbor: Clone + EnabledTabu,
    MoveToNeighbor::TabuMap: Default,
{
    pub fn new(
        stop_condition: StopCondition,
        tabu_tenure: (u64, u64),
        tabu_map: Option<MoveToNeighbor::TabuMap>,
    ) -> Self {
        Self {
            stop_condition,
            tabu_tenure,
            tabu_map: RwLock::new(tabu_map.unwrap_or(MoveToNeighbor::TabuMap::default())),
        }
    }

    pub fn borrow_tabu_map(&self) -> std::sync::RwLockReadGuard<'_, MoveToNeighbor::TabuMap> {
        self.tabu_map.read().unwrap()
    }

    pub fn borrow_mut_tabu_map(&self) -> std::sync::RwLockWriteGuard<'_, MoveToNeighbor::TabuMap> {
        self.tabu_map.write().unwrap()
    }

    pub fn take_tabu_map(&self) -> MoveToNeighbor::TabuMap {
        let mut write_guard = self.tabu_map.write().unwrap();
        std::mem::take(&mut *write_guard)
    }

    pub fn set_tabu_map(&self, tabu_map: MoveToNeighbor::TabuMap) {
        let mut write_guard = self.tabu_map.write().unwrap();
        *write_guard = tabu_map;
    }
}

impl<Problem, MoveToNeighbor> Heuristic<Problem> for TabuSearch<MoveToNeighbor>
where
    Problem: ProblemTrait,
    for<'a> SearchState<'a, Problem>: EnumerateMoveToNeighbor<MoveToNeighbor>,
    MoveToNeighbor: Clone + EnabledTabu,
    MoveToNeighbor::TabuMap: Default,
{
    fn is_done<'a>(&self, state: &SearchState<'a, Problem>) -> bool {
        self.stop_condition.is_done(state)
    }
    fn run_once<'a>(
        &self,
        state: &mut SearchState<'a, Problem>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let best_move_option =
            state.get_best_move(state.iter_on_move_to_neighbor().filter(|neighbor| {
                neighbor.is_move_enabled(&self.borrow_tabu_map(), state.iteration)
                    || state.is_move_to_be_better_than_best(&neighbor)
            }));

        if let Some(best_move) = best_move_option {
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

impl<Problem, MoveToNeighbor> ParallelHeuristic<Problem> for TabuSearch<MoveToNeighbor>
where
    Problem: ProblemTrait + Sync,
    Problem::Solution: Sync,
    Problem::Objective: Sync,
    for<'a> SearchState<'a, Problem>: EnumerateMoveToNeighbor<MoveToNeighbor>,
    MoveToNeighbor: Clone + EnabledTabu + Send + Sync,
    MoveToNeighbor::TabuMap: Default + Sync + Send,
{
    fn run_once_par<'a>(
        &self,
        state: &mut SearchState<'a, Problem>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let best_move_option = state.get_best_move_par_chunks(
            state.iter_on_move_to_neighbor().filter(|neighbor| {
                neighbor.is_move_enabled(&self.borrow_tabu_map(), state.iteration)
                    || state.is_move_to_be_better_than_best(&neighbor)
            }),
            10000,
        );

        if let Some(best_move) = best_move_option {
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
