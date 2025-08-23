use super::{Heuristic, StopCondition};
use crate::search_state::{EnabledTabu, EnumerateMoveToNeighbor, ProblemTrait, SearchState};
use std::cell::RefCell;

pub struct TabuSearch<MoveToNeighbor>
where
    MoveToNeighbor: Clone + EnabledTabu,
{
    pub stop_condition: StopCondition,
    pub tabu_tenure: (u64, u64),
    tabu_map: RefCell<MoveToNeighbor::TabuMap>,
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
            tabu_map: RefCell::new(tabu_map.unwrap_or(MoveToNeighbor::TabuMap::default())),
        }
    }

    pub fn borrow_tabu_map(&self) -> std::cell::Ref<'_, MoveToNeighbor::TabuMap> {
        self.tabu_map.borrow()
    }

    pub fn borrow_mut_tabu_map(&self) -> std::cell::RefMut<'_, MoveToNeighbor::TabuMap> {
        self.tabu_map.borrow_mut()
    }

    pub fn take_tabu_map(&self) -> MoveToNeighbor::TabuMap {
        self.tabu_map.take()
    }

    pub fn set_tabu_map(&self, tabu_map: MoveToNeighbor::TabuMap) {
        *self.tabu_map.borrow_mut() = tabu_map;
    }
}

impl<Problem, MoveToNeighbor> Heuristic<Problem> for TabuSearch<MoveToNeighbor>
where
    Problem: ProblemTrait,
    for<'a> SearchState<'a, Problem>: EnumerateMoveToNeighbor<MoveToNeighbor>,
    MoveToNeighbor: Clone + EnabledTabu,
{
    fn is_done<'a>(&self, state: &SearchState<'a, Problem>) -> bool {
        self.stop_condition.is_done(state)
    }
    fn run_once<'a>(
        &self,
        state: &mut SearchState<'a, Problem>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let mut best_move_option = None;
        for neighbor in state.iter_on_move_to_neighbor() {
            let is_enabled = neighbor.is_move_enabled(&self.tabu_map.borrow(), state.iteration);
            let is_updating_best = state.is_move_to_be_better_than_best(&neighbor);

            if !is_enabled && !is_updating_best {
                continue;
            }

            if let Some(best_move) = best_move_option {
                if state.is_first_move_better_than_second(&neighbor, &best_move) {
                    best_move_option = Some(neighbor);
                } else {
                    best_move_option = Some(best_move);
                }
            } else {
                best_move_option = Some(neighbor);
            }
        }

        if let Some(best_move) = best_move_option {
            best_move.add_to_tabu_map(
                &mut self.tabu_map.borrow_mut(),
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
