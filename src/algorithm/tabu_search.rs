use super::{Heuristic, StopCondition};
use crate::search_state::{EnumerateMoveToNeighbor, ProblemTrait, SearchState};
use rand::seq::IndexedRandom;
use rand::Rng;
use std::cell::RefCell;
use std::collections::HashMap;

pub struct TabuSearch<MoveToNeighbor>
where
    MoveToNeighbor: Clone + std::hash::Hash + std::cmp::Eq,
{
    pub stop_condition: StopCondition,
    pub max_tabu_tenure: u64,
    pub min_tabu_tenure: u64,
    tabu_map: RefCell<HashMap<MoveToNeighbor, u64>>,
}

impl<MoveToNeighbor> TabuSearch<MoveToNeighbor>
where
    MoveToNeighbor: Clone + std::hash::Hash + std::cmp::Eq,
{
    pub fn new(
        stop_condition: StopCondition,
        max_tabu_tenure: u64,
        min_tabu_tenure: u64,
        tabu_map: Option<HashMap<MoveToNeighbor, u64>>,
    ) -> Self {
        Self {
            stop_condition,
            max_tabu_tenure,
            min_tabu_tenure,
            tabu_map: RefCell::new(tabu_map.unwrap_or_else(HashMap::new)),
        }
    }
}

impl<Problem, MoveToNeighbor> Heuristic<Problem> for TabuSearch<MoveToNeighbor>
where
    Problem: ProblemTrait,
    for<'a> SearchState<'a, Problem>: EnumerateMoveToNeighbor<MoveToNeighbor>,
    MoveToNeighbor: Clone + std::hash::Hash + std::cmp::Eq,
{
    fn clear(&mut self) {
        self.tabu_map.borrow_mut().clear();
    }
    fn is_done<'a>(&self, state: &SearchState<'a, Problem>) -> bool {
        self.stop_condition.is_done(state)
    }
    fn run_once<'a>(
        &self,
        state: &mut SearchState<'a, Problem>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let mut best_move_list = vec![];
        for neighbor in state.iter_on_move_to_neighbor() {
            let within_tabu_tenure = self
                .tabu_map
                .borrow()
                .get(&neighbor)
                .map_or(false, |&v| v > state.iteration);
            let is_updating_best = state.is_move_to_be_better_than_best(&neighbor);

            if within_tabu_tenure && !is_updating_best {
                continue;
            }

            if best_move_list.is_empty() {
                best_move_list.push(neighbor);
            } else {
                let sample = &best_move_list[0];
                // neighbor is better than the current best move
                if state.is_first_move_better_than_second(&neighbor, &sample) {
                    best_move_list.clear();
                    best_move_list.push(neighbor);
                // neighbor is equal to the current best move
                } else if !state.is_first_move_better_than_second(&sample, &neighbor) {
                    best_move_list.push(neighbor);
                }
            }
        }

        if best_move_list.is_empty() {
            tracing::warn!("No best move found");
            return Ok(());
        }

        let best_neighbor = {
            if let Some(best_move) = best_move_list.choose(&mut rand::rng()) {
                best_move
            } else {
                return Ok(());
            }
        };

        state.is_move_to_be_better_than_best(&best_neighbor);
        self.tabu_map.borrow_mut().insert(
            best_neighbor.clone(),
            state.iteration + rand::rng().random_range(self.min_tabu_tenure..self.max_tabu_tenure),
        );

        state.apply(&best_neighbor);
        return Ok(());
    }
}
