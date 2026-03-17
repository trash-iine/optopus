use super::{Heuristic, StopCondition};
use crate::search_state::{EnumerateMoveToNeighbor, ProblemTrait, SearchState};
use std::cell::RefCell;

pub struct MonteCarloPolicyGradient<MoveToNeighbor> {
    pub stop_condition: StopCondition,
    phantom_neighbor: std::marker::PhantomData<MoveToNeighbor>,
    no_best_move: RefCell<bool>,
}

impl<MoveToNeighbor> MonteCarloPolicyGradient<MoveToNeighbor> {
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

impl<Problem, MoveToNeighbor> Heuristic<Problem> for MonteCarloPolicyGradient<MoveToNeighbor>
where
    Problem: ProblemTrait,
    for<'a> SearchState<'a, Problem>: EnumerateMoveToNeighbor<MoveToNeighbor>,
{
    fn run_once<'a>(
        &self,
        state: &mut SearchState<'a, Problem>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let mut best_move_option = None;
        for neighbor in state.iter_on_move_to_neighbor() {
            if !state.is_move_to_be_better_than_currernt(&neighbor) {
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
            state.apply(&best_move)?;
        } else {
            *self.no_best_move.borrow_mut() = true;
        }

        Ok(())
    }

    fn is_done<'a>(&self, state: &SearchState<'a, Problem>) -> bool {
        self.stop_condition.is_done(state) || *self.no_best_move.borrow()
    }
}
