use super::{Heuristic, StopCondition};
use crate::search_state::{EnumerateMoveToNeighbor, ProblemTrait, SearchState};
use rand::seq::IteratorRandom;

pub struct RandomWalk<MoveToNeighbor> {
    pub stop_condition: StopCondition,
    phantom_neighbor: std::marker::PhantomData<MoveToNeighbor>,
}

impl<MoveToNeighbor> RandomWalk<MoveToNeighbor> {
    pub fn new(stop_condition: StopCondition) -> Self {
        Self {
            stop_condition,
            phantom_neighbor: std::marker::PhantomData,
        }
    }
}

impl<Problem, MoveToNeighbor> Heuristic<Problem> for RandomWalk<MoveToNeighbor>
where
    Problem: ProblemTrait,
    for<'a> SearchState<'a, Problem>: EnumerateMoveToNeighbor<MoveToNeighbor>,
{
    fn is_done<'a>(&self, state: &SearchState<'a, Problem>) -> bool {
        self.stop_condition.is_done(state)
    }
    fn run_once<'a>(
        &self,
        state: &mut SearchState<'a, Problem>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        while !self.stop_condition.is_done(state) {
            let neighbor = state
                .iter_on_move_to_neighbor()
                .choose(&mut rand::rng())
                .ok_or("No neighbor found")?;

            state.apply(&neighbor);
        }

        Ok(())
    }
}
