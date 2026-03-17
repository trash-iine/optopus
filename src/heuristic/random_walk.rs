use super::{Heuristic, StopCondition};
use crate::error::OptError;
use crate::search_state::{MoveToNeigbor, ProblemTrait, Rankable, SearchState};
use rand::seq::IteratorRandom;

pub struct RandomWalk<N> {
    pub stop_condition: StopCondition,
    phantom_neighbor: std::marker::PhantomData<N>,
}

impl<N> RandomWalk<N> {
    pub fn new(stop_condition: StopCondition) -> Self {
        Self {
            stop_condition,
            phantom_neighbor: std::marker::PhantomData,
        }
    }
}

impl<P, N> Heuristic<P> for RandomWalk<N>
where
    P: ProblemTrait,
    N: MoveToNeigbor<P> + Rankable,
{
    fn is_done<'a>(&self, state: &SearchState<'a, P>) -> bool {
        self.stop_condition.is_done(state)
    }
    fn run_once<'a>(
        &self,
        state: &mut SearchState<'a, P>,
    ) -> Result<(), OptError> {
        while !self.stop_condition.is_done(state) {
            let neighbor = N::iter(&state.instance, &state.solution)
                .choose(&mut rand::rng())
                .ok_or_else(|| OptError::InvalidState("No neighbor found".to_string()))?;

            state.apply(&neighbor)?;
        }

        Ok(())
    }
}
