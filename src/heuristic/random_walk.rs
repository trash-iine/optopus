use super::{Heuristic, StopCondition};
use crate::error::OptError;
use crate::search_state::{MoveToNeigbor, ProblemTrait, Rankable, SearchState};
use rand::seq::IteratorRandom;

/// Random walk heuristic.
///
/// At each iteration a uniformly random neighbor is selected and applied unconditionally,
/// regardless of whether it improves or worsens the current solution.
/// The best solution encountered during the walk is recorded in [`SearchState::best_solution`].
pub struct RandomWalk<N> {
    pub stop_condition: StopCondition,
    phantom_neighbor: std::marker::PhantomData<N>,
}

impl<N> RandomWalk<N> {
    /// Create a new [`RandomWalk`] with the given stopping condition.
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
    fn run_once<'a>(&mut self, state: &mut SearchState<'a, P>) -> Result<(), OptError> {
        let neighbor = N::iter(&state.instance, &state.solution)
            .choose(&mut rand::rng())
            .ok_or_else(|| OptError::InvalidState("No neighbor found".to_string()))?;

        state.apply(&neighbor)?;

        Ok(())
    }
}
