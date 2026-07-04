use super::{Heuristic, StopCondition};
use crate::error::OptError;
use crate::search_state::SearchState;
use crate::trait_defs::{MoveToNeighbor, ProblemTrait, Rankable};

/// Random walk heuristic.
///
/// At each iteration a uniformly random neighbor is selected and applied unconditionally,
/// regardless of whether it improves or worsens the current solution.
/// The best solution encountered during the walk is recorded in [`SearchState::best_solution`].
pub struct RandomWalk<N> {
    pub stop_condition: StopCondition,
    _neighbor: std::marker::PhantomData<N>,
}

impl<N> RandomWalk<N> {
    /// Create a new [`RandomWalk`] with the given stopping condition.
    pub fn new(stop_condition: StopCondition) -> Self {
        Self {
            stop_condition,
            _neighbor: std::marker::PhantomData,
        }
    }
}

impl<P, N> Heuristic<P> for RandomWalk<N>
where
    P: ProblemTrait,
    N: MoveToNeighbor<P> + Rankable,
{
    fn stop_condition(&self) -> &StopCondition {
        &self.stop_condition
    }

    fn run_once<'a>(&mut self, state: &mut SearchState<'a, P>) -> Result<(), OptError> {
        let neighbor: N = state.random_neighbor("RandomWalk")?;
        state.apply(&neighbor)?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::problem::{MaxCut, MaxCutFlipNeighbor};

    #[test]
    fn random_walk_accepts_every_move() {
        let mc = MaxCut::from_edges([(0, 1, 1.0), (1, 2, 1.0), (0, 2, 1.0)]);
        let mut state = SearchState::new_with_seed(&mc, 42);
        let mut rw = RandomWalk::<MaxCutFlipNeighbor>::new(StopCondition::iterations(100));
        rw.run(&mut state).unwrap();

        assert_eq!(state.iteration, 100);
        assert_eq!(state.n_accepted, 100);
        assert_eq!(state.n_rejected, 0);
    }
}
