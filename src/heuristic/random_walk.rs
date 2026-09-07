use super::{Heuristic, StopCondition};
use crate::error::OptError;
use crate::search_state::SearchState;
use crate::trait_defs::{MoveToNeighbor, ProblemTrait, Rankable};

/// Random walk heuristic.
///
/// At each iteration a uniformly random neighbor is selected and applied unconditionally,
/// regardless of whether it improves or worsens the current solution.
/// The best solution encountered during the walk is recorded in [`SearchState::best_solution`].
///
/// An iteration that finds no move steps the counter and returns. An empty
/// neighborhood is a state a walk can legitimately be handed, a MaxCut
/// sub-problem with no edges among others, and it is not the walk's to
/// report as a failure. Stepping the counter is also what lets an outer
/// budget terminate, which returning without it would not.
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
        match N::random_neighbor(state.instance, &state.solution, &mut state.rng) {
            Some(neighbor) => state.apply(&neighbor),
            None => {
                state.progress_iteration();
                Ok(())
            }
        }
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

    /// A walk over a problem with nothing to move has to step its counter and
    /// return, not error. `SubProblemBasedCrossover` hands MaxCut an edgeless
    /// sub-problem whenever the two parents disagree only on an independent
    /// set, and a walk that failed there would take the whole search with it.
    /// Returning without the counter step is no better, since the budget below
    /// would never be reached.
    #[test]
    fn an_empty_neighborhood_steps_the_counter_rather_than_failing() {
        let mc = MaxCut::new(crate::common::Graph::new());
        let mut state = SearchState::new_with_seed(&mc, 1);
        let mut rw = RandomWalk::<MaxCutFlipNeighbor>::new(StopCondition::iterations(10));
        rw.run(&mut state)
            .expect("an empty neighborhood is not a failure");

        assert_eq!(state.iteration, 10);
        assert_eq!(state.n_accepted, 0, "nothing was applied");
        assert_eq!(state.n_rejected, 10);
    }
}
