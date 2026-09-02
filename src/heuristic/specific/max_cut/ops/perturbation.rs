//! The random kick a heuristic can run once the descent has nothing left.
//!
//! The other two are elsewhere: the tabu walk is a walk in its own right and
//! lives in [`tabu_walk`](super::tabu_walk), and the directed swap is a
//! [`TabuSearch`](crate::heuristic::TabuSearch) step on each side of
//! [`MaxCutSideFlipNeighbor`](crate::problem::MaxCutSideFlipNeighbor).

use crate::error::OptError;
use crate::problem::MaxCut;
use crate::problem::max_cut::MaxCutFlipNeighbor;
use crate::search_state::SearchState;

/// Applies `l` random flip moves (the paper's *strong* perturbation).
///
/// Skips `update_best` per move; the caller updates best after the phase.
///
/// On a graph with no edged vertices (e.g. an empty sub-MaxCut extracted by
/// [`SubProblemBasedCrossover`](crate::heuristic::SubProblemBasedCrossover)
/// when the parents disagree only on an independent set) there is nothing to
/// flip, so this just advances the iteration counter — mirroring how
/// [`tabu_walk`](super::tabu_walk::tabu_walk) progresses when it finds no move
/// — so the outer stop condition
/// still terminates instead of the sampler panicking on an empty range.
pub(crate) fn random_flips(l: u64, state: &mut SearchState<'_, MaxCut>) -> Result<(), OptError> {
    if state.instance.graph.vertices.is_empty() {
        for _ in 0..l {
            state.progress_iteration();
        }
        return Ok(());
    }
    for _ in 0..l {
        let neighbor =
            MaxCutFlipNeighbor::random_neighbor(state.instance, &state.solution, &mut state.rng);

        state.apply_move_only(&neighbor)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::super::tests::state_with_tabu;
    use super::*;

    /// On a graph with no edged vertices — as produced by
    /// `SubProblemBasedCrossover` when the two parents disagree only on an
    /// independent set — the random-flip kick must advance iterations without
    /// panicking (`random_neighbor` samples an empty range).
    #[test]
    fn random_flips_progress_on_an_edgeless_graph() {
        let mc = MaxCut::new(crate::common::Graph::new());
        let mut state = state_with_tabu(&mc, 0, (3, 15));

        let before = state.iteration;
        random_flips(5, &mut state).unwrap();
        assert_eq!(state.iteration - before, 5);
    }
}
