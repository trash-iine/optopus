//! Gain-indexed greedy descent: the only operator here that is monotone.

use super::keep_best;
use crate::error::OptError;
use crate::problem::MaxCut;
use crate::problem::max_cut::MaxCutFlipNeighbor;
use crate::search_state::SearchState;

/// Runs greedy local search until no improving flip move exists; the state
/// records each applied move in its tabu memory.
///
/// Instead of scanning all `n` flip neighbors, this iterates only over vertices
/// currently in `solution.positive_gain` — every improving flip must have
/// strictly positive gain, so this set is a superset of the improving moves. On
/// G-set instances the set shrinks rapidly as the search approaches a local
/// optimum, turning the inner loop from O(n) into effectively
/// O(improving_moves).
pub(crate) fn descent(state: &mut SearchState<'_, MaxCut>) -> Result<(), OptError> {
    state.solution.enable_positive_gain_index();
    loop {
        let mut best_move_option = None;
        for &v in state.solution.positive_gain.as_slice() {
            let gain = state.solution.gain[v];
            keep_best(&mut best_move_option, MaxCutFlipNeighbor { i: v, gain });
        }

        if let Some(best_move) = best_move_option {
            state.apply_move_only_with_tabu(&best_move)?;
        } else {
            // The descent only ever takes strictly positive gains, so the
            // point it stops at is the best it passed through: one update
            // here is worth the same as one per move, and costs one clone
            // instead of one per improving move. It has to happen before
            // returning, because the weak perturbations read
            // `best_solution.objective` for their aspiration test.
            state.update_best();
            return Ok(());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::tests::{small_instance, state_with_tabu};
    use super::*;

    /// The descent must stop exactly at a local optimum — no vertex left with a
    /// positive flip gain — and leave the vertices it moved behind in the
    /// state's tabu memory, which is what stops the following perturbation
    /// undoing it.
    #[test]
    fn descent_reaches_a_local_optimum_and_fills_the_tabu_memory() {
        let mc = small_instance();
        let mut state = state_with_tabu(&mc, 3, (3, 15));

        let before = state.solution.objective;
        descent(&mut state).unwrap();

        assert!(state.solution.objective >= before, "descent must not lose");
        for v in 0..state.solution.x.len() {
            assert!(
                mc.calculate_gain(&state.solution.x, v) <= 0.0,
                "vertex {v} still improves, so this is not a local optimum"
            );
        }
        assert!(
            (0..state.solution.x.len())
                .any(|v| { !state.tabu_allows(&MaxCutFlipNeighbor::new(&mc, &state.solution, v)) }),
            "the moves it applied must be recorded"
        );
        assert_eq!(
            state.best_solution.objective, state.solution.objective,
            "the local optimum has to be published before returning"
        );
    }
}
