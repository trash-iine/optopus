//! The tabu walk: a best-non-tabu descent that is allowed to go uphill.

use super::keep_best;
use crate::error::OptError;
use crate::problem::MaxCut;
use crate::problem::max_cut::MaxCutFlipNeighbor;
use crate::search_state::SearchState;
use crate::trait_defs::MoveToNeighbor;

/// Runs a tabu search over flip moves for `l` iterations against the state's
/// tabu memory.
///
/// The walk keeps whatever it lands on — which is what a perturbation wants,
/// and the reason this is not phrased as an improvement phase: it climbs out of
/// a local optimum and back in, and where its budget runs out is not where it
/// was best.
///
/// Uses scalar best tracking to avoid the overhead of constructing a
/// [`TabuSearch`](crate::heuristic::TabuSearch) object and its per-iteration
/// `filter_best` Vec allocation.
pub(crate) fn tabu_walk(l: u64, state: &mut SearchState<'_, MaxCut>) -> Result<(), OptError> {
    let end_iter = state.iteration + l;
    while state.iteration < end_iter {
        let mut best = None;
        for neighbor in MaxCutFlipNeighbor::iter(state.instance, &state.solution) {
            let enabled = state.tabu_allows(&neighbor);
            // Aspiration: accept a tabu move if it improves the global best.
            if !enabled && neighbor.gain + state.solution.objective <= state.best_solution.objective
            {
                continue;
            }
            keep_best(&mut best, neighbor);
        }
        // No eligible move: every flip is tabu and none aspires. BLS as
        // published leaves this open; see "Cases the original scheme leaves
        // open" in docs/heuristics/breakout_local_search.md.
        let Some(best_move) = best else {
            state.progress_iteration();
            continue;
        };
        state.apply_move_only(&best_move)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::super::tests::{small_instance, state_with_tabu};
    use super::*;

    /// A walk of `l` iterations must consume exactly `l` iterations, whether or
    /// not it found an eligible move on each of them.
    #[test]
    fn a_walk_spends_its_whole_budget() {
        let mc = small_instance();
        let mut state = state_with_tabu(&mc, 5, (3, 15));

        let before = state.iteration;
        tabu_walk(37, &mut state).unwrap();
        assert_eq!(state.iteration - before, 37);
    }

    /// The walk must respect the memory it writes: a move it just recorded
    /// cannot be the move it makes next.
    #[test]
    fn the_walk_does_not_immediately_repeat_a_move() {
        let mc = small_instance();
        let mut state = state_with_tabu(&mc, 4, (5, 5));

        let before = state.solution.x.clone();
        tabu_walk(2, &mut state).unwrap();
        let flipped: Vec<usize> = (0..before.len())
            .filter(|&v| before[v] != state.solution.x[v])
            .collect();
        assert_eq!(flipped.len(), 2, "two iterations must flip two vertices");
    }
}
