//! Two of the three kicks a heuristic can run once the descent has nothing
//! left: the random one and the directed swap. The third — the tabu walk — is
//! a walk in its own right and lives in [`tabu_walk`](super::tabu_walk).

use super::keep_best;
use crate::error::OptError;
use crate::problem::max_cut::MaxCutFlipNeighbor;
use crate::problem::{MaxCut, MaxCutSwapNeighbor};
use crate::search_state::SearchState;
use crate::trait_defs::MoveToNeighbor;

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

        state.apply_move_only_with_tabu(&neighbor)?;
    }
    Ok(())
}

/// Applies `l` swap moves guided by the state's tabu memory (the paper's
/// *weak swap*).
///
/// This is Benlic & Hao's set `A2` — the highest-gain move of the `M2`
/// operator that is *not tabu*, with a tabu move admitted only when the
/// resulting swap would beat the global best (aspiration). `M2` takes one
/// vertex per partition side, so a single scan tracks two candidates per
/// side: the best non-tabu vertex, which is what `A2` normally selects,
/// and the best vertex overall, which is consulted only for the aspiration
/// test and as the fallback for a side that has no non-tabu vertex left.
///
/// Uses scalar best tracking per side instead of collecting tied-best lists
/// into Vecs; the tie rule itself is [`keep_best`](super::keep_best), and it is
/// deliberate — see its measurement record.
pub(crate) fn best_swap(l: u64, state: &mut SearchState<'_, MaxCut>) -> Result<(), OptError> {
    for _ in 0..l {
        let mut free_v0 = None;
        let mut free_v1 = None;
        let mut any_v0 = None;
        let mut any_v1 = None;

        for neighbor in MaxCutFlipNeighbor::iter(state.instance, &state.solution) {
            let on_side0 = state.solution.x[neighbor.i];

            keep_best(if on_side0 { &mut any_v0 } else { &mut any_v1 }, neighbor);

            if state.tabu_allows(&neighbor) {
                keep_best(if on_side0 { &mut free_v0 } else { &mut free_v1 }, neighbor);
            }
        }

        let (Some(any0), Some(any1)) = (any_v0, any_v1) else {
            // One side is empty, so no swap exists. Two counter steps match
            // the swap's `+2` accounting. Paper-undefined; see
            // docs/heuristics/breakout_local_search.md.
            state.progress_iteration();
            state.progress_iteration();
            continue;
        };

        // Aspiration is tested on the unrestricted best swap: that is the
        // only way a tabu vertex may enter `A2`.
        let aspiration = MaxCutSwapNeighbor::new(state.instance, &state.solution, any0.i, any1.i);
        let swap = if state.is_neighbor_better_than_best(&aspiration) {
            aspiration
        } else {
            // A side with no non-tabu vertex falls back to its best one,
            // breaking tabu without aspiration. Paper-undefined and
            // unreachable on the G-set; see
            // docs/heuristics/breakout_local_search.md.
            let i = free_v0.map_or(any0.i, |b| b.i);
            let j = free_v1.map_or(any1.i, |b| b.i);
            MaxCutSwapNeighbor::new(state.instance, &state.solution, i, j)
        };

        state.apply_move_only_with_tabu(&swap)?;
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

    /// A swap moves one vertex per side, so it must leave the partition sizes
    /// untouched — that is the whole reason `M2` exists next to the flips.
    #[test]
    fn a_swap_keeps_the_partition_sizes() {
        let mc = super::super::tests::small_instance();
        let mut state = state_with_tabu(&mc, 2, (3, 15));

        let side0 = |x: &[bool]| x.iter().filter(|&&b| b).count();
        let before = side0(&state.solution.x);
        best_swap(4, &mut state).unwrap();
        assert_eq!(side0(&state.solution.x), before);
    }

    /// A swap moves one vertex per side in a single move, so the partition
    /// sizes survive even when one side has nothing free left.
    ///
    /// This pins the case that broke a generalisation of this operator into two
    /// independent one-step tabu searches: those succeed or fail separately, so
    /// a side with no eligible move left the other vertex moved on its own.
    /// Two vertices, one unit edge, `[true, false]`, vertex 1 forbidden.
    #[test]
    fn a_swap_keeps_the_partition_sizes_even_with_a_side_fully_tabu() {
        let mc = MaxCut::from_edges([(0, 1, 1.0)]);
        let sol = crate::problem::MaxCutSolution::new_from_assignment(&mc, vec![true, false]);
        let mut state = SearchState::with_solution_and_seed(&mc, sol, 1);
        state.set_tabu_tenure((50, 50));
        state.reserve_tabu_vars(2);

        // Forbid vertex 1, which is the whole of the `false` side.
        state.record_tabu(&MaxCutFlipNeighbor::new(&mc, &state.solution, 1));
        assert!(!state.tabu_allows(&MaxCutFlipNeighbor::new(&mc, &state.solution, 1)));

        let on_true = |x: &[bool]| x.iter().filter(|&&b| b).count();
        let before = on_true(&state.solution.x);
        best_swap(1, &mut state).unwrap();
        assert_eq!(
            on_true(&state.solution.x),
            before,
            "a fully tabu side must not leave the swap half-applied"
        );
    }
}
