//! The MaxCut search engine shared by every heuristic in this directory.
//!
//! [`MaxCutSearchOps`] bundles one tabu map with the operators that read and
//! write it: the gain-indexed greedy descent, the tabu walk, and the three
//! perturbations. They live together because they share that tabu state — the
//! entries the descent writes are the entries the weak perturbations consume —
//! not because any one heuristic owns them.
//!
//! What "tabu" *means* is not decided here. Every operator marks and tests
//! moves through the moves' own `EnabledTabu` impls, so the engine forbids
//! exactly what a generic `TabuSearch` over the same neighborhood would.

use crate::common::VecTabuMap;
use crate::error::OptError;
use crate::problem::max_cut::MaxCutFlipNeighbor;
use crate::problem::{MaxCut, MaxCutSwapNeighbor};
use crate::search_state::SearchState;
use crate::trait_defs::{EnabledTabu, MoveToNeighbor};

// The positive-gain index attached to `MaxCutSolution` lets the local-search
// phase skip the O(n) neighborhood scan: any improving flip must be a vertex
// with strictly positive gain, so we only need to iterate `positive_gain`.

/// Keeps `candidate` in `slot` when it beats what is already there.
///
/// Ties keep the incumbent, i.e. the first candidate the scan met, which on the
/// G-set means the lowest vertex index — see
/// [`weak_swap_perturbation`](MaxCutSearchOps::weak_swap_perturbation) for why
/// that is deliberate rather than an accident of iteration order.
fn keep_best(slot: &mut Option<MaxCutFlipNeighbor>, candidate: MaxCutFlipNeighbor) {
    if slot.is_none_or(|best| candidate.gain > best.gain) {
        *slot = Some(candidate);
    }
}

/// One of the perturbation operators [`MaxCutSearchOps`] can apply.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum PerturbationType {
    /// Strong perturbation: apply `l` random flip moves.
    Strong,
    /// Weak flip perturbation: run tabu search for `l` iterations.
    WeakFlip,
    /// Weak swap perturbation: apply `l` swap moves guided by the tabu map.
    WeakSwap,
}

/// The vertex-level search operators for MaxCut, sharing one tabu map: a
/// gain-indexed greedy descent, a tabu walk, and three perturbations.
///
/// The tabu state written during descent is the same state consumed by the
/// weak perturbations, which is why these operators live together rather than
/// as independent heuristics. Both heuristics in this directory drive them —
/// [`BreakoutLocalSearch`](super::bls::BreakoutLocalSearch) and
/// [`RlBreakoutLocalSearch`](super::rl_bls::RlBreakoutLocalSearch) — so the
/// name deliberately does not mention either of them.
pub(super) struct MaxCutSearchOps {
    /// Which vertices the operators must currently leave alone, and for how
    /// long. Every operator reads or writes it — that sharing is the reason
    /// they are one type.
    ///
    /// The policy on it is not defined here: the operators go through
    /// [`MaxCutFlipNeighbor`]'s and [`MaxCutSwapNeighbor`]'s own
    /// [`EnabledTabu`] impls, which is the same policy a generic
    /// [`TabuSearch`](crate::heuristic::TabuSearch) over MaxCut applies.
    tabu: VecTabuMap,
    /// Tenure range handed to those impls on every move.
    tenure: (u64, u64),
}

impl MaxCutSearchOps {
    pub(super) fn new(tabu_tenure: (u64, u64)) -> Self {
        Self {
            tabu: VecTabuMap::new(),
            tenure: tabu_tenure,
        }
    }

    /// Resets the tabu state (for a new episode).
    pub(super) fn clear(&mut self) {
        self.tabu.clear();
    }

    /// Ensures the tabu map is large enough for the given problem instance.
    pub(super) fn ensure_capacity(&mut self, n: usize) {
        self.tabu.ensure_capacity(n);
    }

    /// Runs greedy local search until no improving flip move exists,
    /// recording each applied move in the tabu vec.
    ///
    /// Instead of scanning all `n` flip neighbors, this iterates only over
    /// vertices currently in `solution.positive_gain` — every improving flip
    /// must have strictly positive gain, so this set is a superset of the
    /// improving moves. On G-set instances the set shrinks rapidly as the
    /// search approaches a local optimum, turning the inner loop from O(n)
    /// into effectively O(improving_moves).
    pub(super) fn descent(&mut self, state: &mut SearchState<'_, MaxCut>) -> Result<(), OptError> {
        state.solution.enable_positive_gain_index();
        loop {
            let mut best_move_option = None;
            for &v in state.solution.positive_gain.as_slice() {
                let gain = state.solution.gain[v];
                keep_best(&mut best_move_option, MaxCutFlipNeighbor { i: v, gain });
            }

            if let Some(best_move) = best_move_option {
                best_move.add_to_tabu_map(
                    &mut self.tabu,
                    state.iteration,
                    self.tenure,
                    &mut state.rng,
                );
                state.apply_move_only(&best_move)?;
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

    /// Applies the strong perturbation: `l` random flip moves.
    /// Skips `update_best` per move; caller updates best after the phase.
    ///
    /// On a graph with no edged vertices (e.g. an empty sub-MaxCut extracted by
    /// [`SubProblemBasedCrossover`](crate::heuristic::SubProblemBasedCrossover)
    /// when the parents disagree only on an independent set) there is nothing to
    /// flip, so this just advances the iteration counter — mirroring how
    /// `weak_flip_perturbation` progresses when it finds no move — so the outer
    /// stop condition still terminates instead of the sampler panicking on an
    /// empty range.
    pub(super) fn strong_perturbation(
        &mut self,
        l: u64,
        state: &mut SearchState<'_, MaxCut>,
    ) -> Result<(), OptError> {
        if state.instance.graph.vertices.is_empty() {
            for _ in 0..l {
                state.progress_iteration();
            }
            return Ok(());
        }
        for _ in 0..l {
            let neighbor = MaxCutFlipNeighbor::random_neighbor(
                state.instance,
                &state.solution,
                &mut state.rng,
            );

            neighbor.add_to_tabu_map(&mut self.tabu, state.iteration, self.tenure, &mut state.rng);
            state.apply_move_only(&neighbor)?;
        }
        Ok(())
    }

    /// Applies the weak flip perturbation: inline tabu search for `l` iterations.
    ///
    /// Uses the BLS tabu map directly and scalar best tracking to avoid the
    /// overhead of constructing a `TabuSearch` object and its per-iteration
    /// `filter_best` Vec allocation.
    pub(super) fn weak_flip_perturbation(
        &mut self,
        l: u64,
        state: &mut SearchState<'_, MaxCut>,
    ) -> Result<(), OptError> {
        let end_iter = state.iteration + l;
        while state.iteration < end_iter {
            let mut best = None;
            for neighbor in MaxCutFlipNeighbor::iter(state.instance, &state.solution) {
                let enabled = neighbor.is_move_enabled(&self.tabu, state.iteration);
                // Aspiration: accept a tabu move if it improves the global best.
                if !enabled
                    && neighbor.gain + state.solution.objective <= state.best_solution.objective
                {
                    continue;
                }
                keep_best(&mut best, neighbor);
            }
            // No eligible move: every flip is tabu and none aspires. BLS as
            // published leaves this open; see "Cases the original scheme leaves
            // open" in docs/heuristics/breakout_local_search.md.
            if let Some(best_move) = best {
                best_move.add_to_tabu_map(
                    &mut self.tabu,
                    state.iteration,
                    self.tenure,
                    &mut state.rng,
                );
                state.apply_move_only(&best_move)?;
            } else {
                state.progress_iteration();
            }
        }
        Ok(())
    }

    /// Applies the weak swap perturbation: `l` swap moves guided by the tabu map.
    ///
    /// This is Benlic & Hao's set `A2` — the highest-gain move of the `M2`
    /// operator that is *not tabu*, with a tabu move admitted only when the
    /// resulting swap would beat the global best (aspiration). `M2` takes one
    /// vertex per partition side, so a single scan tracks two candidates per
    /// side: the best non-tabu vertex, which is what `A2` normally selects,
    /// and the best vertex overall, which is consulted only for the aspiration
    /// test and as the fallback for a side that has no non-tabu vertex left.
    ///
    /// Uses scalar best tracking per side instead of collecting tied-best
    /// lists into Vecs.
    ///
    /// Ties keep the first candidate the scan meets, i.e. the lowest vertex
    /// index. That looks like an arbitrary bias — every G-set weight is ±1, so
    /// gains are small integers and the degree-4 toroidal instances admit only
    /// five distinct values, putting hundreds of vertices in one tie — but
    /// sampling the tie uniformly was **measured and rejected**: over
    /// G11/G12/G13/G32-G34 it lost 6 cut points and turned three exact matches
    /// of the paper's best into misses, while gaining only on one planar
    /// instance. Index order on a toroidal grid tracks position, so taking the
    /// lowest index walks the lattice coherently; randomising it scatters the
    /// perturbation instead.
    pub(super) fn weak_swap_perturbation(
        &mut self,
        l: u64,
        state: &mut SearchState<'_, MaxCut>,
    ) -> Result<(), OptError> {
        for _ in 0..l {
            let mut free_v0 = None;
            let mut free_v1 = None;
            let mut any_v0 = None;
            let mut any_v1 = None;

            for neighbor in MaxCutFlipNeighbor::iter(state.instance, &state.solution) {
                let on_side0 = state.solution.x[neighbor.i];

                keep_best(if on_side0 { &mut any_v0 } else { &mut any_v1 }, neighbor);

                if neighbor.is_move_enabled(&self.tabu, state.iteration) {
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
            let aspiration =
                MaxCutSwapNeighbor::new(state.instance, &state.solution, any0.i, any1.i);
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

            swap.add_to_tabu_map(&mut self.tabu, state.iteration, self.tenure, &mut state.rng);
            state.apply_move_only(&swap)?;
        }
        Ok(())
    }

    /// Applies the given perturbation type with strength `l`.
    pub(super) fn perturb(
        &mut self,
        perturbation_type: PerturbationType,
        l: u64,
        state: &mut SearchState<'_, MaxCut>,
    ) -> Result<(), OptError> {
        match perturbation_type {
            PerturbationType::Strong => self.strong_perturbation(l, state),
            PerturbationType::WeakFlip => self.weak_flip_perturbation(l, state),
            PerturbationType::WeakSwap => self.weak_swap_perturbation(l, state),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::problem::MaxCut;
    use crate::search_state::SearchState;

    /// Builds a small toroidal-like graph (degree 4, unit weights) that has both
    /// partition sides populated throughout the search.
    fn small_instance() -> MaxCut {
        let n = 30usize;
        let mut edges = Vec::new();
        for i in 0..n {
            edges.push((i, (i + 1) % n, 1.0));
            edges.push((i, (i + 2) % n, 1.0));
        }
        MaxCut::from_edges(edges)
    }

    /// Property test: after hundreds of mixed perturbations of all three types,
    /// the incrementally maintained gain vector and both gain indexes must
    /// agree with a from-scratch recomputation.
    #[test]
    fn mixed_perturbations_keep_gains_and_indexes_consistent() {
        use PerturbationType::*;
        let mc = small_instance();
        let mut ops = MaxCutSearchOps::new((3, 15));
        let mut state = SearchState::new_with_seed(&mc, 7);
        ops.ensure_capacity(mc.graph.len());
        state.solution.enable_positive_gain_index();
        state.solution.enable_zero_gain_index();

        let schedule = [Strong, WeakFlip, WeakSwap];
        for round in 0..60 {
            for &ptype in &schedule {
                ops.perturb(ptype, 3, &mut state).unwrap();
            }
            ops.descent(&mut state).unwrap();

            for v in 0..state.solution.x.len() {
                let expected = mc.calculate_gain(&state.solution.x, v);
                assert_eq!(
                    state.solution.gain[v], expected,
                    "gain[{v}] diverged after round {round}"
                );
                assert_eq!(
                    state.solution.positive_gain.contains(v),
                    expected > 0.0,
                    "positive_gain membership of {v} wrong after round {round}"
                );
                assert_eq!(
                    state.solution.zero_gain.contains(v),
                    expected == 0.0,
                    "zero_gain membership of {v} wrong after round {round}"
                );
            }
            let expected_objective = mc.calculate_cut_size(&state.solution.x);
            assert_eq!(
                state.solution.objective, expected_objective,
                "objective diverged after round {round}"
            );
        }
    }

    /// On a graph with no edged vertices — as produced by
    /// `SubProblemBasedCrossover` when the two parents disagree only on an
    /// independent set — the strong perturbation must advance iterations
    /// without panicking (`random_neighbor` samples an empty range).
    #[test]
    fn perturbations_progress_on_edgeless_graph() {
        let mc = MaxCut::new(crate::common::Graph::new());
        let mut ops = MaxCutSearchOps::new((3, 15));
        let mut state = SearchState::new_with_seed(&mc, 0);
        ops.ensure_capacity(mc.graph.len());
        let before = state.iteration;
        ops.strong_perturbation(5, &mut state).unwrap();
        assert_eq!(state.iteration - before, 5);
    }
}
