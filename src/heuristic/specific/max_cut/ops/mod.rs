//! The vertex-level search operators for MaxCut that no generic heuristic
//! covers, shared by every heuristic in this directory.
//!
//! Each is a free function over a [`SearchState`](crate::search_state::SearchState),
//! one module per role: [`tabu_walk`] walks through local optima and
//! [`perturbation`] kicks. They are independent — neither calls into the other
//! — and what they share is the state's own tabu memory, recorded by
//! [`apply`](crate::search_state::SearchState::apply) itself rather than by
//! each operator.
//!
//! **Neither the descent nor the directed swap is here.** Both used to be, as
//! `ops::descent` and `ops::best_swap`; Breakout Local Search now descends with
//! the generic [`LocalSearch`](crate::heuristic::LocalSearch) and swaps with a
//! [`TabuSearch`](crate::heuristic::TabuSearch) step on each side of
//! [`MaxCutSideFlipNeighbor`](crate::problem::MaxCutSideFlipNeighbor). See
//! below for what each cost.
//!
//! `keep_best` is the tie rule [`tabu_walk`] selects with; it stayed here
//! rather than moving into that module because the record of *why* ties keep
//! the lowest index is a measurement, not an implementation detail.
//!
//! That sharing is the point, and it is automatic here: a
//! [`TabuMemory`](crate::common::TabuMemory) keys its slots by the *shape* of a
//! [`TabuKey`](crate::common::TabuKey), and both [`MaxCutFlipNeighbor`] and
//! [`MaxCutSwapNeighbor`](crate::problem::MaxCutSwapNeighbor) key on
//! [`TabuKey::Var`](crate::common::TabuKey::Var) — so in Breakout Local Search
//! the entries the descent writes are exactly the entries the weak
//! perturbations must not undo. That is the paper's structure, not this
//! implementation's licence: in Benlic & Hao's Algorithm 1 the tabu list update
//! `H ← Iter + γ` sits *inside* the descent loop, and the same `H` is what
//! `Perturbation(C, L, H, Iter, ω)` is handed. The descent writes it and never
//! reads it — its own move selection takes the best applicable move, with no
//! diversification. A caller wanting the two phases isolated runs them on
//! separate states.
//!
//! What "tabu" *means* is decided nowhere here. Every operator marks and tests
//! moves through the moves' own [`EnabledTabu`](crate::trait_defs::EnabledTabu)
//! impls, so these operators forbid exactly what a generic
//! [`TabuSearch`](crate::heuristic::TabuSearch) over the same neighborhood
//! would; they only decide *which* moves to try.
//!
//! # What the descent cost to give up, and why the rest stays
//!
//! Sharing was never the reason these are hand-written. Once
//! [`apply`](crate::search_state::SearchState::apply) became what records a
//! move, `LocalSearch` writes the same memory an operator does and reads it
//! just as little. The reasons are narrower, and the descent's turned out to be
//! affordable while the other two are not.
//!
//! **The descent was integrated, at a measured price.** A specialised descent
//! enumerated only the improving flips through `MaxCutSolution`'s optional
//! `positive_gain` index, where `LocalSearch` rescans all `n` vertices per move
//! — both selecting from the same set, since
//! `is_neighbor_better_than_current` on a flip is `gain > 0`. One descent to a
//! local optimum takes **1.8-2.3x** as long generically (10 seeds, medians, on
//! G1 / G22 / G32 / G55 / G70 / G81 — flat across n from 800 to 20000 and
//! degree from 2 to 48), of which the scan is 87% and `update_best`'s per-move
//! clone the other 13%. Through a whole BLS run that is **-40.0 total average
//! cut** (G-set panel of ten, 30s x 5 runs, seed 42; better on 3, worse on 5)
//! and 0.74-0.93x as many moves. The 2x is the price of the generic contract,
//! not slack in it: `clone_from` in `update_best` changed nothing (`derive`
//! does not specialise it), folding the best update to the end of the descent
//! recovers only 8-10% and coarsens the anytime trajectory, and the same
//! traversal written as a hand loop ran **3.3x slower** than the iterator
//! chain. Cutting the scan needs to know which gains moved, which is problem
//! knowledge.
//!
//! **`best_swap` was replaced too — by restricting the neighborhood on the
//! right axis.** `TabuSearch<MaxCutSwapNeighbor>` is out because its `iter`
//! enumerates every cross-side pair, 4·10⁸ against 2·10⁴ per step on G81. The
//! first attempt at a restriction — a move type yielding only the pairs
//! touching each side's *highest-gain* vertex, O(n) and always containing the
//! greedy pair — **was built, measured and failed badly**: −452.6 total average
//! cut on the ten-instance G-set panel, 8 of 10 worse, at 0.05-0.36x the moves,
//! with `TabuSearch` reporting no eligible move 3.8x more often than it found
//! one.
//!
//! The cause was structural. Applying a move inverts the sign of its vertex's
//! gain, and this phase starts from a local optimum where every gain is ≤ 0 —
//! so the vertices a search just moved become the highest-gain ones, and they
//! are exactly the vertices the tabu memory now forbids. **Ranking a restricted
//! neighborhood by gain is anti-correlated with a recency-based tabu list**,
//! and widening to the top `k` per side does not escape it: `k` would have to
//! exceed the tenure (up to 600 on the sparse instances), and `k²` then
//! overtakes the `n` it was meant to replace.
//!
//! Restricting by **partition side** instead has no such correlation: a side
//! holds about n/2 vertices and loses only the handful recently moved. That is
//! [`MaxCutSideFlipNeighbor`](crate::problem::MaxCutSideFlipNeighbor), and a
//! [`TabuSearch`](crate::heuristic::TabuSearch) step on each side makes the
//! swap. Measured against the operator it replaced: **+20.8 total average cut**
//! (better on 5, worse on 4), and **not one** "no eligible move" on any
//! instance, against 5.6M on G1 alone for the gain-ranked attempt.
//!
//! Three details of the operator did not survive, and none of them cost the
//! panel: aspiration is now tested per flip rather than on the combined swap;
//! the second flip is chosen from gains already updated by the first, so the
//! `2·w(i, j)` correction happens by construction instead of by formula; and a
//! side with no free vertex now yields no move instead of breaking tabu (a
//! branch already recorded as paper-undefined and unreached on the G-set).
//!
//! **`random_flips` is free to replace and still here.** Measured the same way,
//! swapping it for [`RandomWalk`](crate::heuristic::RandomWalk) moved the total
//! by +0.4 with throughput unchanged. What keeps it is the edgeless
//! sub-instances [`SubProblemBasedCrossover`](crate::heuristic::SubProblemBasedCrossover)
//! produces, where `RandomWalk` fails with `InvalidState`: the guard it holds
//! would only move into the caller. It is now the only thing in
//! [`perturbation`], so replacing it would empty that module.

mod perturbation;
mod tabu_walk;

pub(super) use perturbation::random_flips;
pub(super) use tabu_walk::tabu_walk;

use crate::problem::max_cut::MaxCutFlipNeighbor;

/// Keeps `candidate` in `slot` when it beats what is already there.
///
/// Ties keep the incumbent, i.e. the first candidate the scan met, which on the
/// G-set means the lowest vertex index. That looks like an arbitrary bias —
/// every G-set weight is ±1, so gains are small integers and the degree-4
/// toroidal instances admit only five distinct values, putting hundreds of
/// vertices in one tie — but sampling the tie uniformly was **measured and
/// rejected**: over G11/G12/G13/G32-G34 it lost 6 cut points and turned three
/// exact matches of the paper's best into misses, while gaining only on one
/// planar instance. Index order on a toroidal grid tracks position, so taking
/// the lowest index walks the lattice coherently; randomising it scatters the
/// perturbation instead.
///
/// Every operator that picks a move selects with this, which is why the rule
/// lives here rather than in any one of them.
fn keep_best(slot: &mut Option<MaxCutFlipNeighbor>, candidate: MaxCutFlipNeighbor) {
    if slot.is_none_or(|best| candidate.gain > best.gain) {
        *slot = Some(candidate);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::OptError;
    use crate::heuristic::{Heuristic, LocalSearch, StopCondition, TabuSearch};
    use crate::problem::{MaxCut, MaxCutSideFlipNeighbor};
    use crate::search_state::SearchState;

    /// The shape every perturbation operator shares.
    type Op = fn(u64, &mut SearchState<'_, MaxCut>) -> Result<(), OptError>;

    /// The directed swap, as `BreakoutLocalSearch::kick` runs it: one
    /// `TabuSearch` step per side of the cut, `l` times over.
    fn best_swap_via_side_flips(
        l: u64,
        state: &mut SearchState<'_, MaxCut>,
    ) -> Result<(), OptError> {
        let cond = StopCondition::new(None, None, None);
        let mut on_true = TabuSearch::<MaxCutSideFlipNeighbor<true>>::new(cond.clone(), (3, 15));
        let mut on_false = TabuSearch::<MaxCutSideFlipNeighbor<false>>::new(cond, (3, 15));
        for _ in 0..l {
            on_true.run_once(state)?;
            on_false.run_once(state)?;
        }
        Ok(())
    }

    /// Prepares a state the way [`BreakoutLocalSearch`](super::super::bls::BreakoutLocalSearch)
    /// does before handing it to an operator: the tenure the records draw from,
    /// and a tabu map grown to the instance up front.
    pub(super) fn state_with_tabu(
        mc: &MaxCut,
        seed: u64,
        tenure: (u64, u64),
    ) -> SearchState<'_, MaxCut> {
        let mut state = SearchState::new_with_seed(mc, seed);
        state.set_tabu_tenure(tenure);
        state.reserve_tabu_vars(mc.graph.len());
        state
    }

    /// Builds a small toroidal-like graph (degree 4, unit weights) that has both
    /// partition sides populated throughout the search.
    pub(super) fn small_instance() -> MaxCut {
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
    ///
    /// The `zero_gain` index is not read by any operator here — it is
    /// maintained for
    /// [`PopulationAnnealing`](super::super::population_annealing::PopulationAnnealing) —
    /// so this is the only place its incremental updates are checked against a
    /// recomputation under these moves.
    #[test]
    fn mixed_perturbations_keep_gains_and_indexes_consistent() {
        let mc = small_instance();
        let mut state = state_with_tabu(&mc, 7, (3, 15));
        state.solution.enable_positive_gain_index();
        state.solution.enable_zero_gain_index();

        let schedule: [Op; 3] = [random_flips, tabu_walk, best_swap_via_side_flips];
        for round in 0..60 {
            for op in schedule {
                op(3, &mut state).unwrap();
            }
            // The descent between rounds is the generic `LocalSearch` — the
            // same one `BreakoutLocalSearch::descend` drives — so this checks
            // the indexes stay consistent under it too.
            LocalSearch::<MaxCutFlipNeighbor>::new(StopCondition::new(None, None, None))
                .run(&mut state)
                .unwrap();

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
}
