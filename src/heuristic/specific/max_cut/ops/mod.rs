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
//! **The descent is not here.** It used to be, as `ops::descent`; Breakout
//! Local Search now descends with the generic
//! [`LocalSearch`](crate::heuristic::LocalSearch). See below for what that
//! cost.
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
//! **`best_swap` has no generic counterpart at all.**
//! `TabuSearch<MaxCutSwapNeighbor>` would enumerate O(n²) vertex pairs where
//! `best_swap` scans each partition side once, O(n) — 4·10⁸ against 2·10⁴ per
//! step on G81 — and its `A2` selection rule (aspiration tested on the
//! unrestricted best swap, a side with no free vertex falling back to its own
//! best) is not a tabu search's.
//!
//! **`random_flips` is free to replace and still here.** Measured the same way,
//! swapping it for [`RandomWalk`](crate::heuristic::RandomWalk) moved the total
//! by +0.4 with throughput unchanged. It deletes no file — [`perturbation`]
//! holds `best_swap` too — and `RandomWalk` fails with `InvalidState` on the
//! edgeless sub-instances
//! [`SubProblemBasedCrossover`](crate::heuristic::SubProblemBasedCrossover)
//! produces, so the guard it holds would only move into the caller.

mod perturbation;
mod tabu_walk;

pub(super) use perturbation::{best_swap, random_flips};
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
    use crate::heuristic::{Heuristic, LocalSearch, StopCondition};
    use crate::problem::MaxCut;
    use crate::search_state::SearchState;

    /// The shape every perturbation operator shares.
    type Op = fn(u64, &mut SearchState<'_, MaxCut>) -> Result<(), OptError>;

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

        let schedule: [Op; 3] = [random_flips, tabu_walk, best_swap];
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
