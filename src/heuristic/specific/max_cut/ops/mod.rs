//! The vertex-level search operators for MaxCut, shared by every heuristic in
//! this directory.
//!
//! Each operator is a free function over a [`SearchState`](crate::search_state::SearchState),
//! one module per role: [`descent`] walks downhill, [`tabu_walk`] walks through
//! local optima, and [`perturbation`] kicks. They are independent — none calls
//! into another — and what they share is the state's own tabu memory, recorded
//! by [`apply`](crate::search_state::SearchState::apply) itself rather than by
//! each operator.
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
//! # Why these are not the generic heuristics
//!
//! Sharing is *not* the answer to that question. Once
//! [`apply`](crate::search_state::SearchState::apply) became what records a
//! move, [`LocalSearch`](crate::heuristic::LocalSearch) and
//! [`RandomWalk`](crate::heuristic::RandomWalk) write the same memory these
//! operators do, and neither reads it — exactly as the paper's descent does.
//! The reasons are narrower, and two of them were measured (BLS, 30s x 5 runs,
//! seed 42, ten G-set instances, `l0 = 0.01|V|`, density-scaled tenure):
//!
//! - **[`descent`] against `LocalSearch`.** Both take the same candidate set —
//!   `is_neighbor_better_than_current` on a flip is `gain > 0`, which is what
//!   `positive_gain` indexes — but `LocalSearch` rescans all `n` vertices per
//!   move where [`descent`] enumerates only the improving ones. In the same
//!   wall clock it made **0.74-0.93x as many moves** on all ten instances, for
//!   **-40.0 total average cut** (better on 3, worse on 5): -21.2 on G81,
//!   -12.4 on G63, -8.4 on G70, -6.4 on G55, against +8.2 on G60. Two further
//!   differences ride along and are not separated by that number: `max_by`
//!   breaks gain ties toward the *last* candidate where [`keep_best`] keeps the
//!   first, and `LocalSearch` spends one extra `progress_iteration` per
//!   descent.
//! - **`random_flips` against `RandomWalk`.** Measured the same way, this one
//!   is **free** — swapping it on top of the above moved the total by +0.4 with
//!   throughput unchanged. It stays anyway: it cannot delete
//!   [`perturbation`] (which holds `best_swap` too), and `RandomWalk` fails
//!   with `InvalidState` on the edgeless sub-instances
//!   [`SubProblemBasedCrossover`](crate::heuristic::SubProblemBasedCrossover)
//!   produces, so the guard would only move into the caller.
//! - **`best_swap` has no generic counterpart.** `TabuSearch<MaxCutSwapNeighbor>`
//!   would enumerate O(n^2) vertex pairs where `best_swap` scans each partition
//!   side once, O(n) — 4e8 against 2e4 per step on G81.
//! - **`LocalSearch` cannot express this descent as constructed.** Its
//!   `new` forces `max_failed_update = Some(1)`, and after a kick the solution
//!   is worse than the global best, so `Heuristic::run` returns before taking a
//!   move. Reaching the phase BLS wants means clearing that field by hand.

mod descent;
mod perturbation;
mod tabu_walk;

pub(super) use descent::descent;
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
            descent(&mut state).unwrap();

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
