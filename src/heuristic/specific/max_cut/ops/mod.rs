//! The vertex-level search operators for MaxCut, shared by every heuristic in
//! this directory.
//!
//! Each operator is a free function over a [`Ledger`], one module per role:
//! [`descent`] walks downhill, [`tabu_walk`] walks through local optima, and
//! [`perturbation`] kicks. They are independent — none calls into another
//! except through the ledger — so what they share is visible in their
//! signatures rather than hidden in a receiver: whoever hands them the same
//! ledger is the one deciding they should see each other's prohibitions.
//!
//! That sharing is the point. In Breakout Local Search the entries the descent
//! writes are the entries the weak perturbations must not undo, so the two run
//! against one ledger; a caller wanting them isolated just passes two.
//!
//! What "tabu" *means* is decided nowhere here. Every operator marks and tests
//! moves through the moves' own [`EnabledTabu`](crate::trait_defs::EnabledTabu)
//! impls, so these operators forbid exactly what a generic
//! [`TabuSearch`](crate::heuristic::TabuSearch) over the same neighborhood
//! would; they only decide *which* moves to try.

mod descent;
mod perturbation;
mod tabu_walk;

pub(super) use descent::descent;
pub(super) use perturbation::{best_swap, random_flips};
pub(super) use tabu_walk::tabu_walk;

use crate::common::{TabuLedger, VecTabuMap};
use crate::problem::max_cut::MaxCutFlipNeighbor;

/// The prohibitions the operators read and write, and the tenure they record
/// with. MaxCut's variables are a dense `0..n`, hence the `Vec` backing.
pub(super) type Ledger = TabuLedger<VecTabuMap>;

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
    type Op = fn(&mut Ledger, u64, &mut SearchState<'_, MaxCut>) -> Result<(), OptError>;

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
        let mut tabu = Ledger::new((3, 15));
        let mut state = SearchState::new_with_seed(&mc, 7);
        tabu.ensure_capacity(mc.graph.len());
        state.solution.enable_positive_gain_index();
        state.solution.enable_zero_gain_index();

        let schedule: [Op; 3] = [random_flips, tabu_walk, best_swap];
        for round in 0..60 {
            for op in schedule {
                op(&mut tabu, 3, &mut state).unwrap();
            }
            descent(&mut tabu, &mut state).unwrap();

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
