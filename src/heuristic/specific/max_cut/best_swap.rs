//! The directed swap kick, which moves one vertex per partition side in a
//! single move.
//!
//! It is the one perturbation here with no generic equivalent in the library,
//! because a pair of independent one-step searches cannot express that single
//! move. They succeed or fail separately, so a side with nothing eligible
//! leaves the other vertex moved on its own, which the last test below pins.
//! The other two kicks are generic heuristics, a
//! [`RandomWalk`](crate::heuristic::RandomWalk) for the strong one and a
//! [`TabuSearch`](crate::heuristic::TabuSearch) for the weak flip.

use crate::error::OptError;
use crate::heuristic::{Heuristic, StopCondition};
use crate::problem::max_cut::MaxCutFlipNeighbor;
use crate::problem::{MaxCut, MaxCutSwapNeighbor};
use crate::search_state::SearchState;
use crate::trait_defs::MoveToNeighbor;

/// The directed swap as a one-step heuristic, so Breakout Local Search can hold
/// it in the same bank as the two generic kicks.
///
/// One `run_once` is one swap, two flips and two iterations, which is what the
/// perturbation length counts.
pub(crate) struct BestSwap {
    stop_condition: StopCondition,
    tabu_tenure: (u64, u64),
}

impl BestSwap {
    pub(crate) fn new(tabu_tenure: (u64, u64)) -> Self {
        Self {
            // Never consulted: BLS steps this with `run_once` for exactly the
            // perturbation length.
            stop_condition: StopCondition::iterations(u64::MAX),
            tabu_tenure,
        }
    }
}

impl Heuristic<MaxCut> for BestSwap {
    /// A swap takes one vertex per partition side, and this applies the
    /// highest-gain such swap that is not tabu, admitting a tabu one only when
    /// the result would beat the global best (aspiration). Benlic & Hao call
    /// the operator `M2` and the eligible set `A2`.
    ///
    /// A single scan therefore tracks two candidates per side, the best
    /// non-tabu vertex, which is what normally gets picked, and the best
    /// vertex overall, which is consulted only for the aspiration test and as
    /// the fallback for a side that has no non-tabu vertex left.
    ///
    /// Tracks a scalar best per side rather than collecting tied-best lists into
    /// Vecs. Among candidates of equal gain it keeps the first one the scan met.
    fn run_once<'a>(&mut self, state: &mut SearchState<'a, MaxCut>) -> Result<(), OptError> {
        // As `TabuSearch` does, this operator reads and writes the tabu memory,
        // so it turns recording on itself rather than trusting the caller.
        state.start_record_tabu(self.tabu_tenure);

        let mut free_v0 = None;
        let mut free_v1 = None;
        let mut any_v0 = None;
        let mut any_v1 = None;

        for neighbor in MaxCutFlipNeighbor::iter(state.instance, &state.solution) {
            // The comparisons below are `>` and not `>=`, so equal gains keep the
            // lowest vertex index. Sampling ties uniformly was measured and lost;
            // see decisions/0005.
            let on_side0 = state.solution.x[neighbor.i];

            let any = if on_side0 { &mut any_v0 } else { &mut any_v1 };
            if any.is_none_or(|best: MaxCutFlipNeighbor| neighbor.gain > best.gain) {
                *any = Some(neighbor);
            }

            if state.tabu_allows(&neighbor) {
                let free = if on_side0 { &mut free_v0 } else { &mut free_v1 };
                if free.is_none_or(|best: MaxCutFlipNeighbor| neighbor.gain > best.gain) {
                    *free = Some(neighbor);
                }
            }
        }

        let (Some(any0), Some(any1)) = (any_v0, any_v1) else {
            // One side is empty, so no swap exists. Two counter steps match
            // the swap's `+2` accounting. Paper-undefined; see
            // docs/heuristics/breakout_local_search.md.
            state.progress_iteration();
            state.progress_iteration();
            return Ok(());
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

        state.apply_move_only(&swap)
    }

    fn stop_condition(&self) -> &StopCondition {
        &self.stop_condition
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::heuristic::{LocalSearch, RandomWalk};
    use crate::problem::max_cut::test_fixtures::small_instance;

    /// Prepares a state the way
    /// [`BreakoutLocalSearch`](crate::heuristic::BreakoutLocalSearch) does
    /// before handing it to an operator: the tenure the records draw from, and
    /// a tabu map grown to the instance up front.
    fn state_with_tabu(mc: &MaxCut, seed: u64, tenure: (u64, u64)) -> SearchState<'_, MaxCut> {
        let mut state = SearchState::new_with_seed(mc, seed);
        state.reserve_tabu_vars(mc.graph.len());
        state.start_record_tabu(tenure);
        state
    }

    /// Property test: after hundreds of rounds mixing random flips, directed
    /// swaps and a descent, the incrementally maintained gain vector and
    /// objective must still agree with a from-scratch recomputation.
    ///
    /// Every move here updates `gain` in O(degree) rather than recomputing it,
    /// and the swap selects on those gains, so a drift would be silent.
    #[test]
    fn mixed_perturbations_keep_gains_consistent() {
        let mc = small_instance();
        let mut state = state_with_tabu(&mc, 7, (3, 15));

        for round in 0..60 {
            let budget = state.iterations_this_run() + 3;
            RandomWalk::<MaxCutFlipNeighbor>::new(StopCondition::iterations(budget))
                .run(&mut state)
                .unwrap();
            for _ in 0..3 {
                BestSwap::new((3, 15)).run_once(&mut state).unwrap();
            }
            LocalSearch::<MaxCutFlipNeighbor>::new(StopCondition::iterations(u64::MAX))
                .run(&mut state)
                .unwrap();

            for v in 0..state.solution.x.len() {
                let expected = mc.calculate_gain(&state.solution.x, v);
                assert_eq!(
                    state.solution.gain[v], expected,
                    "gain[{v}] diverged after round {round}"
                );
            }
            let expected_objective = mc.calculate_cut_size(&state.solution.x);
            assert_eq!(
                state.solution.objective, expected_objective,
                "objective diverged after round {round}"
            );
        }
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
        // A tenure long enough that the prohibition written by hand below
        // outlives the one swap this test takes.
        state.start_record_tabu((50, 50));
        state.reserve_tabu_vars(2);

        // Forbid vertex 1, which is the whole of the `false` side.
        state.record_tabu(&MaxCutFlipNeighbor::new(&mc, &state.solution, 1));
        assert!(!state.tabu_allows(&MaxCutFlipNeighbor::new(&mc, &state.solution, 1)));

        let on_true = |x: &[bool]| x.iter().filter(|&&b| b).count();
        let before = on_true(&state.solution.x);
        BestSwap::new((50, 50)).run_once(&mut state).unwrap();
        assert_eq!(
            on_true(&state.solution.x),
            before,
            "a fully tabu side must not leave the swap half-applied"
        );
    }
}
