//! Breakout Local Search wired up for MaxCut.
//!
//! The search itself is generic
//! ([`BreakoutLocalSearch`](crate::heuristic::BreakoutLocalSearch)). What is
//! here is the three operators MaxCut fills its perturbation bank with, and the
//! reading of `tabu_tenure` that Benlic & Hao's schedule asks for.

use super::best_swap::BestSwap;
use crate::heuristic::{
    AdaptivePerturbation, BreakoutLocalSearch, Heuristic, LocalSearch, RandomWalk, StopCondition,
    TabuSearch,
};
use crate::problem::{MaxCut, MaxCutFlipNeighbor};

/// Breakout Local Search on MaxCut, with Benlic & Hao's schedule.
pub type BreakoutLocalSearchForMaxCut = BreakoutLocalSearch<MaxCut, AdaptivePerturbation<MaxCut>>;

/// One of the perturbation operators, in Benlic & Hao's vocabulary.
///
/// A weak perturbation is directed by the gains and the tabu memory, a strong
/// one is random. This names which one to build, and is what
/// [`max_cut_perturbation`] takes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PerturbationType {
    /// Random flip moves, ignoring gains.
    Strong = 0,
    /// Tabu flip moves, each taking the best non-tabu move or one that
    /// satisfies the aspiration rule.
    WeakFlip = 1,
    /// Tabu swap moves, each taking the best non-tabu vertex per partition
    /// side, or one that satisfies the aspiration rule.
    WeakSwap = 2,
}

/// The descent.
///
/// The library's own [`LocalSearch`]: it selects the best strictly improving
/// flip, and because recording is a mode the search turns on, its `apply`
/// writes the prohibitions the kick then reads. What ends a descent is the
/// local optimum it reports, not the budget, so the cap below is never reached.
/// An unreachable cap rather than an empty `StopCondition`, which says the same
/// thing and measured 7% slower. See decisions/0003.
pub fn max_cut_descent() -> Box<dyn Heuristic<MaxCut>> {
    Box::new(LocalSearch::<MaxCutFlipNeighbor>::new(
        StopCondition::iterations(u64::MAX),
    ))
}

/// One of MaxCut's three kicks, by name.
///
/// Two of the three are generic heuristics. Only the directed swap is
/// hand-written, because `M2` moves one vertex per partition side in a single
/// move and a pair of independent one-step searches cannot express that.
///
/// The two that read the tabu memory carry the tenure themselves, as
/// `TabuSearch` does everywhere, so each turns recording on when it steps.
///
/// A schedule of your own builds the operators it wants from here. Asking by
/// name is the whole interface, so there is no bank order to agree on with
/// anyone.
pub fn max_cut_perturbation(
    kind: PerturbationType,
    tabu_tenure: (u64, u64),
) -> Box<dyn Heuristic<MaxCut>> {
    // Never consulted: BLS steps each of these with `run_once` for exactly the
    // perturbation length.
    let unreachable = StopCondition::iterations(u64::MAX);
    match kind {
        PerturbationType::Strong => Box::new(RandomWalk::<MaxCutFlipNeighbor>::new(unreachable)),
        PerturbationType::WeakFlip => Box::new(TabuSearch::<MaxCutFlipNeighbor>::new(
            unreachable,
            tabu_tenure,
        )),
        PerturbationType::WeakSwap => Box::new(BestSwap::new(tabu_tenure)),
    }
}

/// Breakout Local Search on MaxCut, with Benlic & Hao's `omega` / `l` schedule.
///
/// # Parameters
///
/// - `tabu_tenure`, the paper's `gamma`, doubled on the way in so a vertex
///   stays forbidden for `2 gamma` as the paper has it
/// - `t`, period of the `omega` counter before it resets
/// - `l0`, initial perturbation length
/// - `p0`, minimum probability of a directed perturbation
/// - `q`, fraction of directed perturbations that flip rather than swap
///
/// # Panics
///
/// Panics if `tabu_tenure.0 > tabu_tenure.1` (an empty range). The range is
/// only sampled from once the search is running, so it is checked here.
pub fn bls_for_max_cut(
    stop_condition: StopCondition,
    tabu_tenure: (u64, u64),
    t: u64,
    l0: u64,
    p0: f64,
    q: f64,
) -> BreakoutLocalSearchForMaxCut {
    let effective = paper_effective_tenure(tabu_tenure);
    BreakoutLocalSearch::new(
        stop_condition,
        effective,
        max_cut_descent(),
        AdaptivePerturbation::new(
            t,
            l0,
            p0,
            max_cut_perturbation(PerturbationType::Strong, effective),
            vec![
                (
                    max_cut_perturbation(PerturbationType::WeakFlip, effective),
                    q,
                ),
                (
                    max_cut_perturbation(PerturbationType::WeakSwap, effective),
                    1.0 - q,
                ),
            ],
        ),
    )
}

/// Converts Benlic & Hao's tenure parameter `gamma` into the prohibition length
/// the engine's tabu map actually stores.
///
/// The paper's tabu list `H` holds "the iteration when the vertex was last
/// moved plus gamma", and the eligibility predicate of the directed
/// perturbations then asks for `(H_m + gamma) < Iter`, so `gamma` is counted
/// twice and a vertex stays forbidden for `2 gamma`.
/// [`TabuMemory`](crate::common::TabuMemory) stores the first iteration at which
/// a move is allowed again, one tenure exactly, so reproducing the paper means
/// handing it twice the caller's range. `tabu_tenure` therefore keeps the
/// paper's meaning, `rand[3, |V|/10]` on the G-set, instead of silently meaning
/// something else.
///
/// Doubling only the upper bound does not reproduce the paper. The whole range
/// has to scale.
fn paper_effective_tenure((min, max): (u64, u64)) -> (u64, u64) {
    (min * 2, max * 2)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::Graph;
    use crate::problem::max_cut::test_fixtures::small_instance;
    use crate::search_state::SearchState;

    fn bls(iterations: u64) -> BreakoutLocalSearchForMaxCut {
        bls_for_max_cut(
            StopCondition::iterations(iterations),
            (5, 15),
            1_000,
            8,
            0.8,
            0.5,
        )
    }

    /// Seeded per iteration rather than drawn from the OS, so a failure here
    /// is reproducible, and claiming the search leaves its own starting point
    /// behind rather than merely reaching a positive cut, which any assignment
    /// on this instance does.
    #[test]
    fn bls_improves_the_cut() {
        let mc = small_instance();
        for seed in 0..8 {
            let mut state = SearchState::new_with_seed(&mc, seed);
            let initial = state.solution.objective;
            bls(2_000).run(&mut state).unwrap();
            assert!(
                state.best_solution.objective > initial,
                "seed {seed}: BLS did not improve on its start ({initial})"
            );
        }
    }

    #[test]
    fn seeded_runs_are_deterministic() {
        let mc = small_instance();
        let run = || {
            let mut state = SearchState::new_with_seed(&mc, 42);
            bls(1_500).run(&mut state).unwrap();
            (
                state.best_solution.objective,
                state.best_iteration,
                state.best_solution.x.clone(),
            )
        };
        assert_eq!(run(), run());
    }

    /// Each kick has to move the assignment or, where there is nothing to move,
    /// still advance the counter so an outer budget terminates.
    #[test]
    fn every_kick_advances_the_search() {
        let mc = small_instance();
        for kind in [
            PerturbationType::Strong,
            PerturbationType::WeakFlip,
            PerturbationType::WeakSwap,
        ] {
            let mut state = SearchState::new_with_seed(&mc, 3);
            state.start_record_tabu((3, 9));
            let mut op = max_cut_perturbation(kind, (3, 9));
            let before = state.iteration;
            for _ in 0..4 {
                op.run_once(&mut state).unwrap();
            }
            assert!(
                state.iteration > before,
                "{kind:?} left the counter where it was"
            );
        }
    }

    /// The edgeless sub-instance `SubProblemBasedCrossover` can build. The
    /// strong kick has nothing to sample and must step the counter rather than
    /// fail, and the whole loop over the same graph must terminate, which
    /// additionally puts the descent through the empty neighborhood.
    #[test]
    fn an_edgeless_graph_neither_stalls_the_kick_nor_the_loop() {
        let mc = MaxCut::new(Graph::new());
        let mut state = SearchState::new_with_seed(&mc, 1);
        let mut strong = max_cut_perturbation(PerturbationType::Strong, (3, 9));
        let before = state.iteration;
        for _ in 0..5 {
            strong.run_once(&mut state).unwrap();
        }
        assert_eq!(state.iteration, before + 5);

        // `LocalSearch` has to raise `no_best_move` on the empty neighborhood
        // rather than spin, or this never returns.
        let mut whole = bls_for_max_cut(
            StopCondition::iterations(10_000).with_failed_updates(500),
            (3, 15),
            1_000,
            5,
            0.8,
            0.5,
        );
        whole
            .run(&mut state)
            .expect("BLS must terminate on an edgeless graph");
        assert!(state.iteration > before + 5, "the loop charged nothing");
    }
}
