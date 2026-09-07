//! Breakout Local Search wired up for MaxCut.
//!
//! The search itself is generic
//! ([`BreakoutLocalSearch`](crate::heuristic::BreakoutLocalSearch)). What is
//! here is the three operators MaxCut fills its perturbation bank with, and the
//! reading of `tabu_tenure` that Benlic & Hao's schedule asks for.

use super::best_swap::BestSwap;
use crate::error::OptError;
use crate::heuristic::{
    AdaptivePerturbation, BreakoutLocalSearch, Heuristic, LocalSearch, RandomWalk, StopCondition,
    TabuSearch,
};
use crate::problem::{MaxCut, MaxCutFlipNeighbor};
use crate::search_state::SearchState;

/// Breakout Local Search on MaxCut, with Benlic & Hao's schedule.
pub type BreakoutLocalSearchForMaxCut = BreakoutLocalSearch<MaxCut, AdaptivePerturbation<MaxCut>>;

/// Breakout Local Search on MaxCut with no schedule, for a caller that steps
/// [`descend`](BreakoutLocalSearch::descend) and
/// [`kick`](BreakoutLocalSearch::kick) itself.
///
/// `()` does not implement
/// [`PerturbationSchedule`](crate::heuristic::PerturbationSchedule), so this is
/// not a [`Heuristic`] and `run_once` cannot be reached. A caller supplying
/// lines 5 to 7 of the framework itself has nothing for the loop to consult,
/// and the type says so.
pub type ExternallyDrivenBlsForMaxCut = BreakoutLocalSearch<MaxCut, ()>;

/// One of the perturbation operators, in Benlic & Hao's vocabulary.
///
/// A weak perturbation is directed by the gains and the tabu memory, a strong
/// one is random. The values are the positions the perturbation bank is filled
/// in, which is what [`kick`](BreakoutLocalSearch::kick) indexes with.
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

impl From<PerturbationType> for usize {
    fn from(p: PerturbationType) -> usize {
        p as usize
    }
}

/// A [`RandomWalk`] that steps the counter instead of failing when there is
/// nothing to flip.
///
/// [`SubProblemBasedCrossover`](crate::heuristic::SubProblemBasedCrossover)
/// builds an edgeless sub-MaxCut when the parents disagree only on an
/// independent set, and `MaxCutFlipNeighbor`'s sampler then has an empty range
/// to draw from. Advancing the counter keeps the outer stop condition
/// terminating. The weak flip needs no such guard: `TabuSearch` finds no move,
/// says so, and steps the counter itself.
struct RandomFlips {
    inner: RandomWalk<MaxCutFlipNeighbor>,
}

impl Heuristic<MaxCut> for RandomFlips {
    fn run_once<'a>(&mut self, state: &mut SearchState<'a, MaxCut>) -> Result<(), OptError> {
        if state.instance.graph.vertices.is_empty() {
            state.progress_iteration();
            return Ok(());
        }
        self.inner.run_once(state)
    }

    fn stop_condition(&self) -> &StopCondition {
        self.inner.stop_condition()
    }
}

/// The descent, Algorithm 1's line 4.
///
/// The library's own [`LocalSearch`]: it selects the best strictly improving
/// flip, and because recording is a mode the search turns on, its `apply`
/// writes the prohibitions the kick then reads. What ends a descent is the
/// local optimum it reports, not the budget, so the cap below is never reached.
/// An unreachable cap rather than an empty `StopCondition`, which says the same
/// thing and measured 7% slower. See decisions/0003.
fn descent() -> Box<dyn Heuristic<MaxCut>> {
    Box::new(LocalSearch::<MaxCutFlipNeighbor>::new(
        StopCondition::iterations(u64::MAX),
    ))
}

/// The three kicks, in the order [`PerturbationType`] numbers them.
///
/// Two of the three are generic heuristics. Only the directed swap is
/// hand-written, because `M2` moves one vertex per partition side in a single
/// move and a pair of independent one-step searches cannot express that.
///
/// The two that read the tabu memory carry the tenure themselves, as
/// `TabuSearch` does everywhere, so each turns recording on when it steps.
fn perturbation_bank(tabu_tenure: (u64, u64)) -> Vec<Box<dyn Heuristic<MaxCut>>> {
    // Never consulted: BLS steps each of these with `run_once` for exactly the
    // perturbation length.
    let unreachable = || StopCondition::iterations(u64::MAX);
    vec![
        Box::new(RandomFlips {
            inner: RandomWalk::<MaxCutFlipNeighbor>::new(unreachable()),
        }),
        Box::new(TabuSearch::<MaxCutFlipNeighbor>::new(
            unreachable(),
            tabu_tenure,
        )),
        Box::new(BestSwap::new(tabu_tenure)),
    ]
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
pub fn breakout_local_search_for_max_cut(
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
        descent(),
        perturbation_bank(effective),
        AdaptivePerturbation::new(t, l0, p0, q),
    )
}

/// A MaxCut BLS whose kicks the caller chooses, stepped through
/// [`descend`](BreakoutLocalSearch::descend) and
/// [`kick`](BreakoutLocalSearch::kick).
///
/// `tabu_tenure` is taken literally, without the doubling
/// [`breakout_local_search_for_max_cut`] applies: `2 gamma` is a property of
/// Benlic & Hao's perturbation rule, and a controller that replaces that rule
/// brings its own tenure.
///
/// # Panics
///
/// Panics if `tabu_tenure.0 > tabu_tenure.1`.
pub fn externally_driven_bls_for_max_cut(
    stop_condition: StopCondition,
    tabu_tenure: (u64, u64),
) -> ExternallyDrivenBlsForMaxCut {
    BreakoutLocalSearch::new(
        stop_condition,
        tabu_tenure,
        descent(),
        perturbation_bank(tabu_tenure),
        (),
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
    use crate::common::{Graph, seeded_rng};
    use crate::trait_defs::ProblemTrait;

    fn small_instance() -> MaxCut {
        let n = 30usize;
        let mut edges = Vec::new();
        for i in 0..n {
            edges.push((i, (i + 1) % n, 1.0));
            edges.push((i, (i + 2) % n, 1.0));
        }
        MaxCut::from_edges(edges)
    }

    fn bls(iterations: u64) -> BreakoutLocalSearchForMaxCut {
        breakout_local_search_for_max_cut(
            StopCondition::iterations(iterations),
            (5, 15),
            1_000,
            8,
            0.8,
            0.5,
        )
    }

    #[test]
    fn bls_improves_the_cut() {
        let mc = small_instance();
        for seed in 0..8 {
            let mut state = SearchState::new_with_seed(&mc, seed);
            let initial = state.solution.objective;
            bls(2_000).run(&mut state).unwrap();
            assert!(
                state.best_solution.objective >= initial,
                "seed {seed}: {} < {initial}",
                state.best_solution.objective
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

    /// The bank positions have to line up with the vocabulary, since the
    /// schedule returns an index and `PerturbationType` names one.
    #[test]
    fn the_bank_is_indexed_by_the_perturbation_vocabulary() {
        let bls = bls(1);
        assert_eq!(bls.num_perturbations(), 3);
        assert_eq!(usize::from(PerturbationType::Strong), 0);
        assert_eq!(usize::from(PerturbationType::WeakFlip), 1);
        assert_eq!(usize::from(PerturbationType::WeakSwap), 2);
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
            let mut bls = externally_driven_bls_for_max_cut(StopCondition::iterations(100), (3, 9));
            bls.descend(&mut state).unwrap();
            let before = state.iteration;
            bls.kick(&mut state, kind.into(), 4).unwrap();
            assert!(
                state.iteration > before,
                "{kind:?} left the counter where it was"
            );
        }
    }

    /// The edgeless sub-instance `SubProblemBasedCrossover` can build: the
    /// strong kick has nothing to sample, and must step the counter rather than
    /// fail.
    #[test]
    fn the_strong_kick_survives_an_edgeless_instance() {
        let mc = MaxCut::new(Graph::new());
        let mut rng = seeded_rng(1);
        let sol = mc.new_solution(&mut rng);
        let mut state = SearchState::with_solution_and_seed(&mc, sol, 1);
        let mut bls = externally_driven_bls_for_max_cut(StopCondition::iterations(10), (3, 9));
        let before = state.iteration;
        bls.kick(&mut state, PerturbationType::Strong.into(), 5)
            .unwrap();
        assert_eq!(state.iteration, before + 5);
    }
}
