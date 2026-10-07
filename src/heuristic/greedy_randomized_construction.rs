//! The construction phase of GRASP over any [`Ruinable`] problem.

use crate::building_blocks::search::randomized_greedy_insertion;
use crate::error::OptError;
use crate::heuristic::{Heuristic, StopCondition};
use crate::search_state::SearchState;
use crate::trait_defs::{Evaluate, Ruinable};

/// Builds one new solution from nothing by GRASP's randomized greedy rule, and
/// replaces the incumbent with it.
///
/// Every element is taken out, and the empty solution is filled again one
/// element at a time, each drawn from the restricted candidate list of the
/// ones whose cheapest placement is within `alpha` of the cheapest of all.
/// See [`randomized_greedy_insertion`].
///
/// It is one step, not a search. A run builds exactly one solution and stops,
/// so it is meant to sit inside a composition. GRASP itself is
///
/// ```text
/// Iterated { search: LocalSearch, perturbation: GreedyRandomizedConstruction }
/// ```
///
/// since `Iterated` alternates a descent with this restart and keeps the best
/// of every cycle, which is the whole of the procedure Feo and Resende state.
/// No GRASP type exists for that reason.
///
/// # References
///
/// - Feo, T. A. and Resende, M. G. C. "Greedy randomized adaptive search
///   procedures." *Journal of Global Optimization*, 6(2), 109-133, 1995.
///   [DOI](https://doi.org/10.1007/BF01096763)
pub struct GreedyRandomizedConstruction<P: Ruinable> {
    stop_condition: StopCondition,
    alpha: f64,
    built: bool,
    scratch: Vec<P::Element>,
}

impl<P: Ruinable> GreedyRandomizedConstruction<P> {
    /// `alpha` in `[0, 1]` sets how greedy the construction is, `0` taking
    /// the cheapest element every time and `1` any element.
    ///
    /// The stop condition can end a run before its one construction, an
    /// iteration budget of zero for instance, but never makes it build more.
    ///
    /// # Panics
    ///
    /// Panics if `alpha` is outside `[0, 1]`.
    pub fn new(stop_condition: StopCondition, alpha: f64) -> Self {
        assert!((0.0..=1.0).contains(&alpha), "alpha must be in [0, 1]");
        Self {
            stop_condition,
            alpha,
            built: false,
            scratch: Vec::new(),
        }
    }
}

impl<P> Heuristic<P> for GreedyRandomizedConstruction<P>
where
    P: Ruinable,
    P::Solution: Evaluate,
{
    fn clear(&mut self) {
        self.built = false;
    }

    fn stop_condition(&self) -> &StopCondition {
        &self.stop_condition
    }

    /// Done once the one solution is built.
    fn is_done<'a>(&self, state: &SearchState<'a, P>) -> bool {
        self.built || self.stop_condition.is_done(state)
    }

    fn run_once<'a>(&mut self, state: &mut SearchState<'a, P>) -> Result<(), OptError> {
        let prob: &P = state.instance;
        let mut partial = prob.to_partial(&state.solution);
        prob.elements(&partial, &mut self.scratch);
        let all = std::mem::take(&mut self.scratch);
        prob.remove_all(&mut partial, &all);
        randomized_greedy_insertion(prob, &mut partial, all, self.alpha, &mut state.rng);
        state.solution = prob.finish(&partial);
        state.iteration += 1;
        state.n_accepted += 1;
        state.update_best();
        self.built = true;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::heuristic::{Iterated, LocalSearch};
    use crate::problem::{BinPacking, BinPackingRelocateNeighbor, FlowShop};
    use rand::rngs::SmallRng;
    use rand::{Rng, SeedableRng};

    fn bins() -> BinPacking {
        let mut rng = SmallRng::seed_from_u64(1);
        let sizes = (0..60).map(|_| rng.random_range(20..100)).collect();
        BinPacking::new("random", 150, sizes)
    }

    #[test]
    fn builds_exactly_one_valid_solution_per_run() {
        let bp = bins();
        let mut state = SearchState::new_with_seed(&bp, 2);
        let mut step = GreedyRandomizedConstruction::<BinPacking>::new(
            StopCondition::new(None, None, None),
            0.3,
        );
        step.run(&mut state).unwrap();
        assert_eq!(state.iteration, 1);
        assert!(state.solution.loads.iter().all(|&l| l <= bp.capacity));
        step.run(&mut state).unwrap();
        assert_eq!(state.iteration, 2);
    }

    /// Pure greedy still builds a valid solution on every problem, and with
    /// the elements of equal cost tied it can still vary with the seed, so
    /// only validity is asserted.
    #[test]
    fn alpha_zero_builds_valid_solutions() {
        let bp = bins();
        let mut state = SearchState::new_with_seed(&bp, 1);
        GreedyRandomizedConstruction::<BinPacking>::new(StopCondition::iterations(1), 0.0)
            .run(&mut state)
            .unwrap();
        assert!(state.solution.loads.iter().all(|&l| l <= bp.capacity));

        let fs = FlowShop::load_file("data/instances/flow_shop/taillard/ta001.txt").unwrap();
        let mut state = SearchState::new_with_seed(&fs, 3);
        GreedyRandomizedConstruction::<FlowShop>::new(StopCondition::iterations(1), 0.0)
            .run(&mut state)
            .unwrap();
        assert_eq!(
            state.solution.objective,
            fs.makespan(&state.solution.sequence)
        );
    }

    /// GRASP is the composition, and it terminates on its outer budget.
    #[test]
    fn grasp_is_iterated_local_search_with_this_as_the_restart() {
        let bp = bins();
        let mut state = SearchState::new_with_seed(&bp, 4);
        Iterated::<BinPacking>::new(
            StopCondition::iterations(2_000),
            Box::new(LocalSearch::<BinPackingRelocateNeighbor>::new(
                StopCondition::iterations(2_000),
            )),
            Box::new(GreedyRandomizedConstruction::<BinPacking>::new(
                StopCondition::new(None, None, None),
                0.3,
            )),
        )
        .run(&mut state)
        .unwrap();
        assert!(state.best_solution.num_bins() as u64 >= bp.lower_bound());
        assert!(state.best_solution.loads.iter().all(|&l| l <= bp.capacity));
    }
}
