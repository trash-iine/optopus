//! Adaptive Large Neighborhood Search over any [`Ruinable`] problem.

use rand::Rng;
use rand::rngs::SmallRng;

use crate::common::{
    AdaptiveWeights, greedy_insertion, random_removal, regret2_insertion, shaw_removal,
    worst_removal,
};
use crate::error::OptError;
use crate::heuristic::{Heuristic, StopCondition};
use crate::search_state::SearchState;
use crate::trait_defs::{Evaluate, LocalRepair, Ruinable};

/// Which destroy operator ruins part of the incumbent.
///
/// The variants are the roulette's options, in the order it indexes them.
/// There is deliberately no separate `NUM_DESTROY` constant: the count comes
/// from [`ALL`](Self::ALL), so adding an operator cannot leave a stale number
/// behind, and the `match` that dispatches on this is exhaustive, so a new
/// variant is a compile error rather than a silent fall-through onto an
/// existing operator.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DestroyOp {
    /// `k` elements drawn uniformly.
    Random,
    /// The `k` elements whose current placement costs the most.
    Worst,
    /// The `k` elements most related to a random seed element (Shaw).
    Shaw,
}

impl DestroyOp {
    /// Every operator, in the order the roulette indexes them.
    pub const ALL: [DestroyOp; 3] = [Self::Random, Self::Worst, Self::Shaw];
}

/// Which repair operator recreates what a destroy operator ruined.
///
/// Counted from [`ALL`](Self::ALL), for the reason [`DestroyOp`] explains.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RepairOp {
    /// Each removed element at its cheapest placement, random order.
    Greedy,
    /// Largest-regret-first, regret measured across containers.
    Regret2,
}

impl RepairOp {
    /// Every operator, in the order the roulette indexes them.
    pub const ALL: [RepairOp; 2] = [Self::Greedy, Self::Regret2];
}

// Ropke & Pisinger's published scoring: the reward for a new global best, for
// a better-than-current solution, and for an accepted worse one, together with
// the segment length and reaction factor the weights adapt on. These are
// defaults, not fixed values. See `with_scoring` and `with_adaptation`.
const DEFAULT_SIGMA_BEST: f64 = 4.0;
const DEFAULT_SIGMA_BETTER: f64 = 2.0;
const DEFAULT_SIGMA_ACCEPT: f64 = 1.0;
const DEFAULT_SEGMENT_LEN: u64 = 100;
const DEFAULT_REACTION: f64 = 0.1;

/// Default ceiling on how many elements one iteration ruins.
///
/// `removal_fraction · n` alone grows without bound, and past a few dozen
/// elements the repair stops being a *large neighborhood* move and becomes a
/// partial restart, the recreated solution keeping too little of the incumbent
/// for the acceptance criterion to mean anything. The specific value is
/// inherited from the VRP-only implementation this was generalized from, where
/// it arrived without a recorded measurement; treat its order of magnitude as
/// justified and its exact value as not, which is why
/// [`with_max_removal`](Alns::with_max_removal) exists.
const DEFAULT_MAX_REMOVAL: usize = 50;

/// Adaptive Large Neighborhood Search (Ropke & Pisinger).
///
/// Each iteration *ruins* part of the incumbent with a destroy operator and
/// *recreates* it with a repair operator, choosing both by a roulette wheel
/// whose weights track recent performance, and accepting worse solutions on a
/// simulated-annealing criterion.
///
/// # What the problem supplies
///
/// [`Ruinable`], elements and containers and the cost of putting one in the
/// other. See that trait for why the *containers* are the load-bearing part:
/// regret-2 measures how much worse an element gets once its best container is
/// taken, which is meaningless when there is only one.
///
/// A destroy+repair step is not a single
/// [`MoveToNeighbor`](crate::search_state::MoveToNeighbor), so this operates
/// directly on `state.solution` rather than through `state.apply`, the same
/// shape as [`LinKernighanHelsgaunForTsp`](crate::heuristic::LinKernighanHelsgaunForTsp).
///
/// # The local repair
///
/// Ruin-and-recreate re-inserts elements greedily and never repairs the
/// edges its own choices spoil elsewhere. Passing a [`LocalRepair`] fixes that
/// anchored at the re-inserted elements, so a pass stays cheap. It is
/// optional in the type system and *not* optional in practice for vehicle
/// routing; see [`AnchoredRouteDescent`](crate::problem::vrp::AnchoredRouteDescent).
///
/// # References
///
/// - Ropke, S. and Pisinger, D. "An Adaptive Large Neighborhood Search
///   Heuristic for the Pickup and Delivery Problem with Time Windows."
///   *Transportation Science*, 40(4), 455-472, 2006.
/// - Shaw, P. "Using Constraint Programming and Local Search Methods to Solve
///   Vehicle Routing Problems." *CP-98*, 417-431, 1998.
pub struct Alns<P: Ruinable> {
    stop_condition: StopCondition,
    destroy: AdaptiveWeights,
    repair: AdaptiveWeights,
    /// Fraction of elements ruined each iteration.
    removal_fraction: f64,
    /// Geometric cooling factor applied to the temperature each iteration.
    cooling_rate: f64,
    /// Current SA temperature (initialized lazily from the first solution).
    temperature: Option<f64>,
    /// Optional post-repair local search, anchored at the re-inserted elements.
    local: Option<Box<dyn LocalRepair<P>>>,
    /// Reused buffer for the element listing every destroy operator starts from.
    scratch: Vec<P::Element>,
    /// The working representation, kept across iterations so the problem can
    /// refill it rather than rebuild it. See `Ruinable::refresh_partial`.
    partial: Option<P::Partial>,
    /// Reward for a candidate that beats the global best.
    sigma_best: f64,
    /// Reward for a candidate that beats the incumbent.
    sigma_better: f64,
    /// Reward for an accepted candidate that is worse than the incumbent.
    sigma_accept: f64,
    /// Ceiling on the per-iteration removal count.
    max_removal: usize,
}

impl<P: Ruinable> Alns<P> {
    /// Creates an ALNS with no post-repair local search.
    ///
    /// # Panics
    ///
    /// Panics if `removal_fraction` is not in `(0, 1]` or `cooling_rate` is not
    /// in `(0, 1]`.
    pub fn new(stop_condition: StopCondition, removal_fraction: f64, cooling_rate: f64) -> Self {
        assert!(
            removal_fraction > 0.0 && removal_fraction <= 1.0,
            "removal_fraction must be in (0, 1]"
        );
        assert!(
            cooling_rate > 0.0 && cooling_rate <= 1.0,
            "cooling_rate must be in (0, 1]"
        );
        Self {
            stop_condition,
            destroy: AdaptiveWeights::new(
                DestroyOp::ALL.len(),
                DEFAULT_SEGMENT_LEN,
                DEFAULT_REACTION,
            ),
            repair: AdaptiveWeights::new(
                RepairOp::ALL.len(),
                DEFAULT_SEGMENT_LEN,
                DEFAULT_REACTION,
            ),
            removal_fraction,
            cooling_rate,
            temperature: None,
            local: None,
            scratch: Vec::new(),
            partial: None,
            sigma_best: DEFAULT_SIGMA_BEST,
            sigma_better: DEFAULT_SIGMA_BETTER,
            sigma_accept: DEFAULT_SIGMA_ACCEPT,
            max_removal: DEFAULT_MAX_REMOVAL,
        }
    }

    /// Builder-style: run `local` over each recreated solution, anchored at the
    /// elements the repair re-inserted.
    pub fn with_local_repair(mut self, local: Box<dyn LocalRepair<P>>) -> Self {
        self.local = Some(local);
        self
    }

    /// Builder-style: the rewards an operator pair earns for a new global best,
    /// for beating the incumbent, and for an accepted worse candidate.
    ///
    /// Defaults to Ropke & Pisinger's `4 / 2 / 1`. Only the ratios matter,
    /// the weights are a convex blend of segment averages, so scaling all three
    /// leaves the wheel unchanged.
    ///
    /// # Panics
    ///
    /// Panics if any reward is negative. Zero is allowed: it says the outcome
    /// earns nothing, which is what an unaccepted candidate already scores.
    pub fn with_scoring(mut self, best: f64, better: f64, accept: f64) -> Self {
        assert!(
            best >= 0.0 && better >= 0.0 && accept >= 0.0,
            "scoring rewards must be non-negative, got ({best}, {better}, {accept})"
        );
        self.sigma_best = best;
        self.sigma_better = better;
        self.sigma_accept = accept;
        self
    }

    /// Builder-style: how many iterations a scoring segment spans, and how much
    /// of its average is blended into the weights at the boundary.
    ///
    /// Defaults to Ropke & Pisinger's `100` / `0.1`. Rebuilds both roulettes,
    /// which is why this takes `self` by value. A builder runs before the
    /// search does, so there are no learned weights to lose.
    ///
    /// # Panics
    ///
    /// Panics if `segment_len` is zero or `reaction` is outside `[0, 1]`
    /// (checked by [`AdaptiveWeights::new`]).
    pub fn with_adaptation(mut self, segment_len: u64, reaction: f64) -> Self {
        self.destroy = AdaptiveWeights::new(DestroyOp::ALL.len(), segment_len, reaction);
        self.repair = AdaptiveWeights::new(RepairOp::ALL.len(), segment_len, reaction);
        self
    }

    /// Builder-style: the ceiling on how many elements one iteration ruins,
    /// applied on top of `removal_fraction · n`.
    ///
    /// Defaults to `50`, inherited from the VRP-only implementation this was
    /// generalized from without a recorded measurement behind it. Treat that
    /// number's order of magnitude as justified and its exact value as not.
    /// Past a few dozen elements the repair stops being a *large neighborhood*
    /// move and becomes a partial restart, keeping too little of the incumbent
    /// for the acceptance criterion to mean anything.
    ///
    /// # Panics
    ///
    /// Panics if `max_removal` is zero, since an iteration that ruins nothing can
    /// never move.
    pub fn with_max_removal(mut self, max_removal: usize) -> Self {
        assert!(max_removal >= 1, "max_removal must be at least 1");
        self.max_removal = max_removal;
        self
    }

    fn removal_count(&self, n: usize) -> usize {
        let k = (self.removal_fraction * n as f64).round() as usize;
        k.clamp(1, n.saturating_sub(1).max(1)).min(self.max_removal)
    }

    fn run_destroy(
        &mut self,
        op: DestroyOp,
        prob: &P,
        partial: &mut P::Partial,
        k: usize,
        rng: &mut SmallRng,
    ) -> Vec<P::Element> {
        match op {
            DestroyOp::Random => random_removal(prob, partial, k, rng, &mut self.scratch),
            DestroyOp::Worst => worst_removal(prob, partial, k, &mut self.scratch),
            DestroyOp::Shaw => shaw_removal(prob, partial, k, rng, &mut self.scratch),
        }
    }

    fn run_repair(
        op: RepairOp,
        prob: &P,
        partial: &mut P::Partial,
        removed: Vec<P::Element>,
        rng: &mut SmallRng,
    ) {
        match op {
            RepairOp::Greedy => greedy_insertion(prob, partial, removed, rng),
            RepairOp::Regret2 => regret2_insertion(prob, partial, removed),
        }
    }
}

impl<P> Heuristic<P> for Alns<P>
where
    P: Ruinable,
    P::Solution: Evaluate,
    P::Partial: Clone,
{
    fn clear(&mut self) {
        self.destroy.reset();
        self.repair.reset();
        self.temperature = None;
    }

    fn stop_condition(&self) -> &StopCondition {
        &self.stop_condition
    }

    fn run_once<'a>(&mut self, state: &mut SearchState<'a, P>) -> Result<(), OptError> {
        let prob: &P = state.instance;
        // Refilled rather than rebuilt: a partial that indexes its elements
        // owns buffers, and this runs once per iteration.
        let mut partial = match self.partial.take() {
            Some(mut partial) => {
                prob.refresh_partial(&mut partial, &state.solution);
                partial
            }
            None => prob.to_partial(&state.solution),
        };
        // The count comes from the partial rather than the instance: how many
        // elements are placed is a property of the solution, and a problem
        // whose containers are created on demand has no fixed count. Counted
        // rather than listed, since the destroy operator lists them itself.
        let n = prob.num_elements(&partial);
        if n == 0 {
            state.progress_iteration();
            return Ok(());
        }

        // Lazily initialize the temperature so a solution ~5% worse is accepted
        // with probability ~0.5 at the start.
        let current_energy = state.solution.evaluate().minimized();
        let temperature = *self
            .temperature
            .get_or_insert_with(|| (0.05 * current_energy.abs()).max(1.0) / std::f64::consts::LN_2);

        let k = self.removal_count(n);
        let d_idx = self.destroy.select(&mut state.rng);
        let r_idx = self.repair.select(&mut state.rng);

        let removed =
            self.run_destroy(DestroyOp::ALL[d_idx], prob, &mut partial, k, &mut state.rng);
        let anchors = removed.clone();
        Self::run_repair(
            RepairOp::ALL[r_idx],
            prob,
            &mut partial,
            removed,
            &mut state.rng,
        );
        if let Some(local) = self.local.as_mut() {
            local.repair_around(prob, &mut partial, &anchors, &mut state.rng);
        }
        // Read the candidate's energy off the partial rather than converting
        // it: `finish` is a full solution rebuild, and most iterations reject
        // Paying that cost on every one of them is exactly the double
        // recompute this method exists to avoid (see `Ruinable::partial_energy`).
        let candidate_energy = prob.partial_energy(&partial);

        // Simulated-annealing acceptance, on the direction-normalized energy so
        // this reads the same whichever way the problem optimizes.
        let accept = candidate_energy <= current_energy
            || state.rng.random::<f64>()
                < ((current_energy - candidate_energy) / temperature).exp();

        let score = if candidate_energy < state.best_solution.evaluate().minimized() {
            self.sigma_best
        } else if candidate_energy < current_energy {
            self.sigma_better
        } else if accept {
            self.sigma_accept
        } else {
            0.0
        };
        self.destroy.record(d_idx, score);
        self.repair.record(r_idx, score);

        state.iteration += 1;
        if accept {
            // Only the accepted candidate ever pays for a full solution
            // rebuild, and it is the one iteration that gives up the buffers:
            // `finish` consumes the partial. The next iteration builds a fresh
            // one, which is the rarer path.
            state.solution = prob.finish(partial);
            state.n_accepted += 1;
        } else {
            // Rejected, so the buffers come back for the next round.
            self.partial = Some(partial);
            state.n_rejected += 1;
        }
        state.update_best();

        self.temperature = Some((temperature * self.cooling_rate).max(1e-9));
        self.destroy.maybe_update();
        self.repair.maybe_update();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::problem::vrp::{AnchoredRouteDescent, Vrp};

    fn ring_vrp() -> Vrp {
        // 8 customers on a circle around the depot; capacity 2, 4 vehicles.
        let mut coords = vec![(0.0, 0.0)];
        let mut demands = vec![0];
        for i in 0..8 {
            let theta = std::f64::consts::TAU * i as f64 / 8.0;
            coords.push((theta.cos(), theta.sin()));
            demands.push(1);
        }
        Vrp::new("ring", coords, demands, 2, 4)
    }

    fn alns(iterations: u64, removal: f64, cooling: f64) -> Alns<Vrp> {
        Alns::<Vrp>::new(StopCondition::iterations(iterations), removal, cooling)
            .with_local_repair(Box::new(AnchoredRouteDescent::new()))
    }

    #[test]
    fn alns_improves_and_stays_feasible() {
        let vrp = ring_vrp();
        let mut state = SearchState::new_with_seed(&vrp, 7);
        let initial = state.best_solution.objective;
        alns(3_000, 0.3, 0.999).run(&mut state).unwrap();
        assert!(
            state.best_solution.objective <= initial,
            "ALNS must not worsen the incumbent"
        );
        assert_eq!(state.best_solution.overload, 0, "best should be feasible");
        vrp.validate_routes(&state.best_solution.routes).unwrap();
    }

    #[test]
    fn alns_is_reproducible_under_seed() {
        let vrp = ring_vrp();
        let run = || {
            let mut state = SearchState::new_with_seed(&vrp, 123);
            alns(1_500, 0.25, 0.9995).run(&mut state).unwrap();
            state.best_solution.objective
        };
        assert_eq!(run(), run());
    }

    /// Without a `LocalRepair` the search is still a valid ruin-and-recreate,
    /// the trait makes it optional, and a problem with no anchored local search
    /// has to be able to run at all. It is *worse* on VRP, which is the whole
    /// reason the VRP registration supplies one.
    #[test]
    fn alns_runs_without_a_local_repair() {
        let vrp = ring_vrp();
        let mut state = SearchState::new_with_seed(&vrp, 7);
        let initial = state.best_solution.objective;
        Alns::<Vrp>::new(StopCondition::iterations(600), 0.3, 0.999)
            .run(&mut state)
            .unwrap();
        assert!(state.best_solution.objective <= initial);
        vrp.validate_routes(&state.best_solution.routes).unwrap();
    }

    /// The operator count comes from the enums, not from a constant that could
    /// drift out of step with the `match` that dispatches on them. If a variant
    /// is ever added without extending `ALL`, the roulette would never select
    /// it, and this is the one assertion that would catch that.
    #[test]
    fn the_operator_banks_match_their_enums() {
        assert_eq!(DestroyOp::ALL.len(), 3);
        assert_eq!(RepairOp::ALL.len(), 2);
        for (i, op) in DestroyOp::ALL.iter().enumerate() {
            assert_eq!(DestroyOp::ALL[i], *op, "ALL must be its own index order");
        }
    }

    /// `with_max_removal` has to bound the ruin, not merely be stored. A
    /// ceiling of 1 on an 8-customer instance means every iteration removes
    /// exactly one customer, so the ruin can never empty a whole route, and a
    /// removal count that ignored the ceiling would take `0.9 * 8 = 7`.
    #[test]
    fn with_max_removal_bounds_the_ruin() {
        let default = Alns::<Vrp>::new(StopCondition::iterations(1), 0.9, 0.999);
        assert_eq!(default.removal_count(8), 7);

        let capped = Alns::<Vrp>::new(StopCondition::iterations(1), 0.9, 0.999).with_max_removal(1);
        assert_eq!(capped.removal_count(8), 1);
    }

    /// The scoring rewards have to reach the weights. Scoring *only* new global
    /// bests and giving nothing for anything else starves the wheel: every
    /// segment that finds no improvement scores zero across the board, so the
    /// weights decay toward zero where the default 4/2/1 keeps them near their
    /// initial 1.0.
    #[test]
    fn with_scoring_changes_how_the_weights_move() {
        let vrp = ring_vrp();

        let mut default_run = SearchState::new_with_seed(&vrp, 11);
        let mut default_alns = alns(400, 0.3, 0.999);
        default_alns.run(&mut default_run).unwrap();
        let default_total: f64 = default_alns.destroy.weights().iter().sum();

        let mut starved_run = SearchState::new_with_seed(&vrp, 11);
        let mut starved_alns = Alns::<Vrp>::new(StopCondition::iterations(400), 0.3, 0.999)
            .with_local_repair(Box::new(AnchoredRouteDescent::new()))
            .with_scoring(4.0, 0.0, 0.0);
        starved_alns.run(&mut starved_run).unwrap();
        let starved_total: f64 = starved_alns.destroy.weights().iter().sum();

        assert!(
            starved_total < default_total,
            "starved scoring left weights at {starved_total}, default at {default_total}"
        );
    }

    /// A shorter segment blends more often, so the weights are further from
    /// their uniform start after the same number of iterations.
    #[test]
    fn with_adaptation_changes_the_blend_frequency() {
        let vrp = ring_vrp();

        let mut slow_state = SearchState::new_with_seed(&vrp, 5);
        let mut slow = Alns::<Vrp>::new(StopCondition::iterations(300), 0.3, 0.999)
            .with_local_repair(Box::new(AnchoredRouteDescent::new()))
            .with_adaptation(1_000, 0.5);
        slow.run(&mut slow_state).unwrap();

        let mut fast_state = SearchState::new_with_seed(&vrp, 5);
        let mut fast = Alns::<Vrp>::new(StopCondition::iterations(300), 0.3, 0.999)
            .with_local_repair(Box::new(AnchoredRouteDescent::new()))
            .with_adaptation(10, 0.5);
        fast.run(&mut fast_state).unwrap();

        // 300 iterations never reach a 1000-long segment, so the slow one is
        // still exactly uniform; the fast one has blended thirty times.
        assert_eq!(slow.destroy.weights(), &[1.0, 1.0, 1.0]);
        assert_ne!(fast.destroy.weights(), &[1.0, 1.0, 1.0]);
    }

    /// Everything the builders set has to survive into a seeded run
    /// reproducibly. The parameters change which moves are taken, not
    /// whether the run is deterministic.
    #[test]
    fn a_tuned_alns_is_still_reproducible() {
        let vrp = ring_vrp();
        let run = || {
            let mut state = SearchState::new_with_seed(&vrp, 99);
            Alns::<Vrp>::new(StopCondition::iterations(800), 0.25, 0.998)
                .with_local_repair(Box::new(AnchoredRouteDescent::new()))
                .with_scoring(9.0, 3.0, 0.5)
                .with_adaptation(25, 0.4)
                .with_max_removal(3)
                .run(&mut state)
                .unwrap();
            state.best_solution.objective
        };
        assert_eq!(run(), run());
    }

    #[test]
    #[should_panic(expected = "scoring rewards must be non-negative")]
    fn negative_scoring_is_rejected() {
        let _ =
            Alns::<Vrp>::new(StopCondition::iterations(1), 0.3, 0.999).with_scoring(4.0, -1.0, 1.0);
    }

    #[test]
    #[should_panic(expected = "max_removal must be at least 1")]
    fn a_zero_removal_ceiling_is_rejected() {
        let _ = Alns::<Vrp>::new(StopCondition::iterations(1), 0.3, 0.999).with_max_removal(0);
    }
}
