//! VRP's answer to [`Ruinable`]: customers are the elements, vehicles the
//! containers, and capacity the resource they compete for.
//!
//! This is the shape ruin-and-recreate was designed around, so the mapping is
//! direct. The working representation is the descent's own `RouteState`, so
//! the anchored descent at the bottom runs on the partial in place, with no
//! conversion on either side of it.

use rand::rngs::SmallRng;

use super::ops::pricing::SlotEdit;
use super::ops::{Descent, RouteState};
use super::problem::{Vrp, VrpSolution};
use crate::trait_defs::{LocalRepair, Ruinable};

/// A route partition mid-ruin, priced at the penalty every placement is
/// charged with.
///
/// The descent's own `RouteState` inside carries the routes, every cache of
/// the solution they form, and the position index that makes
/// [`Ruinable::removal_gain`] an O(1) question. The trait asks for the gain of
/// removing a customer, not of removing position `i` of route `r`, so without
/// an index worst-removal would scan for each of the `n` candidates and cost
/// O(n²). The index is refreshed per touched route by the same `reindex` the
/// descent uses.
///
/// The state's penalty is `Vrp::penalty_weight()`, read once per conversion.
/// The weight is behind a `OnceLock`, so reading it is an atomic load, and
/// `insertion_cost` is called once per candidate place. Regret-2 alone asks
/// O(k² · places) times per iteration, which is where an atomic load stops
/// being free. Everything that prices a move within one iteration reads it
/// from the state, the two `Ruinable` methods that charge violations and the
/// anchored descent alike, so nothing can improve against one weight and
/// score against another.
pub struct VrpPartial {
    state: RouteState,
}

impl Vrp {
    /// The edit that puts `element` before `place` of route `bucket`, which
    /// [`Ruinable::insertion_cost`] prices and [`Ruinable::insert`] applies.
    fn placement(
        &self,
        state: &RouteState,
        bucket: usize,
        place: usize,
        element: usize,
    ) -> [SlotEdit; 1] {
        [SlotEdit::insert(
            bucket,
            state.insertion_cost(self, bucket, place, element, element),
            self.demands[element],
            self.service_times[element],
            1,
        )]
    }
}

impl Ruinable for Vrp {
    type Element = usize;
    type Partial = VrpPartial;

    fn to_partial(&self, sol: &VrpSolution) -> VrpPartial {
        VrpPartial {
            state: RouteState::from_solution(self, sol.clone(), self.penalty_weight()),
        }
    }

    fn finish(&self, partial: &VrpPartial) -> VrpSolution {
        // The solution alone, not the whole state: its position indexes and
        // cumulative distances are the descent's, and cloning them here would
        // allocate one vector per route to throw away.
        let mut sol = partial.state.sol.clone();
        sol.objective = self.objective_under(&sol, self.penalty_weight());
        sol
    }

    fn elements(&self, partial: &VrpPartial, out: &mut Vec<usize>) {
        out.clear();
        out.extend(partial.state.routes().iter().flatten().copied());
    }

    fn num_elements(&self, partial: &VrpPartial) -> usize {
        partial.state.routes().iter().map(Vec::len).sum()
    }

    fn remove_all(&self, partial: &mut VrpPartial, set: &[usize]) {
        let state = &mut partial.state;
        let mut by_route: Vec<Vec<usize>> = vec![Vec::new(); state.routes().len()];
        for &c in set {
            let (r, pos) = state.locate(c);
            by_route[r].push(pos);
        }
        for (r, positions) in by_route.iter_mut().enumerate() {
            if positions.is_empty() {
                continue;
            }
            // Descending, so removing one position never invalidates a
            // position still queued in the same route.
            positions.sort_unstable_by(|a, b| b.cmp(a));
            for &pos in positions.iter() {
                // Priced and applied per position rather than summed and
                // applied once, since the gains are read against the route
                // as it stands between removals and floating-point addition
                // is not associative.
                let c = state.routes()[r][pos];
                let gain = state.removal_gain(self, r, pos, 1);
                let edit = [SlotEdit::remove(
                    r,
                    -gain,
                    self.demands[c],
                    self.service_times[c],
                    1,
                )];
                let price = state.price(self, &edit);
                state.sol.routes[r].remove(pos);
                state.apply_unindexed(self, &edit, &price);
            }
            // The positions were removed back to front, so only the state
            // the route is left in is ever read.
            state.reindex(r);
        }
    }

    fn removal_gain(&self, partial: &VrpPartial, element: usize) -> f64 {
        let (r, pos) = partial.state.locate(element);
        partial.state.removal_gain(self, r, pos, 1)
    }

    /// Shaw's relatedness: near in space and alike in demand.
    fn relatedness(&self, a: usize, b: usize) -> f64 {
        self.distance(a, b) + (self.demands[a] - self.demands[b]).unsigned_abs() as f64
    }

    /// The fleet is fixed, so this is constant per instance, but an unused
    /// vehicle is an empty route, which is what satisfies the trait's
    /// "a bucket may be empty" and lets a customer open a fresh route.
    fn num_buckets(&self, partial: &VrpPartial) -> usize {
        partial.state.routes().len()
    }

    /// Every gap of the route, including both ends. An empty slot offers one
    /// place, and only the first empty slot of its vehicle type does: the
    /// others would offer the same route under another label.
    fn num_places(&self, partial: &VrpPartial, bucket: usize) -> usize {
        let len = partial.state.routes()[bucket].len();
        if len == 0 && !partial.state.is_idle_representative(self, bucket) {
            return 0;
        }
        len + 1
    }

    fn insertion_cost(
        &self,
        partial: &VrpPartial,
        bucket: usize,
        place: usize,
        element: usize,
    ) -> f64 {
        let edit = self.placement(&partial.state, bucket, place, element);
        partial.state.price(self, &edit).gain
    }

    fn insert(&self, partial: &mut VrpPartial, bucket: usize, place: usize, element: usize) {
        let edit = self.placement(&partial.state, bucket, place, element);
        let state = &mut partial.state;
        let price = state.price(self, &edit);
        state.sol.routes[bucket].insert(place, element);
        state.apply(self, &edit, &price);
    }

    /// The objective under the partial's penalty, read off the running caches
    /// [`remove_all`](Self::remove_all) / [`insert`](Self::insert) / the
    /// anchored descent maintain instead of rebuilding a [`VrpSolution`] to
    /// ask.
    fn partial_energy(&self, partial: &VrpPartial) -> f64 {
        partial.state.energy(self)
    }
}

/// The granular descent, anchored at the customers a ruin just re-inserted.
///
/// Holds the descent because it owns instance-derived candidate lists worth
/// keeping across iterations, and the three tunables the anchored form needs,
/// each behind a builder with a published default. Ruin-and-recreate without
/// this is worse on VRP, since greedy re-insertion puts each customer where
/// it is cheapest at the time and never repairs the edges that choice spoils.
/// The cost of leaving it out grows with the instance, and on the largest it
/// is several percent.
pub struct AnchoredRouteDescent {
    descent: Descent,
    granularity: usize,
    max_passes: usize,
}

impl AnchoredRouteDescent {
    /// Nearest partners considered per customer, unless
    /// [`with_granularity`](Self::with_granularity) says otherwise. Matches
    /// the default Hybrid Genetic Search uses.
    pub const DEFAULT_GRANULARITY: usize = 20;

    /// Descent passes over a recreated solution, unless
    /// [`with_max_passes`](Self::with_max_passes) says otherwise. The descent
    /// is anchored at the re-inserted customers, so a pass is cheap and a
    /// handful of them is enough to settle the routes the ruin disturbed.
    pub const DEFAULT_MAX_PASSES: usize = 4;

    /// A descent with the published defaults.
    pub fn new() -> Self {
        Self {
            descent: Descent::new(),
            granularity: Self::DEFAULT_GRANULARITY,
            max_passes: Self::DEFAULT_MAX_PASSES,
        }
    }

    /// Builder-style: how many nearest partners of each customer the moves
    /// consider. Defaults to [`DEFAULT_GRANULARITY`](Self::DEFAULT_GRANULARITY).
    ///
    /// # Panics
    ///
    /// Panics if `granularity` is zero, since a descent with no partners
    /// can make no move.
    pub fn with_granularity(mut self, granularity: usize) -> Self {
        assert!(granularity >= 1, "granularity must be at least 1");
        self.granularity = granularity;
        self
    }

    /// Builder-style: how many nearest partners of each anchor the sweep
    /// visits along with it. Defaults to [`AnchoredSweep::DEFAULT_RING`](crate::common::AnchoredSweep::DEFAULT_RING), and
    /// zero sweeps the anchors alone.
    pub fn with_ring(mut self, ring: usize) -> Self {
        self.descent.set_ring(ring);
        self
    }

    /// Builder-style: how many passes over the sweep list a repair may
    /// spend. Defaults to [`DEFAULT_MAX_PASSES`](Self::DEFAULT_MAX_PASSES).
    ///
    /// # Panics
    ///
    /// Panics if `max_passes` is zero, since a repair with no pass repairs
    /// nothing.
    pub fn with_max_passes(mut self, max_passes: usize) -> Self {
        assert!(max_passes >= 1, "max_passes must be at least 1");
        self.max_passes = max_passes;
        self
    }
}

impl Default for AnchoredRouteDescent {
    fn default() -> Self {
        Self::new()
    }
}

impl LocalRepair<Vrp> for AnchoredRouteDescent {
    fn repair_around(
        &mut self,
        prob: &Vrp,
        partial: &mut VrpPartial,
        anchors: &[usize],
        rng: &mut SmallRng,
    ) {
        self.descent.ensure(prob, self.granularity);
        let penalty = partial.state.penalty;
        self.descent.run_around(
            &mut partial.state,
            prob,
            anchors,
            rng,
            penalty,
            self.max_passes,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::trait_defs::Ruinable;
    use rand::SeedableRng;

    fn ring_vrp() -> Vrp {
        let mut coords = vec![(0.0, 0.0)];
        let mut demands = vec![0];
        for i in 0..12 {
            let theta = std::f64::consts::TAU * i as f64 / 12.0;
            coords.push((theta.cos(), theta.sin()));
            demands.push(1);
        }
        Vrp::new("ring", coords, demands, 3, 4)
    }

    /// The builders have to reach the descent, not merely be stored. The
    /// narrowest descent, one partner, no ring, one pass, still runs on a
    /// ruined partition and leaves it valid.
    #[test]
    fn the_narrowest_descent_still_repairs() {
        let vrp = ring_vrp();
        let mut rng = SmallRng::seed_from_u64(3);
        let sol = crate::search_state::ProblemTrait::new_solution(&vrp, &mut rng);
        let mut partial = vrp.to_partial(&sol);
        let anchors = vec![1usize, 2, 3];
        let mut descent = AnchoredRouteDescent::new()
            .with_granularity(1)
            .with_ring(0)
            .with_max_passes(1);
        descent.repair_around(&vrp, &mut partial, &anchors, &mut rng);
        let back = vrp.finish(&partial);
        vrp.validate_routes(&back.routes).unwrap();
        assert!(back.objective <= sol.objective);
    }

    #[test]
    #[should_panic(expected = "granularity must be at least 1")]
    fn zero_granularity_is_rejected() {
        let _ = AnchoredRouteDescent::new().with_granularity(0);
    }

    #[test]
    #[should_panic(expected = "max_passes must be at least 1")]
    fn zero_passes_are_rejected() {
        let _ = AnchoredRouteDescent::new().with_max_passes(0);
    }
}
