//! VRP's answer to [`Ruinable`]: customers are the elements, vehicles the
//! containers, and capacity the resource they compete for.
//!
//! This is the shape ruin-and-recreate was designed around, so the mapping is
//! direct. The working representation is the descent's own `RouteState`, so
//! the anchored descent at the bottom runs on the partial in place, with no
//! conversion on either side of it.

use rand::rngs::SmallRng;

use super::ops::{self, Descent, RouteState};
use super::problem::{Vrp, VrpSolution, overload_of};
use crate::trait_defs::{LocalRepair, Ruinable};

/// Nearest partners considered per customer by the post-repair descent.
/// Matches the default Hybrid Genetic Search uses.
const GRANULARITY: usize = 20;

/// Descent passes over a recreated solution. The descent is anchored at the
/// re-inserted customers, so a pass is cheap and a handful of them is enough to
/// settle the routes the ruin disturbed.
const MAX_LS_PASSES: usize = 4;

/// A route partition mid-ruin, with the capacity penalty every placement is
/// priced at.
///
/// The descent's own `RouteState` inside carries the routes, their loads, the
/// running distance and excess totals, and the position index that makes
/// [`Ruinable::removal_gain`] an O(1) question. The trait asks for the gain of
/// removing a customer, not of removing position `i` of route `r`, so without
/// an index worst-removal would scan for each of the `n` candidates and cost
/// O(n²). The index is refreshed per touched route by the same `reindex` the
/// descent uses.
pub struct VrpPartial {
    state: RouteState,
    /// `Vrp::penalty_weight()`, read once per conversion.
    ///
    /// The weight is behind a `OnceLock`, so reading it is an atomic load, and
    /// `insertion_cost` is called once per candidate place. Regret-2 alone asks
    /// O(k² · places) times per iteration, which is where an atomic load stops
    /// being free.
    ///
    /// **Everything that prices a move within one iteration reads it from
    /// here**, the two `Ruinable` methods that charge overload and the anchored
    /// descent alike. A caller that read the weight from the problem instead
    /// would agree with this field today, since the weight is a constant of the
    /// instance, and would stop agreeing the moment anything moved it. A
    /// descent that improves against one weight and a candidate scored against
    /// another both look correct from the outside.
    penalty: f64,
}

impl Ruinable for Vrp {
    type Element = usize;
    type Partial = VrpPartial;

    fn to_partial(&self, sol: &VrpSolution) -> VrpPartial {
        VrpPartial {
            state: RouteState::from_routes(self, sol.routes.clone()),
            penalty: self.penalty_weight(),
        }
    }

    fn finish(&self, partial: &VrpPartial) -> VrpSolution {
        self.solution_from_routes(partial.state.routes.clone())
    }

    fn elements(&self, partial: &VrpPartial, out: &mut Vec<usize>) {
        out.clear();
        out.extend(partial.state.routes.iter().flatten().copied());
    }

    /// Summed over the routes rather than by listing the customers, so this is
    /// a pass over the fleet instead of over the instance.
    fn num_elements(&self, partial: &VrpPartial) -> usize {
        partial.state.routes.iter().map(Vec::len).sum()
    }

    fn remove_all(&self, partial: &mut VrpPartial, set: &[usize]) {
        let state = &mut partial.state;
        // Group by route and read every position before removing anything
        // in that route. Removal shifts everything after it, so a position
        // read after an earlier removal in the same route would be stale.
        let mut by_route: Vec<Vec<usize>> = vec![Vec::new(); state.routes.len()];
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
            let mut removed_demand = 0i64;
            for &pos in positions.iter() {
                state.distance -= state.removal_gain(self, r, pos, 1);
                removed_demand += self.demands[state.routes[r][pos]];
                state.routes[r].remove(pos);
            }
            let old_load = state.loads[r];
            state.loads[r] -= removed_demand;
            state.excess +=
                overload_of(state.loads[r], self.capacity) - overload_of(old_load, self.capacity);
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
    /// "somewhere new to put an element" requirement without a special case.
    fn num_buckets(&self, partial: &VrpPartial) -> usize {
        partial.state.routes.len()
    }

    /// A route is a sequence, so a customer can go before any of its stops or
    /// after the last one.
    fn num_places(&self, partial: &VrpPartial, bucket: usize) -> usize {
        partial.state.routes[bucket].len() + 1
    }

    /// The detour, plus the capacity overflow this insertion would add at the
    /// weight the objective already charges overload at. Folding the penalty in
    /// is what keeps a placement available even when every vehicle is full.
    fn insertion_cost(
        &self,
        partial: &VrpPartial,
        bucket: usize,
        place: usize,
        element: usize,
    ) -> f64 {
        let state = &partial.state;
        let detour = state.insertion_cost(self, bucket, place, element, element);
        let overflow =
            ops::excess_delta_insert(self.capacity, state.loads[bucket], self.demands[element]);
        detour + partial.penalty * overflow as f64
    }

    fn insert(&self, partial: &mut VrpPartial, bucket: usize, place: usize, element: usize) {
        let state = &mut partial.state;
        let demand = self.demands[element];
        state.distance += state.insertion_cost(self, bucket, place, element, element);
        state.excess += ops::excess_delta_insert(self.capacity, state.loads[bucket], demand);
        state.loads[bucket] += demand;
        state.routes[bucket].insert(place, element);
        state.reindex(bucket);
    }

    /// `distance + penalty_weight * excess`, exactly [`VrpSolution::objective`]'s
    /// definition, read off the running totals [`remove_all`](Self::remove_all)
    /// / [`insert`](Self::insert) / the anchored descent maintain instead of
    /// rebuilding a [`VrpSolution`] to ask.
    fn partial_energy(&self, partial: &VrpPartial) -> f64 {
        partial.state.distance + partial.penalty * partial.state.excess as f64
    }
}

/// The granular descent, anchored at the customers a ruin just re-inserted.
///
/// Holds the descent because it owns instance-derived candidate lists worth
/// keeping across iterations, and the two tunables the anchored form needs.
/// Ruin-and-recreate without this is measurably worse on VRP, since greedy
/// re-insertion puts each customer where it is cheapest at the time and never
/// repairs the edges that choice spoils.
pub struct AnchoredRouteDescent {
    descent: Descent,
}

impl AnchoredRouteDescent {
    pub fn new() -> Self {
        Self {
            descent: Descent::new(),
        }
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
        self.descent.ensure(prob, GRANULARITY);
        self.descent.run_around(
            &mut partial.state,
            prob,
            anchors,
            rng,
            partial.penalty,
            MAX_LS_PASSES,
        );
    }
}
