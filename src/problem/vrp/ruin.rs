//! VRP's answer to [`Ruinable`]: customers are the elements, vehicles the
//! containers, and capacity the resource they compete for.
//!
//! This is the shape ruin-and-recreate was designed around, so the mapping is
//! direct. What is worth reading is [`VrpPartial`]'s position index and the
//! [`LocalRepair`] impl at the bottom, the two places where "cheap to edit
//! repeatedly" had to be arranged deliberately.

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

/// A route partition mid-ruin: the routes, their loads, and where each customer
/// currently sits.
///
/// The position index is what makes [`Ruinable::removal_gain`] an O(1) question.
/// The trait asks for the gain of removing a customer, not of removing
/// position `i` of route `r`, so without an index worst-removal would scan for
/// each of the `n` candidates and cost O(n²). It is maintained by
/// [`insert`](Ruinable::insert) over the same suffix `Vec::insert` already
/// shifts, so keeping it costs the same order as the insertion itself.
///
/// Deliberately not `RouteState`, the descent's own representation. That one
/// also carries the distance and excess totals, which nothing between destroy
/// and repair reads, since every placement is priced by a local delta. It is
/// built once, for the descent.
///
/// `Clone` is what `Ruinable::partial_energy`'s default implementation needs
/// to be callable in a context generic over `P: Ruinable`, and VRP's own override
/// never clones it, but `Alns<P>` is written once for every `Ruinable`
/// problem and has to satisfy the default's signature to compile.
#[derive(Clone)]
pub struct VrpPartial {
    routes: Vec<Vec<usize>>,
    loads: Vec<i64>,
    /// `route_of[c]` / `pos_in[c]`: where customer `c` sits. Stale for a
    /// customer currently in the removed pool, which is harmless, since nothing
    /// asks.
    route_of: Vec<usize>,
    pos_in: Vec<usize>,
    /// True total travel distance, maintained incrementally by
    /// [`Ruinable::remove_all`] / [`Ruinable::insert`] via the same
    /// [`ops::removal_gain`] / [`ops::insertion_cost`] deltas that already
    /// price a move, and overwritten with the exact total whenever a
    /// [`LocalRepair`] runs (it computes one anyway, to descend on).
    distance: f64,
    /// Total capacity overflow, maintained the same way.
    excess: i64,
    /// `Vrp::penalty_weight()`, read once per conversion.
    ///
    /// The weight is behind a `OnceLock`, so reading it is an atomic load, and
    /// `insertion_cost` is called once per candidate place. Regret-2 alone asks
    /// O(k² · places) times per iteration, which is where an atomic load stops
    /// being free.
    penalty: f64,
}

impl VrpPartial {
    fn reindex_route(&mut self, r: usize, from: usize) {
        for pos in from..self.routes[r].len() {
            let c = self.routes[r][pos];
            self.route_of[c] = r;
            self.pos_in[c] = pos;
        }
    }

    fn reindex_all(&mut self) {
        for r in 0..self.routes.len() {
            self.reindex_route(r, 0);
        }
    }
}

impl Ruinable for Vrp {
    type Element = usize;
    type Partial = VrpPartial;

    fn to_partial(&self, sol: &VrpSolution) -> VrpPartial {
        let n = self.get_n();
        let mut partial = VrpPartial {
            routes: sol.routes.clone(),
            loads: sol.route_loads.clone(),
            route_of: vec![usize::MAX; n + 1],
            pos_in: vec![usize::MAX; n + 1],
            distance: sol.distance,
            excess: sol.overload,
            penalty: self.penalty_weight(),
        };
        partial.reindex_all();
        partial
    }

    fn finish(&self, partial: VrpPartial) -> VrpSolution {
        self.solution_from_routes(partial.routes)
    }

    fn elements(&self, partial: &VrpPartial, out: &mut Vec<usize>) {
        out.clear();
        out.extend(partial.routes.iter().flatten().copied());
    }

    /// Keeps the four buffers and refills them, which is what makes the
    /// per-iteration conversion allocation-free once the search is warm. The
    /// position index is sized to the instance and rewritten in full, so no
    /// entry survives from the solution before.
    fn refresh_partial(&self, partial: &mut VrpPartial, sol: &VrpSolution) {
        let n = self.get_n();
        partial.routes.clone_from(&sol.routes);
        partial.loads.clone_from(&sol.route_loads);
        partial.distance = sol.distance;
        partial.excess = sol.overload;
        partial.penalty = self.penalty_weight();
        partial.route_of.clear();
        partial.route_of.resize(n + 1, usize::MAX);
        partial.pos_in.clear();
        partial.pos_in.resize(n + 1, usize::MAX);
        partial.reindex_all();
    }

    /// Summed over the routes rather than by listing the customers, so this is
    /// a pass over the fleet instead of over the instance.
    fn num_elements(&self, partial: &VrpPartial) -> usize {
        partial.routes.iter().map(Vec::len).sum()
    }

    fn remove_all(&self, partial: &mut VrpPartial, set: &[usize]) {
        // Group by route and read every position before removing anything
        // in that route. Removal shifts everything after it, so a position
        // read after an earlier removal in the same route would be stale.
        let mut by_route: Vec<Vec<usize>> = vec![Vec::new(); partial.routes.len()];
        for &c in set {
            by_route[partial.route_of[c]].push(partial.pos_in[c]);
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
                partial.distance -= ops::removal_gain(self, &partial.routes[r], pos, 1);
                removed_demand += self.demands[partial.routes[r][pos]];
                partial.routes[r].remove(pos);
            }
            let old_load = partial.loads[r];
            partial.loads[r] -= removed_demand;
            partial.excess +=
                overload_of(partial.loads[r], self.capacity) - overload_of(old_load, self.capacity);
        }
        partial.reindex_all();
    }

    fn removal_gain(&self, partial: &VrpPartial, element: usize) -> f64 {
        let (r, pos) = (partial.route_of[element], partial.pos_in[element]);
        ops::removal_gain(self, &partial.routes[r], pos, 1)
    }

    /// Shaw's relatedness: near in space and alike in demand.
    fn relatedness(&self, a: usize, b: usize) -> f64 {
        self.distance(a, b) + (self.demands[a] - self.demands[b]).unsigned_abs() as f64
    }

    /// The fleet is fixed, so this is constant per instance, but an unused
    /// vehicle is an empty route, which is what satisfies the trait's
    /// "somewhere new to put an element" requirement without a special case.
    fn num_buckets(&self, partial: &VrpPartial) -> usize {
        partial.routes.len()
    }

    /// A route is a sequence, so a customer can go before any of its stops or
    /// after the last one.
    fn num_places(&self, partial: &VrpPartial, bucket: usize) -> usize {
        partial.routes[bucket].len() + 1
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
        let detour = ops::insertion_cost(self, &partial.routes[bucket], place, element, element);
        let overflow =
            ops::excess_delta_insert(self.capacity, partial.loads[bucket], self.demands[element]);
        detour + partial.penalty * overflow as f64
    }

    fn insert(&self, partial: &mut VrpPartial, bucket: usize, place: usize, element: usize) {
        let demand = self.demands[element];
        partial.distance +=
            ops::insertion_cost(self, &partial.routes[bucket], place, element, element);
        partial.excess += ops::excess_delta_insert(self.capacity, partial.loads[bucket], demand);
        partial.loads[bucket] += demand;
        partial.routes[bucket].insert(place, element);
        partial.reindex_route(bucket, place);
    }

    /// `distance + penalty_weight * excess`, exactly [`VrpSolution::objective`]'s
    /// definition, read off the running totals [`remove_all`](Self::remove_all)
    /// / [`insert`](Self::insert) / the anchored descent maintain instead of
    /// rebuilding a [`VrpSolution`] to ask.
    fn partial_energy(&self, partial: &VrpPartial) -> f64 {
        partial.distance + partial.penalty * partial.excess as f64
    }
}

/// The granular descent, anchored at the customers a ruin just re-inserted.
///
/// Holds the descent because it owns instance-derived candidate lists worth
/// keeping across iterations, and the two tunables the anchored form needs.
/// Ruin-and-recreate without this is measurably worse on VRP: greedy
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
        // `RouteState` is built here rather than kept in the partial because
        // the descent is the only thing that reads its position indexes while
        // it runs, and what it computes along the way (distance, excess, loads)
        // is read back below instead of being thrown away and re-derived.
        let mut state = RouteState::from_routes(prob, std::mem::take(&mut partial.routes));
        self.descent.run_around(
            &mut state,
            prob,
            anchors,
            rng,
            prob.penalty_weight(),
            MAX_LS_PASSES,
        );
        partial.distance = state.distance;
        partial.excess = state.excess;
        partial.loads.clone_from(&state.loads);
        partial.routes = state.into_routes();
        // The position index is left stale on purpose. Nothing reads it after a
        // repair: what follows is `partial_energy`, which reads the totals
        // above, and then either `finish`, which reads only the routes, or the
        // candidate being dropped. The next iteration's `to_partial` builds a
        // fresh one, and rebuilding it here costs O(n) per iteration for a
        // question no caller asks.
    }
}
