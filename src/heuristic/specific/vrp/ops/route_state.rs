//! A route partition under local search, with the caches a descent needs.

use super::{
    excess_delta_exchange, excess_delta_transfer, insertion_cost, removal_gain, route_loads,
    segment_demand, segment_ends,
};
use crate::problem::Vrp;
use crate::problem::vrp::overload_of;

/// A route partition under local search, with the position indexes the granular
/// move evaluation needs.
///
/// `route_of` / `pos_in` make "where does customer `v` currently sit?" an O(1)
/// question, which is what lets a candidate neighbor `v` be turned into a
/// concrete move without scanning. They are rebuilt for the (at most two) routes
/// a move touches.
///
/// `distance` and `excess` are maintained incrementally from each accepted
/// move's deltas; [`RouteState::from_routes`] is the only place they are
/// computed from the routes themselves.
#[derive(Debug, Clone)]
pub(crate) struct RouteState {
    pub routes: Vec<Vec<usize>>,
    pub loads: Vec<i64>,
    /// True total travel distance.
    pub distance: f64,
    /// Total capacity overflow `Σ max(0, load_r − Q)`.
    pub excess: i64,
    route_of: Vec<usize>,
    pos_in: Vec<usize>,
}

impl RouteState {
    /// Builds the search state from a route partition, computing all caches.
    pub(crate) fn from_routes(prob: &Vrp, routes: Vec<Vec<usize>>) -> Self {
        let n = prob.get_n();
        let mut loads = vec![0i64; routes.len()];
        route_loads(prob, &routes, &mut loads);
        let distance: f64 = routes.iter().map(|r| prob.route_distance(r)).sum();
        let excess: i64 = loads.iter().map(|&l| overload_of(l, prob.capacity)).sum();
        let mut state = Self {
            routes,
            loads,
            distance,
            excess,
            route_of: vec![usize::MAX; n + 1],
            pos_in: vec![usize::MAX; n + 1],
        };
        for r in 0..state.routes.len() {
            state.reindex(r);
        }
        state
    }

    /// The penalized objective this search descends on. The descent itself only
    /// ever works with deltas, so this exists for the tests that check monotonicity.
    #[cfg(test)]
    pub(crate) fn cost(&self, penalty: f64) -> f64 {
        self.distance + penalty * self.excess as f64
    }

    /// Consumes the state, yielding the route partition.
    pub(crate) fn into_routes(self) -> Vec<Vec<usize>> {
        self.routes
    }

    /// The route holding customer `c`, and its position within it.
    #[inline]
    pub(super) fn locate(&self, c: usize) -> (usize, usize) {
        (self.route_of[c], self.pos_in[c])
    }

    /// Refreshes `route_of` / `pos_in` for a single route.
    pub(super) fn reindex(&mut self, r: usize) {
        for pos in 0..self.routes[r].len() {
            let c = self.routes[r][pos];
            self.route_of[c] = r;
            self.pos_in[c] = pos;
        }
    }

    /// `(before, first, last, after)` around the segment `routes[r][pos..pos+len]`.
    #[inline]
    pub(super) fn segment_ends(
        &self,
        r: usize,
        pos: usize,
        len: usize,
    ) -> (usize, usize, usize, usize) {
        segment_ends(&self.routes[r], pos, len)
    }

    /// Distance saved by lifting `routes[r][pos..pos+len]` out of its route.
    #[inline]
    pub(super) fn removal_gain(&self, prob: &Vrp, r: usize, pos: usize, len: usize) -> f64 {
        removal_gain(prob, &self.routes[r], pos, len)
    }

    /// Distance added by inserting the segment `first…last` *before* position
    /// `pos` of route `r`.
    #[inline]
    pub(super) fn insertion_cost(
        &self,
        prob: &Vrp,
        r: usize,
        pos: usize,
        first: usize,
        last: usize,
    ) -> f64 {
        insertion_cost(prob, &self.routes[r], pos, first, last)
    }

    /// Total demand of `routes[r][pos..pos+len]`.
    #[inline]
    pub(super) fn segment_demand(&self, prob: &Vrp, r: usize, pos: usize, len: usize) -> i64 {
        segment_demand(prob, &self.routes[r], pos, len)
    }

    /// Change in total overflow when `demand` moves from route `from` to `to`.
    #[inline]
    pub(super) fn excess_delta_transfer(
        &self,
        prob: &Vrp,
        from: usize,
        to: usize,
        demand: i64,
    ) -> i64 {
        excess_delta_transfer(prob.capacity, self.loads[from], self.loads[to], demand)
    }

    /// Change in total overflow when routes `a` and `b` exchange `demand_a` for
    /// `demand_b`.
    #[inline]
    pub(super) fn excess_delta_exchange(
        &self,
        prob: &Vrp,
        a: usize,
        b: usize,
        demand_a: i64,
        demand_b: i64,
    ) -> i64 {
        excess_delta_exchange(
            prob.capacity,
            self.loads[a],
            self.loads[b],
            demand_a,
            demand_b,
        )
    }

    /// Commits a move's cached deltas.
    #[inline]
    pub(super) fn commit(&mut self, delta_distance: f64, delta_excess: i64) {
        self.distance += delta_distance;
        self.excess += delta_excess;
    }

    /// Recomputes every cache from the routes. Used by `debug_assert!` to catch
    /// incremental-update drift, and by the tests.
    #[cfg(debug_assertions)]
    pub(super) fn assert_caches_consistent(&self, prob: &Vrp) {
        let fresh = RouteState::from_routes(prob, self.routes.clone());
        debug_assert!(
            (fresh.distance - self.distance).abs() < 1e-6,
            "distance drifted: incremental {} vs recomputed {}",
            self.distance,
            fresh.distance
        );
        debug_assert_eq!(fresh.excess, self.excess, "excess drifted");
        debug_assert_eq!(fresh.loads, self.loads, "loads drifted");
    }
}
