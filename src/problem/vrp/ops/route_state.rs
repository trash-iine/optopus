//! A route partition under local search, with the caches a descent needs.

use super::pricing::{EditPrice, SlotEdit, apply, price};
use super::{insertion_cost, removal_gain, segment_ends};
use crate::problem::Vrp;
use crate::problem::vrp::VrpSolution;

/// A route partition under local search, with the position indexes the granular
/// move evaluation needs.
///
/// The routes and every cache live in a [`VrpSolution`], whose `objective` is
/// kept under `penalty` rather than under [`Vrp::penalty_weight`], since the
/// searches that hold a `RouteState` price their moves under a penalty of
/// their own. [`RouteState::into_solution`] hands the solution back with the
/// objective the problem defines.
///
/// `route_of` / `pos_in` make "where does customer `v` currently sit?" an O(1)
/// question, which is what lets a candidate neighbor `v` be turned into a
/// concrete move without scanning, and are rebuilt for the (at most two)
/// routes a move touches. `cum_distance[r][i]` is the distance from the depot
/// to `routes[r][i]` along the route, so the tail exchange of 2-opt* prices
/// the tail it moves without walking it. Only the descent reads it, and it is
/// rebuilt on demand ([`RouteState::cum`]), so a route the ruin operators
/// edited and no move ever prices against is never walked.
///
/// `idle_representative[t]` is the first empty slot of vehicle type `t`, if
/// any: the one slot a move that opens a route of that type considers, see
/// [`Vrp::idle_representatives`].
#[derive(Debug, Clone)]
pub(crate) struct RouteState {
    pub sol: VrpSolution,
    pub penalty: f64,
    route_of: Vec<usize>,
    pos_in: Vec<usize>,
    cum_distance: Vec<Vec<f64>>,
    /// Whether `cum_distance[r]` is behind `routes[r]`.
    cum_stale: Vec<bool>,
    idle_representative: Vec<Option<usize>>,
}

impl RouteState {
    /// Builds the state from a route partition, every cache computed from
    /// scratch and the objective priced under `penalty`.
    pub(crate) fn from_routes(prob: &Vrp, routes: Vec<Vec<usize>>, penalty: f64) -> Self {
        Self::from_solution(prob, prob.solution_from_routes(routes), penalty)
    }

    /// Builds the state around a solution whose caches are already current,
    /// re-pricing only its objective under `penalty`.
    pub(crate) fn from_solution(prob: &Vrp, mut sol: VrpSolution, penalty: f64) -> Self {
        let n = prob.get_n();
        let slots = sol.routes.len();
        sol.objective = prob.objective_under(&sol, penalty);
        let mut state = Self {
            sol,
            penalty,
            route_of: vec![usize::MAX; n + 1],
            pos_in: vec![usize::MAX; n + 1],
            cum_distance: vec![Vec::new(); slots],
            cum_stale: vec![true; slots],
            idle_representative: vec![None; prob.vehicle_types().len()],
        };
        for r in 0..slots {
            state.reindex(r);
        }
        state.refresh_idle_representatives(prob);
        state
    }

    /// Re-prices the objective under a new penalty.
    pub(crate) fn set_penalty(&mut self, prob: &Vrp, penalty: f64) {
        self.penalty = penalty;
        self.sol.objective = prob.objective_under(&self.sol, penalty);
    }

    /// The objective under `penalty`, computed from the aggregates rather
    /// than read from the incrementally moved `sol.objective`, so that two
    /// states with the same caches compare equal however they got there.
    pub(crate) fn energy(&self, prob: &Vrp) -> f64 {
        prob.objective_under(&self.sol, self.penalty)
    }

    /// The routes, read-only. A caller that edits them does so through
    /// `sol.routes` and must then price and [`RouteState::apply`] the edit,
    /// which is the whole reason the two spellings differ.
    #[inline]
    pub(crate) fn routes(&self) -> &[Vec<usize>] {
        &self.sol.routes
    }

    /// `(route, position)` of customer `c`.
    #[inline]
    pub(crate) fn locate(&self, c: usize) -> (usize, usize) {
        (self.route_of[c], self.pos_in[c])
    }

    /// Whether `slot` is the empty slot a move opening a route of its type
    /// considers.
    #[inline]
    pub(crate) fn is_idle_representative(&self, prob: &Vrp, slot: usize) -> bool {
        self.idle_representative[prob.type_of_slot(slot)] == Some(slot)
    }

    /// The empty slot a move opening a route of type `t` considers, if that
    /// type has an idle vehicle.
    #[inline]
    pub(crate) fn idle_representative_of(&self, t: usize) -> Option<usize> {
        self.idle_representative[t]
    }

    /// Whether any vehicle type has an idle vehicle at all.
    #[inline]
    pub(crate) fn has_idle_vehicle(&self) -> bool {
        self.idle_representative.iter().any(Option::is_some)
    }

    /// Rebuilds the position index of route `r` after its customers changed,
    /// and marks its cumulative distances stale.
    pub(crate) fn reindex(&mut self, r: usize) {
        for (pos, &c) in self.sol.routes[r].iter().enumerate() {
            self.route_of[c] = r;
            self.pos_in[c] = pos;
        }
        self.cum_stale[r] = true;
    }

    /// The distance from the depot to each customer of route `r` along the
    /// route, rebuilt if the route has changed since it was last asked for.
    #[inline(always)]
    fn cum(&mut self, prob: &Vrp, r: usize) -> &[f64] {
        if self.cum_stale[r] {
            let route = &self.sol.routes[r];
            let cum = &mut self.cum_distance[r];
            cum.clear();
            let mut prev = 0;
            let mut d = 0.0;
            for &c in route {
                d += prob.distance(prev, c);
                cum.push(d);
                prev = c;
            }
            self.cum_stale[r] = false;
        }
        &self.cum_distance[r]
    }

    fn refresh_idle_representatives(&mut self, prob: &Vrp) {
        self.idle_representative.fill(None);
        for s in prob.idle_representatives(&self.sol.routes) {
            self.idle_representative[prob.type_of_slot(s)] = Some(s);
        }
    }

    /// Prices `edits` against the current routes under this state's penalty.
    #[inline]
    pub(crate) fn price<const N: usize>(&self, prob: &Vrp, edits: &[SlotEdit; N]) -> EditPrice {
        price(prob, &self.sol, self.penalty, edits)
    }

    /// Moves the caches by a price after the caller edited the routes, then
    /// reindexes the edited routes. Their cumulative distances go stale; a
    /// descent refreshes them with [`RouteState::refresh_cum`].
    pub(crate) fn apply<const N: usize>(
        &mut self,
        prob: &Vrp,
        edits: &[SlotEdit; N],
        price: &EditPrice,
    ) {
        self.apply_unindexed(prob, edits, price);
        for e in edits {
            self.reindex(e.slot);
        }
    }

    /// [`RouteState::apply`] without the position reindex, for a caller
    /// editing one route several times in a row: only the last of those
    /// edits leaves an index anyone reads, so it calls
    /// [`RouteState::reindex`] itself once at the end.
    pub(crate) fn apply_unindexed<const N: usize>(
        &mut self,
        prob: &Vrp,
        edits: &[SlotEdit; N],
        price: &EditPrice,
    ) {
        apply(prob, &mut self.sol, edits, price);
        let occupancy_changed = edits.iter().any(|e| {
            e.delta_count != 0 && {
                let len = self.sol.routes[e.slot].len();
                len == 0 || len == e.delta_count as usize
            }
        });
        if occupancy_changed {
            self.refresh_idle_representatives(prob);
        }
    }

    /// `(before, first, last, after)` around `routes[r][pos..pos + len]`.
    #[inline]
    pub(crate) fn segment_ends(
        &self,
        r: usize,
        pos: usize,
        len: usize,
    ) -> (usize, usize, usize, usize) {
        segment_ends(&self.sol.routes[r], pos, len)
    }

    /// Distance saved by lifting `routes[r][pos..pos + len]` out.
    #[inline]
    pub(crate) fn removal_gain(&self, prob: &Vrp, r: usize, pos: usize, len: usize) -> f64 {
        removal_gain(prob, &self.sol.routes[r], pos, len)
    }

    /// Distance added by inserting `first…last` before `pos` of route `r`.
    #[inline]
    pub(crate) fn insertion_cost(
        &self,
        prob: &Vrp,
        r: usize,
        pos: usize,
        first: usize,
        last: usize,
    ) -> f64 {
        insertion_cost(prob, &self.sol.routes[r], pos, first, last)
    }

    /// Total demand and service time of `routes[r][pos..pos + len]`.
    #[inline(always)]
    pub(crate) fn segment_load(&self, prob: &Vrp, r: usize, pos: usize, len: usize) -> (i64, f64) {
        self.sol.routes[r][pos..pos + len]
            .iter()
            .fold((0, 0.0), |(d, s), &c| {
                (d + prob.demands[c], s + prob.service_times[c])
            })
    }

    /// The edges inside `routes[r][pos..pos + len]`, which a segment carries
    /// with it when it moves between routes and which [`removal_gain`] and
    /// [`insertion_cost`] both leave out.
    #[inline(always)]
    pub(crate) fn segment_internal(&mut self, prob: &Vrp, r: usize, pos: usize, len: usize) -> f64 {
        if len == 1 {
            return 0.0;
        }
        let cum = self.cum(prob, r);
        cum[pos + len - 1] - cum[pos]
    }

    /// What the tail after position `pos` of route `r` costs on its own: its
    /// internal edges plus the return to the depot, `0.0` for an empty tail.
    /// The edge joining `routes[r][pos]` to the tail is not included, since a
    /// tail exchange replaces exactly that edge.
    #[inline(always)]
    pub(crate) fn tail_cost(&mut self, prob: &Vrp, r: usize, pos: usize) -> f64 {
        let last = self.sol.routes[r].len() - 1;
        if pos >= last {
            return 0.0;
        }
        let depot_leg = prob.distance(self.sol.routes[r][last], 0);
        let cum = self.cum(prob, r);
        cum[last] - cum[pos + 1] + depot_leg
    }

    /// Recomputes every cache from the routes. Used by `debug_assert!` to catch
    /// incremental-update drift, and by the tests.
    #[cfg(debug_assertions)]
    pub(crate) fn assert_caches_consistent(&self, prob: &Vrp) {
        let fresh = RouteState::from_routes(prob, self.sol.routes.clone(), self.penalty);
        let close = |a: f64, b: f64, what: &str| {
            debug_assert!(
                (a - b).abs() < 1e-6,
                "{what} drifted: incremental {a} vs recomputed {b}"
            );
        };
        debug_assert_eq!(fresh.sol.route_loads, self.sol.route_loads, "loads drifted");
        debug_assert_eq!(
            fresh.sol.used_count, self.sol.used_count,
            "used_count drifted"
        );
        debug_assert_eq!(fresh.sol.overload, self.sol.overload, "overload drifted");
        debug_assert_eq!(
            fresh.sol.min_count_shortfall, self.sol.min_count_shortfall,
            "shortfall drifted"
        );
        for r in 0..self.sol.routes.len() {
            close(
                fresh.sol.route_distance[r],
                self.sol.route_distance[r],
                "route_distance",
            );
            close(
                fresh.sol.route_time[r],
                self.sol.route_time[r],
                "route_time",
            );
            if !self.cum_stale[r] {
                debug_assert_eq!(
                    fresh.sol.routes[r].len(),
                    self.cum_distance[r].len(),
                    "cum_distance drifted"
                );
            }
        }
        close(fresh.sol.total_time, self.sol.total_time, "total_time");
        close(fresh.sol.makespan, self.sol.makespan, "makespan");
        close(fresh.sol.time_excess, self.sol.time_excess, "time_excess");
        close(fresh.sol.total_cost, self.sol.total_cost, "total_cost");
        close(fresh.sol.objective, self.sol.objective, "objective");
        debug_assert_eq!(
            fresh.idle_representative, self.idle_representative,
            "idle representatives drifted"
        );
        for c in 1..=prob.get_n() {
            let (r, pos) = self.locate(c);
            debug_assert_eq!(self.sol.routes[r][pos], c, "position index drifted");
        }
    }
}
