//! Granular local search over CVRP routes, in place and penalty-parameterized.
//!
//! This is deliberately *not* built on [`crate::problem::VrpRelocateNeighbor`] and
//! friends. Those bake [`Vrp::penalty_weight`] — a fixed, deliberately enormous
//! constant — into every gain, whereas Hybrid Genetic Search needs to search
//! under a penalty it adapts at runtime. They also enumerate the full O(n²)
//! neighborhood and offer neither intra-route relocation nor 2-opt\*.
//!
//! Moves are restricted to *granular* candidate pairs: for each customer `u`,
//! only its `granularity` nearest customers `v` are considered as partners. The
//! move set follows Vidal's HGS-CVRP:
//!
//! | Move | Kind |
//! |---|---|
//! | relocate segment of 1–2 customers, optionally reversed | inter-route, intra-route |
//! | swap segments of 1–2 customers | inter-route |
//! | 2-opt (reverse a sub-path) | intra-route |
//! | 2-opt\* (exchange route tails) | inter-route |
//!
//! Every move is evaluated in O(1) from the distances at its endpoints and
//! accepted on first improvement of `distance + penalty · excess`.

use rand::rngs::SmallRng;
use rand::seq::SliceRandom;

use crate::problem::Vrp;

/// Improvements smaller than this are treated as numerical noise, which keeps
/// the descent from cycling on ties.
const MIN_IMPROVEMENT: f64 = 1e-10;

/// Longest customer segment relocated or swapped as a unit.
const MAX_SEGMENT: usize = 2;

/// Returns the capacity overflow of a route load: `max(0, load - capacity)`.
///
/// Mirrors the private helper in [`crate::problem::vrp`]; duplicated here (three
/// lines) rather than widening that module's visibility for one caller.
#[inline]
fn overload_of(load: i64, capacity: i64) -> i64 {
    (load - capacity).max(0)
}

/// For each customer, its `granularity` nearest customers in ascending distance.
///
/// Index `0` (the depot) is present but empty so the list can be indexed by
/// customer id directly. Costs O(n²) once per instance; callers cache it.
pub(super) fn build_neighbor_lists(prob: &Vrp, granularity: usize) -> Vec<Vec<usize>> {
    let n = prob.get_n();
    let mut lists = vec![Vec::new(); n + 1];
    let mut buffer: Vec<(f64, usize)> = Vec::with_capacity(n);
    for (c, list) in lists.iter_mut().enumerate().skip(1) {
        buffer.clear();
        buffer.extend(
            (1..=n)
                .filter(|&o| o != c)
                .map(|o| (prob.distance(c, o), o)),
        );
        let keep = granularity.min(buffer.len());
        if keep < buffer.len() {
            buffer.select_nth_unstable_by(keep, |a, b| a.0.total_cmp(&b.0));
            buffer.truncate(keep);
        }
        buffer.sort_by(|a, b| a.0.total_cmp(&b.0));
        *list = buffer.iter().map(|&(_, o)| o).collect();
    }
    lists
}

/// A route partition under local search, with the position indexes the granular
/// move evaluation needs.
///
/// `route_of` / `pos_in` make "where does customer `v` currently sit?" an O(1)
/// question, which is what lets a candidate neighbor `v` be turned into a
/// concrete move without scanning. They are rebuilt for the (at most two) routes
/// a move touches.
#[derive(Debug, Clone)]
pub(super) struct LsState {
    pub routes: Vec<Vec<usize>>,
    pub loads: Vec<i64>,
    /// True total travel distance.
    pub distance: f64,
    /// Total capacity overflow `Σ max(0, load_r − Q)`.
    pub excess: i64,
    route_of: Vec<usize>,
    pos_in: Vec<usize>,
}

impl LsState {
    /// Builds the search state from a route partition, computing all caches.
    pub(super) fn from_routes(prob: &Vrp, routes: Vec<Vec<usize>>) -> Self {
        let n = prob.get_n();
        let loads: Vec<i64> = routes
            .iter()
            .map(|r| r.iter().map(|&c| prob.demands[c]).sum())
            .collect();
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
    pub(super) fn cost(&self, penalty: f64) -> f64 {
        self.distance + penalty * self.excess as f64
    }

    /// Consumes the state, yielding the route partition.
    pub(super) fn into_routes(self) -> Vec<Vec<usize>> {
        self.routes
    }

    /// Refreshes `route_of` / `pos_in` for a single route.
    fn reindex(&mut self, r: usize) {
        for pos in 0..self.routes[r].len() {
            let c = self.routes[r][pos];
            self.route_of[c] = r;
            self.pos_in[c] = pos;
        }
    }

    /// Node preceding position `pos` of route `r` (the depot at the start).
    #[inline]
    fn before(&self, r: usize, pos: usize) -> usize {
        if pos == 0 { 0 } else { self.routes[r][pos - 1] }
    }

    /// Node following position `pos` of route `r` (the depot at the end).
    #[inline]
    fn after(&self, r: usize, pos: usize) -> usize {
        let route = &self.routes[r];
        if pos + 1 >= route.len() {
            0
        } else {
            route[pos + 1]
        }
    }

    /// `(before, first, last, after)` around the segment `routes[r][pos..pos+len]`.
    #[inline]
    fn segment_ends(&self, r: usize, pos: usize, len: usize) -> (usize, usize, usize, usize) {
        let route = &self.routes[r];
        let before = if pos == 0 { 0 } else { route[pos - 1] };
        let after = if pos + len >= route.len() {
            0
        } else {
            route[pos + len]
        };
        (before, route[pos], route[pos + len - 1], after)
    }

    /// Distance saved by lifting `routes[r][pos..pos+len]` out of its route.
    #[inline]
    fn removal_gain(&self, prob: &Vrp, r: usize, pos: usize, len: usize) -> f64 {
        let (before, first, last, after) = self.segment_ends(r, pos, len);
        prob.distance(before, first) + prob.distance(last, after) - prob.distance(before, after)
    }

    /// Distance added by inserting the segment `first…last` *before* position
    /// `pos` of route `r`.
    #[inline]
    fn insertion_cost(&self, prob: &Vrp, r: usize, pos: usize, first: usize, last: usize) -> f64 {
        let route = &self.routes[r];
        let before = if pos == 0 { 0 } else { route[pos - 1] };
        let after = if pos >= route.len() { 0 } else { route[pos] };
        prob.distance(before, first) + prob.distance(last, after) - prob.distance(before, after)
    }

    /// Total demand of `routes[r][pos..pos+len]`.
    #[inline]
    fn segment_demand(&self, prob: &Vrp, r: usize, pos: usize, len: usize) -> i64 {
        self.routes[r][pos..pos + len]
            .iter()
            .map(|&c| prob.demands[c])
            .sum()
    }

    /// Change in total overflow when `demand` moves from route `from` to `to`.
    #[inline]
    fn excess_delta_transfer(&self, prob: &Vrp, from: usize, to: usize, demand: i64) -> i64 {
        let cap = prob.capacity;
        overload_of(self.loads[from] - demand, cap) - overload_of(self.loads[from], cap)
            + overload_of(self.loads[to] + demand, cap)
            - overload_of(self.loads[to], cap)
    }

    /// Change in total overflow when routes `a` and `b` exchange `demand_a` for
    /// `demand_b`.
    #[inline]
    fn excess_delta_exchange(
        &self,
        prob: &Vrp,
        a: usize,
        b: usize,
        demand_a: i64,
        demand_b: i64,
    ) -> i64 {
        let cap = prob.capacity;
        overload_of(self.loads[a] - demand_a + demand_b, cap) - overload_of(self.loads[a], cap)
            + overload_of(self.loads[b] - demand_b + demand_a, cap)
            - overload_of(self.loads[b], cap)
    }

    /// Commits a move's cached deltas.
    #[inline]
    fn commit(&mut self, delta_distance: f64, delta_excess: i64) {
        self.distance += delta_distance;
        self.excess += delta_excess;
    }

    /// Recomputes every cache from the routes. Used by `debug_assert!` to catch
    /// incremental-update drift, and by the tests.
    #[cfg(debug_assertions)]
    fn assert_caches_consistent(&self, prob: &Vrp) {
        let fresh = LsState::from_routes(prob, self.routes.clone());
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

/// Runs the descent to a local optimum of `distance + penalty · excess`.
///
/// `order` is a scratch buffer of the customers `1..=n`; it is shuffled in place
/// each pass so the caller can reuse the allocation across calls. Stops at the
/// first pass with no improving move, or after `max_passes`.
pub(super) fn local_search(
    state: &mut LsState,
    prob: &Vrp,
    neighbors: &[Vec<usize>],
    order: &mut Vec<usize>,
    rng: &mut SmallRng,
    penalty: f64,
    max_passes: usize,
) {
    let n = prob.get_n();
    if n == 0 {
        return;
    }
    order.clear();
    order.extend(1..=n);

    for _ in 0..max_passes {
        order.shuffle(rng);
        let mut improved = false;
        for &u in order.iter() {
            if improve_around(state, prob, neighbors, u, penalty) {
                improved = true;
            }
        }
        if !improved {
            break;
        }
    }
    #[cfg(debug_assertions)]
    state.assert_caches_consistent(prob);
}

/// Tries every granular move anchored at `u`, applying the first improving one.
fn improve_around(
    state: &mut LsState,
    prob: &Vrp,
    neighbors: &[Vec<usize>],
    u: usize,
    penalty: f64,
) -> bool {
    for &v in &neighbors[u] {
        for len in 1..=MAX_SEGMENT {
            for reverse in [false, true] {
                if try_relocate(state, prob, u, v, len, reverse, penalty) {
                    return true;
                }
            }
        }
        for len_u in 1..=MAX_SEGMENT {
            for len_v in 1..=MAX_SEGMENT {
                if try_swap(state, prob, u, v, len_u, len_v, penalty) {
                    return true;
                }
            }
        }
        if try_two_opt(state, prob, u, v, penalty) {
            return true;
        }
        if try_two_opt_star(state, prob, u, v, penalty) {
            return true;
        }
    }
    // A fixed fleet means idle vehicles are free capacity; give the segment
    // anchored at `u` a chance to claim one.
    for len in 1..=MAX_SEGMENT {
        if try_relocate_to_idle(state, prob, u, len, penalty) {
            return true;
        }
    }
    false
}

/// Moves the `len` customers starting at `u` to just after `v`, optionally
/// reversed. Handles both the inter-route and the intra-route (Or-opt) case.
fn try_relocate(
    state: &mut LsState,
    prob: &Vrp,
    u: usize,
    v: usize,
    len: usize,
    reverse: bool,
    penalty: f64,
) -> bool {
    if reverse && len == 1 {
        return false; // Reversing a single customer is the same move.
    }
    let from = state.route_of[u];
    let pos = state.pos_in[u];
    if pos + len > state.routes[from].len() {
        return false;
    }
    let to = state.route_of[v];
    let target = state.pos_in[v] + 1;
    // The insertion point must lie outside the segment, and re-inserting it
    // exactly where it came from is a no-op.
    if from == to && target >= pos && target <= pos + len {
        return false;
    }

    let (_, first, last, _) = state.segment_ends(from, pos, len);
    let (head, tail) = if reverse {
        (last, first)
    } else {
        (first, last)
    };
    let gain = state.removal_gain(prob, from, pos, len);

    let (delta_distance, delta_excess) = if from == to {
        // Same route: the removal and insertion edges are disjoint (checked
        // above), so the two O(1) deltas still compose.
        (state.insertion_cost(prob, to, target, head, tail) - gain, 0)
    } else {
        let demand = state.segment_demand(prob, from, pos, len);
        (
            state.insertion_cost(prob, to, target, head, tail) - gain,
            state.excess_delta_transfer(prob, from, to, demand),
        )
    };
    let delta = delta_distance + penalty * delta_excess as f64;
    if delta > -MIN_IMPROVEMENT {
        return false;
    }

    apply_relocate(
        state,
        prob,
        &Relocation {
            from,
            pos,
            len,
            to,
            target,
            reverse,
        },
    );
    state.commit(delta_distance, delta_excess);
    true
}

/// Moves the `len` customers starting at `u` into an empty route.
fn try_relocate_to_idle(
    state: &mut LsState,
    prob: &Vrp,
    u: usize,
    len: usize,
    penalty: f64,
) -> bool {
    let from = state.route_of[u];
    let pos = state.pos_in[u];
    if pos + len > state.routes[from].len() || state.routes[from].len() == len {
        // Emptying one route to fill another is a relabeling, not a move.
        return false;
    }
    let Some(to) = state.routes.iter().position(|r| r.is_empty()) else {
        return false;
    };

    let (_, first, last, _) = state.segment_ends(from, pos, len);
    let gain = state.removal_gain(prob, from, pos, len);
    let demand = state.segment_demand(prob, from, pos, len);
    let delta_distance = prob.distance(0, first) + prob.distance(last, 0) - gain;
    let delta_excess = state.excess_delta_transfer(prob, from, to, demand);
    let delta = delta_distance + penalty * delta_excess as f64;
    if delta > -MIN_IMPROVEMENT {
        return false;
    }

    apply_relocate(
        state,
        prob,
        &Relocation {
            from,
            pos,
            len,
            to,
            target: 0,
            reverse: false,
        },
    );
    state.commit(delta_distance, delta_excess);
    true
}

/// The `len` customers at `routes[from][pos..]` move to just before position
/// `target` of route `to`, reversed if `reverse`.
struct Relocation {
    from: usize,
    pos: usize,
    len: usize,
    to: usize,
    target: usize,
    reverse: bool,
}

/// Splices the segment described by `mv` into its target route.
fn apply_relocate(state: &mut LsState, prob: &Vrp, mv: &Relocation) {
    let mut segment: Vec<usize> = state.routes[mv.from]
        .drain(mv.pos..mv.pos + mv.len)
        .collect();
    if mv.reverse {
        segment.reverse();
    }
    let demand: i64 = segment.iter().map(|&c| prob.demands[c]).sum();
    // Removing from earlier in the same route shifts the insertion point left.
    let target = if mv.from == mv.to && mv.target > mv.pos {
        mv.target - mv.len
    } else {
        mv.target
    };
    state.routes[mv.to].splice(target..target, segment);
    if mv.from != mv.to {
        state.loads[mv.from] -= demand;
        state.loads[mv.to] += demand;
    }
    state.reindex(mv.from);
    state.reindex(mv.to);
}

/// Exchanges the segment of `len_u` customers at `u` with the segment of `len_v`
/// customers at `v`. Inter-route only: an intra-route exchange of overlapping
/// segments needs its own case analysis and is already covered by relocation.
fn try_swap(
    state: &mut LsState,
    prob: &Vrp,
    u: usize,
    v: usize,
    len_u: usize,
    len_v: usize,
    penalty: f64,
) -> bool {
    let ru = state.route_of[u];
    let rv = state.route_of[v];
    if ru == rv {
        return false;
    }
    let pu = state.pos_in[u];
    let pv = state.pos_in[v];
    if pu + len_u > state.routes[ru].len() || pv + len_v > state.routes[rv].len() {
        return false;
    }

    let (bu, fu, lu, au) = state.segment_ends(ru, pu, len_u);
    let (bv, fv, lv, av) = state.segment_ends(rv, pv, len_v);
    let delta_distance = prob.distance(bu, fv) + prob.distance(lv, au)
        - prob.distance(bu, fu)
        - prob.distance(lu, au)
        + prob.distance(bv, fu)
        + prob.distance(lu, av)
        - prob.distance(bv, fv)
        - prob.distance(lv, av);
    let demand_u = state.segment_demand(prob, ru, pu, len_u);
    let demand_v = state.segment_demand(prob, rv, pv, len_v);
    let delta_excess = state.excess_delta_exchange(prob, ru, rv, demand_u, demand_v);
    let delta = delta_distance + penalty * delta_excess as f64;
    if delta > -MIN_IMPROVEMENT {
        return false;
    }

    let seg_u: Vec<usize> = state.routes[ru][pu..pu + len_u].to_vec();
    let seg_v: Vec<usize> = state.routes[rv][pv..pv + len_v].to_vec();
    state.routes[ru].splice(pu..pu + len_u, seg_v);
    state.routes[rv].splice(pv..pv + len_v, seg_u);
    state.loads[ru] += demand_v - demand_u;
    state.loads[rv] += demand_u - demand_v;
    state.reindex(ru);
    state.reindex(rv);
    state.commit(delta_distance, delta_excess);
    true
}

/// Reverses the sub-path between `u` and `v` within their shared route.
fn try_two_opt(state: &mut LsState, prob: &Vrp, u: usize, v: usize, penalty: f64) -> bool {
    let r = state.route_of[u];
    if state.route_of[v] != r {
        return false;
    }
    let (i, j) = {
        let (a, b) = (state.pos_in[u], state.pos_in[v]);
        if a < b { (a, b) } else { (b, a) }
    };
    if i == j {
        return false;
    }

    let before = state.before(r, i);
    let after = state.after(r, j);
    let first = state.routes[r][i];
    let last = state.routes[r][j];
    let delta_distance = prob.distance(before, last) + prob.distance(first, after)
        - prob.distance(before, first)
        - prob.distance(last, after);
    // Reversal keeps the route's customer set, so loads and excess are untouched.
    if delta_distance > -MIN_IMPROVEMENT {
        return false;
    }
    let _ = penalty;

    state.routes[r][i..=j].reverse();
    state.reindex(r);
    state.commit(delta_distance, 0);
    true
}

/// Exchanges the tails of the routes of `u` and `v`: the customers after `u` are
/// served by `v`'s vehicle and vice versa.
fn try_two_opt_star(state: &mut LsState, prob: &Vrp, u: usize, v: usize, penalty: f64) -> bool {
    let ru = state.route_of[u];
    let rv = state.route_of[v];
    if ru == rv {
        return false;
    }
    let pu = state.pos_in[u];
    let pv = state.pos_in[v];
    let tail_u = state.after(ru, pu);
    let tail_v = state.after(rv, pv);
    if tail_u == 0 && tail_v == 0 {
        return false; // Both tails empty: nothing to exchange.
    }

    let delta_distance = prob.distance(u, tail_v) + prob.distance(v, tail_u)
        - prob.distance(u, tail_u)
        - prob.distance(v, tail_v);
    let moved_u: i64 = state.routes[ru][pu + 1..]
        .iter()
        .map(|&c| prob.demands[c])
        .sum();
    let moved_v: i64 = state.routes[rv][pv + 1..]
        .iter()
        .map(|&c| prob.demands[c])
        .sum();
    let delta_excess = state.excess_delta_exchange(prob, ru, rv, moved_u, moved_v);
    let delta = delta_distance + penalty * delta_excess as f64;
    if delta > -MIN_IMPROVEMENT {
        return false;
    }

    let suffix_u: Vec<usize> = state.routes[ru].split_off(pu + 1);
    let suffix_v: Vec<usize> = state.routes[rv].split_off(pv + 1);
    state.routes[ru].extend(suffix_v);
    state.routes[rv].extend(suffix_u);
    state.loads[ru] += moved_v - moved_u;
    state.loads[rv] += moved_u - moved_v;
    state.reindex(ru);
    state.reindex(rv);
    state.commit(delta_distance, delta_excess);
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::problem::split_giant_tour;
    use rand::{Rng, SeedableRng};

    fn random_vrp(rng: &mut SmallRng, n: usize, capacity: i64, fleet: usize) -> Vrp {
        let mut coordinates = vec![(0.0, 0.0)];
        let mut demands = vec![0i64];
        for _ in 0..n {
            coordinates.push((rng.random_range(-50.0..50.0), rng.random_range(-50.0..50.0)));
            demands.push(rng.random_range(1..=4));
        }
        Vrp::new("rnd", coordinates, demands, capacity, fleet)
    }

    fn random_tour(rng: &mut SmallRng, n: usize) -> Vec<usize> {
        let mut giant: Vec<usize> = (1..=n).collect();
        giant.shuffle(rng);
        giant
    }

    /// Descends from a random start and returns the state plus its starting cost.
    fn descend(prob: &Vrp, rng: &mut SmallRng, penalty: f64, granularity: usize) -> (LsState, f64) {
        let giant = random_tour(rng, prob.get_n());
        let routes = split_giant_tour(prob, &giant, penalty);
        let mut state = LsState::from_routes(prob, routes);
        let before = state.cost(penalty);
        let neighbors = build_neighbor_lists(prob, granularity);
        let mut order = Vec::new();
        local_search(&mut state, prob, &neighbors, &mut order, rng, penalty, 64);
        (state, before)
    }

    #[test]
    fn local_search_is_monotone() {
        for seed in 0..30u64 {
            let mut rng = SmallRng::seed_from_u64(seed);
            let prob = random_vrp(&mut rng, 25, 10, 5);
            let penalty = 100.0;
            let (state, before) = descend(&prob, &mut rng, penalty, 8);
            assert!(
                state.cost(penalty) <= before + 1e-9,
                "seed {seed}: cost rose from {before} to {}",
                state.cost(penalty)
            );
        }
    }

    #[test]
    fn caches_stay_consistent() {
        for seed in 0..30u64 {
            let mut rng = SmallRng::seed_from_u64(1000 + seed);
            let prob = random_vrp(&mut rng, 25, 10, 5);
            let (state, _) = descend(&prob, &mut rng, 50.0, 8);
            let fresh = LsState::from_routes(&prob, state.routes.clone());
            assert!(
                (fresh.distance - state.distance).abs() < 1e-6,
                "seed {seed}: distance {} vs recomputed {}",
                state.distance,
                fresh.distance
            );
            assert_eq!(fresh.excess, state.excess, "seed {seed}: excess");
            assert_eq!(fresh.loads, state.loads, "seed {seed}: loads");
        }
    }

    #[test]
    fn local_search_keeps_a_valid_partition() {
        for seed in 0..30u64 {
            let mut rng = SmallRng::seed_from_u64(2000 + seed);
            let prob = random_vrp(&mut rng, 25, 10, 5);
            let (state, _) = descend(&prob, &mut rng, 50.0, 8);
            assert_eq!(state.routes.len(), prob.num_vehicles);
            prob.validate_routes(&state.routes).unwrap();
        }
    }

    #[test]
    fn a_large_penalty_restores_feasibility() {
        // 20 customers of demand 1..=4 (~50 total) into 8 × capacity 10 = 80:
        // slack enough that a feasible assignment is always reachable.
        for seed in 0..20u64 {
            let mut rng = SmallRng::seed_from_u64(3000 + seed);
            let prob = random_vrp(&mut rng, 20, 10, 8);
            let (state, _) = descend(&prob, &mut rng, 1e6, 10);
            assert_eq!(
                state.excess, 0,
                "seed {seed}: an expensive penalty left {} units of overload",
                state.excess
            );
        }
    }

    /// Six customers on a circle, one vehicle with ample capacity: the optimum is
    /// the convex-hull cycle, which 2-opt alone reaches from any start.
    #[test]
    fn reaches_the_optimum_on_a_ring() {
        let n = 6;
        let mut coordinates = vec![(0.0, 0.0)];
        let mut demands = vec![0i64];
        for i in 0..n {
            let theta = 2.0 * std::f64::consts::PI * i as f64 / n as f64;
            coordinates.push((10.0 * theta.cos(), 10.0 * theta.sin()));
            demands.push(1);
        }
        let prob = Vrp::new("ring", coordinates, demands, 100, 1);
        let optimum = prob.route_distance(&(1..=n).collect::<Vec<_>>());

        let neighbors = build_neighbor_lists(&prob, n);
        for seed in 0..20u64 {
            let mut rng = SmallRng::seed_from_u64(4000 + seed);
            let giant = random_tour(&mut rng, n);
            let mut state = LsState::from_routes(&prob, vec![giant]);
            let mut order = Vec::new();
            local_search(&mut state, &prob, &neighbors, &mut order, &mut rng, 1.0, 64);
            assert!(
                (state.distance - optimum).abs() < 1e-9,
                "seed {seed}: got {}, optimum {optimum}",
                state.distance
            );
        }
    }

    #[test]
    fn granularity_one_still_terminates() {
        let mut rng = SmallRng::seed_from_u64(5);
        let prob = random_vrp(&mut rng, 30, 12, 6);
        let (state, _) = descend(&prob, &mut rng, 100.0, 1);
        prob.validate_routes(&state.routes).unwrap();
    }

    /// Splitting a route never shortens it (triangle inequality), so an idle
    /// vehicle only earns its keep by absorbing overload — which is exactly what
    /// the fixed-fleet encoding needs it for.
    #[test]
    fn an_idle_vehicle_absorbs_overload() {
        let prob = Vrp::new(
            "overloaded",
            vec![
                (0.0, 0.0),
                (10.0, 0.0),
                (11.0, 0.0),
                (12.0, 0.0),
                (13.0, 0.0),
            ],
            vec![0, 1, 1, 1, 1],
            2,
            2,
        );
        let mut state = LsState::from_routes(&prob, vec![vec![1, 2, 3, 4], Vec::new()]);
        assert_eq!(
            state.excess, 2,
            "one vehicle cannot carry four unit demands"
        );

        let neighbors = build_neighbor_lists(&prob, 4);
        let mut order = Vec::new();
        let mut rng = SmallRng::seed_from_u64(0);
        local_search(
            &mut state, &prob, &neighbors, &mut order, &mut rng, 100.0, 64,
        );

        assert_eq!(
            state.excess, 0,
            "the idle vehicle should have taken the excess: {:?}",
            state.routes
        );
        assert!(state.routes.iter().all(|r| !r.is_empty()));
    }
}
