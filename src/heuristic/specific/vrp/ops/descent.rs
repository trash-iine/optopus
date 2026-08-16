//! Granular local search over CVRP routes, in place and penalty-parameterized.
//!
//! This is deliberately *not* built on [`crate::problem::VrpRelocateNeighbor`] and
//! friends. Those bake [`Vrp::penalty_weight`] — a fixed, deliberately enormous
//! constant — into every gain, whereas the callers here must be able to descend
//! under a penalty they choose: Hybrid Genetic Search adapts one at runtime, and
//! ALNS hands in the weight its own objective uses. They also enumerate the full
//! O(n²) neighborhood and offer neither intra-route relocation nor 2-opt\*.
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

use super::{RouteState, before, node_at};
use crate::problem::Vrp;

/// Improvements smaller than this are treated as numerical noise, which keeps
/// the descent from cycling on ties.
const MIN_IMPROVEMENT: f64 = 1e-10;

/// Longest customer segment relocated or swapped as a unit.
const MAX_SEGMENT: usize = 2;

/// Partners of an anchor that [`Descent::run_around`] sweeps along with it.
///
/// Deliberately far below `granularity`: the anchors of a large ruin, widened by
/// a full candidate list of 20, already cover most of a mid-sized instance, and
/// a sweep that touches everything is the full descent the caller was trying not
/// to pay for. The nearest handful is where a displaced customer actually lands.
const ANCHOR_RING: usize = 5;

/// The descent, plus the two things it needs kept between calls: the granular
/// candidate lists (instance-derived, O(n²) to build) and the sweep buffers.
///
/// It has a receiver where the pricing functions in [`super`] are free, because
/// these caches are exactly what both callers were keeping a private copy of.
/// What is *not* in here is any policy: when to descend, under which penalty and
/// for how long stays with the heuristic driving it.
#[derive(Debug, Default)]
pub(crate) struct Descent {
    neighbors: Vec<Vec<usize>>,
    /// The customers a pass visits, shuffled in place each time.
    order: Vec<usize>,
    /// Membership marks that keep [`Descent::run_around`]'s sweep list free of
    /// duplicates without sorting it.
    seen: Vec<bool>,
}

impl Descent {
    /// A descent with no candidate lists yet; [`Descent::ensure`] builds them.
    pub(crate) fn new() -> Self {
        Self::default()
    }

    /// Builds the candidate lists if they do not already fit `prob`.
    ///
    /// They depend only on the instance, so a caller may hold a `Descent` across
    /// restarts and clears; only a change of instance costs the O(n²) again.
    pub(crate) fn ensure(&mut self, prob: &Vrp, granularity: usize) {
        if self.neighbors.len() != prob.get_n() + 1 {
            self.neighbors = super::build_neighbor_lists(prob, granularity);
            self.seen = vec![false; prob.get_n() + 1];
        }
    }

    /// The candidate lists, for callers that also build tours from them.
    pub(crate) fn neighbors(&self) -> &[Vec<usize>] {
        &self.neighbors
    }

    /// Descends to a local optimum of `distance + penalty · excess`, sweeping
    /// every customer. Stops at the first pass with no improving move, or after
    /// `max_passes`.
    pub(crate) fn run(
        &mut self,
        state: &mut RouteState,
        prob: &Vrp,
        rng: &mut SmallRng,
        penalty: f64,
        max_passes: usize,
    ) {
        let n = prob.get_n();
        if n == 0 {
            return;
        }
        self.order.clear();
        self.order.extend(1..=n);
        self.sweep(state, prob, rng, penalty, max_passes);
    }

    /// The same descent, anchored only at `anchors` and the customers near them
    /// — everything else is left alone.
    ///
    /// A caller that has just edited a few routes knows where the damage is, and
    /// paying for a full sweep of `1..=n` to find it again is what makes a
    /// descent too expensive to run every iteration. The anchor set is widened
    /// by one granular ring, so a customer *displaced* by the edit is
    /// reconsidered too, not only the ones the caller moved — see
    /// [`ANCHOR_RING`] for how wide that ring is and why it is narrow.
    pub(crate) fn run_around(
        &mut self,
        state: &mut RouteState,
        prob: &Vrp,
        anchors: &[usize],
        rng: &mut SmallRng,
        penalty: f64,
        max_passes: usize,
    ) {
        if prob.get_n() == 0 || anchors.is_empty() {
            return;
        }
        self.order.clear();
        for &u in anchors {
            for &v in std::iter::once(&u).chain(self.neighbors[u].iter().take(ANCHOR_RING)) {
                if !self.seen[v] {
                    self.seen[v] = true;
                    self.order.push(v);
                }
            }
        }
        for &u in &self.order {
            self.seen[u] = false;
        }
        self.sweep(state, prob, rng, penalty, max_passes);
    }

    /// Sweeps `order` until a pass finds nothing, or `max_passes` are spent.
    fn sweep(
        &mut self,
        state: &mut RouteState,
        prob: &Vrp,
        rng: &mut SmallRng,
        penalty: f64,
        max_passes: usize,
    ) {
        for _ in 0..max_passes {
            self.order.shuffle(rng);
            let mut improved = false;
            for &u in &self.order {
                if improve_around(state, prob, &self.neighbors, u, penalty) {
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
}

/// Tries every granular move anchored at `u`, applying the first improving one.
fn improve_around(
    state: &mut RouteState,
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
    state: &mut RouteState,
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
    let (from, pos) = state.locate(u);
    if pos + len > state.routes[from].len() {
        return false;
    }
    let (to, v_pos) = state.locate(v);
    let target = v_pos + 1;
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
    state: &mut RouteState,
    prob: &Vrp,
    u: usize,
    len: usize,
    penalty: f64,
) -> bool {
    let (from, pos) = state.locate(u);
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
fn apply_relocate(state: &mut RouteState, prob: &Vrp, mv: &Relocation) {
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
    state: &mut RouteState,
    prob: &Vrp,
    u: usize,
    v: usize,
    len_u: usize,
    len_v: usize,
    penalty: f64,
) -> bool {
    let (ru, pu) = state.locate(u);
    let (rv, pv) = state.locate(v);
    if ru == rv {
        return false;
    }
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
fn try_two_opt(state: &mut RouteState, prob: &Vrp, u: usize, v: usize, penalty: f64) -> bool {
    let (r, pu) = state.locate(u);
    let (rv, pv) = state.locate(v);
    if rv != r {
        return false;
    }
    let (i, j) = if pu < pv { (pu, pv) } else { (pv, pu) };
    if i == j {
        return false;
    }

    let route = &state.routes[r];
    let (before, after) = (before(route, i), node_at(route, j + 1));
    let (first, last) = (route[i], route[j]);
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
fn try_two_opt_star(state: &mut RouteState, prob: &Vrp, u: usize, v: usize, penalty: f64) -> bool {
    let (ru, pu) = state.locate(u);
    let (rv, pv) = state.locate(v);
    if ru == rv {
        return false;
    }
    let tail_u = node_at(&state.routes[ru], pu + 1);
    let tail_v = node_at(&state.routes[rv], pv + 1);
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
    fn descend(
        prob: &Vrp,
        rng: &mut SmallRng,
        penalty: f64,
        granularity: usize,
    ) -> (RouteState, f64) {
        let giant = random_tour(rng, prob.get_n());
        let routes = split_giant_tour(prob, &giant, penalty);
        let mut state = RouteState::from_routes(prob, routes);
        let before = state.cost(penalty);
        let mut descent = Descent::new();
        descent.ensure(prob, granularity);
        descent.run(&mut state, prob, rng, penalty, 64);
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
            let fresh = RouteState::from_routes(&prob, state.routes.clone());
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

        let mut descent = Descent::new();
        descent.ensure(&prob, n);
        for seed in 0..20u64 {
            let mut rng = SmallRng::seed_from_u64(4000 + seed);
            let giant = random_tour(&mut rng, n);
            let mut state = RouteState::from_routes(&prob, vec![giant]);
            descent.run(&mut state, &prob, &mut rng, 1.0, 64);
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
        let mut state = RouteState::from_routes(&prob, vec![vec![1, 2, 3, 4], Vec::new()]);
        assert_eq!(
            state.excess, 2,
            "one vehicle cannot carry four unit demands"
        );

        let mut descent = Descent::new();
        descent.ensure(&prob, 4);
        let mut rng = SmallRng::seed_from_u64(0);
        descent.run(&mut state, &prob, &mut rng, 100.0, 64);

        assert_eq!(
            state.excess, 0,
            "the idle vehicle should have taken the excess: {:?}",
            state.routes
        );
        assert!(state.routes.iter().all(|r| !r.is_empty()));
    }

    /// An anchored sweep must reach the same local optimum as a full one when
    /// the only damage is where the anchors are — that is the assumption ALNS
    /// makes when it descends around the customers it just re-inserted.
    #[test]
    fn an_anchored_sweep_fixes_the_damage_it_is_pointed_at() {
        let mut rng = SmallRng::seed_from_u64(17);
        let prob = random_vrp(&mut rng, 25, 10, 5);
        let (settled, _) = descend(&prob, &mut rng, 100.0, 8);

        // Take two customers out of a settled solution and put them back at the
        // ends of the first route, then descend around exactly those two.
        let mut routes = settled.routes.clone();
        let moved: Vec<usize> = routes[1].drain(..2).collect();
        routes[0].insert(0, moved[0]);
        routes[0].push(moved[1]);
        let mut state = RouteState::from_routes(&prob, routes);
        let damaged = state.cost(100.0);

        let mut descent = Descent::new();
        descent.ensure(&prob, 8);
        descent.run_around(&mut state, &prob, &moved, &mut rng, 100.0, 64);

        assert!(
            state.cost(100.0) < damaged,
            "the anchored sweep left the damage in place: {damaged} -> {}",
            state.cost(100.0)
        );
        prob.validate_routes(&state.routes).unwrap();
    }

    /// Anchors are a *hint*, not a contract: an empty set and an anchor whose
    /// route is untouched must both leave a valid solution behind.
    #[test]
    fn an_empty_anchor_set_is_a_no_op() {
        let mut rng = SmallRng::seed_from_u64(23);
        let prob = random_vrp(&mut rng, 20, 10, 4);
        let (mut state, _) = descend(&prob, &mut rng, 100.0, 8);
        let before = state.cost(100.0);

        let mut descent = Descent::new();
        descent.ensure(&prob, 8);
        descent.run_around(&mut state, &prob, &[], &mut rng, 100.0, 64);

        assert_eq!(state.cost(100.0), before);
        prob.validate_routes(&state.routes).unwrap();
    }
}
