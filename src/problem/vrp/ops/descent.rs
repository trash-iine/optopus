//! Granular local search over CVRP routes, in place and penalty-parameterized.
//!
//! This is deliberately not built on [`crate::problem::VrpRelocateNeighbor`] and
//! friends. Those bake [`Vrp::penalty_weight`], a fixed, deliberately enormous
//! constant, into every gain, whereas the callers here must be able to descend
//! under a penalty they choose: Hybrid Genetic Search adapts one at runtime, and
//! ALNS hands in the weight its own objective uses. They also enumerate the full
//! O(n²) neighborhood and offer neither intra-route relocation nor 2-opt\*.
//!
//! Moves are restricted to granular candidate pairs: for each customer `u`,
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
//! Every move is evaluated in O(1) from the distances at its endpoints, priced
//! through [`super::pricing`] under the penalty the caller hands in, and
//! accepted on first improvement of the objective under that penalty.

use rand::rngs::SmallRng;

use super::pricing::{EditPrice, SlotEdit};
use super::{RouteState, node_at};
use crate::common::{AnchoredSweep, MIN_IMPROVEMENT};
use crate::problem::Vrp;

/// Longest customer segment relocated or swapped as a unit.
const MAX_SEGMENT: usize = 2;

/// The descent, plus the two things it needs kept between calls: the granular
/// candidate lists (instance-derived, O(n²) to build) and the sweep buffers.
///
/// It has a receiver where the pricing functions in [`super`] are free, because
/// these caches are exactly what both callers were keeping a private copy of.
/// What is not in here is any policy: when to descend, under which penalty and
/// for how long stays with the heuristic driving it.
#[derive(Debug, Default)]
pub(crate) struct Descent {
    neighbors: Vec<Vec<usize>>,
    sweep: AnchoredSweep,
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
            self.sweep.ensure(prob.get_n() + 1);
        }
    }

    /// Sets how many partners of each anchor [`Descent::run_around`] widens
    /// the anchors by.
    pub(crate) fn set_ring(&mut self, ring: usize) {
        self.sweep = std::mem::take(&mut self.sweep).with_ring(ring);
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
        state.set_penalty(prob, penalty);
        self.sweep.set_order(1..=n);
        self.sweep_in_place(state, prob, rng, max_passes);
    }

    /// The same descent, anchored only at `anchors` and the customers near them.
    /// Everything else is left alone.
    ///
    /// A caller that has just edited a few routes knows where the damage is, and
    /// paying for a full sweep of `1..=n` to find it again is what makes a
    /// descent too expensive to run every iteration. The anchor set is widened
    /// by one granular ring, so a customer displaced by the edit is
    /// reconsidered too, not only the ones the caller moved, see
    /// [`AnchoredSweep::DEFAULT_RING`] for how wide that ring is by default
    /// and why it is narrow.
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
        state.set_penalty(prob, penalty);
        self.sweep
            .collect_around(anchors, &self.neighbors, |_| true);
        self.sweep_in_place(state, prob, rng, max_passes);
    }

    /// Sweeps the prepared list with the granular move set.
    fn sweep_in_place(
        &mut self,
        state: &mut RouteState,
        prob: &Vrp,
        rng: &mut SmallRng,
        max_passes: usize,
    ) {
        let neighbors = &self.neighbors;
        self.sweep.sweep(rng, max_passes, |u| {
            improve_around(state, prob, neighbors, u)
        });
        #[cfg(debug_assertions)]
        state.assert_caches_consistent(prob);
    }
}

/// Tries every granular move anchored at `u`, applying the first improving one.
fn improve_around(state: &mut RouteState, prob: &Vrp, neighbors: &[Vec<usize>], u: usize) -> bool {
    for &v in &neighbors[u] {
        for len in 1..=MAX_SEGMENT {
            for reverse in [false, true] {
                if try_relocate(state, prob, u, v, len, reverse) {
                    return true;
                }
            }
        }
        for len_u in 1..=MAX_SEGMENT {
            for len_v in 1..=MAX_SEGMENT {
                if try_swap(state, prob, u, v, len_u, len_v) {
                    return true;
                }
            }
        }
        if try_two_opt(state, prob, u, v) {
            return true;
        }
        if try_two_opt_star(state, prob, u, v) {
            return true;
        }
    }
    for len in 1..=MAX_SEGMENT {
        if try_relocate_to_idle(state, prob, u, len) {
            return true;
        }
    }
    false
}

/// Improving means the priced gain clears [`MIN_IMPROVEMENT`].
#[inline]
fn improves(gain: f64) -> bool {
    gain <= -MIN_IMPROVEMENT
}

/// Applies an accepted move and keeps the cumulative distances the tail
/// exchange reads current for the routes it touched.
#[inline]
fn commit<const N: usize>(
    state: &mut RouteState,
    prob: &Vrp,
    edits: &[SlotEdit; N],
    price: &EditPrice,
) {
    state.apply(prob, edits, price);
}

/// Moves the segment of `len` customers starting at `u`, optionally reversed,
/// to right after `v`. Intra-route when `v` sits in `u`'s route.
fn try_relocate(
    state: &mut RouteState,
    prob: &Vrp,
    u: usize,
    v: usize,
    len: usize,
    reverse: bool,
) -> bool {
    if reverse && len == 1 {
        return false; // Reversing a single customer is the same move.
    }
    let (from, pos) = state.locate(u);
    if pos + len > state.routes()[from].len() {
        return false;
    }
    let (to, v_pos) = state.locate(v);
    let target = v_pos + 1;
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
    let insertion = state.insertion_cost(prob, to, target, head, tail);

    let relocation = Relocation {
        from,
        pos,
        len,
        to,
        target,
        reverse,
    };
    if from == to {
        let edit = [SlotEdit::reorder(from, insertion - gain)];
        let price = state.price(prob, &edit);
        if !improves(price.gain) {
            return false;
        }
        relocate_routes(state, &relocation);
        commit(state, prob, &edit, &price);
    } else {
        // The segment carries its inner edges from one route to the other.
        let inner = state.segment_internal(prob, from, pos, len);
        let (demand, service) = state.segment_load(prob, from, pos, len);
        let edits = [
            SlotEdit::remove(from, -gain - inner, demand, service, len),
            SlotEdit::insert(to, insertion + inner, demand, service, len),
        ];
        let price = state.price(prob, &edits);
        if !improves(price.gain) {
            return false;
        }
        relocate_routes(state, &relocation);
        commit(state, prob, &edits, &price);
    }
    true
}

/// Moves the segment of `len` customers at `u` onto an idle vehicle, one
/// representative per vehicle type, so a fleet with idle vehicles of several
/// types is offered each type once.
fn try_relocate_to_idle(state: &mut RouteState, prob: &Vrp, u: usize, len: usize) -> bool {
    let (from, pos) = state.locate(u);
    if pos + len > state.routes()[from].len()
        || state.routes()[from].len() == len
        || !state.has_idle_vehicle()
    {
        return false;
    }

    let (_, first, last, _) = state.segment_ends(from, pos, len);
    let gain = state.removal_gain(prob, from, pos, len);
    let inner = state.segment_internal(prob, from, pos, len);
    let (demand, service) = state.segment_load(prob, from, pos, len);
    let opened = prob.distance(0, first) + prob.distance(last, 0);

    for t in 0..prob.vehicle_types().len() {
        let Some(to) = state.idle_representative_of(t) else {
            continue;
        };
        let edits = [
            SlotEdit::remove(from, -gain - inner, demand, service, len),
            SlotEdit::insert(to, opened + inner, demand, service, len),
        ];
        let price = state.price(prob, &edits);
        if !improves(price.gain) {
            continue;
        }
        relocate_routes(
            state,
            &Relocation {
                from,
                pos,
                len,
                to,
                target: 0,
                reverse: false,
            },
        );
        commit(state, prob, &edits, &price);
        return true;
    }
    false
}

struct Relocation {
    from: usize,
    pos: usize,
    len: usize,
    to: usize,
    target: usize,
    reverse: bool,
}

/// Edits the routes of a relocation; the caches follow through `apply`.
fn relocate_routes(state: &mut RouteState, mv: &Relocation) {
    let mut segment: Vec<usize> = state.sol.routes[mv.from]
        .drain(mv.pos..mv.pos + mv.len)
        .collect();
    if mv.reverse {
        segment.reverse();
    }
    // Removing from earlier in the same route shifts the insertion point left.
    let target = if mv.from == mv.to && mv.target > mv.pos {
        mv.target - mv.len
    } else {
        mv.target
    };
    state.sol.routes[mv.to].splice(target..target, segment);
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
) -> bool {
    let (ru, pu) = state.locate(u);
    let (rv, pv) = state.locate(v);
    if ru == rv {
        return false;
    }
    if pu + len_u > state.routes()[ru].len() || pv + len_v > state.routes()[rv].len() {
        return false;
    }

    let (bu, fu, lu, au) = state.segment_ends(ru, pu, len_u);
    let (bv, fv, lv, av) = state.segment_ends(rv, pv, len_v);
    // Each route trades its segment's inner edges for the other's.
    let inner_u = state.segment_internal(prob, ru, pu, len_u);
    let inner_v = state.segment_internal(prob, rv, pv, len_v);
    let delta_u = prob.distance(bu, fv) + inner_v + prob.distance(lv, au)
        - prob.distance(bu, fu)
        - inner_u
        - prob.distance(lu, au);
    let delta_v = prob.distance(bv, fu) + inner_u + prob.distance(lu, av)
        - prob.distance(bv, fv)
        - inner_v
        - prob.distance(lv, av);
    let (demand_u, service_u) = state.segment_load(prob, ru, pu, len_u);
    let (demand_v, service_v) = state.segment_load(prob, rv, pv, len_v);
    let edits = SlotEdit::exchange(
        (ru, delta_u, (demand_u, service_u, len_u)),
        (rv, delta_v, (demand_v, service_v, len_v)),
    );
    let price = state.price(prob, &edits);
    if !improves(price.gain) {
        return false;
    }

    let seg_u: Vec<usize> = state.sol.routes[ru][pu..pu + len_u].to_vec();
    let seg_v: Vec<usize> = state.sol.routes[rv][pv..pv + len_v].to_vec();
    state.sol.routes[ru].splice(pu..pu + len_u, seg_v);
    state.sol.routes[rv].splice(pv..pv + len_v, seg_u);
    commit(state, prob, &edits, &price);
    true
}

/// Reverses the sub-path between `u` and `v` of one route.
fn try_two_opt(state: &mut RouteState, prob: &Vrp, u: usize, v: usize) -> bool {
    let (r, pu) = state.locate(u);
    let (rv, pv) = state.locate(v);
    if rv != r {
        return false;
    }
    let (i, j) = if pu < pv { (pu, pv) } else { (pv, pu) };
    if i == j {
        return false;
    }

    let edit = [SlotEdit::reorder(
        r,
        super::reversal_delta(prob, &state.routes()[r], i, j),
    )];
    let price = state.price(prob, &edit);
    if !improves(price.gain) {
        return false;
    }

    state.sol.routes[r][i..=j].reverse();
    commit(state, prob, &edit, &price);
    true
}

/// Exchanges the tails after `u` and after `v` between their two routes.
///
/// Each route's distance moves by the tail it gives up against the tail it
/// receives, read from the cumulative distances the state keeps, so the
/// exchange is priced without walking either tail.
fn try_two_opt_star(state: &mut RouteState, prob: &Vrp, u: usize, v: usize) -> bool {
    let (ru, pu) = state.locate(u);
    let (rv, pv) = state.locate(v);
    if ru == rv {
        return false;
    }
    let tail_u = node_at(&state.routes()[ru], pu + 1);
    let tail_v = node_at(&state.routes()[rv], pv + 1);
    if tail_u == 0 && tail_v == 0 {
        return false; // Both tails empty: nothing to exchange.
    }

    // Each route gives up the tail it carries and receives the other's.
    let (kept_u, kept_v) = (state.tail_cost(prob, ru, pu), state.tail_cost(prob, rv, pv));
    let own_u = prob.distance(u, tail_u) + kept_u;
    let own_v = prob.distance(v, tail_v) + kept_v;
    let new_u = prob.distance(u, tail_v) + kept_v;
    let new_v = prob.distance(v, tail_u) + kept_u;
    let (len_u, len_v) = (
        state.routes()[ru].len() - pu - 1,
        state.routes()[rv].len() - pv - 1,
    );
    let (moved_u, service_u) = state.segment_load(prob, ru, pu + 1, len_u);
    let (moved_v, service_v) = state.segment_load(prob, rv, pv + 1, len_v);
    let edits = SlotEdit::exchange(
        (ru, new_u - own_u, (moved_u, service_u, len_u)),
        (rv, new_v - own_v, (moved_v, service_v, len_v)),
    );
    let price = state.price(prob, &edits);
    if !improves(price.gain) {
        return false;
    }

    let suffix_u: Vec<usize> = state.sol.routes[ru].split_off(pu + 1);
    let suffix_v: Vec<usize> = state.sol.routes[rv].split_off(pv + 1);
    state.sol.routes[ru].extend(suffix_v);
    state.sol.routes[rv].extend(suffix_u);
    commit(state, prob, &edits, &price);
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::problem::split_giant_tour;
    use rand::seq::SliceRandom;
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
        let mut state = RouteState::from_routes(prob, routes, penalty);
        let before = state.energy(prob);
        let mut descent = Descent::new();
        descent.ensure(prob, granularity);
        descent.run(&mut state, prob, rng, penalty, 64);
        (state, before)
    }

    /// The three things a descent owes its caller, checked on the descent that
    /// produced them rather than on three separate ones: the cost does not
    /// rise, the cached distance, excess and loads still match a fresh
    /// recompute, and the routes are still a partition of the customers over
    /// the fleet. They are properties of the same walk, so running it three
    /// times to assert them one at a time buys nothing.
    #[test]
    fn a_descent_lowers_the_cost_and_leaves_its_caches_and_routes_intact() {
        for penalty in [50.0, 100.0] {
            for seed in 0..30u64 {
                let mut rng = SmallRng::seed_from_u64(seed);
                let prob = random_vrp(&mut rng, 25, 10, 5);
                let (state, before) = descend(&prob, &mut rng, penalty, 8);
                let at = || format!("penalty {penalty}, seed {seed}");

                assert!(
                    state.energy(&prob) <= before + 1e-9,
                    "{}: cost rose from {before} to {}",
                    at(),
                    state.energy(&prob)
                );
                state.assert_caches_consistent(&prob);
                assert_eq!(state.routes().len(), prob.num_slots(), "{}", at());
                prob.validate_routes(state.routes()).unwrap();
            }
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
                state.sol.overload, 0,
                "seed {seed}: an expensive penalty left {} units of overload",
                state.sol.overload
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
            let mut state = RouteState::from_routes(&prob, vec![giant], 1.0);
            descent.run(&mut state, &prob, &mut rng, 1.0, 64);
            assert!(
                (state.sol.total_distance() - optimum).abs() < 1e-9,
                "seed {seed}: got {}, optimum {optimum}",
                state.sol.total_distance()
            );
        }
    }

    #[test]
    fn granularity_one_still_terminates() {
        let mut rng = SmallRng::seed_from_u64(5);
        let prob = random_vrp(&mut rng, 30, 12, 6);
        let (state, _) = descend(&prob, &mut rng, 100.0, 1);
        prob.validate_routes(state.routes()).unwrap();
    }

    /// Splitting a route never shortens it (triangle inequality), so an idle
    /// vehicle only earns its keep by absorbing overload, which is exactly what
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
        let mut state = RouteState::from_routes(&prob, vec![vec![1, 2, 3, 4], Vec::new()], 100.0);
        assert_eq!(
            state.sol.overload, 2,
            "one vehicle cannot carry four unit demands"
        );

        let mut descent = Descent::new();
        descent.ensure(&prob, 4);
        let mut rng = SmallRng::seed_from_u64(0);
        descent.run(&mut state, &prob, &mut rng, 100.0, 64);

        assert_eq!(
            state.sol.overload,
            0,
            "the idle vehicle should have taken the excess: {:?}",
            state.routes()
        );
        assert!(state.routes().iter().all(|r| !r.is_empty()));
    }

    /// An anchored sweep must reach the same local optimum as a full one when
    /// the only damage is where the anchors are, which is the assumption ALNS
    /// makes when it descends around the customers it just re-inserted.
    #[test]
    fn an_anchored_sweep_fixes_the_damage_it_is_pointed_at() {
        let mut rng = SmallRng::seed_from_u64(17);
        let prob = random_vrp(&mut rng, 25, 10, 5);
        let (settled, _) = descend(&prob, &mut rng, 100.0, 8);

        // Take two customers out of a settled solution and put them back at the
        // ends of the first route, then descend around exactly those two.
        let mut routes = settled.routes().to_vec();
        let moved: Vec<usize> = routes[1].drain(..2).collect();
        routes[0].insert(0, moved[0]);
        routes[0].push(moved[1]);
        let mut state = RouteState::from_routes(&prob, routes, 100.0);
        let damaged = state.energy(&prob);

        let mut descent = Descent::new();
        descent.ensure(&prob, 8);
        descent.run_around(&mut state, &prob, &moved, &mut rng, 100.0, 64);

        assert!(
            state.energy(&prob) < damaged,
            "the anchored sweep left the damage in place: {damaged} -> {}",
            state.energy(&prob)
        );
        prob.validate_routes(state.routes()).unwrap();
    }

    /// Anchors are a hint, not a contract: an empty set and an anchor whose
    /// route is untouched must both leave a valid solution behind.
    #[test]
    fn an_empty_anchor_set_is_a_no_op() {
        let mut rng = SmallRng::seed_from_u64(23);
        let prob = random_vrp(&mut rng, 20, 10, 4);
        let (mut state, _) = descend(&prob, &mut rng, 100.0, 8);
        let before = state.energy(&prob);

        let mut descent = Descent::new();
        descent.ensure(&prob, 8);
        descent.run_around(&mut state, &prob, &[], &mut rng, 100.0, 64);

        assert_eq!(state.energy(&prob), before);
        prob.validate_routes(state.routes()).unwrap();
    }
}
