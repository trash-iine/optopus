//! The route-level operators for the CVRP, shared by both heuristics in this
//! directory.
//!
//! Four layers, each a module: this one prices a route edit, [`route_state`]
//! keeps a route partition and its caches, [`granular`] says which customer
//! pairs are worth considering at all, and [`descent`] walks downhill over the
//! moves those pairs allow.
//!
//! Everything here is parameterized by the capacity **penalty** rather than
//! reading [`Vrp::penalty_weight`](crate::problem::vrp::Vrp::penalty_weight),
//! which is what lets one descent serve both callers: ALNS hands it the fixed
//! weight its objective already uses, HGS hands it the weight it is currently
//! tuning, and neither has to own a copy of the move set. What each caller
//! still decides for itself is *when* to descend and over which customers —
//! HGS over every one of a freshly decoded offspring, ALNS only around the ones
//! it has just re-inserted.
//!
//! The pricing functions below are free rather than methods because both
//! callers hold their routes differently — in a [`RouteState`] with position
//! indexes while descending, in the plain `Vec<Vec<usize>>` ALNS ruins and
//! recreates — and what they share is the arithmetic, not the container.
//! Sharing it is not cosmetic: these are the formulas that decide what a move
//! costs, so a second copy is a second answer to "how long is this route".
//! [`Descent`] does have a receiver, because it owns caches (the candidate
//! lists, the sweep buffers) that both callers were otherwise keeping their own
//! copy of.

mod descent;
mod granular;
mod route_state;

pub(super) use descent::Descent;
use granular::build_neighbor_lists;
pub(super) use route_state::RouteState;

use crate::problem::Vrp;
use crate::problem::vrp::overload_of;

/// The node at `pos` of `route`, or the depot when `pos` is past its end.
#[inline]
pub(super) fn node_at(route: &[usize], pos: usize) -> usize {
    route.get(pos).copied().unwrap_or(0)
}

/// The node preceding `pos`, or the depot when `pos` is the start of the route.
#[inline]
pub(super) fn before(route: &[usize], pos: usize) -> usize {
    if pos == 0 { 0 } else { route[pos - 1] }
}

/// `(before, first, last, after)` around the segment `route[pos..pos + len]`,
/// with the depot standing in at either end of the route.
#[inline]
pub(super) fn segment_ends(
    route: &[usize],
    pos: usize,
    len: usize,
) -> (usize, usize, usize, usize) {
    (
        before(route, pos),
        route[pos],
        route[pos + len - 1],
        node_at(route, pos + len),
    )
}

/// Distance saved by lifting `route[pos..pos + len]` out of its route.
///
/// The segment's own edges are *not* counted: it is lifted out to be put back
/// somewhere, and it carries its internal edges with it. Removal and insertion
/// therefore compose into the cost of a relocation without either of them ever
/// naming the segment's length.
#[inline]
pub(super) fn removal_gain(prob: &Vrp, route: &[usize], pos: usize, len: usize) -> f64 {
    let (before, first, last, after) = segment_ends(route, pos, len);
    prob.distance(before, first) + prob.distance(last, after) - prob.distance(before, after)
}

/// Distance added by inserting the segment `first…last` *before* position `pos`
/// of `route`, its internal edges excluded for the reason
/// [`removal_gain`] excludes them.
#[inline]
pub(super) fn insertion_cost(
    prob: &Vrp,
    route: &[usize],
    pos: usize,
    first: usize,
    last: usize,
) -> f64 {
    let (before, after) = (before(route, pos), node_at(route, pos));
    prob.distance(before, first) + prob.distance(last, after) - prob.distance(before, after)
}

/// Total demand of `route[pos..pos + len]`.
#[inline]
pub(super) fn segment_demand(prob: &Vrp, route: &[usize], pos: usize, len: usize) -> i64 {
    route[pos..pos + len].iter().map(|&c| prob.demands[c]).sum()
}

/// Recomputes every route's load in place.
pub(super) fn route_loads(prob: &Vrp, routes: &[Vec<usize>], loads: &mut [i64]) {
    for (r, route) in routes.iter().enumerate() {
        loads[r] = route.iter().map(|&c| prob.demands[c]).sum();
    }
}

/// Change in total overflow when `demand` is added to a route carrying `load`.
#[inline]
pub(super) fn excess_delta_insert(capacity: i64, load: i64, demand: i64) -> i64 {
    overload_of(load + demand, capacity) - overload_of(load, capacity)
}

/// Change in total overflow when `demand` moves from one route to another.
#[inline]
pub(super) fn excess_delta_transfer(capacity: i64, from: i64, to: i64, demand: i64) -> i64 {
    overload_of(from - demand, capacity) - overload_of(from, capacity)
        + overload_of(to + demand, capacity)
        - overload_of(to, capacity)
}

/// Change in total overflow when two routes exchange `demand_a` for `demand_b`.
#[inline]
pub(super) fn excess_delta_exchange(
    capacity: i64,
    load_a: i64,
    load_b: i64,
    demand_a: i64,
    demand_b: i64,
) -> i64 {
    overload_of(load_a - demand_a + demand_b, capacity) - overload_of(load_a, capacity)
        + overload_of(load_b - demand_b + demand_a, capacity)
        - overload_of(load_b, capacity)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Five customers on a line right of the depot, so every distance is exact
    /// in binary and a hand-computed expectation is meaningful.
    fn line_vrp() -> Vrp {
        Vrp::new(
            "line",
            vec![
                (0.0, 0.0),
                (1.0, 0.0),
                (2.0, 0.0),
                (4.0, 0.0),
                (8.0, 0.0),
                (16.0, 0.0),
            ],
            vec![0, 1, 2, 3, 4, 5],
            10,
            2,
        )
    }

    /// Both deltas must equal the difference of two from-scratch route lengths,
    /// up to the segment's internal edges they deliberately leave out — that
    /// equality is the whole reason a heuristic may trust an O(1) gain.
    #[test]
    fn removal_and_insertion_price_the_edit_they_describe() {
        let prob = line_vrp();
        let route = vec![1, 2, 3, 4, 5];
        let full = prob.route_distance(&route);

        for len in 1..=3 {
            for pos in 0..=route.len() - len {
                let mut cut = route.clone();
                let segment: Vec<usize> = cut.drain(pos..pos + len).collect();
                // The edges inside the segment travel with it, so neither delta
                // counts them and both comparisons have to add them back.
                let inside: f64 = segment.windows(2).map(|w| prob.distance(w[0], w[1])).sum();
                let shortened = prob.route_distance(&cut);
                let gain = removal_gain(&prob, &route, pos, len);
                assert!(
                    (full - shortened - inside - gain).abs() < 1e-9,
                    "removal_gain({pos}, {len}) = {gain}, routes say {}",
                    full - shortened - inside
                );

                // Putting it back anywhere costs exactly what insertion_cost says.
                let (first, last) = (segment[0], segment[len - 1]);
                for target in 0..=cut.len() {
                    let mut rebuilt = cut.clone();
                    rebuilt.splice(target..target, segment.iter().copied());
                    let cost = insertion_cost(&prob, &cut, target, first, last);
                    assert!(
                        (prob.route_distance(&rebuilt) - shortened - inside - cost).abs() < 1e-9,
                        "insertion_cost({target}) = {cost} for segment {segment:?}"
                    );
                }

                // Removing and re-inserting where it came from is free.
                let round_trip = gain - insertion_cost(&prob, &cut, pos, first, last);
                assert!(
                    round_trip.abs() < 1e-9,
                    "putting the segment back at {pos} moved the objective by {round_trip}"
                );
            }
        }
    }

    #[test]
    fn segment_ends_and_demand_read_the_route() {
        let prob = line_vrp();
        let route = vec![1, 2, 3, 4, 5];
        assert_eq!(segment_ends(&route, 0, 2), (0, 1, 2, 3));
        assert_eq!(segment_ends(&route, 3, 2), (3, 4, 5, 0));
        assert_eq!(segment_demand(&prob, &route, 1, 3), 2 + 3 + 4);
    }

    /// The three excess deltas must agree with recomputing total overflow, which
    /// is what the accepted move's `excess` cache is later checked against.
    #[test]
    fn excess_deltas_match_a_recompute() {
        let capacity = 10;
        let total = |a: i64, b: i64| overload_of(a, capacity) + overload_of(b, capacity);
        for load_a in [0, 5, 10, 14] {
            for load_b in [0, 8, 12] {
                for demand in [1, 4, 7] {
                    assert_eq!(
                        excess_delta_insert(capacity, load_a, demand),
                        overload_of(load_a + demand, capacity) - overload_of(load_a, capacity)
                    );
                    assert_eq!(
                        excess_delta_transfer(capacity, load_a, load_b, demand),
                        total(load_a - demand, load_b + demand) - total(load_a, load_b)
                    );
                    for other in [2, 6] {
                        assert_eq!(
                            excess_delta_exchange(capacity, load_a, load_b, demand, other),
                            total(load_a - demand + other, load_b - other + demand)
                                - total(load_a, load_b)
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn route_loads_sums_every_route() {
        let prob = line_vrp();
        let routes = vec![vec![1, 2, 3], vec![4, 5], Vec::new()];
        let mut loads = vec![-1; routes.len()];
        route_loads(&prob, &routes, &mut loads);
        assert_eq!(loads, vec![1 + 2 + 3, 4 + 5, 0]);
    }
}
