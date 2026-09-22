//! The route-level operators for the VRP, read by both searches that work on
//! its routes.
//!
//! They live with the problem rather than with either search, because VRP's
//! [`Ruinable`](crate::trait_defs::Ruinable) impl is built out of them and a
//! problem's trait impls have no business reaching into a heuristic.
//!
//! Five layers, each a module: this one is the distance arithmetic a route
//! edit is built from, [`pricing`] turns such an edit into what it costs the
//! objective, [`route_state`] keeps a route partition and its caches,
//! [`granular`] says which customer pairs are worth considering at all, and
//! [`descent`] walks downhill over the moves those pairs allow.
//!
//! A price is taken under a penalty the caller hands in rather than under
//! [`Vrp::penalty_weight`](crate::problem::vrp::Vrp::penalty_weight), which is
//! what lets one descent serve both callers: ALNS hands it the fixed weight
//! its objective already uses, HGS hands it the weight it is currently
//! tuning, and neither has to own a copy of the move set. What each caller
//! still decides for itself is when to descend and over which customers,
//! HGS over every one of a freshly decoded offspring, ALNS only around the ones
//! it has just re-inserted.
//!
//! The functions below are free rather than methods because they are
//! arithmetic over a route slice and nothing else. [`RouteState`] wraps them
//! for the routes it holds, and the tests here check them against from-scratch
//! route lengths with no state in sight. Sharing them is not cosmetic. These
//! are the formulas that decide what an edit moves, so a second copy is a
//! second answer to "how long is this route". [`Descent`] does have a
//! receiver, because it owns caches (the candidate lists, the sweep buffers)
//! that both callers were otherwise keeping their own copy of.

mod descent;
mod granular;
pub(crate) mod pricing;
mod route_state;

pub(crate) use descent::Descent;
use granular::build_neighbor_lists;
pub(crate) use route_state::RouteState;

use crate::common::DistanceStore;
use crate::problem::Vrp;

/// Distance of a single route, `depot → route[0] → … → route[last] → depot`,
/// with the depot at node `0`. An empty route has distance `0`.
pub(crate) fn route_distance(distances: &DistanceStore, route: &[usize]) -> f64 {
    let Some((&first, _)) = route.split_first() else {
        return 0.0;
    };
    let mut d = distances.distance(0, first);
    for w in route.windows(2) {
        d += distances.distance(w[0], w[1]);
    }
    d + distances.distance(route[route.len() - 1], 0)
}

/// The node at `pos` of `route`, or the depot when `pos` is past its end.
#[inline]
pub(crate) fn node_at(route: &[usize], pos: usize) -> usize {
    route.get(pos).copied().unwrap_or(0)
}

/// The node preceding `pos`, or the depot when `pos` is the start of the route.
#[inline]
pub(crate) fn before(route: &[usize], pos: usize) -> usize {
    if pos == 0 { 0 } else { route[pos - 1] }
}

/// `(before, first, last, after)` around the segment `route[pos..pos + len]`,
/// with the depot standing in at either end of the route.
#[inline]
pub(crate) fn segment_ends(
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
/// The segment's own edges are not counted: it is lifted out to be put back
/// somewhere, and it carries its internal edges with it. Removal and insertion
/// therefore compose into the cost of a relocation without either of them ever
/// naming the segment's length.
#[inline]
pub(crate) fn removal_gain(prob: &Vrp, route: &[usize], pos: usize, len: usize) -> f64 {
    let (before, first, last, after) = segment_ends(route, pos, len);
    prob.distance(before, first) + prob.distance(last, after) - prob.distance(before, after)
}

/// Distance added by inserting the segment `first…last` before position `pos`
/// of `route`, its internal edges excluded for the reason
/// [`removal_gain`] excludes them.
#[inline]
pub(crate) fn insertion_cost(
    prob: &Vrp,
    route: &[usize],
    pos: usize,
    first: usize,
    last: usize,
) -> f64 {
    let (before, after) = (before(route, pos), node_at(route, pos));
    prob.distance(before, first) + prob.distance(last, after) - prob.distance(before, after)
}

/// Distance saved by reversing `route[i..=j]`: the two edges at the ends of
/// the segment are exchanged, and its internal edges are traversed backwards,
/// which costs the same on a symmetric instance.
#[inline]
pub(crate) fn reversal_delta(prob: &Vrp, route: &[usize], i: usize, j: usize) -> f64 {
    let (before, after) = (before(route, i), node_at(route, j + 1));
    let (first, last) = (route[i], route[j]);
    prob.distance(before, last) + prob.distance(first, after)
        - prob.distance(before, first)
        - prob.distance(last, after)
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
    /// up to the segment's internal edges they deliberately leave out, that
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
    fn segment_ends_read_the_route() {
        let route = vec![1, 2, 3, 4, 5];
        assert_eq!(segment_ends(&route, 0, 2), (0, 1, 2, 3));
        assert_eq!(segment_ends(&route, 3, 2), (3, 4, 5, 0));
    }
}
