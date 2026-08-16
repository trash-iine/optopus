//! Each customer's route neighbors — the view a relabeling-invariant distance
//! between two CVRP solutions is computed from.
//!
//! A route partition names its routes, and two solutions that differ only in
//! which vehicle drives which route describe the same set of trips. Comparing
//! route *indices* therefore reads a relabeling as a total difference, which is
//! why the diversity metric works on adjacency instead: who each customer is
//! served between.

/// For each customer, the node before and after it along its route (`0` = the
/// depot, so a route's first customer has `pred == 0`).
///
/// Built once per solution and compared in O(n). Callers that compare one
/// solution against many — a population ranking its members by diversity —
/// keep the view rather than rebuilding it per pair.
#[derive(Debug, Clone)]
pub(crate) struct RouteAdjacency {
    succ: Vec<usize>,
    pred: Vec<usize>,
}

impl RouteAdjacency {
    /// Reads the adjacency of a route partition over customers `1..=n`.
    pub(crate) fn from_routes(n: usize, routes: &[Vec<usize>]) -> Self {
        let mut succ = vec![0usize; n + 1];
        let mut pred = vec![0usize; n + 1];
        for route in routes {
            for (pos, &c) in route.iter().enumerate() {
                pred[c] = if pos == 0 { 0 } else { route[pos - 1] };
                succ[c] = if pos + 1 == route.len() {
                    0
                } else {
                    route[pos + 1]
                };
            }
        }
        Self { succ, pred }
    }

    /// Number of customers.
    pub(crate) fn customers(&self) -> usize {
        self.succ.len().saturating_sub(1)
    }

    /// How many of `self`'s adjacencies `other` does not have: the broken-pairs
    /// count of Vidal's HGS-CVRP.
    ///
    /// A pair counts as kept whichever way round `other` drives it, so reversing
    /// a route breaks nothing, and route order is never consulted, so relabeling
    /// breaks nothing either. Starting a route is a structural fact of its own:
    /// a customer that leaves the depot in `self` but is served mid-route in
    /// `other` counts as one more break.
    ///
    /// **Directional.** That last clause is not symmetric — the solution using
    /// more routes has more depot departures to lose — so `a.broken_pairs_from(b)`
    /// and `b.broken_pairs_from(a)` can differ: `[[1, 2, 3], []]` measures 2
    /// against `[[1, 3], [2]]`, which measures 3 back. This is the count Vidal's
    /// biased fitness is defined on and it is kept as-is;
    /// [`Distance`](crate::trait_defs::Distance) symmetrizes it at the trait
    /// boundary, where a caller comparing two solutions has no reason to care
    /// which one it asked about.
    pub(crate) fn broken_pairs_from(&self, other: &Self) -> usize {
        let mut broken = 0;
        for c in 1..=self.customers() {
            if self.succ[c] != other.succ[c] && self.succ[c] != other.pred[c] {
                broken += 1;
            }
            if self.pred[c] == 0 && other.pred[c] != 0 && other.succ[c] != 0 {
                broken += 1;
            }
        }
        broken
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::problem::vrp::Vrp;
    use crate::trait_defs::Distance;
    use rand::seq::SliceRandom;
    use rand::{Rng, SeedableRng, rngs::SmallRng};

    fn adjacency(n: usize, routes: &[Vec<usize>]) -> RouteAdjacency {
        RouteAdjacency::from_routes(n, routes)
    }

    /// The two invariances the metric exists for: a solution is at distance `0`
    /// from a relabeling and from a reversal of itself, because both describe
    /// the same set of trips.
    #[test]
    fn relabeling_and_reversal_break_nothing() {
        let n = 6;
        let a = adjacency(n, &[vec![1, 2, 3], vec![4, 5, 6]]);
        let relabeled = adjacency(n, &[vec![4, 5, 6], vec![1, 2, 3]]);
        let reversed = adjacency(n, &[vec![3, 2, 1], vec![6, 5, 4]]);
        let other = adjacency(n, &[vec![1, 4, 5], vec![2, 3, 6]]);

        assert_eq!(a.broken_pairs_from(&a), 0);
        assert_eq!(a.broken_pairs_from(&relabeled), 0);
        assert_eq!(a.broken_pairs_from(&reversed), 0);
        assert!(a.broken_pairs_from(&other) > 0);
    }

    /// The asymmetry that makes [`Distance`] symmetrize rather than delegate.
    ///
    /// Splitting customer 2 out of `[[1, 2, 3]]` costs it a depot departure it
    /// did not have, so the split solution counts one break the joined one does
    /// not count back. Found by exhaustive search over every partition of three
    /// and four customers, which is also how the claim that the count is
    /// otherwise symmetric was rejected.
    #[test]
    fn the_directional_count_is_asymmetric_at_the_depot() {
        let joined = adjacency(3, &[vec![1, 2, 3], Vec::new()]);
        let split = adjacency(3, &[vec![1, 3], vec![2]]);
        assert_eq!(joined.broken_pairs_from(&split), 2);
        assert_eq!(split.broken_pairs_from(&joined), 3);
    }

    fn random_solution(prob: &Vrp, rng: &mut SmallRng) -> Vec<Vec<usize>> {
        let n = prob.get_n();
        let mut customers: Vec<usize> = (1..=n).collect();
        customers.shuffle(rng);
        let mut routes = vec![Vec::new(); prob.num_vehicles];
        for c in customers {
            let r = rng.random_range(0..prob.num_vehicles);
            routes[r].push(c);
        }
        routes
    }

    /// Over random pairs: the trait's distance is symmetric, zero exactly on
    /// solutions describing the same trips, and never exceeds what a solution
    /// has to lose.
    #[test]
    fn the_trait_distance_is_symmetric_and_bounded() {
        let mut rng = SmallRng::seed_from_u64(11);
        let coordinates: Vec<(f64, f64)> = (0..13)
            .map(|_| (rng.random_range(-50.0..50.0), rng.random_range(-50.0..50.0)))
            .collect();
        let prob = Vrp::new(
            "rnd",
            coordinates,
            vec![0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            5,
            4,
        );

        for _ in 0..200 {
            let first = prob.solution_from_routes(random_solution(&prob, &mut rng));
            let second = prob.solution_from_routes(random_solution(&prob, &mut rng));

            assert_eq!(first.distance(&second), second.distance(&first));
            assert_eq!(first.distance(&first), 0);
            assert!(first.distance(&second) <= 2 * prob.get_n());

            // Reversing every route keeps the trips, so it keeps the distance.
            let flipped: Vec<Vec<usize>> = second
                .routes
                .iter()
                .map(|r| r.iter().rev().copied().collect())
                .collect();
            let flipped = prob.solution_from_routes(flipped);
            assert_eq!(second.distance(&flipped), 0);
            assert_eq!(first.distance(&flipped), first.distance(&second));
        }
    }
}
