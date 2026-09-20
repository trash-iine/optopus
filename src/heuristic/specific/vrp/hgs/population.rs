//! Diversity-aware population management for Hybrid Genetic Search.
//!
//! Individuals are ranked by *biased fitness*: a blend of how good a solution is
//! and how much diversity it contributes. Selecting purely on cost collapses the
//! population onto one basin within a few hundred generations; blending in a
//! diversity rank is what lets HGS keep searching for millions of them.

use crate::common::{BiasedFitnessPopulation, CostFn};
use crate::problem::vrp::RouteAdjacency;

pub(super) use crate::common::binary_tournament;

/// A member of the population: a route partition plus the adjacency view the
/// diversity metric is measured on.
///
/// The view is kept per member rather than rebuilt per comparison because a
/// newcomer is measured against every incumbent, so building it once turns the
/// arrival's O(N·n) into N cheap comparisons.
#[derive(Debug, Clone)]
pub(super) struct Individual {
    pub routes: Vec<Vec<usize>>,
    pub distance: f64,
    pub excess: i64,
    adjacency: RouteAdjacency,
}

impl Individual {
    /// Builds an individual from a route partition and its evaluated caches.
    pub(super) fn new(n: usize, routes: Vec<Vec<usize>>, distance: f64, excess: i64) -> Self {
        let adjacency = RouteAdjacency::from_routes(n, &routes);
        Self {
            routes,
            distance,
            excess,
            adjacency,
        }
    }

    /// The penalized objective under the caller's current capacity penalty.
    pub(super) fn cost(&self, penalty: f64) -> f64 {
        self.distance + penalty * self.excess as f64
    }

    pub(super) fn is_feasible(&self) -> bool {
        self.excess == 0
    }

    /// Flattens the routes into a giant tour (route boundaries are dropped).
    pub(super) fn giant_tour(&self) -> Vec<usize> {
        self.routes.iter().flatten().copied().collect()
    }

    /// Broken-pairs distance in `[0, 1]`: the share of customers whose route
    /// neighbors differ between the two individuals.
    ///
    /// [`RouteAdjacency::broken_pairs_from`] does the counting; normalizing by
    /// `n` is what makes the value comparable across instances, which is all
    /// biased fitness needs of it, the ranking itself only reads the order.
    ///
    /// This is the directional count, the form Vidal's biased fitness is
    /// defined on, not the symmetrized one
    /// [`Distance`](crate::search_state::Distance) exposes.
    pub(super) fn broken_pairs_distance(&self, other: &Self) -> f64 {
        let n = self.adjacency.customers();
        if n == 0 {
            return 0.0;
        }
        self.adjacency.broken_pairs_from(&other.adjacency) as f64 / n as f64
    }
}

/// A sub-population (feasible or infeasible), ranked by biased fitness.
///
/// [`BiasedFitnessPopulation`] holds the ranking itself. What is HGS's own is
/// the pair of quantities it reads, stated in one place by
/// [`sub_population`]. The distance is the broken-pairs count, and the cost is
/// penalized by a capacity weight the search adapts as it runs.
pub(super) type Subpopulation = BiasedFitnessPopulation<Individual>;

/// The diversity metric, as the shared component wants it.
fn broken_pairs(a: &Individual, b: &Individual) -> f64 {
    a.broken_pairs_distance(b)
}

/// The cost the ranking is ordered by, under `penalty`.
///
/// `HybridGeneticSearch::set_penalty` is the only place a penalty other than
/// the starting one reaches this, which is what keeps the ranking and the
/// weight it ranks through in step.
pub(super) fn penalized(penalty: f64) -> CostFn<Individual> {
    Box::new(move |m| m.cost(penalty))
}

/// An empty sub-population ranked under `penalty`, with Vidal's `nbElite` and
/// `nbClose` as the search was built with.
pub(super) fn sub_population(penalty: f64, n_elite: usize, n_closest: usize) -> Subpopulation {
    BiasedFitnessPopulation::new(
        n_elite,
        n_closest,
        penalized(penalty),
        Box::new(broken_pairs),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn individual(n: usize, routes: Vec<Vec<usize>>, distance: f64, excess: i64) -> Individual {
        Individual::new(n, routes, distance, excess)
    }

    #[test]
    fn broken_pairs_distance_is_the_normalized_count() {
        // What the metric means is pinned next to it, in
        // `problem/vrp/adjacency.rs`; here only the normalization is at stake,
        // since biased fitness compares these values across sub-populations.
        let n = 6;
        let a = individual(n, vec![vec![1, 2, 3], vec![4, 5, 6]], 10.0, 0);
        let same = individual(n, vec![vec![1, 2, 3], vec![4, 5, 6]], 10.0, 0);
        let other = individual(n, vec![vec![1, 4, 5], vec![2, 3, 6]], 12.0, 0);

        assert_eq!(a.broken_pairs_distance(&same), 0.0);
        let d = a.broken_pairs_distance(&other);
        assert!(d > 0.0 && d <= 2.0, "distance {d} left [0, 2]");
        assert_eq!(
            d * n as f64,
            (d * n as f64).round(),
            "the value must be a whole count divided by n, got {d}"
        );
    }

    /// The ranking itself lives in `common::BiasedFitnessPopulation` and is
    /// tested there. What is this wrapper's own is that the penalty reaches the
    /// cost, so a member that is cheap only while infeasibility is cheap stops
    /// winning once the penalty rises, and that `set_cost` is what moves it.
    #[test]
    fn the_penalty_reaches_the_cost_the_ranking_reads() {
        let n = 6;
        let mut pop = sub_population(1.0, 4, 5);
        // Short but over capacity, against long but feasible.
        let short_infeasible = individual(n, vec![vec![1, 2, 3], vec![4, 5, 6]], 10.0, 5);
        let long_feasible = individual(n, vec![vec![1, 4, 5], vec![2, 3, 6]], 20.0, 0);
        pop.push(short_infeasible);
        pop.push(long_feasible);

        assert_eq!(
            pop.best().map(|m| m.distance),
            Some(10.0),
            "at a penalty of 1 the excess costs 5, so the short route wins"
        );

        pop.set_cost(penalized(10.0));
        assert_eq!(
            pop.best().map(|m| m.distance),
            Some(20.0),
            "at a penalty of 10 the excess costs 50, so the feasible route wins"
        );
    }
}
