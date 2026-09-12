//! Diversity-aware population management for Hybrid Genetic Search.
//!
//! Individuals are ranked by *biased fitness*: a blend of how good a solution is
//! and how much diversity it contributes. Selecting purely on cost collapses the
//! population onto one basin within a few hundred generations; blending in a
//! diversity rank is what lets HGS keep searching for millions of them.

use crate::common::BiasedFitnessPopulation;
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

/// One of the two sub-populations (feasible / infeasible), kept ranked by
/// biased fitness.
///
/// A thin adapter over [`BiasedFitnessPopulation`], which holds the ranking
/// itself. What is HGS's own is the pair of quantities the ranking reads: the
/// distance is the broken-pairs count, and the cost is penalized by a capacity
/// weight the search adapts as it runs, which is why the shared component takes
/// the cost as a closure rather than reading it off the member.
///
/// The penalty reaches it through [`new`](Self::new) and
/// [`set_cost`](Self::set_cost) and nowhere else, so the rest of the surface
/// cannot be handed one that disagrees with the ranking.
pub(super) struct Subpopulation {
    inner: BiasedFitnessPopulation<Individual>,
}

// Vidal's published nbElite and nbClose: how many members the cost rank alone
// keeps alive, and how many nearest members are averaged into a diversity
// contribution. Fixed here rather than exposed, HGS being the algorithm they
// were published for. `GeneticAlgorithm` takes both from its config.
const N_ELITE: usize = 4;
const N_CLOSEST: usize = 5;

/// The diversity metric, as the shared component wants it.
fn broken_pairs(a: &Individual, b: &Individual) -> f64 {
    a.broken_pairs_distance(b)
}

/// The cost the ranking is ordered by, under `penalty`.
fn penalized(penalty: f64) -> Box<dyn Fn(&Individual) -> f64> {
    Box::new(move |m| m.cost(penalty))
}

impl Subpopulation {
    pub(super) fn new(penalty: f64) -> Self {
        Self {
            inner: BiasedFitnessPopulation::new(N_ELITE, N_CLOSEST, penalized(penalty)),
        }
    }

    pub(super) fn len(&self) -> usize {
        self.inner.len()
    }

    pub(super) fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    pub(super) fn clear(&mut self) {
        self.inner.clear();
    }

    pub(super) fn members(&self) -> &[Individual] {
        self.inner.members()
    }

    /// Biased fitness of member `i`; lower is better.
    pub(super) fn fitness(&self, i: usize) -> f64 {
        self.inner.fitness(i)
    }

    /// Adds an individual and re-ranks the sub-population.
    pub(super) fn push(&mut self, individual: Individual) {
        self.inner.push(individual, broken_pairs);
    }

    /// The cheapest member.
    pub(super) fn best(&self) -> Option<&Individual> {
        self.inner.best()
    }

    /// Re-costs every member under `penalty` and re-ranks.
    ///
    /// Needed when the penalty moves: the cost half of the ranking depends on
    /// it, the distance half does not. `HybridGeneticSearch::set_penalty` is
    /// the only caller, which is what keeps the two in step.
    pub(super) fn set_cost(&mut self, penalty: f64) {
        self.inner.set_cost(penalized(penalty));
    }

    /// Shrinks the sub-population to `target` members, evicting clones first and
    /// otherwise the worst biased fitness.
    pub(super) fn trim_to(&mut self, target: usize) {
        self.inner.trim_to(target);
    }
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
        let mut pop = Subpopulation::new(1.0);
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

        pop.set_cost(10.0);
        assert_eq!(
            pop.best().map(|m| m.distance),
            Some(20.0),
            "at a penalty of 10 the excess costs 50, so the feasible route wins"
        );
    }
}
