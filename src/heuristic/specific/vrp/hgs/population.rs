//! Diversity-aware population management for Hybrid Genetic Search.
//!
//! Individuals are ranked by *biased fitness*: a blend of how good a solution is
//! and how much diversity it contributes. Selecting purely on cost collapses the
//! population onto one basin within a few hundred generations; blending in a
//! diversity rank is what lets HGS keep searching for millions of them.

use rand::Rng;
use rand::rngs::SmallRng;

use crate::problem::vrp::RouteAdjacency;

/// Number of nearest individuals averaged into a diversity contribution.
const N_CLOSEST: usize = 5;

/// Individuals whose cost rank alone guarantees survival.
const N_ELITE: usize = 4;

/// Added to a clone's fitness so duplicates are always evicted first. Biased
/// fitness itself lies in `[0, 2]`, so any large constant separates the two.
const CLONE_FITNESS_PENALTY: f64 = 100.0;

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
    /// biased fitness needs of it — the ranking itself only reads the order.
    ///
    /// This is the *directional* count, the form Vidal's biased fitness is
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

/// One of the two sub-populations (feasible / infeasible), kept ranked by biased
/// fitness.
pub(super) struct Subpopulation {
    members: Vec<Individual>,
    fitness: Vec<f64>,
    /// Broken-pairs distances between every pair of `members`, row-major and
    /// `members.len()` wide.
    ///
    /// Cached because biased fitness has to be re-ranked on *every* insertion —
    /// the ranks are relative, so one new member shifts them all — and
    /// recomputing the whole matrix each time would cost O(N²·n) per generation.
    /// Grown one row/column at a time by [`Subpopulation::push`] and squeezed in
    /// place by [`Subpopulation::trim_to`], so each individual's O(N·n) of
    /// distance work is done once, when it arrives.
    distances: Vec<f64>,
}

impl Subpopulation {
    pub(super) fn new() -> Self {
        Self {
            members: Vec::new(),
            fitness: Vec::new(),
            distances: Vec::new(),
        }
    }

    pub(super) fn len(&self) -> usize {
        self.members.len()
    }

    pub(super) fn is_empty(&self) -> bool {
        self.members.is_empty()
    }

    pub(super) fn clear(&mut self) {
        self.members.clear();
        self.fitness.clear();
        self.distances.clear();
    }

    pub(super) fn members(&self) -> &[Individual] {
        &self.members
    }

    /// Biased fitness of member `i`; lower is better.
    ///
    /// Always current: every insertion re-ranks the sub-population. Leaving a
    /// fresh member unranked is not an option — biased fitness lies in `[0, 2]`,
    /// so any placeholder is either the best or the worst value there is, and
    /// [`binary_tournament`] would then either always or never pick it.
    pub(super) fn fitness(&self, i: usize) -> f64 {
        self.fitness[i]
    }

    /// Adds an individual and re-ranks the sub-population under `penalty`.
    ///
    /// Only the newcomer's distances are measured (O(N·n)); the rest of the
    /// matrix is reused, so the re-rank itself touches no routes.
    pub(super) fn push(&mut self, individual: Individual, penalty: f64) {
        let row: Vec<f64> = self
            .members
            .iter()
            .map(|m| m.broken_pairs_distance(&individual))
            .collect();
        self.grow_distances(&row);
        self.members.push(individual);
        self.refresh_fitness(penalty);
    }

    /// Widens the distance matrix by one, `row` being the new member's distances
    /// to the existing ones (and therefore also the new last row).
    fn grow_distances(&mut self, row: &[f64]) {
        let old = row.len();
        let new = old + 1;
        let mut grown = vec![0.0; new * new];
        for i in 0..old {
            grown[i * new..i * new + old].copy_from_slice(&self.distances[i * old..(i + 1) * old]);
            grown[i * new + old] = row[i];
        }
        grown[old * new..old * new + old].copy_from_slice(row);
        self.distances = grown;
    }

    /// The cheapest member under `penalty`.
    pub(super) fn best(&self, penalty: f64) -> Option<&Individual> {
        self.members
            .iter()
            .min_by(|a, b| a.cost(penalty).total_cmp(&b.cost(penalty)))
    }

    /// Recomputes biased fitness for every member from the cached distances.
    ///
    /// Callers outside this module need it when `penalty` moves: the cost half
    /// of the ranking depends on it, the distance half does not.
    pub(super) fn refresh_fitness(&mut self, penalty: f64) {
        let alive = vec![true; self.members.len()];
        self.fitness = biased_fitness(&self.members, &self.distances, &alive, penalty).1;
    }

    /// Shrinks the sub-population to `target` members, evicting clones first and
    /// otherwise the worst biased fitness.
    ///
    /// The cached distance matrix is masked as members are removed, so a trim
    /// costs O(N² log N) per eviction and measures no routes at all.
    pub(super) fn trim_to(&mut self, target: usize, penalty: f64) {
        if self.members.len() <= target {
            return;
        }
        let mut alive = vec![true; self.members.len()];
        let mut alive_count = self.members.len();

        while alive_count > target {
            let (victim, _) = biased_fitness(&self.members, &self.distances, &alive, penalty);
            let victim = victim.expect("a survivor must exist while alive_count > target");
            alive[victim] = false;
            alive_count -= 1;
        }

        self.compact(&alive);
        self.refresh_fitness(penalty);
    }

    /// Drops the dead members and squeezes the distance matrix down to match.
    fn compact(&mut self, alive: &[bool]) {
        let n = self.members.len();
        let survivors: Vec<usize> = (0..n).filter(|&i| alive[i]).collect();
        let m = survivors.len();
        let mut squeezed = vec![0.0; m * m];
        for (a, &i) in survivors.iter().enumerate() {
            for (b, &j) in survivors.iter().enumerate() {
                squeezed[a * m + b] = self.distances[i * n + j];
            }
        }
        self.distances = squeezed;

        let mut index = 0;
        self.members.retain(|_| {
            let keep = alive[index];
            index += 1;
            keep
        });
    }
}

/// Biased fitness of the living members, plus the index of the worst one.
///
/// `fitness = rank_cost / (N-1) + (1 - N_ELITE/N) · rank_diversity / (N-1)`,
/// where `rank_diversity` orders members by decreasing contribution, so a
/// solution earns its place either by being cheap or by being unlike the rest.
/// Clones are pushed to the back of the queue outright.
///
/// The returned vector is indexed like `members`; dead entries hold `0.0`.
fn biased_fitness(
    members: &[Individual],
    distances: &[f64],
    alive: &[bool],
    penalty: f64,
) -> (Option<usize>, Vec<f64>) {
    let n = members.len();
    let living: Vec<usize> = (0..n).filter(|&i| alive[i]).collect();
    let m = living.len();
    let mut fitness = vec![0.0; n];
    if m == 0 {
        return (None, fitness);
    }
    if m == 1 {
        return (Some(living[0]), fitness);
    }

    // Diversity contribution: mean distance to the N_CLOSEST living neighbors.
    let mut contribution = vec![0.0; m];
    let mut is_clone = vec![false; m];
    let mut neighbor_distances: Vec<f64> = Vec::with_capacity(m);
    for (a, &i) in living.iter().enumerate() {
        neighbor_distances.clear();
        neighbor_distances.extend(
            living
                .iter()
                .filter(|&&j| j != i)
                .map(|&j| distances[i * n + j]),
        );
        neighbor_distances.sort_by(f64::total_cmp);
        let keep = N_CLOSEST.min(neighbor_distances.len());
        is_clone[a] = neighbor_distances[0] <= f64::EPSILON;
        contribution[a] = neighbor_distances[..keep].iter().sum::<f64>() / keep as f64;
    }

    let mut by_cost: Vec<usize> = (0..m).collect();
    by_cost.sort_by(|&x, &y| {
        members[living[x]]
            .cost(penalty)
            .total_cmp(&members[living[y]].cost(penalty))
    });
    let mut by_diversity: Vec<usize> = (0..m).collect();
    by_diversity.sort_by(|&x, &y| contribution[y].total_cmp(&contribution[x]));

    let mut rank_cost = vec![0usize; m];
    let mut rank_diversity = vec![0usize; m];
    for (rank, &x) in by_cost.iter().enumerate() {
        rank_cost[x] = rank;
    }
    for (rank, &x) in by_diversity.iter().enumerate() {
        rank_diversity[x] = rank;
    }

    let denominator = (m - 1) as f64;
    let elite_weight = 1.0 - (N_ELITE as f64 / m as f64).min(1.0);
    let mut worst = living[0];
    let mut worst_fitness = f64::NEG_INFINITY;
    for (a, &i) in living.iter().enumerate() {
        let mut value = rank_cost[a] as f64 / denominator
            + elite_weight * rank_diversity[a] as f64 / denominator;
        fitness[i] = value;
        if is_clone[a] {
            value += CLONE_FITNESS_PENALTY;
        }
        if value > worst_fitness {
            worst_fitness = value;
            worst = i;
        }
    }
    (Some(worst), fitness)
}

/// Picks the better of two uniformly drawn candidates (lower fitness wins).
///
/// `candidates` are `(sub-population, index)` pairs; the caller supplies the
/// union of both sub-populations so selection can cross the feasibility border.
pub(super) fn binary_tournament<T: Copy>(candidates: &[(T, f64)], rng: &mut SmallRng) -> Option<T> {
    if candidates.is_empty() {
        return None;
    }
    let a = rng.random_range(0..candidates.len());
    let b = rng.random_range(0..candidates.len());
    let winner = if candidates[a].1 <= candidates[b].1 {
        a
    } else {
        b
    };
    Some(candidates[winner].0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    fn individual(n: usize, routes: Vec<Vec<usize>>, distance: f64, excess: i64) -> Individual {
        Individual::new(n, routes, distance, excess)
    }

    /// All pairwise broken-pairs distances, row-major — what a `Subpopulation`
    /// builds incrementally, written out in full so `biased_fitness` can be
    /// exercised without one.
    fn pairwise_distances(members: &[Individual]) -> Vec<f64> {
        let n = members.len();
        let mut distances = vec![0.0; n * n];
        for i in 0..n {
            for j in (i + 1)..n {
                let d = members[i].broken_pairs_distance(&members[j]);
                distances[i * n + j] = d;
                distances[j * n + i] = d;
            }
        }
        distances
    }

    #[test]
    fn broken_pairs_distance_is_the_normalized_count() {
        // What the metric *means* is pinned next to it, in
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

    #[test]
    fn survivor_selection_keeps_mu() {
        let n = 6;
        let mut pop = Subpopulation::new();
        for k in 0..10 {
            let routes = vec![vec![1, 2, 3], vec![4, 5, 6]];
            pop.push(individual(n, routes, 10.0 + k as f64, 0), 1.0);
        }
        pop.trim_to(4, 1.0);
        assert_eq!(pop.len(), 4);
    }

    #[test]
    fn survivor_selection_removes_clones_first() {
        let n = 6;
        let mut pop = Subpopulation::new();
        // Three identical (but expensive-to-lose) individuals plus two distinct
        // and strictly worse ones. Cost alone would evict the distinct pair.
        for _ in 0..3 {
            pop.push(
                individual(n, vec![vec![1, 2, 3], vec![4, 5, 6]], 10.0, 0),
                1.0,
            );
        }
        pop.push(
            individual(n, vec![vec![1, 4, 5], vec![2, 3, 6]], 20.0, 0),
            1.0,
        );
        pop.push(
            individual(n, vec![vec![1, 3, 5], vec![2, 4, 6]], 21.0, 0),
            1.0,
        );

        pop.trim_to(3, 1.0);
        assert_eq!(pop.len(), 3);
        let clones = pop
            .members()
            .iter()
            .filter(|m| m.routes == vec![vec![1, 2, 3], vec![4, 5, 6]])
            .count();
        assert_eq!(clones, 1, "duplicates should have been evicted first");
    }

    /// The matrix grown one member at a time must equal a from-scratch
    /// recompute, both after insertions and after a trim squeezes it — the
    /// diversity half of the ranking reads it directly.
    #[test]
    fn incremental_distances_match_a_full_recompute() {
        let n = 6;
        let mut pop = Subpopulation::new();
        let members = [
            individual(n, vec![vec![1, 2, 3], vec![4, 5, 6]], 10.0, 0),
            individual(n, vec![vec![1, 4, 5], vec![2, 3, 6]], 20.0, 0),
            individual(n, vec![vec![1, 3, 5], vec![2, 4, 6]], 21.0, 0),
            individual(n, vec![vec![1, 5, 6], vec![2, 3, 4]], 22.0, 0),
            individual(n, vec![vec![1, 6, 4], vec![2, 5, 3]], 23.0, 0),
        ];
        for m in &members {
            pop.push(m.clone(), 1.0);
            assert_eq!(pop.distances, pairwise_distances(pop.members()));
        }

        pop.trim_to(3, 1.0);
        assert_eq!(pop.distances, pairwise_distances(pop.members()));
    }

    /// A newcomer has to be ranked on arrival. Filing it unranked would leave it
    /// holding a placeholder, and since biased fitness lives in `[0, 2]` any
    /// placeholder makes it win or lose every tournament regardless of cost.
    #[test]
    fn a_pushed_individual_is_ranked_immediately() {
        let n = 6;
        let mut pop = Subpopulation::new();
        pop.push(
            individual(n, vec![vec![1, 2, 3], vec![4, 5, 6]], 10.0, 0),
            1.0,
        );
        pop.push(
            individual(n, vec![vec![1, 4, 5], vec![2, 3, 6]], 20.0, 0),
            1.0,
        );
        // The last arrival is by far the most expensive and no more diverse than
        // the others, so it must not come out ahead of the cheapest member.
        pop.push(
            individual(n, vec![vec![1, 3, 5], vec![2, 4, 6]], 99.0, 0),
            1.0,
        );

        let newest = pop.len() - 1;
        assert!(
            pop.fitness(newest) > pop.fitness(0),
            "the newcomer ranked {} against the cheapest member's {}",
            pop.fitness(newest),
            pop.fitness(0)
        );
    }

    #[test]
    fn biased_fitness_favors_cost_and_diversity() {
        let n = 6;
        let members = vec![
            // Cheapest and structurally unique: must rank best.
            individual(n, vec![vec![1, 2, 3], vec![4, 5, 6]], 10.0, 0),
            individual(n, vec![vec![1, 4, 5], vec![2, 3, 6]], 20.0, 0),
            individual(n, vec![vec![1, 4, 6], vec![2, 3, 5]], 21.0, 0),
            individual(n, vec![vec![1, 5, 6], vec![2, 3, 4]], 22.0, 0),
        ];
        let distances = pairwise_distances(&members);
        let alive = vec![true; members.len()];
        let (worst, fitness) = biased_fitness(&members, &distances, &alive, 1.0);

        assert_eq!(
            fitness
                .iter()
                .enumerate()
                .min_by(|a, b| a.1.total_cmp(b.1))
                .map(|(i, _)| i),
            Some(0),
            "the cheapest, most distinct individual should rank best"
        );
        assert_ne!(worst, Some(0), "it must not be the eviction candidate");
    }

    #[test]
    fn fitness_ranks_a_cheaper_clone_pair_by_cost() {
        // With diversity tied, cost decides the order.
        let n = 4;
        let members = vec![
            individual(n, vec![vec![1, 2], vec![3, 4]], 30.0, 0),
            individual(n, vec![vec![1, 3], vec![2, 4]], 10.0, 0),
        ];
        let distances = pairwise_distances(&members);
        let alive = vec![true; 2];
        let (_, fitness) = biased_fitness(&members, &distances, &alive, 1.0);
        assert!(fitness[1] < fitness[0]);
    }

    #[test]
    fn binary_tournament_prefers_lower_fitness() {
        let mut rng = SmallRng::seed_from_u64(3);
        let candidates = [(0usize, 0.0), (1usize, 1.0)];
        let mut wins = [0usize; 2];
        for _ in 0..200 {
            let winner = binary_tournament(&candidates, &mut rng).unwrap();
            wins[winner] += 1;
        }
        assert!(
            wins[0] > wins[1],
            "the fitter candidate should win more often: {wins:?}"
        );
        assert!(binary_tournament::<usize>(&[], &mut rng).is_none());
    }
}
