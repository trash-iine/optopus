//! Diversity-aware population management. Members are ranked by biased
//! fitness, a blend of how good a solution is and how much diversity it
//! contributes.
//!
//! Selecting purely on cost collapses a population onto one basin within a few
//! hundred generations. Blending in a diversity rank is what lets a genetic
//! search keep going for millions of them. The scheme is Vidal's, introduced
//! for HGS on vehicle routing, but nothing in it is about routes. It needs a
//! cost per member and a distance between two members, and no more.
//!
//! The cost is a closure rather than a trait method, because it is not always
//! a property of the member alone. HGS adapts a capacity penalty as it runs, so
//! the same individual is worth different amounts at different generations, and
//! a trait would have to smuggle the penalty in through the member.
//!
//! It is given once, to [`BiasedFitnessPopulation::new`], and can only be
//! replaced through [`set_cost`](BiasedFitnessPopulation::set_cost), which
//! re-ranks. No other method takes one, so two calls on one population cannot
//! rank against two different costs, and a cost cannot be changed while the
//! order still reflects the old one.
//!
//! The distance stays an argument to [`push`](BiasedFitnessPopulation::push)
//! instead, that being the only place it is measured: what a member is worth
//! can move under the population, but how far apart two members are cannot.

use rand::Rng;

/// Added to a clone's fitness so duplicates are always evicted first. Biased
/// fitness itself lies in `[0, 2]`, so any large constant separates the two.
const CLONE_FITNESS_PENALTY: f64 = 100.0;

/// A population kept ranked by biased fitness.
///
/// The pairwise distance matrix is cached because biased fitness has to be
/// re-ranked on every insertion, the ranks being relative, so one new member
/// shifts them all. Recomputing it each time would cost O(N²·d) per generation,
/// where `d` is what one distance costs. It is grown one row/column at a time
/// by [`push`](Self::push) and squeezed in place by [`trim_to`](Self::trim_to),
/// so each member's O(N·d) of distance work is done once, when it arrives.
///
/// # References
///
/// - Vidal, T., Crainic, T. G., Gendreau, M., Lahrichi, N. and Rei, W. "A
///   Hybrid Genetic Algorithm for Multidepot and Periodic Vehicle Routing
///   Problems." *Operations Research*, 60(3), 611-624, 2012.
pub struct BiasedFitnessPopulation<T> {
    /// What a member is worth, lower being better. Boxed rather than a type
    /// parameter so that neither `Members` in the genetic algorithm nor HGS's
    /// `Subpopulation` has to carry one.
    cost: Box<dyn Fn(&T) -> f64>,
    members: Vec<T>,
    fitness: Vec<f64>,
    /// Distances between every pair of `members`, row-major and
    /// `members.len()` wide.
    distances: Vec<f64>,
    /// How many nearest members are averaged into a diversity contribution.
    n_closest: usize,
    /// How many members the cost rank alone keeps alive.
    n_elite: usize,
}

impl<T> BiasedFitnessPopulation<T> {
    /// Creates an empty population ranked on the two parameters given.
    ///
    /// `n_elite` is how many members the cost rank alone keeps alive and
    /// `n_closest` is how many nearest members are averaged into a diversity
    /// contribution, Vidal's `nbElite` and `nbClose`, whose published values
    /// are `4` and `5`. Neither is defaulted.
    ///
    /// `n_elite` enters the blend as `1 - n_elite / N`, so **a population of
    /// `n_elite` or fewer ranks on cost alone**. The diversity half is
    /// multiplied by zero and biased fitness degenerates to the cost-only
    /// selection it exists to replace. Keep it well below the population size.
    ///
    /// A population smaller than `n_closest + 1` averages over everyone, which
    /// makes a contribution the mean distance to the whole population rather
    /// than a local one, and the measure loses the locality it is for.
    ///
    /// # Panics
    ///
    /// Panics if either argument is zero.
    pub fn new(n_elite: usize, n_closest: usize, cost: Box<dyn Fn(&T) -> f64>) -> Self {
        assert!(n_elite >= 1, "n_elite must be at least 1");
        assert!(n_closest >= 1, "n_closest must be at least 1");
        Self {
            cost,
            members: Vec::new(),
            fitness: Vec::new(),
            distances: Vec::new(),
            n_closest,
            n_elite,
        }
    }

    /// Replaces the cost and re-ranks, which is the only way to change it.
    ///
    /// Callers need it when the cost moves, as HGS's capacity penalty does,
    /// because the cost half of the ranking depends on it and the distance half
    /// does not. The re-rank is not optional: a population ordered by a cost it
    /// no longer uses looks exactly like one that is ordered correctly.
    pub fn set_cost(&mut self, cost: Box<dyn Fn(&T) -> f64>) {
        self.cost = cost;
        self.rerank();
    }

    pub fn len(&self) -> usize {
        self.members.len()
    }

    pub fn is_empty(&self) -> bool {
        self.members.is_empty()
    }

    pub fn clear(&mut self) {
        self.members.clear();
        self.fitness.clear();
        self.distances.clear();
    }

    pub fn members(&self) -> &[T] {
        &self.members
    }

    /// The cached pairwise distances, row-major and [`len`](Self::len) wide.
    ///
    /// Exposed because the cache is maintained incrementally, grown on every
    /// push and squeezed on every trim, and the diversity half of the ranking
    /// reads it directly, so "does it still equal a from-scratch recompute?" is
    /// a question worth being able to ask from outside.
    pub fn distances(&self) -> &[f64] {
        &self.distances
    }

    /// Biased fitness of member `i`, where lower is better.
    ///
    /// Always current: every insertion re-ranks the population. Leaving a fresh
    /// member unranked is not an option, biased fitness lies in `[0, 2]`, so
    /// any placeholder is either the best or the worst value there is, and
    /// [`binary_tournament`] would then either always or never pick it.
    pub fn fitness(&self, i: usize) -> f64 {
        self.fitness[i]
    }

    /// Adds a member and re-ranks the population.
    ///
    /// Only the newcomer's distances are measured (O(N·d)). The rest of the
    /// matrix is reused, so the re-rank itself measures nothing.
    pub fn push(&mut self, member: T, distance: impl Fn(&T, &T) -> f64) {
        let row: Vec<f64> = self.members.iter().map(|m| distance(m, &member)).collect();
        self.grow_distances(&row);
        self.members.push(member);
        self.rerank();
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

    /// The cheapest member.
    pub fn best(&self) -> Option<&T> {
        self.members
            .iter()
            .min_by(|a, b| (self.cost)(a).total_cmp(&(self.cost)(b)))
    }

    /// Recomputes biased fitness for every member from the cached distances.
    fn rerank(&mut self) {
        let alive = vec![true; self.members.len()];
        self.fitness = biased_fitness(
            &self.members,
            &self.distances,
            &alive,
            self.cost.as_ref(),
            self.n_closest,
            self.n_elite,
        )
        .1;
    }

    /// Shrinks the population to `target` members, evicting clones first and
    /// otherwise the worst biased fitness.
    ///
    /// The cached distance matrix is masked as members are removed, so a trim
    /// costs O(N² log N) per eviction and measures no distances at all.
    pub fn trim_to(&mut self, target: usize) {
        if self.members.len() <= target {
            return;
        }
        let mut alive = vec![true; self.members.len()];
        let mut alive_count = self.members.len();

        while alive_count > target {
            let (victim, _) = biased_fitness(
                &self.members,
                &self.distances,
                &alive,
                self.cost.as_ref(),
                self.n_closest,
                self.n_elite,
            );
            let victim = victim.expect("a survivor must exist while alive_count > target");
            alive[victim] = false;
            alive_count -= 1;
        }

        self.compact(&alive);
        self.rerank();
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
/// `fitness = rank_cost / (N-1) + (1 - n_elite/N) · rank_diversity / (N-1)`,
/// where `rank_diversity` orders members by decreasing contribution, so a
/// solution earns its place either by being cheap or by being unlike the rest.
/// Clones are pushed to the back of the queue outright.
///
/// The returned vector is indexed like `members`, and dead entries hold `0.0`.
fn biased_fitness<T>(
    members: &[T],
    distances: &[f64],
    alive: &[bool],
    cost: &dyn Fn(&T) -> f64,
    n_closest: usize,
    n_elite: usize,
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

    // Diversity contribution: mean distance to the n_closest living neighbors.
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
        let keep = n_closest.min(neighbor_distances.len());
        is_clone[a] = neighbor_distances[0] <= f64::EPSILON;
        contribution[a] = neighbor_distances[..keep].iter().sum::<f64>() / keep as f64;
    }

    let mut by_cost: Vec<usize> = (0..m).collect();
    by_cost.sort_by(|&x, &y| cost(&members[living[x]]).total_cmp(&cost(&members[living[y]])));
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
    let elite_weight = 1.0 - (n_elite as f64 / m as f64).min(1.0);
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
/// `candidates` are `(payload, fitness)` pairs. The payload is whatever the
/// caller needs to identify the winner, so HGS passes a
/// `(sub-population, index)` pair, so selection can cross the feasibility
/// border by handing in the union of both.
pub fn binary_tournament<T: Copy>(candidates: &[(T, f64)], rng: &mut impl Rng) -> Option<T> {
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
    use rand::rngs::SmallRng;

    /// A member with a cost and a position, so diversity is a plain Euclidean
    /// distance. Nothing here needs a problem, which is the point. The ranking
    /// reads a cost and a distance and knows nothing else.
    #[derive(Debug, Clone, Copy, PartialEq)]
    struct Point {
        cost: f64,
        x: f64,
        y: f64,
    }

    fn point(cost: f64, x: f64, y: f64) -> Point {
        Point { cost, x, y }
    }

    fn euclidean(a: &Point, b: &Point) -> f64 {
        ((a.x - b.x).powi(2) + (a.y - b.y).powi(2)).sqrt()
    }

    fn by_cost(p: &Point) -> f64 {
        p.cost
    }

    /// All pairwise distances, row-major, what the population builds
    /// incrementally, written out in full to check it against.
    fn pairwise(members: &[Point]) -> Vec<f64> {
        let n = members.len();
        let mut distances = vec![0.0; n * n];
        for i in 0..n {
            for j in (i + 1)..n {
                let d = euclidean(&members[i], &members[j]);
                distances[i * n + j] = d;
                distances[j * n + i] = d;
            }
        }
        distances
    }

    fn populate(points: &[Point]) -> BiasedFitnessPopulation<Point> {
        let mut pop = BiasedFitnessPopulation::new(4, 5, Box::new(by_cost));
        for p in points {
            pop.push(*p, euclidean);
        }
        pop
    }

    #[test]
    fn survivor_selection_keeps_the_target_size() {
        let mut pop = BiasedFitnessPopulation::new(4, 5, Box::new(by_cost));
        for k in 0..10 {
            pop.push(point(10.0 + k as f64, k as f64, 0.0), euclidean);
        }
        pop.trim_to(4);
        assert_eq!(pop.len(), 4);
    }

    /// Three identical (but cheap) members plus two distinct and strictly worse
    /// ones. Cost alone would evict the distinct pair, and the clone penalty is
    /// what stops it.
    #[test]
    fn survivor_selection_removes_clones_first() {
        let mut pop = BiasedFitnessPopulation::new(4, 5, Box::new(by_cost));
        for _ in 0..3 {
            pop.push(point(10.0, 0.0, 0.0), euclidean);
        }
        pop.push(point(20.0, 5.0, 0.0), euclidean);
        pop.push(point(21.0, 0.0, 5.0), euclidean);

        pop.trim_to(3);
        assert_eq!(pop.len(), 3);
        let clones = pop
            .members()
            .iter()
            .filter(|m| m.x == 0.0 && m.y == 0.0)
            .count();
        assert_eq!(clones, 1, "duplicates should have been evicted first");
    }

    /// The matrix grown one member at a time must equal a from-scratch
    /// recompute, both after insertions and after a trim squeezes it.
    #[test]
    fn incremental_distances_match_a_full_recompute() {
        let members = [
            point(10.0, 0.0, 0.0),
            point(20.0, 3.0, 0.0),
            point(21.0, 0.0, 4.0),
            point(22.0, 6.0, 8.0),
            point(23.0, 1.0, 9.0),
        ];
        let mut pop = BiasedFitnessPopulation::new(4, 5, Box::new(by_cost));
        for m in &members {
            pop.push(*m, euclidean);
            assert_eq!(pop.distances(), pairwise(pop.members()));
        }

        pop.trim_to(3);
        assert_eq!(pop.distances(), pairwise(pop.members()));
    }

    /// A newcomer has to be ranked on arrival. Filing it unranked would leave
    /// it holding a placeholder, and since biased fitness lives in `[0, 2]` any
    /// placeholder makes it win or lose every tournament regardless of cost.
    #[test]
    fn a_pushed_member_is_ranked_immediately() {
        let pop = populate(&[
            point(10.0, 0.0, 0.0),
            point(20.0, 3.0, 0.0),
            // By far the most expensive and no more diverse than the others, so
            // it must not come out ahead of the cheapest member.
            point(99.0, 1.5, 0.5),
        ]);

        let newest = pop.len() - 1;
        assert!(
            pop.fitness(newest) > pop.fitness(0),
            "the newcomer ranked {} against the cheapest member's {}",
            pop.fitness(newest),
            pop.fitness(0)
        );
    }

    /// The cheapest and most isolated member must rank best.
    #[test]
    fn fitness_favors_cost_and_diversity() {
        let pop = populate(&[
            point(10.0, 0.0, 0.0),
            point(20.0, 10.0, 0.0),
            point(21.0, 10.5, 0.0),
            point(22.0, 11.0, 0.0),
        ]);

        let best = (0..pop.len()).min_by(|&a, &b| pop.fitness(a).total_cmp(&pop.fitness(b)));
        assert_eq!(
            best,
            Some(0),
            "the cheapest, most distinct member should rank best"
        );
    }

    /// With diversity tied, cost decides the order.
    #[test]
    fn fitness_ranks_an_equally_diverse_pair_by_cost() {
        let pop = populate(&[point(30.0, 0.0, 0.0), point(10.0, 1.0, 0.0)]);
        assert!(pop.fitness(1) < pop.fitness(0));
    }

    /// `n_elite >= N` weights the diversity half by zero, which is the
    /// degeneracy `new` warns about: the ranking becomes the cost-only
    /// one biased fitness exists to replace, and it changes who is evicted.
    #[test]
    fn an_elite_count_at_the_population_size_ranks_on_cost_alone() {
        // One expensive isolated member and two cheap near-identical ones.
        let points = [
            point(3.0, 10.0, 10.0),
            point(1.0, 0.0, 0.0),
            point(2.0, 0.1, 0.0),
        ];
        let survivors = |n_elite: usize| {
            let mut pop = BiasedFitnessPopulation::new(n_elite, 5, Box::new(by_cost));
            for p in &points {
                pop.push(*p, euclidean);
            }
            pop.trim_to(2);
            let mut costs: Vec<f64> = pop.members().iter().map(|m| m.cost).collect();
            costs.sort_by(f64::total_cmp);
            costs
        };

        // Cost alone: the expensive isolated member goes, however unlike the
        // rest it is.
        assert_eq!(survivors(3), vec![1.0, 2.0]);
        // With the diversity half alive, the isolated one is worth keeping and
        // the crowded near-duplicate goes instead.
        assert_eq!(survivors(1), vec![1.0, 3.0]);
    }

    /// `n_closest` caps how many neighbors a contribution averages, so raising
    /// it past the population size changes nothing.
    #[test]
    fn n_closest_saturates_at_the_population_size() {
        let points = [
            point(1.0, 0.0, 0.0),
            point(2.0, 1.0, 0.0),
            point(3.0, 5.0, 0.0),
        ];
        let mut two = BiasedFitnessPopulation::new(4, 2, Box::new(by_cost));
        let mut many = BiasedFitnessPopulation::new(4, 50, Box::new(by_cost));
        for p in &points {
            two.push(*p, euclidean);
            many.push(*p, euclidean);
        }
        // Each member has exactly two neighbors, so a cap of 2 and a cap of 50
        // average the same set.
        for i in 0..points.len() {
            assert_eq!(two.fitness(i), many.fitness(i));
        }
    }

    #[test]
    #[should_panic(expected = "n_elite must be at least 1")]
    fn a_zero_elite_count_is_rejected() {
        let _ = BiasedFitnessPopulation::<Point>::new(0, 5, Box::new(by_cost));
    }

    #[test]
    #[should_panic(expected = "n_closest must be at least 1")]
    fn a_zero_closest_count_is_rejected() {
        let _ = BiasedFitnessPopulation::<Point>::new(4, 0, Box::new(by_cost));
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
