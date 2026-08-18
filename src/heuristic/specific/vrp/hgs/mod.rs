//! Hybrid Genetic Search (HGS) for the Capacitated VRP.
//!
//! HGS (Vidal et al.) is the strongest known general-purpose CVRP metaheuristic.
//! It combines three ideas, each in its own layer here:
//!
//! - [`Descent`] — the granular descent shared with ALNS,
//!   which turns every offspring into a local optimum under the capacity
//!   penalty this driver adapts at runtime.
//! - [`population`] — *biased fitness*, which ranks individuals by cost **and**
//!   by how much diversity they contribute, so the population does not collapse.
//! - this module — the generational loop, the adaptive penalty, and the
//!   feasible / infeasible split that lets the search cross infeasible ground
//!   between feasible basins.
//!
//! The genetic representation is the **giant tour**: an offspring is a customer
//! permutation, decoded into routes by [`split_giant_tour`], which is optimal for
//! that permutation. The genetic operator therefore only has to get the customer
//! *order* right — the decoder handles the vehicle assignment exactly.
//!
//! # Relationship to the rest of the crate
//!
//! HGS keeps its own individuals rather than [`VrpSolution`]s because
//! [`Vrp::penalty_weight`] is a fixed, deliberately enormous constant chosen so
//! that any optimum is feasible, whereas HGS must search *under a penalty it
//! tunes*. Solutions are converted back with [`Vrp::solution_from_routes`] only
//! when writing to the search state, so reported objectives stay comparable with
//! every other heuristic in the crate.

mod population;

use rand::Rng;
use rand::rngs::SmallRng;
use rand::seq::SliceRandom;

use crate::common::order_crossover;
use crate::error::OptError;
use crate::heuristic::{Heuristic, StopCondition};
use crate::problem::vrp::{Vrp, split_giant_tour};
use crate::search_state::SearchState;

use super::ops::{Descent, RouteState};
use population::{Individual, Subpopulation, binary_tournament};

/// Initial population size, as a multiple of `min_population_size`.
const INITIAL_POPULATION_FACTOR: usize = 4;

/// Share of the initial population built by a nearest-neighbor tour rather than
/// a random one.
const NEAREST_NEIGHBOR_SHARE: usize = 4;

/// Local-search passes before giving up on further improvement.
const MAX_LS_PASSES: usize = 64;

/// Probability of attempting to repair an infeasible offspring.
const REPAIR_PROBABILITY: f64 = 0.5;

/// Penalty multipliers tried, in order, when repairing.
const REPAIR_PENALTY_BOOSTS: [f64; 2] = [10.0, 100.0];

/// Offspring between two adjustments of the capacity penalty.
const PENALTY_UPDATE_PERIOD: u32 = 100;

/// Multiplicative adjustments steering the feasible share toward its target.
const PENALTY_INCREASE: f64 = 1.2;
const PENALTY_DECREASE: f64 = 0.85;

/// Dead band around `target_feasible` in which the penalty is left alone.
const FEASIBILITY_TOLERANCE: f64 = 0.05;

/// Bounds on the penalty, relative to its instance-derived starting value.
const PENALTY_MIN_FACTOR: f64 = 1e-3;
const PENALTY_MAX_FACTOR: f64 = 1e4;

/// Hybrid Genetic Search for the CVRP.
///
/// Each [`Heuristic::run_once`] produces one offspring: two parents are chosen by
/// binary tournament on biased fitness across both sub-populations, recombined
/// with order crossover, decoded by [`split_giant_tour`], improved by granular
/// local search, and filed into the feasible or infeasible sub-population.
/// Sub-populations grow to `min_population_size + generation_size` and are then
/// culled back to `min_population_size`, clones first.
///
/// The capacity penalty is retuned every `PENALTY_UPDATE_PERIOD` offspring to
/// hold the feasible share near `target_feasible`: too few feasible offspring
/// raises it, too many lowers it. Searching at a *deliberately* low feasible rate
/// is the point — the shortest route through solution space between two good
/// feasible solutions usually crosses infeasible ground.
///
/// The first individual is seeded from `state.solution`, so composing HGS inside
/// [`Sequential`](crate::heuristic::Sequential) or
/// [`Restart`](crate::heuristic::Restart) carries the incumbent forward.
///
/// # References
///
/// - Vidal, T., Crainic, T. G., Gendreau, M., Lahrichi, N., and Rei, W. "A Hybrid
///   Genetic Algorithm for Multidepot and Periodic Vehicle Routing Problems."
///   *Operations Research*, 60(3), 611-624, 2012.
/// - Vidal, T. "Hybrid Genetic Search for the CVRP: Open-Source Implementation
///   and SWAP\* Neighborhood." *Computers & Operations Research*, 140, 105643, 2022.
/// - Prins, C. "A Simple and Effective Evolutionary Algorithm for the Vehicle
///   Routing Problem." *Computers & Operations Research*, 31(12), 1985-2002, 2004.
///
/// # Example
///
/// ```
/// use optopus::heuristic::{Heuristic, HybridGeneticSearchForVrp, StopCondition};
/// use optopus::problem::vrp::Vrp;
/// use optopus::search_state::SearchState;
///
/// let vrp = Vrp::new(
///     "demo",
///     vec![(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (-1.0, 0.0), (0.0, -1.0)],
///     vec![0, 1, 1, 1, 1],
///     2,
///     2,
/// );
/// let mut state = SearchState::new_with_seed(&vrp, 42);
/// let mut hgs = HybridGeneticSearchForVrp::new(
///     StopCondition::iterations(500),
///     8,    // min_population_size
///     10,   // generation_size
///     10,   // granularity
///     0.2,  // target_feasible
///     None, // never restart
/// );
/// hgs.run(&mut state).unwrap();
/// ```
pub struct HybridGeneticSearch {
    stop_condition: StopCondition,
    min_population_size: usize,
    generation_size: usize,
    granularity: usize,
    target_feasible: f64,
    restart_generations: Option<u64>,

    feasible: Subpopulation,
    infeasible: Subpopulation,
    /// The granular descent every offspring is put through, holding the
    /// candidate lists it was built for; survives [`Heuristic::clear`].
    descent: Descent,
    penalty_capacity: f64,
    initial_penalty: f64,
    /// Offspring since the last penalty adjustment, and how many were feasible.
    recent_total: u32,
    recent_feasible: u32,
    generations_without_improvement: u64,
    best_cost: f64,
    warm_started: bool,
}

impl HybridGeneticSearch {
    /// Creates a new Hybrid Genetic Search.
    ///
    /// Reasonable defaults follow Vidal: `min_population_size = 25`,
    /// `generation_size = 40`, `granularity = 20`, `target_feasible = 0.2`,
    /// `restart_generations = Some(20_000)`.
    ///
    /// # Panics
    /// Panics if `min_population_size < 4`, if `generation_size` or `granularity`
    /// is zero, or if `target_feasible` is not in `(0, 1)`.
    pub fn new(
        stop_condition: StopCondition,
        min_population_size: usize,
        generation_size: usize,
        granularity: usize,
        target_feasible: f64,
        restart_generations: Option<u64>,
    ) -> Self {
        assert!(
            min_population_size >= 4,
            "min_population_size must be at least 4"
        );
        assert!(generation_size > 0, "generation_size must be positive");
        assert!(granularity > 0, "granularity must be positive");
        assert!(
            target_feasible > 0.0 && target_feasible < 1.0,
            "target_feasible must be in (0, 1)"
        );
        Self {
            stop_condition,
            min_population_size,
            generation_size,
            granularity,
            target_feasible,
            restart_generations,
            feasible: Subpopulation::new(),
            infeasible: Subpopulation::new(),
            descent: Descent::new(),
            penalty_capacity: 1.0,
            initial_penalty: 1.0,
            recent_total: 0,
            recent_feasible: 0,
            generations_without_improvement: 0,
            best_cost: f64::INFINITY,
            warm_started: false,
        }
    }

    /// Starting capacity penalty: the average distance per unit of demand, so it
    /// is on the same scale as the objective regardless of the instance.
    fn scale_free_penalty(prob: &Vrp) -> f64 {
        let n = prob.get_n();
        if n == 0 {
            return 1.0;
        }
        let mean_depot_distance = (1..=n).map(|c| prob.distance(0, c)).sum::<f64>() / n as f64;
        let mean_demand = (1..=n).map(|c| prob.demands[c]).sum::<i64>() as f64 / n as f64;
        (mean_depot_distance / mean_demand.max(1.0)).max(f64::MIN_POSITIVE)
    }

    /// Seeds both sub-populations with `INITIAL_POPULATION_FACTOR × μ` individuals.
    ///
    /// [`Vrp::new_solution`](crate::search_state::ProblemTrait::new_solution) is
    /// not used: it assigns customers to the least-loaded vehicle in random
    /// order, ignoring geometry entirely, which starts the search far from
    /// anything a route-based local search can fix cheaply.
    fn initialize_population(&mut self, state: &mut SearchState<'_, Vrp>) {
        let prob = state.instance;
        let n = prob.get_n();
        let count = INITIAL_POPULATION_FACTOR * self.min_population_size;

        for i in 0..count {
            let tour = if i == 0 && !self.warm_started {
                // Carry the incumbent forward when composed with another heuristic.
                self.warm_started = true;
                state.solution.routes.iter().flatten().copied().collect()
            } else if i % NEAREST_NEIGHBOR_SHARE == 1 {
                nearest_neighbor_tour(prob, self.descent.neighbors(), &mut state.rng)
            } else {
                random_tour(n, &mut state.rng)
            };
            self.spawn(prob, tour, &mut state.rng);
        }

        // The initial population costs real work; charge it to the iteration
        // counter so time-to-best stays meaningful.
        state.iteration += count as u64;
        self.publish_best(state);
    }

    /// Decodes a giant tour, improves it, repairs it, and files the result.
    fn spawn(&mut self, prob: &Vrp, tour: Vec<usize>, rng: &mut SmallRng) {
        let routes = split_giant_tour(prob, &tour, self.penalty_capacity);
        let mut child = RouteState::from_routes(prob, routes);
        self.descent
            .run(&mut child, prob, rng, self.penalty_capacity, MAX_LS_PASSES);

        self.recent_total += 1;
        if child.excess == 0 {
            self.recent_feasible += 1;
            self.absorb(prob, child);
            return;
        }

        // An infeasible child is worth keeping as-is (it may be a useful bridge),
        // but a feasible version of it is worth more.
        if rng.random::<f64>() < REPAIR_PROBABILITY {
            let mut repaired = child.clone();
            for boost in REPAIR_PENALTY_BOOSTS {
                self.descent.run(
                    &mut repaired,
                    prob,
                    rng,
                    self.penalty_capacity * boost,
                    MAX_LS_PASSES,
                );
                if repaired.excess == 0 {
                    self.absorb(prob, repaired);
                    break;
                }
            }
        }
        self.absorb(prob, child);
    }

    /// Files an evaluated child into its sub-population, culling if it overflows.
    fn absorb(&mut self, prob: &Vrp, child: RouteState) {
        let penalty = self.penalty_capacity;
        let capacity = self.min_population_size + self.generation_size;
        let target = self.min_population_size;
        let (distance, excess) = (child.distance, child.excess);
        let individual = Individual::new(prob.get_n(), child.into_routes(), distance, excess);

        let pool = if individual.is_feasible() {
            &mut self.feasible
        } else {
            &mut self.infeasible
        };
        pool.push(individual, penalty);
        if pool.len() > capacity {
            pool.trim_to(target, penalty);
        }
    }

    /// Two parents by binary tournament over the union of both sub-populations.
    fn select_parents(&self, rng: &mut SmallRng) -> Option<(Vec<usize>, Vec<usize>)> {
        let mut candidates: Vec<((bool, usize), f64)> =
            Vec::with_capacity(self.feasible.len() + self.infeasible.len());
        candidates.extend((0..self.feasible.len()).map(|i| ((true, i), self.feasible.fitness(i))));
        candidates
            .extend((0..self.infeasible.len()).map(|i| ((false, i), self.infeasible.fitness(i))));

        let first = binary_tournament(&candidates, rng)?;
        let second = binary_tournament(&candidates, rng)?;
        Some((
            self.individual(first).giant_tour(),
            self.individual(second).giant_tour(),
        ))
    }

    fn individual(&self, key: (bool, usize)) -> &Individual {
        let (feasible, index) = key;
        if feasible {
            &self.feasible.members()[index]
        } else {
            &self.infeasible.members()[index]
        }
    }

    /// The best feasible individual, or the best infeasible one if none exists.
    fn best_individual(&self) -> Option<&Individual> {
        self.feasible
            .best(self.penalty_capacity)
            .or_else(|| self.infeasible.best(self.penalty_capacity))
    }

    /// Copies the population's best into the search state.
    fn publish_best(&mut self, state: &mut SearchState<'_, Vrp>) {
        let Some(best) = self.best_individual() else {
            return;
        };
        let cost = best.cost(self.penalty_capacity);
        let routes = best.routes.clone();
        state.solution = state.instance.solution_from_routes(routes);
        state.update_best();

        if cost < self.best_cost - 1e-9 {
            self.best_cost = cost;
            self.generations_without_improvement = 0;
        } else {
            self.generations_without_improvement += 1;
        }
    }

    /// Steers the capacity penalty toward the target feasible share.
    fn maybe_update_penalty(&mut self) {
        if self.recent_total < PENALTY_UPDATE_PERIOD {
            return;
        }
        let ratio = self.recent_feasible as f64 / self.recent_total as f64;
        if ratio < self.target_feasible - FEASIBILITY_TOLERANCE {
            self.penalty_capacity *= PENALTY_INCREASE;
        } else if ratio > self.target_feasible + FEASIBILITY_TOLERANCE {
            self.penalty_capacity *= PENALTY_DECREASE;
        }
        self.penalty_capacity = self.penalty_capacity.clamp(
            self.initial_penalty * PENALTY_MIN_FACTOR,
            self.initial_penalty * PENALTY_MAX_FACTOR,
        );
        self.recent_total = 0;
        self.recent_feasible = 0;
        // Infeasible individuals are ranked through the penalty, so their
        // fitness is stale the moment it changes.
        self.infeasible.refresh_fitness(self.penalty_capacity);
    }

    /// Wipes the population after a long stall, keeping `state.best_solution`.
    fn maybe_restart(&mut self) {
        let Some(limit) = self.restart_generations else {
            return;
        };
        if self.generations_without_improvement < limit {
            return;
        }
        self.feasible.clear();
        self.infeasible.clear();
        self.penalty_capacity = self.initial_penalty;
        self.generations_without_improvement = 0;
        self.best_cost = f64::INFINITY;
        // A restart must not re-seed from the incumbent, or it lands right back
        // in the basin it just failed to escape.
        self.warm_started = true;
    }
}

impl Heuristic<Vrp> for HybridGeneticSearch {
    fn clear(&mut self) {
        self.feasible.clear();
        self.infeasible.clear();
        self.recent_total = 0;
        self.recent_feasible = 0;
        self.generations_without_improvement = 0;
        self.best_cost = f64::INFINITY;
        self.warm_started = false;
        self.penalty_capacity = self.initial_penalty;
    }

    fn stop_condition(&self) -> &StopCondition {
        &self.stop_condition
    }

    fn run_once(&mut self, state: &mut SearchState<'_, Vrp>) -> Result<(), OptError> {
        let prob = state.instance;
        self.descent.ensure(prob, self.granularity);

        if self.feasible.is_empty() && self.infeasible.is_empty() {
            self.initial_penalty = Self::scale_free_penalty(prob);
            self.penalty_capacity = self.initial_penalty;
            self.initialize_population(state);
            return Ok(());
        }

        let Some((first, second)) = self.select_parents(&mut state.rng) else {
            return Err(OptError::InvalidState(
                "HGS population is empty after initialization".to_string(),
            ));
        };
        let tour = order_crossover(&first, &second, &mut state.rng);

        let feasible_before = self.recent_feasible;
        self.spawn(prob, tour, &mut state.rng);
        // Acceptance counters carry the feasible share, which is what the
        // adaptive penalty is steering.
        if self.recent_feasible > feasible_before {
            state.n_accepted += 1;
        } else {
            state.n_rejected += 1;
        }

        self.maybe_update_penalty();
        state.iteration += 1;
        self.publish_best(state);
        self.maybe_restart();
        Ok(())
    }
}

/// A uniformly random permutation of the customers.
fn random_tour(n: usize, rng: &mut SmallRng) -> Vec<usize> {
    let mut tour: Vec<usize> = (1..=n).collect();
    tour.shuffle(rng);
    tour
}

/// A greedy nearest-neighbor tour from a random start, falling back to a linear
/// scan when the granular candidate list is exhausted.
fn nearest_neighbor_tour(prob: &Vrp, neighbors: &[Vec<usize>], rng: &mut SmallRng) -> Vec<usize> {
    let n = prob.get_n();
    if n == 0 {
        return Vec::new();
    }
    let mut visited = vec![false; n + 1];
    let mut tour = Vec::with_capacity(n);
    let mut current = rng.random_range(1..=n);
    for _ in 0..n {
        visited[current] = true;
        tour.push(current);
        let next = neighbors[current]
            .iter()
            .copied()
            .find(|&c| !visited[c])
            .or_else(|| (1..=n).find(|&c| !visited[c]));
        match next {
            Some(c) => current = c,
            None => break,
        }
    }
    tour
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::search_state::Rankable;
    use rand::SeedableRng;

    /// Customers on a ring, unit demands: a fleet of `fleet` vehicles each
    /// carrying `capacity` of them.
    fn ring_vrp(n: usize, capacity: i64, fleet: usize) -> Vrp {
        let mut coordinates = vec![(0.0, 0.0)];
        let mut demands = vec![0i64];
        for i in 0..n {
            let theta = 2.0 * std::f64::consts::PI * i as f64 / n as f64;
            coordinates.push((10.0 * theta.cos(), 10.0 * theta.sin()));
            demands.push(1);
        }
        Vrp::new("ring", coordinates, demands, capacity, fleet)
    }

    fn hgs(iterations: u64) -> HybridGeneticSearch {
        HybridGeneticSearch::new(StopCondition::iterations(iterations), 8, 10, 10, 0.2, None)
    }

    #[test]
    fn hgs_improves_and_stays_feasible() {
        let prob = ring_vrp(24, 6, 4);
        let mut state = SearchState::new_with_seed(&prob, 7);
        let initial = state.solution.objective;

        let mut search = hgs(400);
        search.run(&mut state).unwrap();

        assert!(
            state.best_solution.objective <= initial,
            "best {} rose above the initial {initial}",
            state.best_solution.objective
        );
        assert_eq!(
            state.best_solution.overload, 0,
            "a feasible fleet must yield a feasible best"
        );
        prob.validate_routes(&state.best_solution.routes).unwrap();
        assert_eq!(state.best_solution.routes.len(), prob.num_vehicles);
    }

    #[test]
    fn hgs_is_reproducible_under_seed() {
        let prob = ring_vrp(20, 5, 4);
        let run = || {
            let mut state = SearchState::new_with_seed(&prob, 99);
            let mut search = hgs(300);
            search.run(&mut state).unwrap();
            (
                state.best_solution.objective,
                state.best_iteration,
                state.best_solution.routes.clone(),
            )
        };
        assert_eq!(run(), run());
    }

    #[test]
    fn clear_resets_the_population() {
        let prob = ring_vrp(16, 4, 4);
        let mut state = SearchState::new_with_seed(&prob, 3);
        let mut search = hgs(200);
        search.run(&mut state).unwrap();
        assert!(!search.feasible.is_empty() || !search.infeasible.is_empty());

        search.clear();
        assert!(search.feasible.is_empty());
        assert!(search.infeasible.is_empty());
        assert_eq!(search.penalty_capacity, search.initial_penalty);
        assert!(
            !search.descent.neighbors().is_empty(),
            "the candidate lists depend only on the instance and should survive"
        );
    }

    #[test]
    fn a_warm_start_is_never_lost() {
        let prob = ring_vrp(20, 5, 4);
        // The ring visited in order is optimal; hand it to HGS as the incumbent.
        let optimal = prob.solution_from_routes(vec![
            vec![1, 2, 3, 4, 5],
            vec![6, 7, 8, 9, 10],
            vec![11, 12, 13, 14, 15],
            vec![16, 17, 18, 19, 20],
        ]);
        let mut state = SearchState::with_solution_and_seed(&prob, optimal.clone(), 5);

        let mut search = hgs(200);
        search.run(&mut state).unwrap();

        assert!(
            !optimal.is_better_than(&state.best_solution),
            "best {} is worse than the warm start {}",
            state.best_solution.objective,
            optimal.objective
        );
    }

    #[test]
    fn penalty_adapts_to_the_feasible_share() {
        let prob = ring_vrp(12, 4, 3);
        let mut search = hgs(10);
        search.initial_penalty = 100.0;
        search.penalty_capacity = 100.0;

        // Every offspring feasible: far above the 0.2 target, so relax.
        search.recent_total = PENALTY_UPDATE_PERIOD;
        search.recent_feasible = PENALTY_UPDATE_PERIOD;
        search.maybe_update_penalty();
        assert!(search.penalty_capacity < 100.0);

        // None feasible: tighten.
        let relaxed = search.penalty_capacity;
        search.recent_total = PENALTY_UPDATE_PERIOD;
        search.recent_feasible = 0;
        search.maybe_update_penalty();
        assert!(search.penalty_capacity > relaxed);

        // The bounds hold however lopsided the run gets.
        for _ in 0..500 {
            search.recent_total = PENALTY_UPDATE_PERIOD;
            search.recent_feasible = 0;
            search.maybe_update_penalty();
        }
        assert!(search.penalty_capacity <= search.initial_penalty * PENALTY_MAX_FACTOR + 1e-6);
        let _ = &prob;
    }

    #[test]
    fn nearest_neighbor_tour_visits_every_customer() {
        let prob = ring_vrp(15, 5, 3);
        let mut descent = Descent::new();
        descent.ensure(&prob, 4);
        let mut rng = SmallRng::seed_from_u64(1);
        let tour = nearest_neighbor_tour(&prob, descent.neighbors(), &mut rng);
        let mut sorted = tour.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, (1..=15).collect::<Vec<_>>());
    }

    #[test]
    fn a_restart_clears_the_population() {
        let prob = ring_vrp(16, 4, 4);
        let mut state = SearchState::new_with_seed(&prob, 11);
        let mut search =
            HybridGeneticSearch::new(StopCondition::iterations(150), 8, 10, 10, 0.2, Some(5));
        search.run(&mut state).unwrap();
        // Restarts must not lose the best solution found before them.
        assert_eq!(state.best_solution.overload, 0);
        prob.validate_routes(&state.best_solution.routes).unwrap();
    }
}
