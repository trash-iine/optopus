//! Hybrid Genetic Search (HGS) for the VRP.
//!
//! HGS is the strongest known general-purpose CVRP metaheuristic, and runs
//! unchanged on the heterogeneous fleet, since the decoder and the descent
//! price every route with the vehicle type of the slot that drives it.
//! It combines three ideas, each in its own layer here:
//!
//! - [`Descent`], the granular descent shared with ALNS,
//!   which turns every offspring into a local optimum under the penalty
//!   this driver adapts at runtime.
//! - [`population`], biased fitness, which ranks individuals by cost and
//!   by how much diversity they contribute, so the population does not collapse.
//! - this module, the generational loop, the adaptive penalty, and the
//!   feasible / infeasible split that lets the search cross infeasible ground
//!   between feasible basins.
//!
//! The genetic representation is the giant tour: an offspring is a customer
//! permutation, decoded into routes by [`split_giant_tour`], which is optimal for
//! that permutation. The genetic operator therefore only has to get the customer
//! order right, the decoder handles the vehicle assignment exactly.
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

use crate::common::{MIN_IMPROVEMENT, order_crossover};
use crate::error::OptError;
use crate::heuristic::{Heuristic, StopCondition};
use crate::problem::vrp::{Vrp, split_giant_tour};
use crate::search_state::SearchState;

use crate::problem::vrp::ops::{Descent, RouteState};
use population::{Individual, Subpopulation, binary_tournament, penalized, sub_population};

/// Penalty multipliers tried, in order, when repairing.
const REPAIR_PENALTY_BOOSTS: [f64; 2] = [10.0, 100.0];

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
/// The penalty, one weight over every violation the objective charges
/// (overload, route-time excess, minimum-count shortfall), is retuned every
/// `penalty_period` offspring to hold the feasible share near
/// `target_feasible`: too few feasible offspring raises it, too many lowers
/// it. Searching at a deliberately low feasible rate
/// is the point, the shortest route through solution space between two good
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
///   Operations Research, 60(3), 611-624, 2012.
/// - Vidal, T. "Hybrid Genetic Search for the CVRP: Open-Source Implementation
///   and SWAP\* Neighborhood." Computers & Operations Research, 140, 105643, 2022.
/// - Prins, C. "A Simple and Effective Evolutionary Algorithm for the Vehicle
///   Routing Problem." Computers & Operations Research, 31(12), 1985-2002, 2004.
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
    n_elite: usize,
    n_closest: usize,
    initial_population_factor: usize,
    nearest_neighbor_every: usize,
    descent_passes: usize,
    repair_probability: f64,
    penalty_period: u32,
    penalty_increase: f64,
    penalty_decrease: f64,
    feasibility_tolerance: f64,

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
    // Vidal's published values for everything `new` does not take. Each is
    // the default behind a builder, so a caller who wants another number
    // chains the builder rather than editing a constant.

    /// Initial population size, as a multiple of `min_population_size`. See
    /// [`with_initial_population`](Self::with_initial_population).
    pub const DEFAULT_INITIAL_POPULATION_FACTOR: usize = 4;

    /// One initial individual in this many is built by a nearest-neighbour
    /// tour rather than a random one. See
    /// [`with_initial_population`](Self::with_initial_population).
    pub const DEFAULT_NEAREST_NEIGHBOR_EVERY: usize = 4;

    /// Local-search passes an offspring gets before the descent gives up on
    /// it. See [`with_descent_passes`](Self::with_descent_passes).
    pub const DEFAULT_DESCENT_PASSES: usize = 64;

    /// Probability of attempting to repair an infeasible offspring. See
    /// [`with_repair_probability`](Self::with_repair_probability).
    pub const DEFAULT_REPAIR_PROBABILITY: f64 = 0.5;

    /// Offspring between two adjustments of the capacity penalty. See
    /// [`with_penalty_adaptation`](Self::with_penalty_adaptation).
    pub const DEFAULT_PENALTY_PERIOD: u32 = 100;

    /// Multiplicative adjustment when too few offspring are feasible. See
    /// [`with_penalty_adaptation`](Self::with_penalty_adaptation).
    pub const DEFAULT_PENALTY_INCREASE: f64 = 1.2;

    /// Multiplicative adjustment when too many offspring are feasible. See
    /// [`with_penalty_adaptation`](Self::with_penalty_adaptation).
    pub const DEFAULT_PENALTY_DECREASE: f64 = 0.85;

    /// Dead band around `target_feasible` in which the penalty is left
    /// alone. See [`with_penalty_adaptation`](Self::with_penalty_adaptation).
    pub const DEFAULT_FEASIBILITY_TOLERANCE: f64 = 0.05;

    /// Members the cost rank alone keeps alive, Vidal's `nbElite`. See
    /// [`with_elite`](Self::with_elite).
    pub const DEFAULT_N_ELITE: usize = 4;

    /// Nearest members averaged into a diversity contribution, Vidal's
    /// `nbClose`. See [`with_elite`](Self::with_elite).
    pub const DEFAULT_N_CLOSEST: usize = 5;

    /// Creates a new Hybrid Genetic Search.
    ///
    /// Reasonable defaults follow Vidal: `min_population_size = 25`,
    /// `generation_size = 40`, `granularity = 20`, `target_feasible = 0.2`,
    /// `restart_generations = Some(20_000)`. Everything else Vidal tunes is a
    /// builder with a published default, `with_elite`,
    /// `with_penalty_adaptation`, `with_repair_probability`,
    /// `with_initial_population` and `with_descent_passes`.
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
            n_elite: Self::DEFAULT_N_ELITE,
            n_closest: Self::DEFAULT_N_CLOSEST,
            initial_population_factor: Self::DEFAULT_INITIAL_POPULATION_FACTOR,
            nearest_neighbor_every: Self::DEFAULT_NEAREST_NEIGHBOR_EVERY,
            descent_passes: Self::DEFAULT_DESCENT_PASSES,
            repair_probability: Self::DEFAULT_REPAIR_PROBABILITY,
            penalty_period: Self::DEFAULT_PENALTY_PERIOD,
            penalty_increase: Self::DEFAULT_PENALTY_INCREASE,
            penalty_decrease: Self::DEFAULT_PENALTY_DECREASE,
            feasibility_tolerance: Self::DEFAULT_FEASIBILITY_TOLERANCE,
            feasible: sub_population(1.0, Self::DEFAULT_N_ELITE, Self::DEFAULT_N_CLOSEST),
            infeasible: sub_population(1.0, Self::DEFAULT_N_ELITE, Self::DEFAULT_N_CLOSEST),
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

    /// Builder-style: Vidal's `nbElite` and `nbClose`, how many members the
    /// cost rank alone keeps alive and how many nearest members are averaged
    /// into a diversity contribution. Defaults to
    /// [`DEFAULT_N_ELITE`](Self::DEFAULT_N_ELITE) /
    /// [`DEFAULT_N_CLOSEST`](Self::DEFAULT_N_CLOSEST). Rebuilds both
    /// sub-populations, which is why it takes `self` by value; a builder runs
    /// before the search does, so there is nothing to lose.
    ///
    /// # Panics
    ///
    /// Panics if either is zero (checked by
    /// [`BiasedFitnessPopulation::new`](crate::common::BiasedFitnessPopulation::new)).
    pub fn with_elite(mut self, n_elite: usize, n_closest: usize) -> Self {
        self.n_elite = n_elite;
        self.n_closest = n_closest;
        self.feasible = sub_population(self.penalty_capacity, n_elite, n_closest);
        self.infeasible = sub_population(self.penalty_capacity, n_elite, n_closest);
        self
    }

    /// Builder-style: how the capacity penalty chases `target_feasible`. Every
    /// `period` offspring the feasible share is read, and the penalty is
    /// multiplied by `increase` when the share is more than `tolerance` below
    /// the target, by `decrease` when it is more than `tolerance` above.
    /// Defaults to [`DEFAULT_PENALTY_PERIOD`](Self::DEFAULT_PENALTY_PERIOD),
    /// [`DEFAULT_PENALTY_INCREASE`](Self::DEFAULT_PENALTY_INCREASE),
    /// [`DEFAULT_PENALTY_DECREASE`](Self::DEFAULT_PENALTY_DECREASE) and
    /// [`DEFAULT_FEASIBILITY_TOLERANCE`](Self::DEFAULT_FEASIBILITY_TOLERANCE).
    ///
    /// # Panics
    ///
    /// Panics if `period` is zero, `increase` is not above `1`, `decrease` is
    /// not in `(0, 1)`, or `tolerance` is not in `[0, 0.5)`.
    pub fn with_penalty_adaptation(
        mut self,
        period: u32,
        increase: f64,
        decrease: f64,
        tolerance: f64,
    ) -> Self {
        assert!(period >= 1, "penalty period must be at least 1");
        assert!(
            increase > 1.0,
            "penalty increase must be above 1, got {increase}"
        );
        assert!(
            decrease > 0.0 && decrease < 1.0,
            "penalty decrease must be in (0, 1), got {decrease}"
        );
        assert!(
            (0.0..0.5).contains(&tolerance),
            "feasibility tolerance must be in [0, 0.5), got {tolerance}"
        );
        self.penalty_period = period;
        self.penalty_increase = increase;
        self.penalty_decrease = decrease;
        self.feasibility_tolerance = tolerance;
        self
    }

    /// Builder-style: the probability an infeasible offspring is also
    /// repaired under a boosted penalty. Defaults to
    /// [`DEFAULT_REPAIR_PROBABILITY`](Self::DEFAULT_REPAIR_PROBABILITY).
    ///
    /// # Panics
    ///
    /// Panics if `probability` is not in `[0, 1]`.
    pub fn with_repair_probability(mut self, probability: f64) -> Self {
        assert!(
            (0.0..=1.0).contains(&probability),
            "repair probability must be in [0, 1], got {probability}"
        );
        self.repair_probability = probability;
        self
    }

    /// Builder-style: how the initial population is seeded. It holds
    /// `factor · min_population_size` individuals, and one in every
    /// `nearest_neighbor_every` of them is built by a nearest-neighbour tour
    /// rather than drawn at random. Defaults to
    /// [`DEFAULT_INITIAL_POPULATION_FACTOR`](Self::DEFAULT_INITIAL_POPULATION_FACTOR)
    /// and [`DEFAULT_NEAREST_NEIGHBOR_EVERY`](Self::DEFAULT_NEAREST_NEIGHBOR_EVERY).
    ///
    /// # Panics
    ///
    /// Panics if either is zero.
    pub fn with_initial_population(mut self, factor: usize, nearest_neighbor_every: usize) -> Self {
        assert!(factor >= 1, "initial population factor must be at least 1");
        assert!(
            nearest_neighbor_every >= 1,
            "nearest_neighbor_every must be at least 1"
        );
        self.initial_population_factor = factor;
        self.nearest_neighbor_every = nearest_neighbor_every;
        self
    }

    /// Builder-style: how many passes the granular descent may spend on one
    /// offspring. Defaults to
    /// [`DEFAULT_DESCENT_PASSES`](Self::DEFAULT_DESCENT_PASSES).
    ///
    /// # Panics
    ///
    /// Panics if `passes` is zero.
    pub fn with_descent_passes(mut self, passes: usize) -> Self {
        assert!(passes >= 1, "descent passes must be at least 1");
        self.descent_passes = passes;
        self
    }

    /// Starting penalty: the average distance per unit of demand, so it is on
    /// the same scale as the objective regardless of the instance.
    fn scale_free_penalty(prob: &Vrp) -> f64 {
        let n = prob.get_n();
        if n == 0 {
            return 1.0;
        }
        let mean_depot_distance = (1..=n).map(|c| prob.distance(0, c)).sum::<f64>() / n as f64;
        let mean_demand = (1..=n).map(|c| prob.demands[c]).sum::<i64>() as f64 / n as f64;
        (mean_depot_distance / mean_demand.max(1.0)).max(f64::MIN_POSITIVE)
    }

    /// Seeds both sub-populations with `initial_population_factor × μ`
    /// individuals.
    ///
    /// [`Vrp::new_solution`](crate::search_state::ProblemTrait::new_solution) is
    /// not used: it assigns customers to the least-loaded vehicle in random
    /// order, ignoring geometry entirely, which starts the search far from
    /// anything a route-based local search can fix cheaply.
    fn initialize_population(&mut self, state: &mut SearchState<'_, Vrp>) {
        let prob = state.instance;
        let n = prob.get_n();
        let count = self.initial_population_factor * self.min_population_size;

        for i in 0..count {
            let tour = if i == 0 && !self.warm_started {
                // Carry the incumbent forward when composed with another heuristic.
                self.warm_started = true;
                state.solution.routes.iter().flatten().copied().collect()
            } else if i % self.nearest_neighbor_every == 1 {
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
        let mut child = RouteState::from_routes(prob, routes, self.penalty_capacity);
        self.descent.run(
            &mut child,
            prob,
            rng,
            self.penalty_capacity,
            self.descent_passes,
        );

        self.recent_total += 1;
        if child.sol.violation() == 0.0 {
            self.recent_feasible += 1;
            self.absorb(prob, child);
            return;
        }

        // An infeasible child is worth keeping as-is (it may be a useful bridge),
        // but a feasible version of it is worth more.
        if rng.random::<f64>() < self.repair_probability {
            let mut repaired = child.clone();
            for boost in REPAIR_PENALTY_BOOSTS {
                self.descent.run(
                    &mut repaired,
                    prob,
                    rng,
                    self.penalty_capacity * boost,
                    self.descent_passes,
                );
                if repaired.sol.violation() == 0.0 {
                    self.absorb(prob, repaired);
                    break;
                }
            }
        }
        self.absorb(prob, child);
    }

    /// Files an evaluated child into its sub-population, culling if it overflows.
    fn absorb(&mut self, prob: &Vrp, child: RouteState) {
        let capacity = self.min_population_size + self.generation_size;
        let target = self.min_population_size;
        let sol = child.sol;
        // Priced at zero penalty, so the individual carries the objective's
        // unpenalized part and the search supplies its own weight.
        let (base, violation) = (prob.objective_under(&sol, 0.0), sol.violation());
        let individual = Individual::new(prob.get_n(), sol.routes, base, violation);

        let pool = if individual.is_feasible() {
            &mut self.feasible
        } else {
            &mut self.infeasible
        };
        pool.push(individual);
        if pool.len() > capacity {
            pool.trim_to(target);
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
        self.feasible.best().or_else(|| self.infeasible.best())
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

        if cost < self.best_cost - MIN_IMPROVEMENT {
            self.best_cost = cost;
            self.generations_without_improvement = 0;
        } else {
            self.generations_without_improvement += 1;
        }
    }

    /// The one place the capacity penalty is written.
    ///
    /// Both sub-populations rank through it, so moving it without telling them
    /// would leave the order reflecting a penalty the search no longer uses.
    /// Routing every write here is what makes that unwritable. The feasible
    /// pool's costs do not actually move, its members having no excess, but it
    /// is told anyway: an exception here is the next thing to get wrong.
    fn set_penalty(&mut self, penalty: f64) {
        self.penalty_capacity = penalty;
        self.feasible.set_cost(penalized(penalty));
        self.infeasible.set_cost(penalized(penalty));
    }

    /// Steers the capacity penalty toward the target feasible share.
    fn maybe_update_penalty(&mut self) {
        if self.recent_total < self.penalty_period {
            return;
        }
        let ratio = self.recent_feasible as f64 / self.recent_total as f64;
        let mut penalty = self.penalty_capacity;
        if ratio < self.target_feasible - self.feasibility_tolerance {
            penalty *= self.penalty_increase;
        } else if ratio > self.target_feasible + self.feasibility_tolerance {
            penalty *= self.penalty_decrease;
        }
        self.recent_total = 0;
        self.recent_feasible = 0;
        self.set_penalty(penalty.clamp(
            self.initial_penalty * PENALTY_MIN_FACTOR,
            self.initial_penalty * PENALTY_MAX_FACTOR,
        ));
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
        self.set_penalty(self.initial_penalty);
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
        self.set_penalty(self.initial_penalty);
    }

    fn stop_condition(&self) -> &StopCondition {
        &self.stop_condition
    }

    fn run_once(&mut self, state: &mut SearchState<'_, Vrp>) -> Result<(), OptError> {
        let prob = state.instance;
        self.descent.ensure(prob, self.granularity);

        if self.feasible.is_empty() && self.infeasible.is_empty() {
            self.initial_penalty = Self::scale_free_penalty(prob);
            self.set_penalty(self.initial_penalty);
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
        assert_eq!(state.best_solution.routes.len(), prob.num_slots());
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
        let period = search.penalty_period;
        search.recent_total = period;
        search.recent_feasible = period;
        search.maybe_update_penalty();
        assert!(search.penalty_capacity < 100.0);

        // None feasible: tighten.
        let relaxed = search.penalty_capacity;
        search.recent_total = period;
        search.recent_feasible = 0;
        search.maybe_update_penalty();
        assert!(search.penalty_capacity > relaxed);

        // The bounds hold however lopsided the run gets.
        for _ in 0..500 {
            search.recent_total = period;
            search.recent_feasible = 0;
            search.maybe_update_penalty();
        }
        assert!(search.penalty_capacity <= search.initial_penalty * PENALTY_MAX_FACTOR + 1e-6);
        let _ = &prob;
    }

    /// The adaptation builder has to reach the update, not merely be stored.
    /// A period of 10 adjusts where the default 100 would still be waiting,
    /// and the factors it adjusts by are the ones given.
    #[test]
    fn with_penalty_adaptation_sets_the_period_and_the_factors() {
        let mut search = hgs(10).with_penalty_adaptation(10, 1.5, 0.5, 0.0);
        search.initial_penalty = 100.0;
        search.penalty_capacity = 100.0;

        search.recent_total = 10;
        search.recent_feasible = 10;
        search.maybe_update_penalty();
        assert_eq!(search.penalty_capacity, 50.0, "decrease factor reached");

        search.recent_total = 10;
        search.recent_feasible = 0;
        search.maybe_update_penalty();
        assert_eq!(search.penalty_capacity, 75.0, "increase factor reached");

        let mut untouched = hgs(10);
        untouched.initial_penalty = 100.0;
        untouched.penalty_capacity = 100.0;
        untouched.recent_total = 10;
        untouched.recent_feasible = 10;
        untouched.maybe_update_penalty();
        assert_eq!(
            untouched.penalty_capacity, 100.0,
            "default period is longer"
        );
    }

    /// `with_elite` rebuilds the sub-populations with the given ranks, and
    /// the seeding builder sizes the first generation.
    #[test]
    fn with_elite_and_with_initial_population_reach_the_population() {
        let prob = ring_vrp(12, 4, 3);
        let mut state = SearchState::new_with_seed(&prob, 5);
        let mut search = hgs(1).with_elite(2, 3).with_initial_population(2, 2);
        search.run(&mut state).unwrap();
        assert_eq!(search.feasible.n_elite(), 2);
        assert_eq!(search.feasible.n_closest(), 3);
        // Two times `min_population_size` individuals were seeded, charged to
        // the iteration counter, plus the one generation the budget allows.
        assert!(
            state.iteration >= 2 * 8,
            "seeded {} individuals",
            state.iteration
        );
    }

    /// Everything the builders set has to survive into a seeded run
    /// reproducibly, and the run has to stay valid.
    #[test]
    fn a_tuned_hgs_is_still_reproducible() {
        let prob = ring_vrp(16, 4, 4);
        let run = || {
            let mut state = SearchState::new_with_seed(&prob, 21);
            hgs(150)
                .with_elite(2, 2)
                .with_penalty_adaptation(20, 1.5, 0.7, 0.1)
                .with_repair_probability(1.0)
                .with_initial_population(2, 3)
                .with_descent_passes(8)
                .run(&mut state)
                .unwrap();
            prob.validate_routes(&state.best_solution.routes).unwrap();
            state.best_solution.objective
        };
        assert_eq!(run(), run());
    }

    #[test]
    #[should_panic(expected = "penalty increase must be above 1")]
    fn a_penalty_increase_of_one_is_rejected() {
        let _ = hgs(1).with_penalty_adaptation(100, 1.0, 0.85, 0.05);
    }

    #[test]
    #[should_panic(expected = "repair probability must be in [0, 1]")]
    fn a_repair_probability_above_one_is_rejected() {
        let _ = hgs(1).with_repair_probability(1.5);
    }

    #[test]
    #[should_panic(expected = "descent passes must be at least 1")]
    fn zero_descent_passes_are_rejected() {
        let _ = hgs(1).with_descent_passes(0);
    }

    #[test]
    #[should_panic(expected = "n_elite must be at least 1")]
    fn zero_elite_is_rejected() {
        let _ = hgs(1).with_elite(0, 5);
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

        // The best is held by the search state, not by the populations, which
        // is what makes a wipe safe.
        let best_before = state.best_solution.clone();
        assert_eq!(best_before.overload, 0);
        prob.validate_routes(&best_before.routes).unwrap();

        // The run above leaves a populated search whether or not it happened
        // to meet the limit, so the wipe itself is driven here. Reading the
        // populations afterwards is the only way to see it: every outward
        // measure of the run is the same with `maybe_restart` emptied out.
        assert!(!search.feasible.is_empty() || !search.infeasible.is_empty());
        search.set_penalty(search.initial_penalty * PENALTY_MAX_FACTOR);
        search.generations_without_improvement = 5;
        search.maybe_restart();

        assert!(search.feasible.is_empty(), "feasible survived the restart");
        assert!(
            search.infeasible.is_empty(),
            "infeasible survived the restart"
        );
        assert_eq!(search.penalty_capacity, search.initial_penalty);
        assert_eq!(search.generations_without_improvement, 0);
        assert_eq!(search.best_cost, f64::INFINITY);
        assert!(
            search.warm_started,
            "a restart must not re-seed from the incumbent"
        );
        assert_eq!(state.best_solution.routes, best_before.routes);
    }
}
