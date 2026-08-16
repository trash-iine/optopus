//! Adaptive Large Neighborhood Search (ALNS) for the Capacitated VRP.
//!
//! ALNS (Ropke & Pisinger, 2006) is the canonical, most effective general-purpose
//! VRP metaheuristic: each iteration *ruins* part of the incumbent with a destroy
//! operator and *recreates* it with a repair operator, choosing operators
//! adaptively by a roulette wheel whose weights track recent performance, and
//! accepting worse solutions with a simulated-annealing criterion.
//!
//! The destroy/repair bank lives in [`AlnsOps`]; the outer
//! [`AdaptiveLargeNeighborhoodSearch`] drives the accept/score/cool loop,
//! operating directly on `state.solution` (like
//! [`super::super::lkh_for_tsp`]) since a destroy+repair step is not a single
//! [`MoveToNeighbor`](crate::search_state::MoveToNeighbor).
//!
//! What a route edit *costs* is not decided here: the insertion and removal
//! deltas come from [`super::ops`], and each recreated solution is handed to the
//! same granular descent Hybrid Genetic Search uses — with
//! [`Vrp::penalty_weight`] as its penalty, the weight this heuristic's objective
//! already charges overload at. Ruin-and-recreate alone re-inserts customers
//! greedily and never repairs the edges it disturbs elsewhere in the route.

use rand::Rng;
use rand::rngs::SmallRng;
use rand::seq::SliceRandom;

use super::ops::{self, RouteState};
use crate::error::OptError;
use crate::heuristic::{Heuristic, StopCondition};
use crate::problem::vrp::Vrp;
use crate::search_state::SearchState;

const NUM_DESTROY: usize = 3; // Random, Worst, Shaw
const NUM_REPAIR: usize = 2; // Greedy, Regret-2

// Adaptive scoring: reward for a global best / a better-than-current / an
// accepted (worse) solution. Segment length and reaction factor follow the
// standard ALNS scheme.
const SIGMA_BEST: f64 = 4.0;
const SIGMA_BETTER: f64 = 2.0;
const SIGMA_ACCEPT: f64 = 1.0;
const SEGMENT_LEN: u64 = 100;
const REACTION: f64 = 0.1;

/// Nearest partners considered per customer by the post-repair descent. Matches
/// the default Hybrid Genetic Search uses.
const GRANULARITY: usize = 20;

/// Descent passes over a recreated solution. The descent is anchored at the
/// re-inserted customers, so a pass is cheap and a handful of them is enough to
/// settle the routes the ruin disturbed.
const MAX_LS_PASSES: usize = 4;

/// Destroy/repair operator bank plus the adaptive-weight bookkeeping.
struct AlnsOps {
    destroy_weights: [f64; NUM_DESTROY],
    repair_weights: [f64; NUM_REPAIR],
    destroy_scores: [f64; NUM_DESTROY],
    repair_scores: [f64; NUM_REPAIR],
    destroy_counts: [u64; NUM_DESTROY],
    repair_counts: [u64; NUM_REPAIR],
    segment_iter: u64,
}

impl AlnsOps {
    fn new() -> Self {
        Self {
            destroy_weights: [1.0; NUM_DESTROY],
            repair_weights: [1.0; NUM_REPAIR],
            destroy_scores: [0.0; NUM_DESTROY],
            repair_scores: [0.0; NUM_REPAIR],
            destroy_counts: [0; NUM_DESTROY],
            repair_counts: [0; NUM_REPAIR],
            segment_iter: 0,
        }
    }

    fn reset(&mut self) {
        *self = Self::new();
    }

    /// Roulette-wheel index selection proportional to `weights`.
    fn roulette(weights: &[f64], rng: &mut SmallRng) -> usize {
        let total: f64 = weights.iter().sum();
        if total <= 0.0 {
            return rng.random_range(0..weights.len());
        }
        let mut r = rng.random::<f64>() * total;
        for (i, &w) in weights.iter().enumerate() {
            if r < w {
                return i;
            }
            r -= w;
        }
        weights.len() - 1
    }

    fn select_destroy(&self, rng: &mut SmallRng) -> usize {
        Self::roulette(&self.destroy_weights, rng)
    }

    fn select_repair(&self, rng: &mut SmallRng) -> usize {
        Self::roulette(&self.repair_weights, rng)
    }

    fn record(&mut self, destroy: usize, repair: usize, score: f64) {
        self.destroy_scores[destroy] += score;
        self.destroy_counts[destroy] += 1;
        self.repair_scores[repair] += score;
        self.repair_counts[repair] += 1;
        self.segment_iter += 1;
    }

    /// At each segment boundary, blend recent average scores into the weights.
    fn maybe_update_weights(&mut self) {
        if self.segment_iter < SEGMENT_LEN {
            return;
        }
        for i in 0..NUM_DESTROY {
            if self.destroy_counts[i] > 0 {
                let avg = self.destroy_scores[i] / self.destroy_counts[i] as f64;
                self.destroy_weights[i] =
                    (1.0 - REACTION) * self.destroy_weights[i] + REACTION * avg;
            }
            self.destroy_scores[i] = 0.0;
            self.destroy_counts[i] = 0;
        }
        for i in 0..NUM_REPAIR {
            if self.repair_counts[i] > 0 {
                let avg = self.repair_scores[i] / self.repair_counts[i] as f64;
                self.repair_weights[i] = (1.0 - REACTION) * self.repair_weights[i] + REACTION * avg;
            }
            self.repair_scores[i] = 0.0;
            self.repair_counts[i] = 0;
        }
        self.segment_iter = 0;
    }

    // ---- Destroy operators: remove `k` customers, returning them ----

    fn destroy(
        &self,
        idx: usize,
        prob: &Vrp,
        routes: &mut [Vec<usize>],
        loads: &mut [i64],
        k: usize,
        rng: &mut SmallRng,
    ) -> Vec<usize> {
        let removed = match idx {
            0 => Self::random_removal(routes, k, rng),
            1 => Self::worst_removal(prob, routes, k),
            _ => Self::shaw_removal(prob, routes, k, rng),
        };
        ops::route_loads(prob, routes, loads);
        removed
    }

    fn remove_set(routes: &mut [Vec<usize>], set: &[bool]) {
        for route in routes.iter_mut() {
            route.retain(|&c| !set[c]);
        }
    }

    fn random_removal(routes: &mut [Vec<usize>], k: usize, rng: &mut SmallRng) -> Vec<usize> {
        let mut all: Vec<usize> = routes.iter().flatten().copied().collect();
        all.shuffle(rng);
        all.truncate(k);
        let mut set = vec![false; Self::max_customer(routes) + 1];
        for &c in &all {
            set[c] = true;
        }
        Self::remove_set(routes, &set);
        all
    }

    fn worst_removal(prob: &Vrp, routes: &mut [Vec<usize>], k: usize) -> Vec<usize> {
        let mut costs: Vec<(f64, usize)> = Vec::new();
        for route in routes.iter() {
            for (i, &c) in route.iter().enumerate() {
                costs.push((ops::removal_gain(prob, route, i, 1), c));
            }
        }
        costs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        let chosen: Vec<usize> = costs.into_iter().take(k).map(|(_, c)| c).collect();
        let mut set = vec![false; Self::max_customer(routes) + 1];
        for &c in &chosen {
            set[c] = true;
        }
        Self::remove_set(routes, &set);
        chosen
    }

    fn shaw_removal(
        prob: &Vrp,
        routes: &mut [Vec<usize>],
        k: usize,
        rng: &mut SmallRng,
    ) -> Vec<usize> {
        let all: Vec<usize> = routes.iter().flatten().copied().collect();
        if all.is_empty() {
            return Vec::new();
        }
        let seed = all[rng.random_range(0..all.len())];
        let mut related: Vec<(f64, usize)> = all
            .iter()
            .map(|&c| {
                let rel = prob.distance(seed, c)
                    + (prob.demands[seed] - prob.demands[c]).unsigned_abs() as f64;
                (rel, c)
            })
            .collect();
        related.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        let chosen: Vec<usize> = related.into_iter().take(k).map(|(_, c)| c).collect();
        let mut set = vec![false; Self::max_customer(routes) + 1];
        for &c in &chosen {
            set[c] = true;
        }
        Self::remove_set(routes, &set);
        chosen
    }

    fn max_customer(routes: &[Vec<usize>]) -> usize {
        routes.iter().flatten().copied().max().unwrap_or(0)
    }

    // ---- Repair operators: re-insert all removed customers ----

    fn repair(
        &self,
        idx: usize,
        prob: &Vrp,
        routes: &mut [Vec<usize>],
        loads: &mut [i64],
        removed: Vec<usize>,
        rng: &mut SmallRng,
    ) {
        match idx {
            0 => Self::greedy_insertion(prob, routes, loads, removed, rng),
            _ => Self::regret2_insertion(prob, routes, loads, removed),
        }
    }

    /// The cheapest (augmented) insertion of `c`, and the cheapest one into a
    /// *different* route, each as `(cost, route, pos)`.
    ///
    /// The two are taken across routes, not across positions, because that is what
    /// regret-k is defined on (Ropke & Pisinger): the gap that matters is how much
    /// worse `c` gets once its best vehicle is taken, and the second-cheapest
    /// *slot* is almost always the one next door in the same route — a gap of
    /// nearly zero for every customer, which would leave [`Self::regret2_insertion`]
    /// indistinguishable from [`Self::greedy_insertion`].
    ///
    /// The augmented cost folds capacity overflow in via the penalty weight, so an
    /// insertion is always available even when every route is full. With a
    /// single-vehicle fleet the second entry stays infinite; the caller reads that
    /// as unbounded regret.
    fn best_two_insertions(
        prob: &Vrp,
        routes: &[Vec<usize>],
        loads: &[i64],
        c: usize,
    ) -> ((f64, usize, usize), (f64, usize, usize)) {
        let dc = prob.demands[c];
        let cap = prob.capacity;
        let pw = prob.penalty_weight();
        let mut b1 = (f64::INFINITY, 0usize, 0usize);
        let mut b2 = (f64::INFINITY, 0usize, 0usize);
        for (r, route) in routes.iter().enumerate() {
            let pen = pw * ops::excess_delta_insert(cap, loads[r], dc) as f64;
            let mut best_here = (f64::INFINITY, r, 0usize);
            for pos in 0..=route.len() {
                let cost = ops::insertion_cost(prob, route, pos, c, c) + pen;
                if cost < best_here.0 {
                    best_here = (cost, r, pos);
                }
            }
            if best_here.0 < b1.0 {
                b2 = b1;
                b1 = best_here;
            } else if best_here.0 < b2.0 {
                b2 = best_here;
            }
        }
        (b1, b2)
    }

    fn greedy_insertion(
        prob: &Vrp,
        routes: &mut [Vec<usize>],
        loads: &mut [i64],
        removed: Vec<usize>,
        rng: &mut SmallRng,
    ) {
        let mut removed = removed;
        removed.shuffle(rng);
        for c in removed {
            let (best, _) = Self::best_two_insertions(prob, routes, loads, c);
            let (_, r, pos) = best;
            routes[r].insert(pos, c);
            loads[r] += prob.demands[c];
        }
    }

    fn regret2_insertion(
        prob: &Vrp,
        routes: &mut [Vec<usize>],
        loads: &mut [i64],
        removed: Vec<usize>,
    ) {
        let mut pool = removed;
        while !pool.is_empty() {
            let mut best_regret = f64::NEG_INFINITY;
            let mut best_idx = 0usize;
            let mut best_place = (0usize, 0usize);
            for (idx, &c) in pool.iter().enumerate() {
                let (b1, b2) = Self::best_two_insertions(prob, routes, loads, c);
                // Larger regret (best other route minus best) → insert this one
                // first, before its one good vehicle fills up.
                let regret = if b2.0.is_finite() {
                    b2.0 - b1.0
                } else {
                    f64::INFINITY
                };
                if regret > best_regret {
                    best_regret = regret;
                    best_idx = idx;
                    best_place = (b1.1, b1.2);
                }
            }
            let c = pool.swap_remove(best_idx);
            routes[best_place.0].insert(best_place.1, c);
            loads[best_place.0] += prob.demands[c];
        }
    }
}

/// Adaptive Large Neighborhood Search for the Capacitated VRP.
///
/// # Example
///
/// ```
/// use optopus::heuristic::{AdaptiveLargeNeighborhoodSearchForVrp, Heuristic, StopCondition};
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
/// let mut alns = AdaptiveLargeNeighborhoodSearchForVrp::new(
///     StopCondition::iterations(2_000),
///     0.2,
///     0.999,
/// );
/// alns.run(&mut state).unwrap();
/// ```
pub struct AdaptiveLargeNeighborhoodSearch {
    stop_condition: StopCondition,
    ops: AlnsOps,
    /// Fraction of customers ruined each iteration.
    removal_fraction: f64,
    /// Geometric cooling factor applied to the temperature each iteration.
    cooling_rate: f64,
    /// Current SA temperature (initialized lazily from the first solution).
    temperature: Option<f64>,
    /// The post-repair descent, holding its instance-derived candidate lists;
    /// they survive [`Heuristic::clear`] because they depend on nothing else.
    descent: ops::Descent,
}

impl AdaptiveLargeNeighborhoodSearch {
    /// Creates a new ALNS.
    ///
    /// # Panics
    /// Panics if `removal_fraction` is not in `(0, 1]` or `cooling_rate` is not
    /// in `(0, 1]`.
    pub fn new(stop_condition: StopCondition, removal_fraction: f64, cooling_rate: f64) -> Self {
        assert!(
            removal_fraction > 0.0 && removal_fraction <= 1.0,
            "removal_fraction must be in (0, 1]"
        );
        assert!(
            cooling_rate > 0.0 && cooling_rate <= 1.0,
            "cooling_rate must be in (0, 1]"
        );
        Self {
            stop_condition,
            ops: AlnsOps::new(),
            removal_fraction,
            cooling_rate,
            temperature: None,
            descent: ops::Descent::new(),
        }
    }

    fn removal_count(&self, n: usize) -> usize {
        let k = (self.removal_fraction * n as f64).round() as usize;
        k.clamp(1, n.saturating_sub(1).max(1)).min(50)
    }
}

impl Heuristic<Vrp> for AdaptiveLargeNeighborhoodSearch {
    fn clear(&mut self) {
        self.ops.reset();
        self.temperature = None;
    }

    fn stop_condition(&self) -> &StopCondition {
        &self.stop_condition
    }

    fn run_once<'a>(&mut self, state: &mut SearchState<'a, Vrp>) -> Result<(), OptError> {
        let prob: &Vrp = state.instance;
        let n = prob.get_n();
        if n == 0 {
            state.progress_iteration();
            return Ok(());
        }
        self.descent.ensure(prob, GRANULARITY);

        // Lazily initialize the temperature so a solution ~5% worse is accepted
        // with probability ~0.5 at the start.
        let temperature = *self.temperature.get_or_insert_with(|| {
            (0.05 * state.solution.objective.abs()).max(1.0) / std::f64::consts::LN_2
        });

        let current = state.solution.clone();
        let k = self.removal_count(n);
        let d_idx = self.ops.select_destroy(&mut state.rng);
        let r_idx = self.ops.select_repair(&mut state.rng);

        let mut routes = current.routes.clone();
        let mut loads = current.route_loads.clone();
        let removed = self
            .ops
            .destroy(d_idx, prob, &mut routes, &mut loads, k, &mut state.rng);
        let anchors = removed.clone();
        self.ops.repair(
            r_idx,
            prob,
            &mut routes,
            &mut loads,
            removed,
            &mut state.rng,
        );
        // Recreation leaves the disturbed routes locally poor: a customer is
        // inserted where it is cheapest *now*, with no chance to fix the edges
        // that choice spoils. The descent is what makes the ruin worth doing —
        // anchored at the re-inserted customers, since the rest of the solution
        // is exactly as locally optimal as the previous iteration left it.
        let mut recreated = RouteState::from_routes(prob, routes);
        self.descent.run_around(
            &mut recreated,
            prob,
            &anchors,
            &mut state.rng,
            prob.penalty_weight(),
            MAX_LS_PASSES,
        );
        let candidate = prob.solution_from_routes(recreated.into_routes());

        // Simulated-annealing acceptance (minimization).
        let accept = candidate.objective <= current.objective
            || state.rng.random::<f64>()
                < ((current.objective - candidate.objective) / temperature).exp();

        let score = if candidate.objective < state.best_solution.objective {
            SIGMA_BEST
        } else if candidate.objective < current.objective {
            SIGMA_BETTER
        } else if accept {
            SIGMA_ACCEPT
        } else {
            0.0
        };
        self.ops.record(d_idx, r_idx, score);

        state.iteration += 1;
        if accept {
            state.solution = candidate;
            state.n_accepted += 1;
        } else {
            state.n_rejected += 1;
        }
        state.update_best();

        self.temperature = Some((temperature * self.cooling_rate).max(1e-9));
        self.ops.maybe_update_weights();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::problem::vrp::Vrp;

    fn ring_vrp() -> Vrp {
        // 8 customers on a circle around the depot; capacity 2, 4 vehicles.
        let mut coords = vec![(0.0, 0.0)];
        let mut demands = vec![0];
        for i in 0..8 {
            let theta = std::f64::consts::TAU * i as f64 / 8.0;
            coords.push((theta.cos(), theta.sin()));
            demands.push(1);
        }
        Vrp::new("ring", coords, demands, 2, 4)
    }

    #[test]
    fn alns_improves_and_stays_feasible() {
        let vrp = ring_vrp();
        let mut state = SearchState::new_with_seed(&vrp, 7);
        let initial = state.best_solution.objective;
        let mut alns =
            AdaptiveLargeNeighborhoodSearch::new(StopCondition::iterations(3_000), 0.3, 0.999);
        alns.run(&mut state).unwrap();
        assert!(
            state.best_solution.objective <= initial,
            "ALNS must not worsen the incumbent"
        );
        assert_eq!(state.best_solution.overload, 0, "best should be feasible");
        vrp.validate_routes(&state.best_solution.routes).unwrap();
    }

    #[test]
    fn alns_is_reproducible_under_seed() {
        let vrp = ring_vrp();
        let run = || {
            let mut state = SearchState::new_with_seed(&vrp, 123);
            let mut alns = AdaptiveLargeNeighborhoodSearch::new(
                StopCondition::iterations(1_500),
                0.25,
                0.9995,
            );
            alns.run(&mut state).unwrap();
            state.best_solution.objective
        };
        assert_eq!(run(), run());
    }
}
