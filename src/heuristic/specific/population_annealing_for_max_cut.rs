use super::super::simulated_annealing::boltzmann_accept;
use super::super::{Heuristic, StopCondition};
use crate::error::OptError;
use crate::problem::max_cut::MaxCutFlipNeighbor;
use crate::problem::{MaxCut, MaxCutSolution};
use crate::search_state::SearchState;
use crate::trait_defs::{Evaluate, MoveToNeighbor, ProblemTrait, Rankable};
use rand::Rng;
use rand::rngs::SmallRng;

/// Population Annealing Monte Carlo (PAMC) for the MaxCut problem, with
/// non-local cluster ("iso-site") moves.
///
/// PAMC maintains a population of `population_size` replicas and cools a shared
/// inverse temperature `β` from `initial_beta` upward in steps of `delta_beta`.
/// Each temperature step:
///
/// 1. **Metropolis sweeps** — every replica is swept `sweeps_per_step` times at
///    the current `β`; a proposed flip with cut change `gain` is accepted with
///    probability `min(1, exp(β · gain))` (reusing [`boltzmann_accept`]).
/// 2. **Non-local cluster move (NCM)** — when `cluster_moves` is set, a maximal
///    independent set of zero-gain ("iso-site") vertices is flipped in each
///    replica. Independence keeps every flip exactly objective-preserving, so
///    the population traverses energy plateaus that single-spin Metropolis
///    cannot cross. This is the mechanism behind the recent G-set best-known
///    updates.
/// 3. **Resampling** — replicas are reweighted for the next temperature by
///    `τ_j = exp(-Δβ (E_j - E_min)) / Z · R` with `E_j = -cut_j`; low-energy
///    (high-cut) replicas are preferentially replicated and the population is
///    restored to exactly `population_size`.
/// 4. **Periodic reset** — every `reset_period` steps `β` is reset to
///    `initial_beta` (population-annealing restart) while the global best is
///    preserved, recovering diversity after the population converges.
///
/// All randomness flows through `state.rng` in a fixed sequential order, so
/// seeded runs are bit-reproducible.
///
/// # References
///
/// - Machta, J. "Population annealing with weighted averages." *Phys. Rev. E*
///   82, 026704, 2010.
/// - Augmented PAMC with adaptive control and non-local cluster moves,
///   arXiv:2606.25203; new G63 best-known via PAMC, arXiv:2510.21105.
///
/// # Parameters
///
/// - `population_size` — number of replicas `R` (>= 2)
/// - `initial_beta` — starting inverse temperature (> 0)
/// - `delta_beta` — inverse-temperature increment per step (> 0)
/// - `sweeps_per_step` — Metropolis sweeps per replica per step (>= 1); one
///   sweep proposes one flip per edged vertex
/// - `reset_period` — reset `β` to `initial_beta` every this many steps
///   (`None` = never reset)
/// - `cluster_moves` — enable the non-local cluster (iso-site) move
pub struct PopulationAnnealing {
    stop_condition: StopCondition,
    population_size: usize,
    initial_beta: f64,
    delta_beta: f64,
    sweeps_per_step: usize,
    reset_period: Option<usize>,
    cluster_moves: bool,
    // ---- episode state (reset by `clear`) ----
    population: Vec<MaxCutSolution>,
    beta: f64,
    step: u64,
    // ---- scratch (allocation-free NCM / resampling) ----
    /// Epoch-stamped vertex marker for the cluster move.
    mark_vec: Vec<u32>,
    mark_epoch: u32,
    /// Snapshot of the current zero-gain members during an NCM.
    members: Vec<usize>,
    /// The independent set selected by the current NCM.
    selected: Vec<usize>,
    /// Expected-copy weights during resampling.
    weights: Vec<f64>,
    /// Rebuilt population buffer during resampling.
    next_population: Vec<MaxCutSolution>,
}

impl PopulationAnnealing {
    /// # Panics
    ///
    /// Panics if `population_size < 2`, `initial_beta <= 0`, `delta_beta <= 0`,
    /// or `sweeps_per_step == 0`.
    pub fn new(
        stop_condition: StopCondition,
        population_size: usize,
        initial_beta: f64,
        delta_beta: f64,
        sweeps_per_step: usize,
        reset_period: Option<usize>,
        cluster_moves: bool,
    ) -> Self {
        assert!(population_size >= 2, "population_size must be at least 2");
        assert!(initial_beta > 0.0, "initial_beta must be positive");
        assert!(delta_beta > 0.0, "delta_beta must be positive");
        assert!(sweeps_per_step >= 1, "sweeps_per_step must be at least 1");
        Self {
            stop_condition,
            population_size,
            initial_beta,
            delta_beta,
            sweeps_per_step,
            reset_period,
            cluster_moves,
            population: Vec::new(),
            beta: initial_beta,
            step: 0,
            mark_vec: Vec::new(),
            mark_epoch: 0,
            members: Vec::new(),
            selected: Vec::new(),
            weights: Vec::new(),
            next_population: Vec::new(),
        }
    }

    /// Initializes the replica population with fresh random solutions, threading
    /// `state.rng` so a seeded run stays reproducible.
    fn initialize_population(&mut self, state: &mut SearchState<'_, MaxCut>) {
        self.population.clear();
        self.population.reserve(self.population_size);
        for _ in 0..self.population_size {
            self.population
                .push(state.instance.new_solution(&mut state.rng));
        }
    }

    /// Starts a fresh mark generation for the cluster move, growing `mark_vec`
    /// to `n` if needed. O(1) except on epoch wrap-around.
    fn next_mark_epoch(&mut self, n: usize) {
        if self.mark_vec.len() < n {
            self.mark_vec.resize(n, 0);
        }
        self.mark_epoch = self.mark_epoch.wrapping_add(1);
        if self.mark_epoch == 0 {
            self.mark_vec.fill(0);
            self.mark_epoch = 1;
        }
    }

    /// Sweeps a single replica `sweeps` times at temperature `T = 1/β`.
    /// One sweep proposes one flip per edged vertex. Free of `self` so the
    /// caller can iterate `self.population` mutably while sweeping.
    fn metropolis_sweeps(
        replica: &mut MaxCutSolution,
        rng: &mut SmallRng,
        prob: &MaxCut,
        temperature: f64,
        sweeps: usize,
    ) {
        let n = prob.graph.vertices.len();
        if n == 0 {
            return;
        }
        for _ in 0..sweeps {
            for _ in 0..n {
                let flip = MaxCutFlipNeighbor::random_neighbor(prob, replica, rng);
                if boltzmann_accept(flip.evaluate(), temperature, rng) {
                    // `apply_to_solution` refreshes gain/objective incrementally.
                    let _ = flip.apply_to_solution(prob, replica);
                }
            }
        }
    }

    /// Applies one non-local cluster move to a replica: flips a maximal
    /// independent set of zero-gain vertices. Objective-preserving.
    fn cluster_move(&mut self, replica: &mut MaxCutSolution, rng: &mut SmallRng, prob: &MaxCut) {
        replica.enable_zero_gain_index();
        if replica.zero_gain_count() == 0 {
            return;
        }
        self.next_mark_epoch(prob.graph.len());

        // Snapshot the zero-gain members so we can mutate the replica while
        // building the independent set.
        self.members.clear();
        self.members.extend_from_slice(replica.zero_gain.as_slice());
        let len = self.members.len();
        let start = rng.random_range(0..len);

        self.selected.clear();
        for off in 0..len {
            let v = self.members[(start + off) % len];
            // Ineligible if already selected or adjacent to a selected vertex.
            if self.mark_vec[v] == self.mark_epoch {
                continue;
            }
            self.mark_vec[v] = self.mark_epoch;
            for &(j, _) in prob.graph.iter_on_adjacency(v) {
                self.mark_vec[j] = self.mark_epoch;
            }
            self.selected.push(v);
        }

        #[cfg(debug_assertions)]
        let objective_before = replica.objective;
        for &v in &self.selected {
            debug_assert_eq!(replica.gain[v], 0.0, "independence must keep gains zero");
            let _ = MaxCutFlipNeighbor { i: v, gain: 0.0 }.apply_to_solution(prob, replica);
        }
        #[cfg(debug_assertions)]
        debug_assert_eq!(objective_before, replica.objective);
    }

    /// Resamples the population for the transition `β → β + Δβ`. Each replica
    /// gets `τ_j = exp(-Δβ (E_j - E_min)) / Z · R` expected copies (with
    /// `E_j = -cut_j`, shifted by `E_min = -cut_max` for numerical stability),
    /// then the population is restored to exactly `population_size`.
    fn resample(&mut self, rng: &mut SmallRng) {
        let r = self.population_size;
        let cut_max = self
            .population
            .iter()
            .map(|s| s.objective)
            .fold(f32::NEG_INFINITY, f32::max) as f64;

        // Unnormalized weights w_j = exp(-Δβ (cut_max - cut_j)) ∈ (0, 1].
        self.weights.clear();
        let mut sum = 0.0f64;
        for s in &self.population {
            let w = (-self.delta_beta * (cut_max - s.objective as f64)).exp();
            self.weights.push(w);
            sum += w;
        }
        // Degenerate guard (all-equal or numerical underflow): keep as-is.
        if sum <= 0.0 || !sum.is_finite() {
            return;
        }
        let scale = r as f64 / sum;

        // Build the next population by stochastic-rounded replication.
        self.next_population.clear();
        self.next_population.reserve(r);
        for (j, s) in self.population.iter().enumerate() {
            let tau = self.weights[j] * scale;
            let mut copies = tau.floor() as usize;
            if rng.random::<f64>() < (tau - tau.floor()) {
                copies += 1;
            }
            for _ in 0..copies {
                self.next_population.push(s.clone());
            }
        }

        // Restore the population to exactly R.
        if self.next_population.is_empty() {
            // Extremely unlikely; fall back to keeping the current population.
            return;
        }
        while self.next_population.len() > r {
            // Drop the lowest-cut replica.
            let (worst, _) = self
                .next_population
                .iter()
                .enumerate()
                .min_by(|(_, a), (_, b)| {
                    a.objective
                        .partial_cmp(&b.objective)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .unwrap();
            self.next_population.swap_remove(worst);
        }
        while self.next_population.len() < r {
            // Duplicate the highest-cut replica.
            let best = self
                .next_population
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| {
                    a.objective
                        .partial_cmp(&b.objective)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .map(|(i, _)| i)
                .unwrap();
            let clone = self.next_population[best].clone();
            self.next_population.push(clone);
        }

        std::mem::swap(&mut self.population, &mut self.next_population);
    }

    /// Index of the highest-cut replica.
    fn best_replica_idx(&self) -> usize {
        self.population
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| {
                if a.is_better_than(b) {
                    std::cmp::Ordering::Greater
                } else {
                    std::cmp::Ordering::Less
                }
            })
            .map(|(i, _)| i)
            .unwrap_or(0)
    }
}

impl Heuristic<MaxCut> for PopulationAnnealing {
    fn clear(&mut self) {
        self.population.clear();
        self.beta = self.initial_beta;
        self.step = 0;
    }

    fn run_once<'a>(&mut self, state: &mut SearchState<'a, MaxCut>) -> Result<(), OptError> {
        if self.population.is_empty() {
            self.initialize_population(state);
        }

        // 1. Metropolis sweeps on every replica at the current β.
        //    `state.instance` is a shared &-ref, so borrowing it alongside
        //    `&mut state.rng` and the population is fine.
        let prob: &MaxCut = state.instance;
        let temperature = 1.0 / self.beta;
        let sweeps = self.sweeps_per_step;
        for replica in &mut self.population {
            Self::metropolis_sweeps(replica, &mut state.rng, prob, temperature, sweeps);
        }

        // 2. Non-local cluster moves (objective-preserving plateau traversal).
        if self.cluster_moves {
            for idx in 0..self.population.len() {
                let mut replica = std::mem::replace(
                    &mut self.population[idx],
                    // cheap placeholder to satisfy the borrow checker
                    MaxCutSolution::new_from_parts(Vec::new(), Vec::new(), 0.0),
                );
                self.cluster_move(&mut replica, &mut state.rng, prob);
                self.population[idx] = replica;
            }
        }

        // 3. Resample for the next temperature, then advance β.
        self.resample(&mut state.rng);
        self.beta += self.delta_beta;
        self.step += 1;
        if let Some(period) = self.reset_period
            && period > 0
            && self.step.is_multiple_of(period as u64)
        {
            self.beta = self.initial_beta;
        }

        // 4. Track the global best. Advance the iteration counter by the sweep
        //    budget so time-to-best and the anytime trajectory are meaningful.
        state.iteration += self.sweeps_per_step as u64;
        let best_idx = self.best_replica_idx();
        state.solution = self.population[best_idx].clone();
        state.update_best();

        Ok(())
    }

    fn stop_condition(&self) -> &StopCondition {
        &self.stop_condition
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::heuristic::Heuristic;
    use crate::problem::MaxCut;
    use crate::search_state::SearchState;

    /// Same toroidal instance used by the BLS tests (degree 4, plateau-rich).
    fn small_instance() -> MaxCut {
        let n = 30usize;
        let mut edges = Vec::new();
        for i in 0..n {
            edges.push((i, (i + 1) % n, 1.0));
            edges.push((i, (i + 2) % n, 1.0));
        }
        MaxCut::from_edges(edges)
    }

    fn new_pa(stop: StopCondition) -> PopulationAnnealing {
        PopulationAnnealing::new(stop, 16, 0.1, 0.05, 5, Some(20), true)
    }

    #[test]
    fn pa_runs_without_error_and_improves() {
        let mc = small_instance();
        for seed in 0..10 {
            let mut state = SearchState::new_with_seed(&mc, seed);
            let mut pa = new_pa(StopCondition::iterations(2_000));
            pa.run(&mut state).expect("PA must not error");
            assert!(
                state.best_solution.objective > 0.0,
                "PA should find a positive cut, got {}",
                state.best_solution.objective
            );
        }
    }

    #[test]
    fn pa_seeded_runs_are_deterministic() {
        let mc = small_instance();
        let run = || {
            let mut state = SearchState::new_with_seed(&mc, 42);
            let mut pa = new_pa(StopCondition::iterations(1_500));
            pa.run(&mut state).unwrap();
            (
                state.best_solution.objective,
                state.best_iteration,
                state.best_solution.x.clone(),
            )
        };
        assert_eq!(run(), run());
    }

    #[test]
    fn cluster_move_preserves_objective() {
        let mc = small_instance();
        let mut pa = new_pa(StopCondition::iterations(1));
        let mut rng = {
            let state = SearchState::new_with_seed(&mc, 3);
            state.rng
        };
        // Build a replica and descend it a little via random flips so a
        // non-trivial zero-gain set exists, then check NCM invariance.
        let mut replica = mc.new_solution(&mut rng);
        replica.enable_zero_gain_index();
        let mut moved = false;
        for _ in 0..50 {
            if replica.zero_gain_count() == 0 {
                break;
            }
            let objective_before = replica.objective;
            let x_before = replica.x.clone();
            pa.cluster_move(&mut replica, &mut rng, &mc);
            assert_eq!(
                replica.objective, objective_before,
                "cluster move must preserve the objective"
            );
            moved |= replica.x != x_before;
        }
        assert!(
            moved,
            "cluster move should change the assignment at least once"
        );
    }

    #[test]
    fn resample_keeps_population_size_and_favors_high_cut() {
        let mc = small_instance();
        let mut pa =
            PopulationAnnealing::new(StopCondition::iterations(1), 16, 0.1, 0.5, 5, None, false);
        let mut state = SearchState::new_with_seed(&mc, 7);
        pa.initialize_population(&mut state);
        // Give replicas a spread of objectives by descending some of them.
        let prob: &MaxCut = state.instance;
        for (k, replica) in pa.population.iter_mut().enumerate() {
            for _ in 0..(k * 3) {
                let f = MaxCutFlipNeighbor::random_neighbor(prob, replica, &mut state.rng);
                if f.gain > 0.0 {
                    let _ = f.apply_to_solution(prob, replica);
                }
            }
        }
        let avg_before: f64 = pa
            .population
            .iter()
            .map(|s| s.objective as f64)
            .sum::<f64>()
            / pa.population.len() as f64;
        pa.resample(&mut state.rng);
        assert_eq!(pa.population.len(), 16, "resampling must restore R");
        let avg_after: f64 = pa
            .population
            .iter()
            .map(|s| s.objective as f64)
            .sum::<f64>()
            / pa.population.len() as f64;
        assert!(
            avg_after >= avg_before,
            "resampling must not lower the mean cut (before {avg_before}, after {avg_after})"
        );
    }

    #[test]
    fn clear_resets_population() {
        let mc = small_instance();
        let mut state = SearchState::new_with_seed(&mc, 1);
        let mut pa = new_pa(StopCondition::iterations(500));
        pa.run(&mut state).unwrap();
        assert!(!pa.population.is_empty());
        pa.clear();
        assert!(pa.population.is_empty());
        assert_eq!(pa.beta, pa.initial_beta);
        assert_eq!(pa.step, 0);
    }
}
