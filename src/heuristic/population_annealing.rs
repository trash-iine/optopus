use crate::error::OptError;
use crate::heuristic::simulated_annealing::boltzmann_accept;
use crate::heuristic::{Heuristic, StopCondition};
use crate::search_state::SearchState;
use crate::trait_defs::{Evaluate, MoveToNeighbor, ProblemTrait, Rankable};
use rand::Rng;
use rand::rngs::SmallRng;
use std::marker::PhantomData;

/// Population Annealing Monte Carlo (PAMC), for any problem whose solution
/// implements [`Evaluate`] and whose move `N` implements it too.
///
/// PAMC maintains a population of `population_size` replicas and cools a shared
/// inverse temperature `β` from `initial_beta` upward in steps of `delta_beta`.
/// One [`run_once`](Heuristic::run_once) is one annealing step, and follows the
/// canonical algorithm: "in each annealing step the population is resampled and
/// then `N_S` sweeps of the Metropolis algorithm are carried out on each
/// replica".
///
/// 1. Resampling, replica `j` is given `τ_j = exp(-Δβ (E_j - E_min)) / Z · R`
///    expected copies (shifted by `E_min` for numerical stability), so
///    low-energy replicas are preferentially replicated. The population is then
///    restored to exactly `population_size`.
/// 2. `β` advances by `delta_beta`, and every `reset_period` steps it returns
///    to `initial_beta` while the global best is preserved, recovering
///    diversity after the population converges.
/// 3. Metropolis sweeps, every replica is swept `sweeps_per_step` times at the
///    new `β`; a proposed move is accepted with probability
///    `min(1, exp(-β · ΔE))` (reusing [`boltzmann_accept`]).
///
/// There is no initial sweep before the first resampling. The paper starts from
/// a random population at `β = 0`, where random *is* the equilibrium
/// distribution, and steps straight into the first resampling.
///
/// The energy `E_j` is read through [`Evaluate`] on the solution, which is what
/// makes the difference `E_j - E_min` meaningful: population annealing computes
/// *with* the objective rather than only comparing solutions, so the direction
/// has to be applied before the arithmetic.
///
/// All randomness flows through `state.rng` in a fixed sequential order, so
/// seeded runs are bit-reproducible.
///
/// # References
///
/// - Wang, Machta, Katzgraber. "Population annealing: Theory and application in
///   spin glasses." Phys. Rev. E 92, 063307, 2015 (arXiv:1508.05647), the
///   algorithm implemented here.
/// - Machta, J. "Population annealing with weighted averages: A Monte Carlo
///   method for rough free-energy landscapes." Phys. Rev. E 82, 026704, 2010.
///
/// # Parameters
///
/// - `population_size`, number of replicas `R` (>= 2)
/// - `initial_beta`, starting inverse temperature (> 0)
/// - `delta_beta`, inverse-temperature increment per step (> 0)
/// - `sweeps_per_step`, Metropolis sweeps per replica per step (>= 1)
/// - `reset_period`, reset `β` to `initial_beta` every this many steps
///   (`None` = never reset)
pub struct PopulationAnnealing<P: ProblemTrait, N>
where
    P::Solution: Evaluate,
    N: MoveToNeighbor<P> + Evaluate,
{
    stop_condition: StopCondition,
    population_size: usize,
    initial_beta: f64,
    delta_beta: f64,
    sweeps_per_step: usize,
    /// Proposals in one sweep. `None` means "count the neighborhood once", see
    /// [`with_sweep_length`](Self::with_sweep_length).
    sweep_length: Option<usize>,
    reset_period: Option<usize>,
    // ---- episode state (reset by `clear`) ----
    population: Vec<P::Solution>,
    beta: f64,
    step: u64,
    /// The resolved sweep length, measured once per episode.
    proposals_per_sweep: usize,
    // ---- scratch (allocation-free resampling) ----
    /// Expected-copy weights during resampling.
    weights: Vec<f64>,
    /// Rebuilt population buffer during resampling.
    next_population: Vec<P::Solution>,
    _neighbor: PhantomData<N>,
}

impl<P: ProblemTrait, N> PopulationAnnealing<P, N>
where
    P::Solution: Evaluate,
    N: MoveToNeighbor<P> + Evaluate,
{
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
            sweep_length: None,
            reset_period,
            population: Vec::new(),
            beta: initial_beta,
            step: 0,
            proposals_per_sweep: 0,
            weights: Vec::new(),
            next_population: Vec::new(),
            _neighbor: PhantomData,
        }
    }

    /// Sets how many moves one sweep proposes, instead of measuring it.
    ///
    /// A sweep is one pass over the system, so by default the length is the
    /// size of `N`'s neighborhood, counted once per episode from
    /// [`MoveToNeighbor::iter`]. That is O(n) for a single-variable move and is
    /// what the physics literature calls a sweep, but it is O(n²) for a
    /// pairwise move such as 2-opt, and the count builds every move only to
    /// discard it. Pin the length here when that matters.
    ///
    /// # Panics
    ///
    /// Panics if `length == 0`.
    #[must_use]
    pub fn with_sweep_length(mut self, length: usize) -> Self {
        assert!(length >= 1, "sweep_length must be at least 1");
        self.sweep_length = Some(length);
        self
    }

    /// Initializes the replica population with fresh random solutions, threading
    /// `state.rng` so a seeded run stays reproducible.
    fn initialize_population(&mut self, state: &mut SearchState<'_, P>) {
        self.population.clear();
        self.population.reserve(self.population_size);
        for _ in 0..self.population_size {
            self.population
                .push(state.instance.new_solution(&mut state.rng));
        }
        self.proposals_per_sweep = self.sweep_length.unwrap_or_else(|| {
            // Measured once per episode: the neighborhood does not change size
            // while the search runs.
            N::iter(state.instance, &self.population[0]).count()
        });
    }

    /// Sweeps a single replica `sweeps` times at temperature `T = 1/β`.
    /// One sweep proposes `proposals` moves. Free of `self` so the caller can
    /// iterate `self.population` mutably while sweeping.
    fn metropolis_sweeps(
        replica: &mut P::Solution,
        rng: &mut SmallRng,
        prob: &P,
        temperature: f64,
        sweeps: usize,
        proposals: usize,
    ) {
        if proposals == 0 {
            return;
        }
        for _ in 0..sweeps {
            for _ in 0..proposals {
                let Some(mv) = N::random_neighbor(prob, replica, rng) else {
                    continue;
                };
                if boltzmann_accept(mv.evaluate(), temperature, rng) {
                    // `apply_to_solution` refreshes gain/objective incrementally.
                    let _ = mv.apply_to_solution(prob, replica);
                }
            }
        }
    }

    /// Resamples the population for the transition `β → β + Δβ`. Each replica
    /// gets `τ_j = exp(-Δβ (E_j - E_min)) / Z · R` expected copies (shifted by
    /// `E_min` for numerical stability), then the population is restored to
    /// exactly `population_size`.
    fn resample(&mut self, rng: &mut SmallRng) {
        let r = self.population_size;
        let e_min = self
            .population
            .iter()
            .map(|s| s.evaluate().minimized())
            .fold(f64::INFINITY, f64::min);

        // Unnormalized weights w_j = exp(-Δβ (E_j - E_min)) ∈ (0, 1].
        self.weights.clear();
        let mut sum = 0.0f64;
        for s in &self.population {
            let w = (-self.delta_beta * (s.evaluate().minimized() - e_min)).exp();
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
        // Both comparisons are reversed so the tie goes the way it has to.
        // `min_by` keeps the first of an equal run and `max_by` the last, and
        // which replica a tie drops or copies is part of the trajectory.
        let worse_first = |a: &P::Solution, b: &P::Solution| {
            b.evaluate()
                .minimized()
                .partial_cmp(&a.evaluate().minimized())
                .unwrap_or(std::cmp::Ordering::Equal)
        };
        while self.next_population.len() > r {
            // Drop the highest-energy replica, the first of them on a tie.
            let (worst, _) = self
                .next_population
                .iter()
                .enumerate()
                .min_by(|(_, a), (_, b)| worse_first(a, b))
                .unwrap();
            self.next_population.swap_remove(worst);
        }
        while self.next_population.len() < r {
            // Duplicate the lowest-energy replica, the last of them on a tie.
            let best = self
                .next_population
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| worse_first(a, b))
                .map(|(i, _)| i)
                .unwrap();
            let clone = self.next_population[best].clone();
            self.next_population.push(clone);
        }

        std::mem::swap(&mut self.population, &mut self.next_population);
    }

    /// Index of the best replica.
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

impl<P: ProblemTrait, N> Heuristic<P> for PopulationAnnealing<P, N>
where
    P::Solution: Evaluate,
    N: MoveToNeighbor<P> + Evaluate,
{
    fn clear(&mut self) {
        self.population.clear();
        self.beta = self.initial_beta;
        self.step = 0;
    }

    fn run_once<'a>(&mut self, state: &mut SearchState<'a, P>) -> Result<(), OptError> {
        if self.population.is_empty() {
            self.initialize_population(state);
        }

        // 1. Resample for the next temperature. This comes first: the paper's
        //    annealing step is "the population is resampled and then N_S sweeps
        //    of the Metropolis algorithm are carried out on each replica".
        self.resample(&mut state.rng);

        // 2. Advance β, and restart the anneal every `reset_period` steps. The
        //    reset is a restart rather than an annealing step, so the weights
        //    above always use `delta_beta` and never the jump a reset makes.
        self.beta += self.delta_beta;
        self.step += 1;
        if let Some(period) = self.reset_period
            && period > 0
            && self.step.is_multiple_of(period as u64)
        {
            self.beta = self.initial_beta;
        }

        // 3. Metropolis sweeps on every replica at the new β. `state.instance`
        //    is a shared &-ref, so borrowing it alongside `&mut state.rng` and
        //    the population is fine.
        let prob: &P = state.instance;
        let temperature = 1.0 / self.beta;
        let sweeps = self.sweeps_per_step;
        let proposals = self.proposals_per_sweep;
        for replica in &mut self.population {
            Self::metropolis_sweeps(
                replica,
                &mut state.rng,
                prob,
                temperature,
                sweeps,
                proposals,
            );
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
    use crate::problem::max_cut::MaxCutFlipNeighbor;
    use crate::problem::qubo::QuboFlipNeighbor;
    use crate::problem::tsp_2d::{TspTwoOptNeighbor, TspWithCoordinates};
    use crate::problem::{MaxCut, Qubo};
    use crate::search_state::SearchState;

    type PaForMaxCut = PopulationAnnealing<MaxCut, MaxCutFlipNeighbor>;

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

    fn new_pa(stop: StopCondition) -> PaForMaxCut {
        PopulationAnnealing::new(stop, 16, 0.1, 0.05, 5, Some(20))
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

    /// One sweep is one pass over the neighborhood, measured once per episode.
    #[test]
    fn sweep_length_defaults_to_the_neighborhood_size() {
        let mc = small_instance();
        let mut state = SearchState::new_with_seed(&mc, 2);
        let mut pa = new_pa(StopCondition::iterations(1));
        pa.initialize_population(&mut state);
        assert_eq!(pa.proposals_per_sweep, 30);

        let mut pinned = new_pa(StopCondition::iterations(1)).with_sweep_length(7);
        pinned.initialize_population(&mut state);
        assert_eq!(pinned.proposals_per_sweep, 7);
    }

    #[test]
    fn resample_keeps_population_size_and_favors_low_energy() {
        let mc = small_instance();
        let mut pa: PaForMaxCut =
            PopulationAnnealing::new(StopCondition::iterations(1), 16, 0.1, 0.5, 5, None);
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
            .map(|s| s.evaluate().minimized())
            .sum::<f64>()
            / pa.population.len() as f64;
        pa.resample(&mut state.rng);
        assert_eq!(pa.population.len(), 16, "resampling must restore R");
        let avg_after: f64 = pa
            .population
            .iter()
            .map(|s| s.evaluate().minimized())
            .sum::<f64>()
            / pa.population.len() as f64;
        assert!(
            avg_after <= avg_before,
            "resampling must not raise the mean energy (before {avg_before}, after {avg_after})"
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

    /// QUBO minimizes where MaxCut maximizes, so this is also what checks the
    /// direction `Evaluate` carries is applied rather than assumed.
    #[test]
    fn pa_runs_on_qubo() {
        use rand::SeedableRng;
        let mut rng = SmallRng::seed_from_u64(5);
        let mut q = Qubo::new();
        for i in 0..30usize {
            q.set_q(i, i, rng.random_range(-10..10));
            for j in (i + 1)..30 {
                if rng.random_bool(0.2) {
                    q.set_q(i, j, rng.random_range(-10..10));
                }
            }
        }
        let mut state = SearchState::new_with_seed(&q, 11);
        let initial = state.solution.objective;
        let mut pa: PopulationAnnealing<Qubo, QuboFlipNeighbor> =
            PopulationAnnealing::new(StopCondition::iterations(1_000), 16, 0.1, 0.05, 5, Some(20));
        pa.run(&mut state).expect("PA must run on QUBO");
        assert!(
            state.best_solution.objective <= initial,
            "QUBO minimizes, so the best objective must not rise"
        );
    }

    /// Population annealing reads an energy and a move, never a variable. TSP
    /// is not a tuned or measured combination; what this states is that the
    /// search compiles and runs with nothing binary in reach.
    #[test]
    fn pa_runs_on_a_non_binary_problem() {
        use rand::SeedableRng;
        let mut rng = SmallRng::seed_from_u64(9);
        let coordinates: Vec<(f64, f64)> = (0..40)
            .map(|_| (rng.random_range(0.0..100.0), rng.random_range(0.0..100.0)))
            .collect();
        let tsp = TspWithCoordinates::new("pa-non-binary".to_string(), coordinates);
        let mut state = SearchState::new_with_seed(&tsp, 4);
        let initial = state.solution.objective;
        // 2-opt has an O(n²) neighborhood, so the sweep length is pinned rather
        // than counted.
        let mut pa: PopulationAnnealing<TspWithCoordinates, TspTwoOptNeighbor> =
            PopulationAnnealing::new(StopCondition::iterations(2_000), 8, 0.05, 0.02, 5, Some(50))
                .with_sweep_length(40);
        pa.run(&mut state).expect("PA must run on TSP");
        assert!(
            state.best_solution.objective <= initial,
            "TSP minimizes, so the best tour length must not rise"
        );
    }
}
