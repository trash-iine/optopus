use crate::building_blocks::search::metropolis_sweeps;
use crate::error::OptError;
use crate::heuristic::{Heuristic, StopCondition};
use crate::search_state::SearchState;
use crate::trait_defs::{Evaluate, MoveToNeighbor, ProblemTrait, rank_cmp};
use rand::Rng;
use std::marker::PhantomData;

/// Parallel tempering (replica exchange Monte Carlo), for any problem whose
/// solution implements [`Evaluate`] and whose move `N` implements it too.
///
/// `num_replicas` replicas sit on a fixed ladder of inverse temperatures,
/// geometric from `beta_min` to `beta_max`. One
/// [`run_once`](Heuristic::run_once) is one round.
///
/// 1. Metropolis sweeps, every replica is swept `sweeps_per_exchange` times
///    at its own temperature, by the same sweep population annealing uses.
/// 2. Exchange, the replicas at neighbouring temperatures `k` and `k + 1`
///    trade places with probability `min(1, exp((β_k − β_{k+1})(E_k − E_{k+1})))`,
///    over the even pairs on one round and the odd pairs on the next. A
///    configuration that a hot replica finds in a good basin walks down the
///    ladder to where it is refined, and one stuck at the cold end walks up to
///    where it can leave.
///
/// Nothing cools. The ladder stays as built, which is the difference from
/// [`PopulationAnnealing`](crate::heuristic::PopulationAnnealing), whose
/// replicas share one temperature that rises and are resampled toward the
/// good ones. The energy is read through [`Evaluate`] on the solution, so the
/// direction is applied before the exchange arithmetic.
///
/// All randomness flows through `state.rng` in a fixed order, so seeded runs
/// are bit-reproducible.
///
/// # References
///
/// - Hukushima, K. and Nemoto, K. "Exchange Monte Carlo method and
///   application to spin glass simulations." *Journal of the Physical Society
///   of Japan*, 65(6), 1604-1608, 1996.
/// - Wang, W., Machta, J. and Katzgraber, H. G. "Comparing Monte Carlo
///   methods for finding ground states of Ising spin glasses: Population
///   annealing, simulated annealing, and parallel tempering." *Physical
///   Review E*, 92, 013303, 2015 (arXiv:1412.2104).
pub struct ParallelTempering<P: ProblemTrait, N>
where
    P::Solution: Evaluate,
    N: MoveToNeighbor<P> + Evaluate,
{
    stop_condition: StopCondition,
    betas: Vec<f64>,
    sweeps_per_exchange: usize,
    sweep_length: Option<usize>,
    // ---- episode state (reset by `clear`) ----
    /// `replicas[k]` is the configuration at `betas[k]`.
    replicas: Vec<P::Solution>,
    /// Whether this round tries the odd pairs `(1, 2), (3, 4), ...`.
    odd_round: bool,
    proposals_per_sweep: usize,
    exchanges_tried: u64,
    exchanges_accepted: u64,
    _neighbor: PhantomData<N>,
}

impl<P: ProblemTrait, N> ParallelTempering<P, N>
where
    P::Solution: Evaluate,
    N: MoveToNeighbor<P> + Evaluate,
{
    /// # Panics
    ///
    /// Panics if `num_replicas < 2`, unless `0 < beta_min < beta_max`, or if
    /// `sweeps_per_exchange == 0`.
    pub fn new(
        stop_condition: StopCondition,
        num_replicas: usize,
        beta_min: f64,
        beta_max: f64,
        sweeps_per_exchange: usize,
    ) -> Self {
        assert!(num_replicas >= 2, "num_replicas must be at least 2");
        assert!(
            beta_min > 0.0 && beta_min < beta_max,
            "need 0 < beta_min < beta_max"
        );
        assert!(
            sweeps_per_exchange >= 1,
            "sweeps_per_exchange must be at least 1"
        );
        let ratio = (beta_max / beta_min).powf(1.0 / (num_replicas - 1) as f64);
        let betas = (0..num_replicas)
            .map(|k| beta_min * ratio.powi(k as i32))
            .collect();
        Self {
            stop_condition,
            betas,
            sweeps_per_exchange,
            sweep_length: None,
            replicas: Vec::new(),
            odd_round: false,
            proposals_per_sweep: 0,
            exchanges_tried: 0,
            exchanges_accepted: 0,
            _neighbor: PhantomData,
        }
    }

    /// Sets how many moves one sweep proposes, instead of counting the
    /// neighborhood once per episode as
    /// [`PopulationAnnealing::with_sweep_length`](crate::heuristic::PopulationAnnealing::with_sweep_length)
    /// explains.
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

    /// The inverse temperatures of the ladder, hottest first.
    pub fn betas(&self) -> &[f64] {
        &self.betas
    }

    /// The share of exchanges accepted so far in this run, `None` before the
    /// first. A ladder whose neighbours rarely swap is too sparse to carry a
    /// configuration from one end to the other.
    pub fn exchange_acceptance(&self) -> Option<f64> {
        (self.exchanges_tried > 0)
            .then(|| self.exchanges_accepted as f64 / self.exchanges_tried as f64)
    }

    fn initialize(&mut self, state: &mut SearchState<'_, P>) {
        self.replicas.clear();
        for _ in 0..self.betas.len() {
            self.replicas
                .push(state.instance.new_solution(&mut state.rng));
        }
        self.proposals_per_sweep = self
            .sweep_length
            .unwrap_or_else(|| N::iter(state.instance, &self.replicas[0]).count());
    }

    fn exchange(&mut self, rng: &mut rand::rngs::SmallRng) {
        let first = usize::from(self.odd_round);
        for k in (first..self.betas.len() - 1).step_by(2) {
            let e_hot = self.replicas[k].evaluate().minimized();
            let e_cold = self.replicas[k + 1].evaluate().minimized();
            let delta = (self.betas[k] - self.betas[k + 1]) * (e_hot - e_cold);
            self.exchanges_tried += 1;
            if delta >= 0.0 || rng.random::<f64>() < delta.exp() {
                self.replicas.swap(k, k + 1);
                self.exchanges_accepted += 1;
            }
        }
        self.odd_round = !self.odd_round;
    }
}

impl<P: ProblemTrait, N> Heuristic<P> for ParallelTempering<P, N>
where
    P::Solution: Evaluate,
    N: MoveToNeighbor<P> + Evaluate,
{
    fn clear(&mut self) {
        self.replicas.clear();
        self.odd_round = false;
        self.exchanges_tried = 0;
        self.exchanges_accepted = 0;
    }

    fn stop_condition(&self) -> &StopCondition {
        &self.stop_condition
    }

    fn run_once<'a>(&mut self, state: &mut SearchState<'a, P>) -> Result<(), OptError> {
        if self.replicas.is_empty() {
            self.initialize(state);
        }
        let prob: &P = state.instance;
        for (replica, beta) in self.replicas.iter_mut().zip(&self.betas) {
            metropolis_sweeps::<P, N>(
                replica,
                &mut state.rng,
                prob,
                1.0 / beta,
                self.sweeps_per_exchange,
                self.proposals_per_sweep,
            );
        }
        self.exchange(&mut state.rng);

        // As population annealing does, the counter advances by the sweeps
        // and the incumbent is the best replica of the round.
        state.iteration += self.sweeps_per_exchange as u64;
        let best = self
            .replicas
            .iter()
            .max_by(|a, b| rank_cmp(*a, *b))
            .expect("at least two replicas")
            .clone();
        state.solution = best;
        state.update_best();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::problem::max_cut::test_fixtures::small_instance;
    use crate::problem::{MaxCut, MaxCutFlipNeighbor, Qubo, QuboFlipNeighbor};

    type PtForMaxCut = ParallelTempering<MaxCut, MaxCutFlipNeighbor>;

    #[test]
    fn the_ladder_is_geometric_from_hot_to_cold() {
        let pt = PtForMaxCut::new(StopCondition::iterations(1), 5, 0.1, 1.6, 1);
        let b = pt.betas();
        assert_eq!(b.len(), 5);
        assert!((b[0] - 0.1).abs() < 1e-12 && (b[4] - 1.6).abs() < 1e-12);
        for w in b.windows(3) {
            assert!((w[1] / w[0] - w[2] / w[1]).abs() < 1e-9);
        }
    }

    #[test]
    fn improves_and_is_reproducible_under_seed() {
        let mc = small_instance();
        let run = || {
            let mut state = SearchState::new_with_seed(&mc, 7);
            let initial = state.best_solution.objective;
            let mut pt = PtForMaxCut::new(StopCondition::iterations(200), 6, 0.2, 4.0, 2);
            pt.run(&mut state).unwrap();
            assert!(state.best_solution.objective >= initial);
            let rate = pt.exchange_acceptance().unwrap();
            assert!((0.0..=1.0).contains(&rate));
            (state.best_solution.x.clone(), state.best_iteration)
        };
        assert_eq!(run(), run());
    }

    /// A cold replica holding a worse configuration than its hot neighbour
    /// always swaps, since the exchange then lowers the ladder's energy where
    /// it is weighted most.
    #[test]
    fn a_better_configuration_always_moves_down_the_ladder() {
        let qubo = Qubo::from_entries([(0, 0, -5), (1, 1, -5), (0, 1, 1)]);
        let mut pt = ParallelTempering::<Qubo, QuboFlipNeighbor>::new(
            StopCondition::iterations(1),
            2,
            0.5,
            2.0,
            1,
        );
        let mut state = SearchState::new_with_seed(&qubo, 1);
        pt.initialize(&mut state);
        pt.replicas[0] = crate::problem::QuboSolution::new_from_assignment(&qubo, vec![true, true]);
        pt.replicas[1] =
            crate::problem::QuboSolution::new_from_assignment(&qubo, vec![false, false]);
        pt.exchange(&mut state.rng);
        assert_eq!(pt.replicas[1].x, vec![true, true]);
        assert_eq!(pt.exchange_acceptance(), Some(1.0));
    }
}
