//! Reinforcement learning heuristic for combinatorial optimization.
//!
//! [`ReinforcementLearningSearch`] uses a learned softmax policy over move features to select
//! which neighborhood move to apply at each step. The policy is trained online
//! via the REINFORCE algorithm with baseline subtraction.
//!
//! # Example
//!
//! ```rust,ignore
//! use optopus::prelude::*;
//!
//! let rl = ReinforcementLearningSearch::<MaxCutFlipNeighbor>::new(
//!     StopCondition::failed_updates(1000),
//!     0.01,   // learning_rate
//!     1.0,    // softmax_temperature
//!     RewardShaping::Normalized,
//!     Some(64), // max_candidates: sample 64 moves per step instead of all
//! );
//! // Wrap in Restart for multi-episode learning
//! let solver = Restart::new(
//!     StopCondition::iterations(1_000_000),
//!     Box::new(rl),
//!     StopCondition::failed_updates(10_000),
//! );
//! ```

pub mod bandit;
pub mod feature;
pub mod policy;

use feature::{EPSILON, NUM_FEATURES, StepStatsAccumulator, extract_features};
use policy::LinearPolicy;

use super::{Heuristic, StopCondition};
use crate::error::OptError;
use crate::search_state::{Evaluate, MoveToNeighbor, ProblemTrait, SearchState};
use rand::Rng;

/// Reward shaping strategy for the RL agent.
#[derive(Clone, Debug)]
pub enum RewardShaping {
    /// Raw gain: `reward = -minimized`.
    Raw,
    /// Normalized by the step's max absolute gain: `reward = -worsening / max_abs`.
    Normalized,
    /// Binary signal: `1.0` when a new best is found, `0.0` otherwise.
    BestImprovement,
}

/// Reinforcement learning heuristic that learns a move selection policy online.
///
/// At each step, all (or a subsample of) neighborhood moves are scored by a linear
/// policy over hand-crafted features. A move is sampled from the resulting softmax
/// distribution and applied. The policy is updated immediately via single-step
/// REINFORCE with baseline subtraction.
///
/// Key property: `clear()` resets per-episode state but preserves the learned
/// weights, so the policy improves across episodes when used inside
/// [`super::Restart`] or [`super::Iterated`].
///
/// # References
///
/// - Williams, R. J. "Simple Statistical Gradient-Following Algorithms for
///   Connectionist Reinforcement Learning." Machine Learning, 8(3-4), 229-256, 1992.
///   [DOI](https://doi.org/10.1007/BF00992696)
pub struct ReinforcementLearningSearch<N> {
    pub stop_condition: StopCondition,
    pub policy: LinearPolicy,
    pub learning_rate: f64,
    pub softmax_temperature: f64,
    pub reward_shaping: RewardShaping,
    /// When set, each step reservoir-samples this many moves from the lazy
    /// neighborhood iterator before evaluating them, so per-step evaluation
    /// and feature cost is O(max_candidates) instead of O(neighborhood).
    /// Step statistics (and therefore the neighborhood-level features) are
    /// computed over the sample only.
    pub max_candidates: Option<usize>,
    _neighbor: std::marker::PhantomData<N>,
    baseline: f64,
    baseline_count: u64,
    // Applied-move ledger for the `improvement_ratio` feature: the summed
    // worsening of applied moves telescopes to the objective delta since the
    // episode start, without needing an objective accessor on the solution.
    cum_worsening: f64,
    cum_abs_worsening: f64,
    // Pre-allocated buffers (reused across iterations)
    buf_moves: Vec<(N, f64)>,
    buf_scores: Vec<f64>,
    buf_features: Vec<[f64; NUM_FEATURES]>,
}

impl<N> ReinforcementLearningSearch<N> {
    /// # Panics
    ///
    /// Panics if `learning_rate` is negative, `softmax_temperature` is not
    /// strictly positive, or `max_candidates` is `Some(0)`.
    pub fn new(
        stop_condition: StopCondition,
        learning_rate: f64,
        softmax_temperature: f64,
        reward_shaping: RewardShaping,
        max_candidates: Option<usize>,
    ) -> Self {
        assert!(learning_rate >= 0.0, "learning_rate must be non-negative");
        assert!(
            softmax_temperature > 0.0,
            "softmax_temperature must be strictly positive"
        );
        assert!(
            max_candidates != Some(0),
            "max_candidates must be at least 1 when set"
        );
        Self {
            stop_condition,
            policy: LinearPolicy::new(),
            learning_rate,
            softmax_temperature,
            reward_shaping,
            max_candidates,
            _neighbor: std::marker::PhantomData,
            baseline: 0.0,
            baseline_count: 0,
            cum_worsening: 0.0,
            cum_abs_worsening: 0.0,
            buf_moves: Vec::new(),
            buf_scores: Vec::new(),
            buf_features: Vec::new(),
        }
    }

    pub fn with_policy_weights(mut self, weights: [f64; NUM_FEATURES]) -> Self {
        self.policy = LinearPolicy::with_weights(weights);
        self
    }

    fn max_iteration_budget(&self) -> f64 {
        self.stop_condition.max_iteration.unwrap_or(1_000_000) as f64
    }
}

/// Sample an index from a categorical distribution defined by `probs`.
fn sample_categorical(probs: &[f64], rng: &mut impl Rng) -> usize {
    let r: f64 = rng.random();
    let mut cumulative = 0.0;
    for (i, &p) in probs.iter().enumerate() {
        cumulative += p;
        if r < cumulative {
            return i;
        }
    }
    probs.len() - 1
}

/// Compute softmax probabilities in-place with numerical stability (log-sum-exp trick).
fn softmax_in_place(scores: &mut [f64]) {
    let max_score = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let mut sum_exp = 0.0;
    for s in scores.iter_mut() {
        *s = (*s - max_score).exp();
        sum_exp += *s;
    }
    for s in scores.iter_mut() {
        *s /= sum_exp;
    }
}

impl<P, N> Heuristic<P> for ReinforcementLearningSearch<N>
where
    P: ProblemTrait,
    N: MoveToNeighbor<P> + Evaluate + Clone,
{
    fn clear(&mut self) {
        self.cum_worsening = 0.0;
        self.cum_abs_worsening = 0.0;
        // Policy weights and baseline are intentionally preserved across episodes.
    }

    fn stop_condition(&self) -> &StopCondition {
        &self.stop_condition
    }

    /// An empty neighborhood only advances the iteration counter; the stop
    /// condition eventually terminates the run.
    fn run_once<'a>(&mut self, state: &mut SearchState<'a, P>) -> Result<(), OptError> {
        // 1. Collect candidate moves. With `max_candidates` set, reservoir-
        //    sample from the lazy iterator (Algorithm R) so that only the
        //    sampled moves are ever evaluated. With no cap the reservoir never
        //    fills, so the whole neighborhood is kept and the RNG is never
        //    consulted.
        self.buf_moves.clear();
        let mut acc = StepStatsAccumulator::new();
        let k = self.max_candidates.unwrap_or(usize::MAX);
        for (i, m) in N::iter(state.instance, &state.solution).enumerate() {
            if i < k {
                self.buf_moves.push((m, 0.0));
            } else {
                let j = state.rng.random_range(0..=i);
                if j < k {
                    self.buf_moves[j] = (m, 0.0);
                }
            }
        }
        for entry in self.buf_moves.iter_mut() {
            entry.1 = entry.0.evaluate().minimized();
            acc.push(entry.1);
        }

        if self.buf_moves.is_empty() {
            state.progress_iteration();
            return Ok(());
        }

        let improvement_ratio = -self.cum_worsening / self.cum_abs_worsening.max(EPSILON);
        let ctx = acc.finalize(
            state.iteration,
            state.start_iteration,
            state.best_iteration,
            self.max_iteration_budget(),
            improvement_ratio,
        );

        // 2. Score moves with approximate rank (O(n) instead of O(n log n) sort),
        //    keeping each move's feature vector for the exact gradient below.
        let inv_temp = 1.0 / self.softmax_temperature;
        let range = ctx.max_worsening - ctx.min_worsening;
        let inv_range = if range > 1e-10 { 1.0 / range } else { 0.0 };
        let min_w = ctx.min_worsening;

        self.buf_scores.clear();
        self.buf_features.clear();
        for &(_, w) in self.buf_moves.iter() {
            let approx_rank = if inv_range > 0.0 {
                (w - min_w) * inv_range
            } else {
                0.5
            };
            let f = extract_features(w, approx_rank, &ctx);
            self.buf_scores.push(self.policy.score(&f) * inv_temp);
            self.buf_features.push(f);
        }
        softmax_in_place(&mut self.buf_scores);
        let selected_idx = sample_categorical(&self.buf_scores, &mut state.rng);

        // 3. Apply the selected move and update the episode ledger
        let selected_worsening = self.buf_moves[selected_idx].1;
        state.apply(&self.buf_moves[selected_idx].0)?;
        self.cum_worsening += selected_worsening;
        self.cum_abs_worsening += selected_worsening.abs();

        // 4. Compute reward and update policy online (single-step REINFORCE)
        let reward = match self.reward_shaping {
            RewardShaping::Raw => -selected_worsening,
            RewardShaping::Normalized => -selected_worsening / ctx.max_abs_worsening.max(1e-10),
            RewardShaping::BestImprovement => {
                if state.best_iteration == state.iteration {
                    1.0
                } else {
                    0.0
                }
            }
        };

        let advantage = reward - self.baseline;
        self.baseline_count += 1;
        self.baseline += (reward - self.baseline) / self.baseline_count as f64;
        if self.learning_rate > 0.0 {
            // Exact softmax policy gradient:
            // ∇w log π(a) = (φ_a − Σ_i π_i φ_i) / τ.
            let mut grad = self.buf_features[selected_idx];
            for (f, &p) in self.buf_features.iter().zip(self.buf_scores.iter()) {
                for (g, &x) in grad.iter_mut().zip(f.iter()) {
                    *g -= p * x;
                }
            }
            for g in grad.iter_mut() {
                *g *= inv_temp;
            }
            self.policy.update(&grad, advantage, self.learning_rate);
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::problem::{MaxCut, MaxCutFlipNeighbor};
    use crate::search_state::SearchState;

    /// The three claims about `softmax_in_place` share one call each: it
    /// normalises, it is monotone in the score, and it survives inputs whose
    /// exponentials would overflow. Only the last needs large numbers, and
    /// only the last is why the log-sum-exp shift is there.
    #[test]
    fn softmax_normalises_and_survives_large_scores() {
        let mut equal = vec![1.0, 1.0, 1.0];
        softmax_in_place(&mut equal);
        for p in &equal {
            assert!((p - 1.0 / 3.0).abs() < 1e-10, "{equal:?}");
        }

        let mut peaked = vec![0.0, 0.0, 100.0];
        softmax_in_place(&mut peaked);
        assert!(peaked[2] > 0.99, "{peaked:?}");

        // Without the shift these exponentiate to infinity and the result is NaN.
        let mut large = vec![1000.0, 1001.0, 999.0];
        softmax_in_place(&mut large);
        let sum: f64 = large.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10, "{large:?}");
        assert!(large[1] > large[0] && large[0] > large[2], "{large:?}");
    }

    /// The ledger is what the improvement-ratio feature reads, and it has to start
    /// empty on every episode or a `Restart` carries the previous one's
    /// worsening into the next. Driving a real search rather than assigning to
    /// the fields is what makes this a test of `run_once` as well: a run that
    /// never touched the ledger would leave it at zero and fail the first
    /// assertion.
    #[test]
    fn the_ledger_accumulates_over_a_run_and_clear_empties_it() {
        let mc = MaxCut::from_edges((0..30).map(|i| (i, (i + 1) % 30, 1.0)));
        let mut state = SearchState::new_with_seed(&mc, 4);
        let mut rl = ReinforcementLearningSearch::<MaxCutFlipNeighbor>::new(
            StopCondition::iterations(200),
            0.1,
            1.0,
            RewardShaping::Normalized,
            None,
        );

        rl.run(&mut state).unwrap();

        assert!(
            rl.cum_abs_worsening > 0.0,
            "the run recorded no move at all"
        );
        let ratio = -rl.cum_worsening / rl.cum_abs_worsening.max(EPSILON);
        assert!((-1.0..=1.0).contains(&ratio), "ratio out of range: {ratio}");
        assert!(
            rl.policy.weights.iter().any(|&w| w != 0.0),
            "the policy never learned anything"
        );

        Heuristic::<MaxCut>::clear(&mut rl);
        assert_eq!(rl.cum_worsening, 0.0);
        assert_eq!(rl.cum_abs_worsening, 0.0);
    }
}
