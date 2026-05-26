//! Reinforcement learning heuristic for combinatorial optimization.
//!
//! [`RLSearch`] uses a learned softmax policy over move features to select
//! which neighborhood move to apply at each step. The policy is trained online
//! via the REINFORCE algorithm with baseline subtraction.
//!
//! # Example
//!
//! ```rust,ignore
//! use optopus::prelude::*;
//!
//! let rl = RLSearch::<MaxCutFlipNeighbor>::new(
//!     StopCondition::failed_updates(1000),
//!     0.01,   // learning_rate
//!     0.99,   // discount
//!     1.0,    // softmax_temperature
//!     RewardShaping::Normalized,
//!     None,   // max_candidates
//! );
//! // Wrap in Restart for multi-episode learning
//! let solver = Restart::new(
//!     StopCondition::iterations(1_000_000),
//!     Box::new(rl),
//!     StopCondition::failed_updates(10_000),
//! );
//! ```

pub mod feature;
pub mod policy;

use feature::{NUM_FEATURES, StepStatsAccumulator, extract_features};
use policy::LinearPolicy;

use super::{Heuristic, StopCondition};
use crate::error::OptError;
use crate::search_state::{Evaluate, MoveToNeighbor, ProblemTrait, SearchState};
use rand::Rng;

/// Reward shaping strategy for the RL agent.
#[derive(Clone, Debug)]
pub enum RewardShaping {
    /// Raw gain: `reward = -worsening_amount`.
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
/// **Key property**: `clear()` resets per-episode state but preserves the learned
/// weights, so the policy improves across episodes when used inside
/// [`super::Restart`] or [`super::Iterated`].
pub struct RLSearch<N> {
    pub stop_condition: StopCondition,
    pub policy: LinearPolicy,
    pub learning_rate: f64,
    pub discount: f64,
    pub softmax_temperature: f64,
    pub reward_shaping: RewardShaping,
    pub max_candidates: Option<usize>,
    phantom_neighbor: std::marker::PhantomData<N>,
    baseline: f64,
    baseline_count: u64,
    initial_worsening_total: Option<f64>,
    // Pre-allocated buffers (reused across iterations)
    buf_moves: Vec<(N, f64)>,
    buf_scores: Vec<f64>,
}

impl<N> RLSearch<N> {
    pub fn new(
        stop_condition: StopCondition,
        learning_rate: f64,
        discount: f64,
        softmax_temperature: f64,
        reward_shaping: RewardShaping,
        max_candidates: Option<usize>,
    ) -> Self {
        Self {
            stop_condition,
            policy: LinearPolicy::new(),
            learning_rate,
            discount,
            softmax_temperature,
            reward_shaping,
            max_candidates,
            phantom_neighbor: std::marker::PhantomData,
            baseline: 0.0,
            baseline_count: 0,
            initial_worsening_total: None,
            buf_moves: Vec::new(),
            buf_scores: Vec::new(),
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

impl<P, N> Heuristic<P> for RLSearch<N>
where
    P: ProblemTrait,
    N: MoveToNeighbor<P> + Evaluate + Clone,
{
    fn clear(&mut self) {
        self.initial_worsening_total = None;
        // Policy weights and baseline are intentionally preserved across episodes.
    }

    fn is_done<'a>(&self, state: &SearchState<'a, P>) -> bool {
        self.stop_condition.is_done(state)
    }

    fn run_once<'a>(&mut self, state: &mut SearchState<'a, P>) -> Result<(), OptError> {
        // 1. Enumerate moves, compute worsenings, and accumulate step context stats in one pass
        self.buf_moves.clear();
        let mut acc = StepStatsAccumulator::new();
        for m in N::iter(state.instance, &state.solution) {
            let w = m.evaluate().worsening_amount();
            acc.push(w);
            self.buf_moves.push((m, w));
        }

        if self.buf_moves.is_empty() {
            state.progress_iteration();
            return Ok(());
        }

        // Subsample if max_candidates is set
        if let Some(max_cand) = self.max_candidates
            && self.buf_moves.len() > max_cand
        {
            let n = self.buf_moves.len();
            for i in 0..max_cand.min(n) {
                let j = state.rng.random_range(i..n);
                self.buf_moves.swap(i, j);
            }
            self.buf_moves.truncate(max_cand);
        }

        if self.initial_worsening_total.is_none() {
            self.initial_worsening_total = Some(0.0);
        }

        let initial_wt = self.initial_worsening_total.unwrap();
        let ctx = acc.finalize(
            state.iteration,
            state.start_iteration,
            state.best_iteration,
            self.max_iteration_budget(),
            initial_wt,
            0.0,
        );

        // 2. Score moves with approximate rank (O(n) instead of O(n log n) sort)
        let inv_temp = 1.0 / self.softmax_temperature;
        let range = ctx.max_worsening - ctx.min_worsening;
        let inv_range = if range > 1e-10 { 1.0 / range } else { 0.0 };
        let min_w = ctx.min_worsening;

        self.buf_scores.clear();
        self.buf_scores.extend(self.buf_moves.iter().map(|&(_, w)| {
            let approx_rank = if inv_range > 0.0 {
                (w - min_w) * inv_range
            } else {
                0.5
            };
            let f = extract_features(w, approx_rank, &ctx);
            self.policy.score(&f) * inv_temp
        }));
        softmax_in_place(&mut self.buf_scores);
        let selected_idx = sample_categorical(&self.buf_scores, &mut state.rng);

        // Recompute features only for the selected move
        let sel_w = self.buf_moves[selected_idx].1;
        let sel_rank = if inv_range > 0.0 {
            (sel_w - min_w) * inv_range
        } else {
            0.5
        };
        let selected_features = extract_features(sel_w, sel_rank, &ctx);

        // 3. Apply the selected move
        let selected_worsening = self.buf_moves[selected_idx].1;
        state.apply(&self.buf_moves[selected_idx].0)?;

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
            self.policy
                .update(&selected_features, advantage, self.learning_rate);
        }

        Ok(())
    }

    fn run<'a>(&mut self, state: &mut SearchState<'a, P>) -> Result<(), OptError> {
        self.clear();
        tracing::debug!("RLSearch run started");

        while !self.is_done(state) {
            self.run_once(state)?;
        }

        tracing::debug!(
            iteration = state.iteration,
            best_iteration = state.best_iteration,
            elapsed_secs = state.duration().as_secs_f64(),
            weights = ?self.policy.weights,
            "RLSearch run completed"
        );
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn compute_rank_ratios_into(
        worsenings: &[f64],
        indexed: &mut Vec<(usize, f64)>,
        ranks: &mut Vec<f64>,
    ) {
        let n = worsenings.len();
        indexed.clear();
        indexed.extend(worsenings.iter().copied().enumerate());
        ranks.clear();
        ranks.resize(n, 0.0);

        if n <= 1 {
            return;
        }

        indexed.sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        let denom = (n - 1) as f64;
        for (rank, &(orig_idx, _)) in indexed.iter().enumerate() {
            ranks[orig_idx] = rank as f64 / denom;
        }
    }

    #[test]
    fn softmax_uniform_for_equal_scores() {
        let mut scores = vec![1.0, 1.0, 1.0];
        softmax_in_place(&mut scores);
        assert_eq!(scores.len(), 3);
        for p in &scores {
            assert!((p - 1.0 / 3.0).abs() < 1e-10);
        }
    }

    #[test]
    fn softmax_concentrates_on_max() {
        let mut scores = vec![0.0, 0.0, 100.0];
        softmax_in_place(&mut scores);
        assert!(scores[2] > 0.99);
    }

    #[test]
    fn softmax_numerical_stability() {
        // Large values should not cause overflow
        let mut scores = vec![1000.0, 1001.0, 999.0];
        softmax_in_place(&mut scores);
        let sum: f64 = scores.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);
    }

    #[test]
    fn rank_ratios_basic() {
        let worsenings = [-3.0, 0.0, 2.0, -1.0];
        let mut indexed = Vec::new();
        let mut ranks = Vec::new();
        compute_rank_ratios_into(&worsenings, &mut indexed, &mut ranks);
        // sorted: -3.0 (best), -1.0, 0.0, 2.0 (worst)
        assert!((ranks[0] - 0.0).abs() < 1e-10); // -3.0 → rank 0
        assert!((ranks[3] - 1.0 / 3.0).abs() < 1e-10); // -1.0 → rank 1
        assert!((ranks[1] - 2.0 / 3.0).abs() < 1e-10); // 0.0 → rank 2
        assert!((ranks[2] - 1.0).abs() < 1e-10); // 2.0 → rank 3
    }

    #[test]
    fn sample_categorical_valid_index() {
        let probs = vec![0.2, 0.3, 0.5];
        let mut rng = rand::rng();
        for _ in 0..100 {
            let idx = sample_categorical(&probs, &mut rng);
            assert!(idx < 3);
        }
    }

    #[test]
    fn online_update_moves_weights() {
        let mut rl = RLSearch::<()>::new(
            StopCondition::iterations(100),
            0.1,
            0.99,
            1.0,
            RewardShaping::Raw,
            None,
        );

        // Simulate online update: positive reward with feature[0] = 1.0
        let features_good = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let reward_good = 1.0;
        let advantage = reward_good - rl.baseline;
        rl.baseline_count += 1;
        rl.baseline += (reward_good - rl.baseline) / rl.baseline_count as f64;
        rl.policy
            .update(&features_good, advantage, rl.learning_rate);

        // Simulate online update: negative reward with feature[1] = 1.0
        let features_bad = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let reward_bad = -1.0;
        let advantage = reward_bad - rl.baseline;
        rl.baseline_count += 1;
        rl.baseline += (reward_bad - rl.baseline) / rl.baseline_count as f64;
        rl.policy.update(&features_bad, advantage, rl.learning_rate);

        // Weight for feature 0 should be positive (positive reward),
        // weight for feature 1 should be negative (negative reward)
        assert!(rl.policy.weights[0] > 0.0);
        assert!(rl.policy.weights[1] < 0.0);
    }
}
