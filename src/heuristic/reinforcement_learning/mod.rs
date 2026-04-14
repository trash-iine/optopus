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

use feature::{NUM_FEATURES, compute_step_context, extract_features};
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

struct TrajectoryEntry {
    features: [f64; NUM_FEATURES],
    reward: f64,
}

/// Reinforcement learning heuristic that learns a move selection policy online.
///
/// At each step, all (or a subsample of) neighborhood moves are scored by a linear
/// policy over hand-crafted features. A move is sampled from the resulting softmax
/// distribution and applied. At the end of each episode ([`Heuristic::run`] call),
/// the policy weights are updated via REINFORCE with baseline subtraction.
///
/// **Key property**: `clear()` resets the trajectory but preserves the learned weights,
/// so the policy improves across episodes when used inside [`super::Restart`] or
/// [`super::Iterated`].
pub struct RLSearch<N> {
    pub stop_condition: StopCondition,
    pub policy: LinearPolicy,
    pub learning_rate: f64,
    pub discount: f64,
    pub softmax_temperature: f64,
    pub reward_shaping: RewardShaping,
    pub max_candidates: Option<usize>,
    phantom_neighbor: std::marker::PhantomData<N>,
    trajectory: Vec<TrajectoryEntry>,
    baseline: f64,
    baseline_count: u64,
    initial_worsening_total: Option<f64>,
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
            trajectory: Vec::new(),
            baseline: 0.0,
            baseline_count: 0,
            initial_worsening_total: None,
        }
    }

    pub fn with_policy_weights(mut self, weights: [f64; NUM_FEATURES]) -> Self {
        self.policy = LinearPolicy::with_weights(weights);
        self
    }

    fn max_iteration_budget(&self) -> f64 {
        self.stop_condition
            .max_iteration
            .unwrap_or(1_000_000) as f64
    }

    /// REINFORCE policy update at the end of an episode.
    fn update_policy(&mut self) {
        let n = self.trajectory.len();
        if n == 0 {
            return;
        }

        // Compute discounted returns G_t = Σ_{k=t}^{T-1} γ^{k-t} · r_k
        let mut returns = vec![0.0; n];
        returns[n - 1] = self.trajectory[n - 1].reward;
        for t in (0..n - 1).rev() {
            returns[t] = self.trajectory[t].reward + self.discount * returns[t + 1];
        }

        // Update baseline (running average of mean return)
        let mean_return = returns.iter().sum::<f64>() / n as f64;
        self.baseline_count += 1;
        self.baseline += (mean_return - self.baseline) / self.baseline_count as f64;

        // REINFORCE gradient: w += lr · (G_t - baseline) · φ_t
        for (t, entry) in self.trajectory.iter().enumerate() {
            let advantage = returns[t] - self.baseline;
            self.policy.update(&entry.features, advantage, self.learning_rate);
        }
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

/// Compute softmax probabilities with numerical stability (log-sum-exp trick).
fn softmax(scores: &[f64]) -> Vec<f64> {
    let max_score = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let exp_scores: Vec<f64> = scores.iter().map(|s| (s - max_score).exp()).collect();
    let sum_exp: f64 = exp_scores.iter().sum();
    exp_scores.iter().map(|e| e / sum_exp).collect()
}

/// Compute rank ratios from worsening values.
/// Lower worsening (= more improvement) gets rank_ratio closer to 0.0.
fn compute_rank_ratios(worsenings: &[f64]) -> Vec<f64> {
    let n = worsenings.len();
    if n <= 1 {
        return vec![0.0; n];
    }

    // Create index-worsening pairs and sort by worsening (ascending = best first)
    let mut indexed: Vec<(usize, f64)> = worsenings.iter().copied().enumerate().collect();
    indexed.sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

    let mut ranks = vec![0.0; n];
    let denom = (n - 1) as f64;
    for (rank, &(orig_idx, _)) in indexed.iter().enumerate() {
        ranks[orig_idx] = rank as f64 / denom;
    }
    ranks
}

impl<P, N> Heuristic<P> for RLSearch<N>
where
    P: ProblemTrait,
    N: MoveToNeighbor<P> + Evaluate + Clone,
{
    fn clear(&mut self) {
        self.trajectory.clear();
        self.initial_worsening_total = None;
        // Policy weights and baseline are intentionally preserved across episodes.
    }

    fn is_done<'a>(&self, state: &SearchState<'a, P>) -> bool {
        self.stop_condition.is_done(state)
    }

    fn run_once<'a>(&mut self, state: &mut SearchState<'a, P>) -> Result<(), OptError> {
        let mut rng = rand::rng();

        // 1. Enumerate moves and compute worsening values
        let mut moves_with_worsening: Vec<(N, f64)> = N::iter(state.instance, &state.solution)
            .map(|m| {
                let w = m.evaluate().worsening_amount();
                (m, w)
            })
            .collect();

        if moves_with_worsening.is_empty() {
            state.progress_iteration();
            return Ok(());
        }

        // Subsample if max_candidates is set
        if let Some(max_cand) = self.max_candidates {
            if moves_with_worsening.len() > max_cand {
                // Fisher-Yates partial shuffle to pick max_cand elements
                let n = moves_with_worsening.len();
                for i in 0..max_cand.min(n) {
                    let j = rng.random_range(i..n);
                    moves_with_worsening.swap(i, j);
                }
                moves_with_worsening.truncate(max_cand);
            }
        }

        // 2. Compute features
        let worsenings: Vec<f64> = moves_with_worsening.iter().map(|(_, w)| *w).collect();
        let rank_ratios = compute_rank_ratios(&worsenings);

        // Use 0.0 as initial worsening if not yet set (first step of episode)
        if self.initial_worsening_total.is_none() {
            self.initial_worsening_total = Some(0.0);
        }

        let ctx = compute_step_context(
            &worsenings,
            state.iteration,
            state.start_iteration,
            state.best_iteration,
            self.max_iteration_budget(),
            self.initial_worsening_total.unwrap(),
            0.0, // current relative to initial is tracked via state
        );

        let features_vec: Vec<[f64; NUM_FEATURES]> = worsenings
            .iter()
            .zip(rank_ratios.iter())
            .map(|(&w, &rr)| extract_features(w, rr, &ctx))
            .collect();

        // 3. Policy scoring + softmax sampling
        let scores: Vec<f64> = features_vec
            .iter()
            .map(|f| self.policy.score(f) / self.softmax_temperature)
            .collect();
        let probs = softmax(&scores);
        let selected_idx = sample_categorical(&probs, &mut rng);

        // 4. Apply the selected move
        let selected_worsening = moves_with_worsening[selected_idx].1;
        state.apply(&moves_with_worsening[selected_idx].0)?;

        // 5. Compute reward and record trajectory
        let reward = match self.reward_shaping {
            RewardShaping::Raw => -selected_worsening,
            RewardShaping::Normalized => {
                -selected_worsening / ctx.max_abs_worsening.max(1e-10)
            }
            RewardShaping::BestImprovement => {
                if state.best_iteration == state.iteration {
                    1.0
                } else {
                    0.0
                }
            }
        };

        self.trajectory.push(TrajectoryEntry {
            features: features_vec[selected_idx],
            reward,
        });

        Ok(())
    }

    fn run<'a>(&mut self, state: &mut SearchState<'a, P>) -> Result<(), OptError> {
        self.clear();
        tracing::debug!("RLSearch run started");

        while !self.is_done(state) {
            self.run_once(state)?;
        }

        // End-of-episode policy update
        if self.learning_rate > 0.0 {
            self.update_policy();
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

    #[test]
    fn softmax_uniform_for_equal_scores() {
        let probs = softmax(&[1.0, 1.0, 1.0]);
        assert_eq!(probs.len(), 3);
        for p in &probs {
            assert!((p - 1.0 / 3.0).abs() < 1e-10);
        }
    }

    #[test]
    fn softmax_concentrates_on_max() {
        let probs = softmax(&[0.0, 0.0, 100.0]);
        assert!(probs[2] > 0.99);
    }

    #[test]
    fn softmax_numerical_stability() {
        // Large values should not cause overflow
        let probs = softmax(&[1000.0, 1001.0, 999.0]);
        let sum: f64 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);
    }

    #[test]
    fn rank_ratios_basic() {
        let ranks = compute_rank_ratios(&[-3.0, 0.0, 2.0, -1.0]);
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
    fn update_policy_with_empty_trajectory() {
        let mut rl = RLSearch::<()>::new(
            StopCondition::iterations(100),
            0.01,
            0.99,
            1.0,
            RewardShaping::Raw,
            None,
        );
        // Should not panic
        rl.update_policy();
    }

    #[test]
    fn update_policy_moves_weights() {
        let mut rl = RLSearch::<()>::new(
            StopCondition::iterations(100),
            0.1,
            0.99,
            1.0,
            RewardShaping::Raw,
            None,
        );

        // Simulate trajectory: good move with feature[0] = 1.0
        rl.trajectory.push(TrajectoryEntry {
            features: [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            reward: 1.0,
        });
        rl.trajectory.push(TrajectoryEntry {
            features: [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            reward: -1.0,
        });

        rl.update_policy();

        // Weight for feature 0 should increase (positive reward),
        // weight for feature 1 should decrease (negative reward)
        assert!(rl.policy.weights[0] > 0.0);
        assert!(rl.policy.weights[1] < 0.0);
    }
}
