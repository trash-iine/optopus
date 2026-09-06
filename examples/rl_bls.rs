//! Advanced example: driving Breakout Local Search with a learned
//! perturbation policy.
//!
//! [`BreakoutLocalSearchForMaxCut`] exposes its round as two halves —
//! `descend` (greedy descent to a local optimum) and `kick` (one perturbation
//! plus the round's `update_best`) — so a controller can act at the point
//! *between* them. This example replaces Benlic & Hao's hand-crafted
//! `omega`-based perturbation rule *and* their strength schedule with a
//! contextual softmax gradient bandit (`SoftmaxBandit`, from the library's
//! reinforcement-learning module): each round the bandit reads search-state
//! features and picks one of `3 x strength_bins.len()` actions — a
//! perturbation type together with a multiplier of `l0`.
//!
//! Nothing here is MaxCut-specific machinery: the descent, the operators and
//! the tabu memory they share all stay inside BLS. What the example owns is
//! the policy — the feature vector, the action decode, and the deferred-reward
//! wiring around the bandit.
//!
//! How to run:
//! ```
//! cargo run --release --example rl_bls
//! ```

use optopus::error::OptError;
use optopus::heuristic::reinforcement_learning::bandit::SoftmaxBandit;
use optopus::prelude::*;

/// Number of context features fed to the perturbation-selection bandit.
///
/// Layout: `[bias, min(w/t, 1), exp(-w/t), descent_improved_best,
/// relative_gap, reward_ema, budget_progress]`, where `w` is the `omega`
/// stagnation counter.
const NUM_CONTEXT_FEATURES: usize = 7;

/// Number of perturbation operators the bandit chooses between
/// (weak flip / weak swap / strong).
const NUM_PERTURBATION_TYPES: usize = 3;

const REWARD_SCALE_FLOOR: f64 = 1e-6;
/// EMA coefficient for the reward-magnitude scale.
const SCALE_BETA: f64 = 0.05;
/// EMA coefficient for the recent-reward feature.
const REWARD_EMA_BETA: f64 = 0.1;

/// A decision whose reward is observed after the *next* descent completes.
struct PendingDecision {
    action: usize,
    features: [f64; NUM_CONTEXT_FEATURES],
    /// Local-optimum objective right before the perturbation was applied.
    localopt_objective: f32,
    /// Global best objective at decision time (for the new-best bonus).
    global_best_objective: f32,
}

/// Breakout Local Search with a learned perturbation policy.
///
/// **Reward** (observed after the next descent): the change in local-optimum
/// objective, normalized by an EMA of its own magnitude and clamped to
/// `[-1, 1]`, plus a `+1` bonus when the global best improved.
///
/// **Multi-episode learning**: `clear()` resets the episode state (omega,
/// the inner BLS, pending decision, reward statistics) but **preserves the
/// bandit weights and baseline**, so the policy keeps improving across
/// `Restart` / `Iterated` episodes.
struct RlBreakoutLocalSearch {
    stop_condition: StopCondition,
    /// The BLS this controller drives: it owns the descent, the perturbation
    /// operators and the tabu memory they share, and is stepped one half-round at a
    /// time so the bandit can decide in between.
    bls: BreakoutLocalSearchForMaxCut,
    bandit: SoftmaxBandit,
    t: u64,
    l0: u64,
    strength_bins: Vec<f64>,
    // ---- episode state (reset by `clear`) ----
    omega: u64,
    prev_solution_objective: Option<f32>,
    pending: Option<PendingDecision>,
    /// EMA of the |local-optimum objective delta|; `0.0` = uninitialized.
    reward_scale: f64,
    reward_ema: f64,
}

impl RlBreakoutLocalSearch {
    /// # Panics
    ///
    /// Panics if `l0` is zero, `strength_bins` is empty or contains a
    /// non-positive multiplier, or the bandit parameters are invalid
    /// (`learning_rate < 0`, `softmax_temperature <= 0`, `exploration`
    /// outside `[0, 1]`).
    #[allow(clippy::too_many_arguments)]
    fn new(
        stop_condition: StopCondition,
        tabu_tenure: (u64, u64),
        t: u64,
        l0: u64,
        strength_bins: Vec<f64>,
        learning_rate: f64,
        softmax_temperature: f64,
        exploration: f64,
    ) -> Self {
        assert!(l0 > 0, "l0 must be at least 1");
        assert!(!strength_bins.is_empty(), "strength_bins must not be empty");
        assert!(
            strength_bins.iter().all(|&b| b > 0.0),
            "strength_bins must be strictly positive"
        );
        let bandit = SoftmaxBandit::new(
            NUM_PERTURBATION_TYPES * strength_bins.len(),
            NUM_CONTEXT_FEATURES,
            learning_rate,
            softmax_temperature,
            exploration,
        );
        Self {
            // `externally_driven` takes the tenure literally: the doubling
            // `BreakoutLocalSearchForMaxCut::new` applies reproduces the
            // paper's `gamma`, which belongs to the schedule this controller
            // replaces.
            bls: BreakoutLocalSearchForMaxCut::externally_driven(
                stop_condition.clone(),
                tabu_tenure,
            ),
            stop_condition,
            bandit,
            t,
            l0,
            strength_bins,
            omega: 0,
            prev_solution_objective: None,
            pending: None,
            reward_scale: 0.0,
            reward_ema: 0.0,
        }
    }

    /// Number of bandit actions (`3 x strength_bins.len()`).
    fn num_actions(&self) -> usize {
        NUM_PERTURBATION_TYPES * self.strength_bins.len()
    }

    /// Seeds the bandit with pre-trained weights (row-major
    /// `num_actions x NUM_CONTEXT_FEATURES`).
    ///
    /// Combine with `learning_rate = 0.0` for frozen-policy evaluation.
    ///
    /// # Panics
    ///
    /// Panics if `weights.len() != num_actions() * NUM_CONTEXT_FEATURES`.
    #[allow(dead_code)]
    fn with_policy_weights(mut self, weights: Vec<f64>) -> Self {
        self.bandit = self.bandit.with_weights(weights);
        self
    }

    /// Current bandit weights, e.g. for warm-starting a later run.
    fn policy_weights(&self) -> &[f64] {
        self.bandit.weights()
    }

    fn context_features(
        &self,
        state: &SearchState<'_, MaxCut>,
        descent_improved_best: bool,
    ) -> [f64; NUM_CONTEXT_FEATURES] {
        let t = self.t.max(1) as f64;
        let omega = self.omega as f64;
        let best = state.best_solution.objective as f64;
        let current = state.solution.objective as f64;
        let gap = ((best - current) / best.abs().max(1e-10)).clamp(-1.0, 1.0);
        let progress = match (
            self.stop_condition.max_duration,
            self.stop_condition.max_iteration,
        ) {
            (Some(d), _) => (state.duration().as_secs_f64() / d.as_secs_f64()).min(1.0),
            (None, Some(mi)) => (state.iterations_this_run() as f64 / mi.max(1) as f64).min(1.0),
            (None, None) => 0.0,
        };
        [
            1.0,
            (omega / t).min(1.0),
            (-omega / t).exp(),
            if descent_improved_best { 1.0 } else { 0.0 },
            gap,
            self.reward_ema.clamp(-1.0, 1.0),
            progress,
        ]
    }

    fn action_to_perturbation(&self, action: usize) -> (MaxCutPerturbation, u64) {
        let ptype = match action / self.strength_bins.len() {
            0 => MaxCutPerturbation::WeakFlip,
            1 => MaxCutPerturbation::WeakSwap,
            _ => MaxCutPerturbation::Strong,
        };
        let mult = self.strength_bins[action % self.strength_bins.len()];
        let l = ((self.l0 as f64 * mult).round() as u64).max(1);
        (ptype, l)
    }
}

impl Heuristic<MaxCut> for RlBreakoutLocalSearch {
    fn clear(&mut self) {
        self.omega = 0;
        self.prev_solution_objective = None;
        self.pending = None;
        self.reward_scale = 0.0;
        self.reward_ema = 0.0;
        self.bls.clear();
        // Bandit weights and baseline are intentionally preserved across episodes.
    }

    fn run_once<'a>(&mut self, state: &mut SearchState<'a, MaxCut>) -> Result<(), OptError> {
        // 1. Greedy descent to a local optimum — the first half of a BLS round.
        let best_before_descent = state.best_solution.objective;
        self.bls.descend(state)?;
        let descent_improved_best = state.best_solution.objective > best_before_descent;

        // 2. Update the stagnation counter (BLS omega rule: consecutive
        //    iterations whose local optimum did not beat the previous one).
        if let Some(prev) = self.prev_solution_objective
            && prev >= state.solution.objective
        {
            self.omega += 1;
        } else {
            self.omega = 0;
        }
        self.prev_solution_objective = Some(state.solution.objective);

        // 3. Observe the reward for the previous decision and update the policy.
        if let Some(pending) = self.pending.take() {
            let delta = f64::from(state.solution.objective - pending.localopt_objective);
            let abs_delta = delta.abs();
            if self.reward_scale <= 0.0 {
                self.reward_scale = abs_delta.max(REWARD_SCALE_FLOOR);
            } else {
                self.reward_scale += SCALE_BETA * (abs_delta - self.reward_scale);
                self.reward_scale = self.reward_scale.max(REWARD_SCALE_FLOOR);
            }
            let mut reward = (delta / self.reward_scale).clamp(-1.0, 1.0);
            if state.best_solution.objective > pending.global_best_objective {
                reward += 1.0;
            }
            self.bandit
                .update(pending.action, &pending.features, reward);
            self.reward_ema += REWARD_EMA_BETA * (reward - self.reward_ema);
        }

        // 4. Select the next perturbation.
        let features = self.context_features(state, descent_improved_best);
        let action = self.bandit.select(&features, &mut state.rng);
        self.pending = Some(PendingDecision {
            action,
            features,
            localopt_objective: state.solution.objective,
            global_best_objective: state.best_solution.objective,
        });
        let (ptype, l) = self.action_to_perturbation(action);

        // 5. The second half of the round: BLS applies the kick against the
        //    same prohibitions its descent wrote, and updates best once.
        self.bls.kick(state, ptype, l)
    }

    fn stop_condition(&self) -> &StopCondition {
        &self.stop_condition
    }
}

fn main() -> Result<(), OptError> {
    let mut rng = seeded_rng(42);
    let mc = MaxCut::new(Graph::erdos_renyi(800, 0.02, &mut rng));
    let iterations = 20_000;

    // Baseline: BLS with Benlic & Hao's schedule. Its `tabu_tenure` is the
    // paper's gamma, so it prohibits for twice this range.
    let mut bls = BreakoutLocalSearchForMaxCut::new(
        StopCondition::iterations(iterations),
        (15, 300),
        1_000,
        20,
        0.8,
        0.5,
    );
    let mut bls_state = SearchState::new_with_seed(&mc, 42);
    bls.run(&mut bls_state)?;

    // The same operators, driven by the learned policy.
    let mut rl = RlBreakoutLocalSearch::new(
        StopCondition::iterations(iterations),
        (15, 300),
        1_000,
        20,
        vec![1.0, 2.0, 4.0],
        0.1,
        1.0,
        0.05,
    );
    let mut rl_state = SearchState::new_with_seed(&mc, 42);
    rl.run(&mut rl_state)?;

    println!("instance: erdos_renyi(800, 0.02), {iterations} iterations, seed 42");
    println!(
        "  BLS    cut = {:>8} (best at iteration {})",
        bls_state.best_solution.objective, bls_state.best_iteration
    );
    println!(
        "  RL-BLS cut = {:>8} (best at iteration {})",
        rl_state.best_solution.objective, rl_state.best_iteration
    );
    println!(
        "  learned weights ({} actions x {NUM_CONTEXT_FEATURES} features): {:.3?}",
        rl.num_actions(),
        &rl.policy_weights()[..NUM_CONTEXT_FEATURES]
    );
    Ok(())
}
