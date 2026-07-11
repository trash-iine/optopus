use super::super::{Heuristic, StopCondition};
use super::bls_for_max_cut::{BlsOps, PerturbationType};
use crate::error::OptError;
use crate::heuristic::reinforcement_learning::bandit::SoftmaxBandit;
use crate::problem::MaxCut;
use crate::search_state::SearchState;

/// Number of context features fed to the perturbation-selection bandit.
///
/// Layout: `[bias, min(ω/t, 1), exp(−ω/t), descent_improved_best,
/// relative_gap, reward_ema, budget_progress, plateau_width]`.
pub const NUM_CONTEXT_FEATURES: usize = 8;

/// Number of perturbation operators the bandit chooses between
/// (weak flip / weak swap / strong / plateau cluster / plateau independent).
pub const NUM_PERTURBATION_TYPES: usize = 5;

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

/// Breakout Local Search with a *learned* perturbation policy for MaxCut.
///
/// Shares the exact descent / perturbation machinery of
/// [`BreakoutLocalSearch`](super::bls_for_max_cut::BreakoutLocalSearch)
/// (positive-gain-indexed greedy descent, flat tabu map, weak-flip /
/// weak-swap / strong operators), but replaces the hand-crafted
/// `omega`-based perturbation rule *and* the strength schedule with a
/// contextual softmax bandit ([`SoftmaxBandit`]): each outer iteration the
/// bandit observes search-state features and picks one of
/// `5 × strength_bins.len()` actions — a perturbation type (weak flip, weak
/// swap, strong, plateau cluster, or plateau independent-set) together with a
/// strength multiplier applied to `l0`.
///
/// **Reward** (observed after the next descent): the change in local-optimum
/// objective, normalized by an EMA of its own magnitude and clamped to
/// `[−1, 1]`, plus a `+1` bonus when the global best improved.
///
/// **Multi-episode learning**: `clear()` resets the episode state (omega,
/// tabu map, pending decision, reward statistics) but **preserves the bandit
/// weights and baseline**, so the policy keeps improving across
/// [`Restart`](crate::heuristic::Restart) /
/// [`Iterated`](crate::heuristic::Iterated) episodes.
///
/// # Parameters
///
/// - `tabu_tenure` — tabu tenure range `(min, max)` in iterations
/// - `t` — omega normalization period for the stagnation features
/// - `l0` — base perturbation length; actions scale it by a strength bin
/// - `strength_bins` — strength multipliers of `l0` (e.g. `[1.0, 2.0, 4.0]`)
/// - `learning_rate` — bandit step size (`0.0` = frozen-policy evaluation)
/// - `softmax_temperature` — bandit softmax temperature
/// - `exploration` — ε-uniform exploration floor in `[0, 1]`
pub struct RlBreakoutLocalSearch {
    stop_condition: StopCondition,
    ops: BlsOps,
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
    pub fn new(
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
            stop_condition,
            ops: BlsOps::new(tabu_tenure),
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

    /// Number of bandit actions (`5 × strength_bins.len()`).
    pub fn num_actions(&self) -> usize {
        NUM_PERTURBATION_TYPES * self.strength_bins.len()
    }

    /// Seeds the bandit with pre-trained weights (row-major
    /// `num_actions × NUM_CONTEXT_FEATURES`).
    ///
    /// Combine with `learning_rate = 0.0` for frozen-policy evaluation.
    ///
    /// # Panics
    ///
    /// Panics if `weights.len() != num_actions() * NUM_CONTEXT_FEATURES`.
    pub fn with_policy_weights(mut self, weights: Vec<f64>) -> Self {
        self.bandit = self.bandit.with_weights(weights);
        self
    }

    /// Current bandit weights, e.g. for warm-starting a later run.
    pub fn policy_weights(&self) -> &[f64] {
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
            (None, Some(mi)) => {
                ((state.iteration - state.start_iteration) as f64 / mi.max(1) as f64).min(1.0)
            }
            (None, None) => 0.0,
        };
        // Fraction of vertices sitting on the plateau (zero flip gain) — tells
        // the bandit how much room the plateau operators have to work with.
        let plateau_width = (state.solution.zero_gain_count() as f64
            / state.solution.x.len().max(1) as f64)
            .min(1.0);
        [
            1.0,
            (omega / t).min(1.0),
            (-omega / t).exp(),
            if descent_improved_best { 1.0 } else { 0.0 },
            gap,
            self.reward_ema.clamp(-1.0, 1.0),
            progress,
            plateau_width,
        ]
    }

    fn action_to_perturbation(&self, action: usize) -> (PerturbationType, u64) {
        let ptype = match action / self.strength_bins.len() {
            0 => PerturbationType::WeakFlip,
            1 => PerturbationType::WeakSwap,
            2 => PerturbationType::Strong,
            3 => PerturbationType::PlateauCluster,
            _ => PerturbationType::PlateauIndependent,
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
        self.ops.clear();
        // Bandit weights and baseline are intentionally preserved across episodes.
    }

    fn run_once<'a>(&mut self, state: &mut SearchState<'a, MaxCut>) -> Result<(), OptError> {
        self.ops.ensure_capacity(state.instance.graph.len());
        // Keep the plateau-width feature defined even before the first
        // plateau action (idempotent after the first call).
        state.solution.enable_zero_gain_index();

        // 1. Greedy descent to a local optimum (same operator as BLS).
        let best_before_descent = state.best_solution.objective;
        self.ops.descent(state)?;
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

        // 4. Select and apply the next perturbation.
        let features = self.context_features(state, descent_improved_best);
        let action = self.bandit.select(&features, &mut state.rng);
        self.pending = Some(PendingDecision {
            action,
            features,
            localopt_objective: state.solution.objective,
            global_best_objective: state.best_solution.objective,
        });

        let (ptype, l) = self.action_to_perturbation(action);
        tracing::debug!(
            iteration = state.iteration,
            omega = self.omega,
            action,
            perturbation = ?ptype,
            l,
            "RL-BLS: perturbation selected"
        );
        self.ops.perturb(ptype, l, state)?;

        // Update best once after the perturbation phase completes.
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

    /// Same toroidal instance as the BLS tests.
    fn small_instance() -> MaxCut {
        let n = 30usize;
        let mut edges = Vec::new();
        for i in 0..n {
            edges.push((i, (i + 1) % n, 1.0));
            edges.push((i, (i + 2) % n, 1.0));
        }
        MaxCut::from_edges(edges)
    }

    fn new_controller(stop: StopCondition) -> RlBreakoutLocalSearch {
        RlBreakoutLocalSearch::new(stop, (3, 15), 1_000, 5, vec![1.0, 2.0, 4.0], 0.1, 1.0, 0.05)
    }

    #[test]
    fn rl_bls_runs_without_error_and_improves() {
        let mc = small_instance();
        for seed in 0..10 {
            let mut state = SearchState::new_with_seed(&mc, seed);
            let mut rl = new_controller(StopCondition::iterations(5_000));
            rl.run(&mut state).expect("RL-BLS must not error");
            assert!(
                state.best_solution.objective > 0.0,
                "RL-BLS should find a positive cut, got {}",
                state.best_solution.objective
            );
        }
    }

    #[test]
    fn seeded_runs_are_deterministic() {
        let mc = small_instance();
        let run = || {
            let mut state = SearchState::new_with_seed(&mc, 42);
            let mut rl = new_controller(StopCondition::iterations(3_000));
            rl.run(&mut state).unwrap();
            (
                state.best_solution.objective,
                state.best_iteration,
                state.best_solution.x.clone(),
            )
        };
        assert_eq!(run(), run());
    }

    #[test]
    fn clear_resets_episode_state_but_keeps_weights() {
        let mc = small_instance();
        let mut state = SearchState::new_with_seed(&mc, 1);
        let mut rl = new_controller(StopCondition::iterations(2_000));
        rl.run(&mut state).unwrap();
        assert!(
            rl.policy_weights().iter().any(|&w| w != 0.0),
            "bandit should have learned something"
        );
        let weights_before = rl.policy_weights().to_vec();
        rl.clear();
        assert!(rl.pending.is_none());
        assert_eq!(rl.omega, 0);
        assert_eq!(rl.reward_scale, 0.0);
        assert_eq!(rl.policy_weights(), &weights_before[..]);
    }

    #[test]
    fn action_mapping_covers_all_types_and_strengths() {
        let rl = new_controller(StopCondition::iterations(1));
        let bins = 3;
        let mut seen_types = std::collections::HashSet::new();
        for action in 0..rl.num_actions() {
            let (ptype, l) = rl.action_to_perturbation(action);
            seen_types.insert(format!("{ptype:?}"));
            let expected = ((5.0 * rl.strength_bins[action % bins]).round() as u64).max(1);
            assert_eq!(l, expected);
        }
        assert_eq!(seen_types.len(), NUM_PERTURBATION_TYPES);
    }
}
