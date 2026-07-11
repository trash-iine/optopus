/// Number of move-level features: normalized_gain, is_improving, gain_rank_ratio.
pub const MOVE_FEATURES: usize = 3;
/// Number of state/neighborhood-level features: progress, stagnation,
/// improvement_ratio, fraction_improving, mean_gain_normalized, std_gain_normalized.
pub const STATE_FEATURES: usize = 6;
/// Number of features per move used by the RL policy.
///
/// The vector is the interaction product `φ_move ⊗ [1, φ_state]`: each
/// move-level feature appears once on its own and once multiplied by every
/// state-level feature. State-level features are constant across the moves of
/// one step, so as plain additive terms they would cancel in the softmax and
/// contribute nothing to move selection; as interactions they let the state
/// *modulate* the policy's move preferences (e.g. "prefer worsening moves
/// when stagnating").
pub const NUM_FEATURES: usize = MOVE_FEATURES * (1 + STATE_FEATURES);
pub(crate) const EPSILON: f64 = 1e-10;

/// Per-step context computed once from the full neighborhood, reused for every move's features.
pub struct StepContext {
    pub max_abs_worsening: f64,
    pub min_worsening: f64,
    pub max_worsening: f64,
    pub fraction_improving: f64,
    pub mean_worsening_normalized: f64,
    pub std_worsening_normalized: f64,
    pub progress: f64,
    pub stagnation: f64,
    pub improvement_ratio: f64,
}

/// Streaming accumulator for the per-step worsening statistics consumed by
/// [`StepContext`]. Allows callers (such as `RlSearch::run_once`) to feed
/// values one at a time during move enumeration without materializing a slice.
pub struct StepStatsAccumulator {
    count: usize,
    max_abs: f64,
    min_w: f64,
    max_w: f64,
    n_improving: usize,
    mean: f64,
    m2: f64,
}

impl StepStatsAccumulator {
    pub fn new() -> Self {
        Self {
            count: 0,
            max_abs: 0.0,
            min_w: f64::INFINITY,
            max_w: f64::NEG_INFINITY,
            n_improving: 0,
            mean: 0.0,
            m2: 0.0,
        }
    }

    pub fn count(&self) -> usize {
        self.count
    }

    /// Add one worsening value to the accumulator (Welford's online algorithm).
    pub fn push(&mut self, w: f64) {
        self.count += 1;
        self.max_abs = self.max_abs.max(w.abs());
        self.min_w = self.min_w.min(w);
        self.max_w = self.max_w.max(w);
        if w < 0.0 {
            self.n_improving += 1;
        }
        let delta = w - self.mean;
        self.mean += delta / self.count as f64;
        let delta2 = w - self.mean;
        self.m2 += delta * delta2;
    }

    /// Finalize the accumulated stats into a [`StepContext`].
    ///
    /// - `iteration`, `start_iteration`, `best_iteration`: from SearchState.
    /// - `max_iteration_budget`: the configured max iterations (for normalization).
    /// - `improvement_ratio`: net episode improvement in `[-1, 1]`, as
    ///   maintained by the caller (e.g. `RlSearch`'s applied-move ledger:
    ///   `-cum_worsening / max(cum_abs_worsening, ε)`).
    pub fn finalize(
        &self,
        iteration: u64,
        start_iteration: u64,
        best_iteration: u64,
        max_iteration_budget: f64,
        improvement_ratio: f64,
    ) -> StepContext {
        debug_assert!(self.count > 0);

        let n = self.count as f64;
        let max_abs = self.max_abs.max(EPSILON);
        let fraction_improving = self.n_improving as f64 / n;
        let mean_normalized = self.mean / max_abs;
        let std_normalized = (self.m2 / n).sqrt() / max_abs;

        let budget = max_iteration_budget.max(1.0);
        let elapsed = (iteration - start_iteration) as f64;
        let progress = (elapsed / budget).min(1.0);
        let stagnation = ((iteration - best_iteration) as f64 / budget).min(1.0);

        StepContext {
            max_abs_worsening: max_abs,
            min_worsening: self.min_w,
            max_worsening: self.max_w,
            fraction_improving,
            mean_worsening_normalized: mean_normalized,
            std_worsening_normalized: std_normalized,
            progress,
            stagnation,
            improvement_ratio,
        }
    }
}

impl Default for StepStatsAccumulator {
    fn default() -> Self {
        Self::new()
    }
}

/// Compute the step context from worsening values of all candidate moves.
///
/// Convenience wrapper around [`StepStatsAccumulator`] for slice-based callers.
pub fn compute_step_context(
    worsenings: &[f64],
    iteration: u64,
    start_iteration: u64,
    best_iteration: u64,
    max_iteration_budget: f64,
    improvement_ratio: f64,
) -> StepContext {
    let mut acc = StepStatsAccumulator::new();
    for &w in worsenings {
        acc.push(w);
    }
    acc.finalize(
        iteration,
        start_iteration,
        best_iteration,
        max_iteration_budget,
        improvement_ratio,
    )
}

/// Build the feature vector for a single move: `φ_move ⊗ [1, φ_state]`
/// (see [`NUM_FEATURES`] for the layout rationale).
///
/// - `worsening`: this move's worsening amount (positive = worse).
/// - `rank_ratio`: this move's rank among all candidates (0.0 = best, 1.0 = worst).
/// - `ctx`: the step context shared across all moves.
pub fn extract_features(worsening: f64, rank_ratio: f64, ctx: &StepContext) -> [f64; NUM_FEATURES] {
    let normalized_gain = -worsening / ctx.max_abs_worsening;
    let is_improving = if worsening < 0.0 { 1.0 } else { 0.0 };

    let move_features = [normalized_gain, is_improving, rank_ratio];
    let state_features = [
        ctx.progress,
        ctx.stagnation,
        ctx.improvement_ratio,
        ctx.fraction_improving,
        ctx.mean_worsening_normalized,
        ctx.std_worsening_normalized,
    ];

    let mut features = [0.0; NUM_FEATURES];
    for (block, &m) in move_features.iter().enumerate() {
        let base = block * (1 + STATE_FEATURES);
        features[base] = m;
        for (j, &s) in state_features.iter().enumerate() {
            features[base + 1 + j] = m * s;
        }
    }
    features
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn step_context_basic() {
        let worsenings = vec![-2.0, -1.0, 0.0, 1.0, 3.0];
        let ctx = compute_step_context(&worsenings, 50, 0, 40, 100.0, -0.2);

        assert!((ctx.max_abs_worsening - 3.0).abs() < 1e-10);
        assert!((ctx.fraction_improving - 0.4).abs() < 1e-10); // 2 out of 5
        assert!((ctx.progress - 0.5).abs() < 1e-10);
        assert!((ctx.stagnation - 0.1).abs() < 1e-10);
        assert!((ctx.improvement_ratio - (-0.2)).abs() < 1e-10); // passed through
    }

    #[test]
    fn extract_features_improving_move() {
        let ctx = StepContext {
            max_abs_worsening: 4.0,
            min_worsening: -4.0,
            max_worsening: 4.0,
            fraction_improving: 0.5,
            mean_worsening_normalized: -0.1,
            std_worsening_normalized: 0.3,
            progress: 0.25,
            stagnation: 0.05,
            improvement_ratio: -0.1,
        };

        let features = extract_features(-2.0, 0.0, &ctx);

        let block = 1 + STATE_FEATURES;
        assert!((features[0] - 0.5).abs() < 1e-10); // normalized_gain = 2/4
        assert!((features[block] - 1.0).abs() < 1e-10); // is_improving
        assert!((features[2 * block] - 0.0).abs() < 1e-10); // rank_ratio = best
        // Interaction terms: move feature × state feature.
        assert!((features[1] - 0.5 * ctx.progress).abs() < 1e-10);
        assert!((features[block + 2] - 1.0 * ctx.stagnation).abs() < 1e-10);
    }

    #[test]
    fn extract_features_worsening_move() {
        let ctx = StepContext {
            max_abs_worsening: 4.0,
            min_worsening: -4.0,
            max_worsening: 4.0,
            fraction_improving: 0.5,
            mean_worsening_normalized: 0.0,
            std_worsening_normalized: 0.5,
            progress: 0.5,
            stagnation: 0.1,
            improvement_ratio: 0.0,
        };

        let features = extract_features(2.0, 1.0, &ctx);

        let block = 1 + STATE_FEATURES;
        assert!((features[0] - (-0.5)).abs() < 1e-10); // normalized_gain = -2/4
        assert!((features[block] - 0.0).abs() < 1e-10); // not improving
        assert!((features[2 * block] - 1.0).abs() < 1e-10); // rank_ratio = worst
        // is_improving = 0 zeroes its whole interaction block.
        for j in 0..STATE_FEATURES {
            assert_eq!(features[block + 1 + j], 0.0);
        }
    }

    #[test]
    fn zero_budget_does_not_panic() {
        let worsenings = vec![1.0];
        let ctx = compute_step_context(&worsenings, 0, 0, 0, 0.0, 0.0);
        assert!(ctx.progress.is_finite());
        assert!(ctx.stagnation.is_finite());
    }
}
