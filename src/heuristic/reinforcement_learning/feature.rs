/// Number of features per move used by the RL policy.
///
/// - 3 move-level: normalized_gain, is_improving, gain_rank_ratio
/// - 3 state-level: progress, stagnation, improvement_ratio
/// - 3 neighborhood-level: fraction_improving, mean_gain_normalized, std_gain_normalized
pub const NUM_FEATURES: usize = 9;

/// Per-step context computed once from the full neighborhood, reused for every move's features.
pub struct StepContext {
    pub max_abs_worsening: f64,
    pub fraction_improving: f64,
    pub mean_worsening_normalized: f64,
    pub std_worsening_normalized: f64,
    pub progress: f64,
    pub stagnation: f64,
    pub improvement_ratio: f64,
}

/// Compute the step context from worsening values of all candidate moves.
///
/// `worsenings`: per-move worsening amounts (positive = worse, negative = improving).
/// `iteration`, `start_iteration`, `best_iteration`: from SearchState.
/// `max_iteration_budget`: the configured max iterations (for normalization).
/// `initial_worsening_total`: worsening measure of the initial solution (for improvement_ratio).
/// `current_worsening_total`: worsening measure of the current solution.
pub fn compute_step_context(
    worsenings: &[f64],
    iteration: u64,
    start_iteration: u64,
    best_iteration: u64,
    max_iteration_budget: f64,
    initial_worsening_total: f64,
    current_worsening_total: f64,
) -> StepContext {
    let n = worsenings.len() as f64;
    debug_assert!(n > 0.0);

    let max_abs = worsenings
        .iter()
        .map(|w| w.abs())
        .fold(0.0_f64, f64::max)
        .max(1e-10);

    let n_improving = worsenings.iter().filter(|&&w| w < 0.0).count() as f64;
    let fraction_improving = n_improving / n;

    let sum: f64 = worsenings.iter().sum();
    let mean = sum / n;
    let mean_normalized = mean / max_abs;

    let variance = worsenings.iter().map(|w| (w - mean).powi(2)).sum::<f64>() / n;
    let std_normalized = variance.sqrt() / max_abs;

    let budget = max_iteration_budget.max(1.0);
    let elapsed = (iteration - start_iteration) as f64;
    let progress = (elapsed / budget).min(1.0);
    let stagnation = ((iteration - best_iteration) as f64 / budget).min(1.0);

    let denom = initial_worsening_total.abs().max(1e-10);
    let improvement_ratio = (current_worsening_total - initial_worsening_total) / denom;

    StepContext {
        max_abs_worsening: max_abs,
        fraction_improving,
        mean_worsening_normalized: mean_normalized,
        std_worsening_normalized: std_normalized,
        progress,
        stagnation,
        improvement_ratio,
    }
}

/// Build the feature vector for a single move.
///
/// `worsening`: this move's worsening amount (positive = worse).
/// `rank_ratio`: this move's rank among all candidates (0.0 = best, 1.0 = worst).
/// `ctx`: the step context shared across all moves.
pub fn extract_features(worsening: f64, rank_ratio: f64, ctx: &StepContext) -> [f64; NUM_FEATURES] {
    let normalized_gain = -worsening / ctx.max_abs_worsening;
    let is_improving = if worsening < 0.0 { 1.0 } else { 0.0 };

    [
        // Move-level
        normalized_gain,
        is_improving,
        rank_ratio,
        // State-level
        ctx.progress,
        ctx.stagnation,
        ctx.improvement_ratio,
        // Neighborhood-level
        ctx.fraction_improving,
        ctx.mean_worsening_normalized,
        ctx.std_worsening_normalized,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn step_context_basic() {
        let worsenings = vec![-2.0, -1.0, 0.0, 1.0, 3.0];
        let ctx = compute_step_context(&worsenings, 50, 0, 40, 100.0, 10.0, 8.0);

        assert!((ctx.max_abs_worsening - 3.0).abs() < 1e-10);
        assert!((ctx.fraction_improving - 0.4).abs() < 1e-10); // 2 out of 5
        assert!((ctx.progress - 0.5).abs() < 1e-10);
        assert!((ctx.stagnation - 0.1).abs() < 1e-10);
        assert!((ctx.improvement_ratio - (-0.2)).abs() < 1e-10); // (8-10)/10
    }

    #[test]
    fn extract_features_improving_move() {
        let ctx = StepContext {
            max_abs_worsening: 4.0,
            fraction_improving: 0.5,
            mean_worsening_normalized: -0.1,
            std_worsening_normalized: 0.3,
            progress: 0.25,
            stagnation: 0.05,
            improvement_ratio: -0.1,
        };

        let features = extract_features(-2.0, 0.0, &ctx);

        assert!((features[0] - 0.5).abs() < 1e-10); // normalized_gain = 2/4
        assert!((features[1] - 1.0).abs() < 1e-10); // is_improving
        assert!((features[2] - 0.0).abs() < 1e-10); // rank_ratio = best
    }

    #[test]
    fn extract_features_worsening_move() {
        let ctx = StepContext {
            max_abs_worsening: 4.0,
            fraction_improving: 0.5,
            mean_worsening_normalized: 0.0,
            std_worsening_normalized: 0.5,
            progress: 0.5,
            stagnation: 0.1,
            improvement_ratio: 0.0,
        };

        let features = extract_features(2.0, 1.0, &ctx);

        assert!((features[0] - (-0.5)).abs() < 1e-10); // normalized_gain = -2/4
        assert!((features[1] - 0.0).abs() < 1e-10); // not improving
        assert!((features[2] - 1.0).abs() < 1e-10); // rank_ratio = worst
    }

    #[test]
    fn zero_budget_does_not_panic() {
        let worsenings = vec![1.0];
        let ctx = compute_step_context(&worsenings, 0, 0, 0, 0.0, 0.0, 0.0);
        assert!(ctx.progress.is_finite());
        assert!(ctx.stagnation.is_finite());
    }
}
