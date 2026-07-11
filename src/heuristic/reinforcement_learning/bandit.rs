//! Contextual softmax gradient bandit.
//!
//! A problem-agnostic learner for "pick one of A actions given a context
//! vector" decisions inside heuristics — e.g. selecting a perturbation
//! operator and strength from search-state features. Uses per-action linear
//! preferences with a softmax policy, an ε-uniform exploration floor, and a
//! one-step REINFORCE update against an EMA baseline. Constant step sizes keep
//! it adaptive under the non-stationary rewards typical of a running search.

use rand::rngs::SmallRng;

use super::sample_categorical;

/// Contextual gradient bandit with a softmax policy over linear preferences.
///
/// - **Selection**: `π(a | φ) = (1 − ε) · softmax(w_a · φ / τ) + ε / A`.
/// - **Update**: `w_a += lr · (r − b) · (1{a = A} − π(a)) · φ` for every
///   action `a`, with baseline `b` tracked as an exponential moving average.
///
/// Call [`select`](Self::select) to choose an action, then — *before the next
/// `select`* — call [`update`](Self::update) with the observed reward; the
/// selection probabilities needed by the gradient are kept internally from
/// the last `select`.
pub struct SoftmaxBandit {
    num_actions: usize,
    dim: usize,
    /// Per-action weight vectors, row-major: `weights[a * dim + j]`.
    weights: Vec<f64>,
    learning_rate: f64,
    temperature: f64,
    exploration: f64,
    baseline: f64,
    baseline_initialized: bool,
    baseline_beta: f64,
    /// Probabilities computed by the most recent [`select`](Self::select).
    probs: Vec<f64>,
}

impl SoftmaxBandit {
    /// # Panics
    ///
    /// Panics if `num_actions` or `dim` is zero, `learning_rate` is negative,
    /// `temperature` is not strictly positive, or `exploration` is outside
    /// `[0, 1]`.
    pub fn new(
        num_actions: usize,
        dim: usize,
        learning_rate: f64,
        temperature: f64,
        exploration: f64,
    ) -> Self {
        assert!(num_actions > 0, "num_actions must be at least 1");
        assert!(dim > 0, "dim must be at least 1");
        assert!(learning_rate >= 0.0, "learning_rate must be non-negative");
        assert!(temperature > 0.0, "temperature must be strictly positive");
        assert!(
            (0.0..=1.0).contains(&exploration),
            "exploration must be within [0, 1]"
        );
        Self {
            num_actions,
            dim,
            weights: vec![0.0; num_actions * dim],
            learning_rate,
            temperature,
            exploration,
            baseline: 0.0,
            baseline_initialized: false,
            baseline_beta: 0.05,
            probs: vec![0.0; num_actions],
        }
    }

    /// Seeds the preference weights (row-major `num_actions × dim`), e.g. from
    /// a previous run.
    ///
    /// # Panics
    ///
    /// Panics if `weights.len() != num_actions * dim`.
    pub fn with_weights(mut self, weights: Vec<f64>) -> Self {
        assert_eq!(
            weights.len(),
            self.num_actions * self.dim,
            "weights must have num_actions * dim elements"
        );
        self.weights = weights;
        self
    }

    pub fn num_actions(&self) -> usize {
        self.num_actions
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Current preference weights (row-major `num_actions × dim`).
    pub fn weights(&self) -> &[f64] {
        &self.weights
    }

    /// Samples an action for the given context vector.
    ///
    /// # Panics
    ///
    /// Panics if `features.len() != dim`.
    pub fn select(&mut self, features: &[f64], rng: &mut SmallRng) -> usize {
        assert_eq!(features.len(), self.dim, "context dimension mismatch");
        let inv_temp = 1.0 / self.temperature;
        for a in 0..self.num_actions {
            let row = &self.weights[a * self.dim..(a + 1) * self.dim];
            let score: f64 = row.iter().zip(features).map(|(w, f)| w * f).sum();
            self.probs[a] = score * inv_temp;
        }
        super::softmax_in_place(&mut self.probs);
        if self.exploration > 0.0 {
            let uniform = self.exploration / self.num_actions as f64;
            for p in self.probs.iter_mut() {
                *p = (1.0 - self.exploration) * *p + uniform;
            }
        }
        sample_categorical(&self.probs, rng)
    }

    /// Updates the policy for `action` (returned by the immediately preceding
    /// [`select`](Self::select)) given the context it was selected in and the
    /// observed reward.
    ///
    /// # Panics
    ///
    /// Panics if `action >= num_actions` or `features.len() != dim`.
    pub fn update(&mut self, action: usize, features: &[f64], reward: f64) {
        assert!(action < self.num_actions, "action out of range");
        assert_eq!(features.len(), self.dim, "context dimension mismatch");
        if !self.baseline_initialized {
            self.baseline = reward;
            self.baseline_initialized = true;
        }
        let advantage = reward - self.baseline;
        self.baseline += self.baseline_beta * (reward - self.baseline);
        if self.learning_rate == 0.0 {
            return;
        }
        for a in 0..self.num_actions {
            let indicator = if a == action { 1.0 } else { 0.0 };
            let coeff = self.learning_rate * advantage * (indicator - self.probs[a]);
            if coeff == 0.0 {
                continue;
            }
            let row = &mut self.weights[a * self.dim..(a + 1) * self.dim];
            for (w, &f) in row.iter_mut().zip(features) {
                *w += coeff * f;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    /// Two contexts, two actions: action 0 pays in context A, action 1 pays in
    /// context B. The bandit must learn the context-dependent best arm.
    #[test]
    fn learns_contextually_best_action() {
        let mut bandit = SoftmaxBandit::new(2, 2, 0.2, 1.0, 0.05);
        let mut rng = SmallRng::seed_from_u64(42);
        let ctx_a = [1.0, 0.0];
        let ctx_b = [0.0, 1.0];
        for i in 0..2000 {
            let ctx = if i % 2 == 0 { &ctx_a } else { &ctx_b };
            let action = bandit.select(ctx, &mut rng);
            let paying = if i % 2 == 0 { 0 } else { 1 };
            let reward = if action == paying { 1.0 } else { -1.0 };
            bandit.update(action, ctx, reward);
        }
        // Greedy action per context must match the paying arm.
        let score = |bandit: &SoftmaxBandit, a: usize, ctx: &[f64]| -> f64 {
            bandit.weights()[a * 2..(a + 1) * 2]
                .iter()
                .zip(ctx)
                .map(|(w, f)| w * f)
                .sum()
        };
        assert!(score(&bandit, 0, &ctx_a) > score(&bandit, 1, &ctx_a));
        assert!(score(&bandit, 1, &ctx_b) > score(&bandit, 0, &ctx_b));
    }

    #[test]
    fn deterministic_under_fixed_seed() {
        let run = || {
            let mut bandit = SoftmaxBandit::new(3, 2, 0.1, 1.0, 0.1);
            let mut rng = SmallRng::seed_from_u64(7);
            let mut actions = Vec::new();
            for i in 0..50 {
                let ctx = [1.0, (i % 3) as f64 / 2.0];
                let action = bandit.select(&ctx, &mut rng);
                bandit.update(action, &ctx, if action == 0 { 0.5 } else { -0.5 });
                actions.push(action);
            }
            (actions, bandit.weights().to_vec())
        };
        assert_eq!(run(), run());
    }

    #[test]
    fn zero_learning_rate_freezes_weights() {
        let mut bandit = SoftmaxBandit::new(2, 1, 0.0, 1.0, 0.0).with_weights(vec![1.0, -1.0]);
        let mut rng = SmallRng::seed_from_u64(1);
        let before = bandit.weights().to_vec();
        for _ in 0..20 {
            let action = bandit.select(&[1.0], &mut rng);
            bandit.update(action, &[1.0], 1.0);
        }
        assert_eq!(bandit.weights(), &before[..]);
    }

    #[test]
    fn exploration_floor_keeps_all_actions_reachable() {
        // Strongly biased weights + ε floor: the disfavored arm must still be
        // sampled occasionally.
        let mut bandit = SoftmaxBandit::new(2, 1, 0.0, 1.0, 0.2).with_weights(vec![100.0, -100.0]);
        let mut rng = SmallRng::seed_from_u64(3);
        let picks_of_1 = (0..1000)
            .filter(|_| bandit.select(&[1.0], &mut rng) == 1)
            .count();
        // Expected ≈ ε/2 = 10%.
        assert!(picks_of_1 > 50, "got {picks_of_1}");
    }

    #[test]
    #[should_panic(expected = "temperature")]
    fn zero_temperature_panics() {
        let _ = SoftmaxBandit::new(2, 2, 0.1, 0.0, 0.1);
    }
}
