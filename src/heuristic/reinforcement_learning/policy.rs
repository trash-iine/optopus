use super::feature::NUM_FEATURES;

/// Linear policy for move scoring: `score(φ) = w · φ`.
///
/// Weights are updated via the REINFORCE policy gradient rule.
/// Zero-initialized weights correspond to a uniform random policy under softmax.
pub struct LinearPolicy {
    pub weights: [f64; NUM_FEATURES],
}

impl Default for LinearPolicy {
    fn default() -> Self {
        Self::new()
    }
}

impl LinearPolicy {
    /// Create a new policy with zero weights (uniform random under softmax).
    pub fn new() -> Self {
        Self {
            weights: [0.0; NUM_FEATURES],
        }
    }

    /// Create a policy with the given weights.
    pub fn with_weights(weights: [f64; NUM_FEATURES]) -> Self {
        Self { weights }
    }

    /// Compute the score for a feature vector: `w · φ`.
    pub fn score(&self, features: &[f64; NUM_FEATURES]) -> f64 {
        self.weights
            .iter()
            .zip(features.iter())
            .map(|(w, f)| w * f)
            .sum()
    }

    /// REINFORCE gradient update: `w += lr * advantage * φ`.
    pub fn update(&mut self, features: &[f64; NUM_FEATURES], advantage: f64, lr: f64) {
        for (w, f) in self.weights.iter_mut().zip(features.iter()) {
            *w += lr * advantage * f;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_weights_give_zero_score() {
        let policy = LinearPolicy::new();
        let mut features = [0.0; NUM_FEATURES];
        for (i, f) in features.iter_mut().enumerate() {
            *f = (i + 1) as f64;
        }
        assert_eq!(policy.score(&features), 0.0);
    }

    #[test]
    fn score_is_dot_product() {
        let mut weights = [0.0; NUM_FEATURES];
        weights[0] = 1.0;
        weights[1] = -1.0;
        weights[2] = 2.0;
        let policy = LinearPolicy::with_weights(weights);
        let mut features = [0.0; NUM_FEATURES];
        features[0] = 3.0;
        features[1] = 2.0;
        features[2] = 1.0;
        assert!((policy.score(&features) - 3.0).abs() < 1e-10);
    }

    #[test]
    fn update_moves_weights_in_gradient_direction() {
        let mut policy = LinearPolicy::new();
        let mut features = [0.0; NUM_FEATURES];
        features[0] = 1.0;

        // Positive advantage → weight should increase
        policy.update(&features, 1.0, 0.1);
        assert!(policy.weights[0] > 0.0);
        assert_eq!(policy.weights[1], 0.0);

        // Negative advantage → weight should decrease
        policy.update(&features, -2.0, 0.1);
        assert!(policy.weights[0] < 0.0);
    }
}
