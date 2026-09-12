//! Roulette-wheel selection over weights that track recent performance.
//!
//! The scheme is Ropke & Pisinger's, introduced for ALNS's destroy and repair
//! banks. Each option starts equally likely, every use is scored, and at fixed
//! segment boundaries the segment's average score is blended into the weight.
//! Nothing in it is about routes or about operators. It is "pick one of `n`
//! things, learn which ones have been paying".
//!
//! [`SoftmaxBandit`](crate::heuristic::reinforcement_learning::bandit::SoftmaxBandit)
//! also picks one of `n` actions and learns from a reward, and is not a drop-in
//! replacement. This updates in batches, averaging a segment before blending,
//! which keeps one lucky iteration from swinging the wheel, and its weights are
//! a non-negative convex blend that cannot collapse onto a single option. The
//! bandit updates per decision and is contextual. Use the bandit when the
//! choice should depend on features of the current state, as
//! `examples/rl_bls.rs` does for perturbation selection.

use rand::Rng;
use rand::rngs::SmallRng;

/// A roulette wheel over `n` options whose weights adapt to observed scores.
///
/// Weights start at `1.0` (uniform). [`record`](Self::record) accumulates a
/// score per option; every `segment_len` records,
/// [`maybe_update`](Self::maybe_update) folds each option's *average* score for
/// the segment into its weight as
/// `w ← (1 − reaction) · w + reaction · avg`, then clears the segment.
///
/// An option not used during a segment keeps its weight, since its average is
/// undefined, and treating an unused option as scoring zero would drive its
/// weight down for not having been picked, which is the opposite of what the
/// wheel should do.
pub struct AdaptiveWeights {
    weights: Vec<f64>,
    scores: Vec<f64>,
    counts: Vec<u64>,
    segment_iter: u64,
    segment_len: u64,
    reaction: f64,
}

impl AdaptiveWeights {
    /// # Panics
    ///
    /// Panics if `num_options` is zero, `segment_len` is zero, or `reaction` is
    /// outside `[0, 1]`.
    pub fn new(num_options: usize, segment_len: u64, reaction: f64) -> Self {
        assert!(num_options > 0, "num_options must be at least 1");
        assert!(segment_len > 0, "segment_len must be at least 1");
        assert!(
            (0.0..=1.0).contains(&reaction),
            "reaction must be within [0, 1], got {reaction}"
        );
        Self {
            weights: vec![1.0; num_options],
            scores: vec![0.0; num_options],
            counts: vec![0; num_options],
            segment_iter: 0,
            segment_len,
            reaction,
        }
    }

    /// Restores the uniform wheel and drops the segment in progress.
    pub fn reset(&mut self) {
        self.weights.fill(1.0);
        self.scores.fill(0.0);
        self.counts.fill(0);
        self.segment_iter = 0;
    }

    /// The current weights, for logging and tests.
    pub fn weights(&self) -> &[f64] {
        &self.weights
    }

    /// Picks an option with probability proportional to its weight.
    ///
    /// Falls back to a uniform draw if every weight has decayed to zero, which
    /// a segment of all-zero scores can produce.
    pub fn select(&self, rng: &mut SmallRng) -> usize {
        let total: f64 = self.weights.iter().sum();
        if total <= 0.0 {
            return rng.random_range(0..self.weights.len());
        }
        let mut r = rng.random::<f64>() * total;
        for (i, &w) in self.weights.iter().enumerate() {
            if r < w {
                return i;
            }
            r -= w;
        }
        // Reachable only through floating-point rounding of the running
        // subtraction; the last option is the one the wheel had landed on.
        self.weights.len() - 1
    }

    /// Credits `option` with `score` and advances the segment.
    pub fn record(&mut self, option: usize, score: f64) {
        self.scores[option] += score;
        self.counts[option] += 1;
        self.segment_iter += 1;
    }

    /// At a segment boundary, blends the segment's average scores into the
    /// weights. A no-op before the boundary.
    pub fn maybe_update(&mut self) {
        if self.segment_iter < self.segment_len {
            return;
        }
        for i in 0..self.weights.len() {
            if self.counts[i] > 0 {
                let avg = self.scores[i] / self.counts[i] as f64;
                self.weights[i] = (1.0 - self.reaction) * self.weights[i] + self.reaction * avg;
            }
            self.scores[i] = 0.0;
            self.counts[i] = 0;
        }
        self.segment_iter = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    fn rng() -> SmallRng {
        SmallRng::seed_from_u64(7)
    }

    #[test]
    fn the_wheel_starts_uniform() {
        let w = AdaptiveWeights::new(3, 100, 0.1);
        assert_eq!(w.weights(), &[1.0, 1.0, 1.0]);
    }

    /// The blend has to move a well-scoring option up and leave the others
    /// alone until they are used, and it must not fire before the boundary.
    #[test]
    fn a_segment_boundary_blends_the_average_in() {
        let mut w = AdaptiveWeights::new(2, 4, 0.5);
        for _ in 0..3 {
            w.record(0, 4.0);
        }
        w.maybe_update();
        assert_eq!(
            w.weights(),
            &[1.0, 1.0],
            "must not fire before the boundary"
        );

        w.record(0, 4.0);
        w.maybe_update();
        // avg = 4.0 over four uses -> 0.5*1.0 + 0.5*4.0
        assert_eq!(w.weights()[0], 2.5);
        assert_eq!(w.weights()[1], 1.0, "an unused option keeps its weight");
    }

    /// Selection has to follow the weights. With one option at 100x the other,
    /// a few hundred draws must land on it overwhelmingly.
    #[test]
    fn selection_is_proportional_to_the_weights() {
        let mut w = AdaptiveWeights::new(2, 1, 1.0);
        w.record(0, 100.0);
        w.maybe_update();
        assert!(w.weights()[0] > 50.0 * w.weights()[1]);

        let mut rng = rng();
        let picks = (0..400).filter(|_| w.select(&mut rng) == 0).count();
        assert!(picks > 380, "the heavy option won only {picks}/400");
    }

    /// A segment in which everything scored zero drives every weight to zero;
    /// the wheel must still return a valid option rather than divide by zero.
    #[test]
    fn an_all_zero_wheel_falls_back_to_uniform() {
        let mut w = AdaptiveWeights::new(3, 1, 1.0);
        for i in 0..3 {
            w.record(i, 0.0);
        }
        w.maybe_update();
        assert_eq!(w.weights(), &[0.0, 0.0, 0.0]);

        let mut rng = rng();
        for _ in 0..20 {
            assert!(w.select(&mut rng) < 3);
        }
    }

    #[test]
    fn reset_restores_the_uniform_wheel() {
        let mut w = AdaptiveWeights::new(2, 1, 1.0);
        w.record(0, 4.0);
        w.maybe_update();
        assert_ne!(w.weights()[0], 1.0);

        w.reset();
        assert_eq!(w.weights(), &[1.0, 1.0]);
    }
}
