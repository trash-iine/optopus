//! Random draws the moves sample their endpoints with.

use rand::Rng;

/// Two distinct positions drawn uniformly from `0..n`, or `None` when `n < 2`.
///
/// The second draw is over `n - 1` values and shifted past the first, so the
/// pair is uniform over all ordered pairs with two RNG draws. The position
/// moves (relocate, swap over a sequence) draw their endpoints here.
pub fn random_distinct_pair(n: usize, rng: &mut impl Rng) -> Option<(usize, usize)> {
    if n < 2 {
        return None;
    }
    let a = rng.random_range(0..n);
    let b = rng.random_range(0..n - 1);
    Some((a, if b >= a { b + 1 } else { b }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    /// The second draw is over `n - 1` values and shifted past the first, so
    /// the shift is an off-by-one waiting to happen: too eager and it walks
    /// off the end, too lazy and it returns the first position twice. Both
    /// show up as a degenerate move rather than a crash.
    #[test]
    fn random_distinct_pair_is_distinct_in_range_and_reaches_every_ordered_pair() {
        assert_eq!(random_distinct_pair(0, &mut rng()), None);
        assert_eq!(random_distinct_pair(1, &mut rng()), None);

        let n = 4;
        let mut rng = rng();
        let mut seen = std::collections::HashSet::new();
        for _ in 0..2_000 {
            let (a, b) = random_distinct_pair(n, &mut rng).unwrap();
            assert!(a < n && b < n, "({a}, {b}) out of range for n = {n}");
            assert_ne!(a, b, "the pair is not distinct");
            seen.insert((a, b));
        }
        assert_eq!(seen.len(), n * (n - 1), "not every ordered pair was drawn");
    }

    fn rng() -> rand::rngs::SmallRng {
        rand::rngs::SmallRng::seed_from_u64(5)
    }
}
