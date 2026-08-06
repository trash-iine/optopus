//! Shared tabu-map helpers for variable-indexed moves.
//!
//! The flip/swap moves of the binary-variable problems all use the same tabu
//! policy: a map from variable index to the iteration the variable becomes
//! movable again. These helpers hold that policy in one place; each move's
//! [`EnabledTabu`](crate::trait_defs::EnabledTabu) impl delegates to them (once
//! per variable the move touches).
//!
//! Two backings, same policy: [`VarTabuMap`] hashes, which suits a sparse or
//! unbounded key space, while [`VecTabuMap`] indexes a flat `Vec`, which suits
//! a dense `0..n` one. Problems whose variables *are* `0..n` and whose search
//! probes the map once per neighbor per iteration — every binary problem here —
//! want the `Vec`.

use rand::Rng;
use rand::rngs::SmallRng;
use std::collections::HashMap;

/// Tabu map from variable index to expiry iteration.
pub type VarTabuMap = HashMap<usize, u64>;

/// Returns `true` if variable `i` is not tabu at `iteration`.
#[inline]
pub fn is_var_enabled(tabu_map: &VarTabuMap, i: usize, iteration: u64) -> bool {
    tabu_map.get(&i).is_none_or(|&expiry| iteration > expiry)
}

/// Marks variable `i` tabu until `iteration` plus a tenure sampled uniformly
/// from `tabu_tenure = (min, max)` using `rng`.
#[inline]
pub fn add_var_to_tabu(
    tabu_map: &mut VarTabuMap,
    i: usize,
    iteration: u64,
    tabu_tenure: (u64, u64),
    rng: &mut SmallRng,
) {
    let tabu_duration = rng.random_range(tabu_tenure.0..=tabu_tenure.1);
    tabu_map.insert(i, iteration + tabu_duration);
}

/// Tabu map over a dense `0..n` variable space, backed by a flat `Vec`.
///
/// Each entry is the **first iteration at which the variable may move again**,
/// so `0` (the value a fresh or cleared entry holds) means "always allowed".
/// Storing the boundary rather than the last forbidden iteration is what keeps
/// the test a plain `iteration >= until`, with no off-by-one to get wrong at
/// each of the call sites.
///
/// This is the counterpart of [`VarTabuMap`] for keys that are known to be
/// small and dense: no hashing, no per-entry allocation, and the whole map
/// clears without giving up its buffer.
///
/// ```
/// use optopus::common::VecTabuMap;
/// use rand::SeedableRng;
///
/// let mut rng = rand::rngs::SmallRng::seed_from_u64(0);
/// let mut tabu = VecTabuMap::default();
/// tabu.ensure_capacity(4);
///
/// assert!(tabu.is_enabled(2, 0));
/// tabu.add(2, 0, (5, 5), &mut rng);   // forbidden for the next 5 iterations
/// assert!(!tabu.is_enabled(2, 5));
/// assert!(tabu.is_enabled(2, 6));
/// ```
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct VecTabuMap {
    /// `until[i]` = first iteration at which variable `i` may move again.
    until: Vec<u64>,
}

impl VecTabuMap {
    /// Creates an empty map. Entries are added on demand by
    /// [`add`](Self::add); [`ensure_capacity`](Self::ensure_capacity) just
    /// front-loads the growth.
    pub fn new() -> Self {
        Self::default()
    }

    /// Grows the map to hold variables `0..n`. Never shrinks.
    pub fn ensure_capacity(&mut self, n: usize) {
        if self.until.len() < n {
            self.until.resize(n, 0);
        }
    }

    /// Frees every variable, keeping the allocation.
    pub fn clear(&mut self) {
        self.until.fill(0);
    }

    /// Returns whether variable `i` may move at `iteration`. Variables the map
    /// has never seen are movable.
    #[inline]
    pub fn is_enabled(&self, i: usize, iteration: u64) -> bool {
        self.until.get(i).is_none_or(|&until| iteration >= until)
    }

    /// Forbids `i` for a tenure drawn from `tabu_tenure`, counted from
    /// `iteration`. Grows the map if `i` is beyond it.
    ///
    /// Any boundary already recorded is overwritten — the most recent move
    /// wins, which is what lets a short tenure release a variable early.
    #[inline]
    pub fn add(&mut self, i: usize, iteration: u64, tabu_tenure: (u64, u64), rng: &mut SmallRng) {
        // Drawn before the growth so the RNG stream does not depend on the
        // map's current capacity.
        let tabu_duration = rng.random_range(tabu_tenure.0..=tabu_tenure.1);
        if i >= self.until.len() {
            self.until.resize(i + 1, 0);
        }
        // A move applied at `iteration` with tenure `d` stays blocked through
        // `iteration + d`, so the first iteration that frees it is one past.
        self.until[i] = iteration + tabu_duration + 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    fn rng() -> SmallRng {
        SmallRng::seed_from_u64(7)
    }

    /// A tenure of `d` drawn at `iteration` must forbid exactly the next `d`
    /// iterations — blocked through `iteration + d`, free at `iteration + d + 1`.
    #[test]
    fn a_tenure_of_d_blocks_exactly_d_iterations() {
        let mut tabu = VecTabuMap::new();
        tabu.add(0, 100, (3, 3), &mut rng());

        for it in 100..=103 {
            assert!(!tabu.is_enabled(0, it), "must still be blocked at {it}");
        }
        assert!(tabu.is_enabled(0, 104));
    }

    /// The `Vec` and `HashMap` backings must agree move for move — this is what
    /// lets a problem switch between them without changing its search.
    #[test]
    fn vec_and_hash_backings_agree() {
        let (mut vec_map, mut hash_map) = (VecTabuMap::new(), VarTabuMap::new());
        let (mut a, mut b) = (rng(), rng());

        for (iteration, i) in [(0usize, 0usize), (3, 2), (7, 0), (11, 5)] {
            let iteration = iteration as u64;
            vec_map.add(i, iteration, (1, 9), &mut a);
            add_var_to_tabu(&mut hash_map, i, iteration, (1, 9), &mut b);

            for probe in iteration..iteration + 12 {
                for v in 0..6 {
                    assert_eq!(
                        vec_map.is_enabled(v, probe),
                        is_var_enabled(&hash_map, v, probe),
                        "disagreement on variable {v} at iteration {probe}"
                    );
                }
            }
        }
    }

    /// Unseen variables are movable, and `clear` returns every variable to that
    /// state without dropping the buffer.
    #[test]
    fn unseen_and_cleared_variables_are_movable() {
        let mut tabu = VecTabuMap::new();
        assert!(
            tabu.is_enabled(999, 0),
            "an unseen variable must be movable"
        );

        tabu.ensure_capacity(4);
        tabu.add(1, 0, (50, 50), &mut rng());
        assert!(!tabu.is_enabled(1, 10));

        tabu.clear();
        assert!(tabu.is_enabled(1, 10));
    }

    /// `add` must grow the map rather than silently dropping the entry, so a
    /// `Default` map (what `TabuSearch::clear` leaves behind) still works.
    #[test]
    fn add_grows_the_map() {
        let mut tabu = VecTabuMap::default();
        tabu.add(5, 0, (2, 2), &mut rng());
        assert!(!tabu.is_enabled(5, 1));
        assert!(tabu.is_enabled(4, 1), "growth must not block its neighbors");
    }
}
