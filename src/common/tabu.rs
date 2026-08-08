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

use crate::trait_defs::EnabledTabu;
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

/// A tabu map together with the tenure range every write to it draws from.
///
/// The two always travel together — a map without the tenure cannot record a
/// move, and a tenure without the map has nothing to record into — yet keeping
/// them as two separate fields means every call site repeats the same four
/// arguments (`map`, `iteration`, `tenure`, `rng`) and every operator that
/// shares the map has to be handed both. The ledger reduces that to two verbs:
/// [`allows`](Self::allows) asks, [`record`](Self::record) writes.
///
/// It is deliberately *not* a tabu policy. The policy lives in each move's
/// [`EnabledTabu`](crate::trait_defs::EnabledTabu) impl, so a ledger forbids
/// exactly what a [`TabuSearch`](crate::heuristic::TabuSearch) over the same
/// neighborhood would — the ledger only owns the storage and the tenure.
///
/// ```
/// use optopus::common::{TabuLedger, VecTabuMap};
/// use optopus::prelude::*;
/// use rand::SeedableRng;
///
/// let mc = MaxCut::from_edges([(0, 1, 1.0), (1, 2, 1.0)]);
/// let sol = MaxCutSolution::new_from_assignment(&mc, vec![false, true, false]);
/// let mv = MaxCutFlipNeighbor::new(&mc, &sol, 1);
///
/// let mut rng = rand::rngs::SmallRng::seed_from_u64(0);
/// let mut ledger: TabuLedger<VecTabuMap> = TabuLedger::new((5, 5));
/// ledger.ensure_capacity(mc.graph.len());
///
/// assert!(ledger.allows(&mv, 0));
/// ledger.record(&mv, 0, &mut rng);   // forbidden for the next 5 iterations
/// assert!(!ledger.allows(&mv, 5));
/// assert!(ledger.allows(&mv, 6));
/// ```
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct TabuLedger<M> {
    map: M,
    tenure: (u64, u64),
}

impl<M: Default> TabuLedger<M> {
    /// Creates a ledger over an empty map.
    ///
    /// # Panics
    ///
    /// Panics if `tenure.0 > tenure.1`, i.e. the range is empty — sampling from
    /// it would panic later, at a point far from the caller that set it.
    pub fn new(tenure: (u64, u64)) -> Self {
        Self::with_map(M::default(), tenure)
    }

    /// Creates a ledger over an existing map, e.g. one handed over from a
    /// previous phase so the prohibitions it recorded stay in force.
    ///
    /// # Panics
    ///
    /// Panics if `tenure.0 > tenure.1`; see [`new`](Self::new).
    pub fn with_map(map: M, tenure: (u64, u64)) -> Self {
        if tenure.0 > tenure.1 {
            panic!(
                "Invalid tabu tenure range: left side should be smaller than or equal to the right side ({} <= {})",
                tenure.0, tenure.1
            );
        }
        Self { map, tenure }
    }

    /// The tenure range handed to every [`record`](Self::record).
    pub fn tenure(&self) -> (u64, u64) {
        self.tenure
    }

    /// Borrows the underlying map.
    pub fn map(&self) -> &M {
        &self.map
    }

    /// Borrows the underlying map mutably, for operators that count a tenure
    /// differently from [`record`](Self::record).
    pub fn map_mut(&mut self) -> &mut M {
        &mut self.map
    }

    /// Takes the map out, leaving an empty one behind.
    pub fn take_map(&mut self) -> M {
        std::mem::take(&mut self.map)
    }

    /// Replaces the map, keeping the tenure.
    pub fn set_map(&mut self, map: M) {
        self.map = map;
    }

    /// Drops every prohibition by replacing the map with a fresh one.
    ///
    /// [`TabuLedger<VecTabuMap>`] overrides this with a version that keeps the
    /// allocation; see [`clear`](TabuLedger::clear).
    pub fn reset(&mut self) {
        self.map = M::default();
    }

    /// Returns whether `mv` may be applied at `iteration`.
    #[inline]
    pub fn allows<N: EnabledTabu<TabuMap = M>>(&self, mv: &N, iteration: u64) -> bool {
        mv.is_move_enabled(&self.map, iteration)
    }

    /// Forbids `mv` for a tenure drawn from [`tenure`](Self::tenure), counted
    /// from `iteration`.
    ///
    /// The draw goes through the caller's `rng` — pass `&mut state.rng` so
    /// seeded runs stay bit-reproducible.
    #[inline]
    pub fn record<N: EnabledTabu<TabuMap = M>>(
        &mut self,
        mv: &N,
        iteration: u64,
        rng: &mut SmallRng,
    ) {
        mv.add_to_tabu_map(&mut self.map, iteration, self.tenure, rng);
    }
}

impl TabuLedger<VecTabuMap> {
    /// Grows the map to hold variables `0..n`. Never shrinks.
    pub fn ensure_capacity(&mut self, n: usize) {
        self.map.ensure_capacity(n);
    }

    /// Frees every variable, keeping the allocation — the reset a new episode
    /// of a long-running heuristic wants, since the instance has not changed.
    pub fn clear(&mut self) {
        self.map.clear();
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

    mod ledger {
        use super::*;
        use crate::problem::{MaxCut, MaxCutFlipNeighbor, MaxCutSolution};

        fn instance() -> MaxCut {
            MaxCut::from_edges([(0, 1, 1.0), (1, 2, 1.0), (2, 3, 1.0)])
        }

        fn flip(mc: &MaxCut, sol: &MaxCutSolution, i: usize) -> MaxCutFlipNeighbor {
            MaxCutFlipNeighbor::new(mc, sol, i)
        }

        /// The ledger must forbid and release exactly what the bare map does —
        /// it is storage plus a tenure, not a second policy.
        #[test]
        fn record_matches_the_bare_map_move_for_move() {
            let mc = instance();
            let sol = MaxCutSolution::new_from_assignment(&mc, vec![false; 4]);

            let mut ledger: TabuLedger<VecTabuMap> = TabuLedger::new((2, 9));
            let mut bare = VecTabuMap::new();
            let (mut a, mut b) = (rng(), rng());

            for (iteration, i) in [(0u64, 1usize), (4, 3), (9, 1)] {
                let mv = flip(&mc, &sol, i);
                ledger.record(&mv, iteration, &mut a);
                mv.add_to_tabu_map(&mut bare, iteration, (2, 9), &mut b);

                for probe in iteration..iteration + 12 {
                    for v in 0..4 {
                        assert_eq!(
                            ledger.allows(&flip(&mc, &sol, v), probe),
                            flip(&mc, &sol, v).is_move_enabled(&bare, probe),
                            "disagreement on vertex {v} at iteration {probe}"
                        );
                    }
                }
            }
        }

        /// `clear` frees every move but keeps the buffer, which is what a new
        /// episode of a long-running heuristic wants.
        #[test]
        fn clear_releases_every_move() {
            let mc = instance();
            let sol = MaxCutSolution::new_from_assignment(&mc, vec![false; 4]);
            let mut ledger: TabuLedger<VecTabuMap> = TabuLedger::new((50, 50));
            ledger.ensure_capacity(4);

            ledger.record(&flip(&mc, &sol, 2), 0, &mut rng());
            assert!(!ledger.allows(&flip(&mc, &sol, 2), 10));

            ledger.clear();
            assert!(ledger.allows(&flip(&mc, &sol, 2), 10));
            assert_eq!(ledger.tenure(), (50, 50), "clear must not touch the tenure");
        }

        /// A map handed in keeps its prohibitions, and can be taken back out.
        #[test]
        fn a_map_can_be_carried_in_and_out() {
            let mc = instance();
            let sol = MaxCutSolution::new_from_assignment(&mc, vec![false; 4]);
            let mut map = VecTabuMap::new();
            flip(&mc, &sol, 0).add_to_tabu_map(&mut map, 0, (7, 7), &mut rng());

            let mut ledger = TabuLedger::with_map(map, (1, 1));
            assert!(!ledger.allows(&flip(&mc, &sol, 0), 3));

            let taken = ledger.take_map();
            assert!(!flip(&mc, &sol, 0).is_move_enabled(&taken, 3));
            assert!(
                ledger.allows(&flip(&mc, &sol, 0), 3),
                "what is left behind must be an empty map"
            );
        }

        #[test]
        #[should_panic(expected = "Invalid tabu tenure range")]
        fn an_empty_tenure_range_panics_at_construction() {
            let _: TabuLedger<VecTabuMap> = TabuLedger::new((9, 2));
        }
    }
}
