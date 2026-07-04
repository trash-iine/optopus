//! Shared tabu-map helpers for variable-indexed moves.
//!
//! The flip/swap moves of the binary-variable problems all use the same tabu
//! policy: a `HashMap` from variable index to expiry iteration. These helpers
//! hold that policy in one place; each move's [`EnabledTabu`](crate::trait_defs::EnabledTabu)
//! impl delegates to them (once per variable the move touches).

use rand::Rng;
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
    rng: &mut rand::rngs::SmallRng,
) {
    let tabu_duration = rng.random_range(tabu_tenure.0..=tabu_tenure.1);
    tabu_map.insert(i, iteration + tabu_duration);
}
