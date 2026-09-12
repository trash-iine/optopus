//! The destroy and repair operators of ruin-and-recreate, over any
//! [`Ruinable`] problem.
//!
//! Five free functions, three that ruin and two that recreate, plus the
//! two-best-insertions scan both repairs select with. They are Ropke &
//! Pisinger's operator bank, and each reads the problem only through
//! [`Ruinable`]: what an element is, which containers it may go in, and what a
//! placement costs.
//!
//! Free functions rather than trait methods with defaults, because there is no
//! reason for a problem to override one. The operators decide which elements to
//! move, the problem decides what moving them costs, and that is exactly what
//! `Ruinable` asks for.
//!
//! Every one of them is generic over `P`, not `dyn Ruinable`. The inner loop of
//! greedy insertion is O(k · buckets · places) and regret-2 is O(k² · buckets ·
//! places); routing `insertion_cost` through a vtable there would be paid on
//! every one of those.

use rand::Rng;
use rand::rngs::SmallRng;
use rand::seq::SliceRandom;

use crate::trait_defs::Ruinable;

/// A placement and what it costs: `(cost, bucket, place)`.
type Placement = (f64, usize, usize);

const NOWHERE: Placement = (f64::INFINITY, 0, 0);

/// Removes `k` elements drawn uniformly, returning them.
pub fn random_removal<P: Ruinable>(
    prob: &P,
    partial: &mut P::Partial,
    k: usize,
    rng: &mut SmallRng,
    scratch: &mut Vec<P::Element>,
) -> Vec<P::Element> {
    prob.elements(partial, scratch);
    scratch.shuffle(rng);
    scratch.truncate(k);
    let removed = scratch.clone();
    prob.remove_all(partial, &removed);
    removed
}

/// Removes the `k` elements with the largest removal gain, those whose
/// current placement costs the most.
///
/// The gains are read once, against the incumbent, and the top `k` are
/// removed together. Removing one at a time and re-scoring would be the
/// textbook operator; it is also `k` times the work, and two adjacent elements
/// both chosen here have individual gains that do not sum to their joint gain.
/// The approximation is deliberate.
pub fn worst_removal<P: Ruinable>(
    prob: &P,
    partial: &mut P::Partial,
    k: usize,
    scratch: &mut Vec<P::Element>,
) -> Vec<P::Element> {
    prob.elements(partial, scratch);
    let mut scored: Vec<(f64, P::Element)> = scratch
        .iter()
        .map(|&e| (prob.removal_gain(partial, e), e))
        .collect();
    scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
    let removed: Vec<P::Element> = scored.into_iter().take(k).map(|(_, e)| e).collect();
    prob.remove_all(partial, &removed);
    removed
}

/// Removes the `k` elements most related to one random seed element (Shaw).
///
/// Taking a *cluster* is the point: a scattered removal can usually be undone
/// by putting everything back where it was, while a cluster leaves room for a
/// genuinely different arrangement.
pub fn shaw_removal<P: Ruinable>(
    prob: &P,
    partial: &mut P::Partial,
    k: usize,
    rng: &mut SmallRng,
    scratch: &mut Vec<P::Element>,
) -> Vec<P::Element> {
    prob.elements(partial, scratch);
    if scratch.is_empty() {
        return Vec::new();
    }
    let seed = scratch[rng.random_range(0..scratch.len())];
    let mut related: Vec<(f64, P::Element)> = scratch
        .iter()
        .map(|&e| (prob.relatedness(seed, e), e))
        .collect();
    related.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    let removed: Vec<P::Element> = related.into_iter().take(k).map(|(_, e)| e).collect();
    prob.remove_all(partial, &removed);
    removed
}

/// The cheapest placement of `element`, and the cheapest one in a different
/// container.
///
/// The two are taken across containers, not across places, because that is
/// what regret-k is defined on (Ropke & Pisinger): the gap that matters is how
/// much worse `element` gets once its best container is taken. The
/// second-cheapest *place* is almost always the one next door in the same
/// container, a gap of nearly zero for every element, which would leave
/// [`regret2_insertion`] indistinguishable from [`greedy_insertion`].
///
/// With a single container the second entry stays infinite; the caller reads
/// that as unbounded regret.
pub fn best_two_insertions<P: Ruinable>(
    prob: &P,
    partial: &P::Partial,
    element: P::Element,
) -> (Placement, Placement) {
    let mut b1 = NOWHERE;
    let mut b2 = NOWHERE;
    for bucket in 0..prob.num_buckets(partial) {
        let mut best_here = (f64::INFINITY, bucket, 0usize);
        for place in 0..prob.num_places(partial, bucket) {
            let cost = prob.insertion_cost(partial, bucket, place, element);
            if cost < best_here.0 {
                best_here = (cost, bucket, place);
            }
        }
        if best_here.0 < b1.0 {
            b2 = b1;
            b1 = best_here;
        } else if best_here.0 < b2.0 {
            b2 = best_here;
        }
    }
    (b1, b2)
}

/// Re-inserts every element at its cheapest placement, in random order.
///
/// The order is randomized rather than left as the destroy operator produced
/// it: worst- and Shaw-removal both return a *sorted* pool, and inserting in
/// that order would make the first element's choice systematically the freest.
pub fn greedy_insertion<P: Ruinable>(
    prob: &P,
    partial: &mut P::Partial,
    removed: Vec<P::Element>,
    rng: &mut SmallRng,
) {
    let mut removed = removed;
    removed.shuffle(rng);
    for element in removed {
        let (best, _) = best_two_insertions(prob, partial, element);
        let (_, bucket, place) = best;
        prob.insert(partial, bucket, place, element);
    }
}

/// Re-inserts elements largest-regret-first: the element that stands to lose
/// the most once its best container fills up goes in while it still can.
pub fn regret2_insertion<P: Ruinable>(
    prob: &P,
    partial: &mut P::Partial,
    removed: Vec<P::Element>,
) {
    let mut pool = removed;
    while !pool.is_empty() {
        let mut best_regret = f64::NEG_INFINITY;
        let mut best_idx = 0usize;
        let mut best_place = (0usize, 0usize);
        for (idx, &element) in pool.iter().enumerate() {
            let (b1, b2) = best_two_insertions(prob, partial, element);
            let regret = if b2.0.is_finite() {
                b2.0 - b1.0
            } else {
                f64::INFINITY
            };
            if regret > best_regret {
                best_regret = regret;
                best_idx = idx;
                best_place = (b1.1, b1.2);
            }
        }
        let element = pool.swap_remove(best_idx);
        prob.insert(partial, best_place.0, best_place.1, element);
    }
}
