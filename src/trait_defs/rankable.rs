/// Is for comparing two solutions by quality.
///
/// This trait is used by almost heuristic algorithms to determine which solution is better.
///
/// [`Rankable::is_better_than`] returns `true` if `self` is strictly better than `other`.
pub trait Rankable {
    fn is_better_than(&self, other: &Self) -> bool;
}

/// Total-order comparator derived from [`Rankable::is_better_than`], for use
/// with `max_by` / `min_by`.
///
/// Ties compare as `Equal`, so `iter.max_by(rank_cmp)` returns the **last**
/// tied-best element — the same element `filter_best(iter).pop()` yields.
#[inline]
pub fn rank_cmp<R: Rankable>(a: &R, b: &R) -> std::cmp::Ordering {
    if a.is_better_than(b) {
        std::cmp::Ordering::Greater
    } else if b.is_better_than(a) {
        std::cmp::Ordering::Less
    } else {
        std::cmp::Ordering::Equal
    }
}

/// Returns all elements that are tied for the best rank among the items yielded by `iter`.
///
/// If `iter` is empty, returns an empty `Vec`.
pub fn filter_best<R: Rankable, T: Iterator<Item = R>>(iter: T) -> Vec<R> {
    const RESERVE_CAPACITY: usize = 16;
    let mut best_list: Vec<R> = Vec::with_capacity(RESERVE_CAPACITY);
    for r in iter {
        if best_list.is_empty() {
            best_list.push(r);
        } else {
            let sample = &best_list[0];
            if r.is_better_than(sample) {
                best_list.clear();
                best_list.push(r);
            } else if !sample.is_better_than(&r) {
                best_list.push(r);
            }
        }
    }

    best_list
}

/// Hamming-style distance between two solutions.
///
/// Used by parent-selection strategies that promote population diversity
/// (e.g. [`crate::heuristic::ParentSelection::DistantTopK`]).
///
/// For bit-vector solutions this is the standard Hamming distance — the
/// number of variables that differ. For other encodings any application-
/// meaningful integer dissimilarity measure works.
pub trait Distance {
    fn distance(&self, other: &Self) -> usize;
}
