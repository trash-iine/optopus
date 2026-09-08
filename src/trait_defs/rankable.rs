use super::Evaluate;

/// Is for comparing two solutions, or two moves, by quality.
///
/// Almost every heuristic uses this to decide which of two is better.
/// [`Rankable::is_better_than`] returns `true` if `self` is strictly better
/// than `other`.
///
/// **Nothing in this crate implements it by hand.** It is derived from
/// [`Evaluate`] by the blanket impl below, so implementing `Evaluate` on a
/// solution or a move is what makes it rankable. The two would otherwise state
/// the same fact twice, once as a number with a direction and once as a
/// comparison, and a problem that changed one and not the other would rank
/// against an objective it no longer optimizes.
///
/// A type that does *not* implement `Evaluate` may still implement this
/// directly, which is the escape hatch for an order that is not a comparison of
/// one number, a lexicographic objective for example. That costs `Evaluate`,
/// and with it every heuristic that computes with the objective rather than
/// only comparing solutions.
pub trait Rankable {
    fn is_better_than(&self, other: &Self) -> bool;
}

/// Better means a lower [`minimized`](super::Evaluable::minimized) value, which
/// is the objective with its direction already applied.
///
/// Strict, so two equal values rank as neither better, which is what
/// [`rank_cmp`] reports as `Equal` and what the tie rules downstream are
/// written against.
impl<T: Evaluate> Rankable for T {
    #[inline]
    fn is_better_than(&self, other: &Self) -> bool {
        self.evaluate().minimized() < other.evaluate().minimized()
    }
}

/// Total-order comparator derived from [`Rankable::is_better_than`], for use
/// with `max_by` / `min_by`.
///
/// Ties compare as `Equal`, so `iter.max_by(rank_cmp)` returns the last
/// tied-best element, the same element `filter_best(iter).pop()` yields.
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
/// For bit-vector solutions this is the standard Hamming distance, the
/// number of variables that differ. For other encodings any application-
/// meaningful integer dissimilarity measure works.
pub trait Distance {
    fn distance(&self, other: &Self) -> usize;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::trait_defs::Evaluable;

    /// The blanket impl derives the same order the direction states.
    #[test]
    fn better_means_a_lower_minimized_value() {
        struct Maximized(f64);
        impl Evaluate for Maximized {
            fn evaluate(&self) -> Evaluable<f64> {
                Evaluable::Maximize(self.0)
            }
        }
        struct Minimized(f64);
        impl Evaluate for Minimized {
            fn evaluate(&self) -> Evaluable<f64> {
                Evaluable::Minimize(self.0)
            }
        }

        assert!(Maximized(2.0).is_better_than(&Maximized(1.0)));
        assert!(!Maximized(1.0).is_better_than(&Maximized(2.0)));
        assert!(Minimized(1.0).is_better_than(&Minimized(2.0)));
        assert!(!Minimized(2.0).is_better_than(&Minimized(1.0)));

        // Strict, so equals rank as neither better and `rank_cmp` says Equal.
        assert!(!Maximized(1.0).is_better_than(&Maximized(1.0)));
        assert_eq!(
            rank_cmp(&Minimized(1.0), &Minimized(1.0)),
            std::cmp::Ordering::Equal
        );
    }

    /// A type that does not implement [`Evaluate`] may still implement
    /// [`Rankable`] by hand, which is the escape hatch for an order that is not
    /// a comparison of one number: a lexicographic objective, say. The blanket
    /// impl does not close it, here or in a downstream crate, because the crate
    /// that owns the type is the one that decides whether it is `Evaluate`.
    #[test]
    fn a_type_without_evaluate_can_still_rank_by_hand() {
        struct Lexicographic(u32, u32);
        impl Rankable for Lexicographic {
            fn is_better_than(&self, other: &Self) -> bool {
                (self.0, self.1) > (other.0, other.1)
            }
        }

        assert!(Lexicographic(1, 2).is_better_than(&Lexicographic(1, 1)));
        assert!(!Lexicographic(1, 1).is_better_than(&Lexicographic(2, 0)));
    }
}
