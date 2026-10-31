use super::Evaluate;

/// Is for comparing two solutions, or two moves, by quality.
///
/// Every heuristic decides which of two is better through this and nothing
/// else. [`Rankable::is_better_than`] returns `true` if `self` is strictly
/// better than `other`.
///
/// It is separate from [`Evaluate`] because an order need not be a number. A
/// problem whose constraints cannot be priced, or whose solutions are compared
/// violations first and cost second, states its order here, and every
/// heuristic that only compares solutions then runs on it. `Evaluate` is the
/// number a heuristic reads where it needs to know how much worse a candidate
/// is, not only whether it is worse, and it may be a surrogate that the order
/// does not follow.
///
/// Where the order is the objective, [`rank_by_objective!`](crate::rank_by_objective)
/// writes this impl from the `Evaluate` one in one line, and the built-in
/// problems use it for every solution and move.
pub trait Rankable {
    fn is_better_than(&self, other: &Self) -> bool;
}

/// Whether `a` is better than `b` by their objectives, which is a lower
/// [`minimized`](super::Evaluable::minimized) value, the objective with its
/// direction already applied.
///
/// Strict, so two equal values rank as neither better, which is what
/// [`rank_cmp`] reports as `Equal` and what the tie rules downstream are
/// written against. [`rank_by_objective!`](crate::rank_by_objective) implements
/// [`Rankable`] with this.
#[inline]
pub fn better_by_objective<T: Evaluate>(a: &T, b: &T) -> bool {
    a.evaluate().minimized() < b.evaluate().minimized()
}

/// Implements [`Rankable`] for each type given by comparing objectives, with
/// [`better_by_objective`].
///
/// A solution or a move of your own whose order is its objective calls
/// `optopus::rank_by_objective!(MySolution, MyMove);` beside its
/// [`Evaluate`] impl. One whose order is something else, violations before
/// cost for example, implements [`Rankable`] by hand instead.
///
/// ```
/// use optopus::prelude::*;
///
/// struct Cost(f64);
/// impl Evaluate for Cost {
///     fn evaluate(&self) -> Evaluable<f64> {
///         Evaluable::Minimize(self.0)
///     }
/// }
/// optopus::rank_by_objective!(Cost);
///
/// assert!(Cost(1.0).is_better_than(&Cost(2.0)));
/// ```
#[macro_export]
macro_rules! rank_by_objective {
    ($($ty:ty),* $(,)?) => {$(
        impl $crate::trait_defs::Rankable for $ty {
            #[inline]
            fn is_better_than(&self, other: &Self) -> bool {
                $crate::trait_defs::better_by_objective(self, other)
            }
        }
    )*};
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

    /// `filter_best` keeps every tie, and `rank_cmp`'s `max_by` keeps the last
    /// of them. `LocalSearch` and `TabuSearch` both rely on the two agreeing,
    /// and on a plateau -- which is where MaxCut spends most of a run -- almost
    /// every candidate is a tie, so a disagreement is not a corner case.
    #[test]
    fn filter_best_keeps_every_tie_and_agrees_with_max_by() {
        #[derive(Clone, Copy, Debug)]
        struct Cut(f64);
        impl Evaluate for Cut {
            fn evaluate(&self) -> Evaluable<f64> {
                Evaluable::Maximize(self.0)
            }
        }
        crate::rank_by_objective!(Cut);

        let items = [Cut(1.0), Cut(3.0), Cut(2.0), Cut(3.0), Cut(3.0)];

        let best = filter_best(items.iter().copied());
        assert_eq!(best.len(), 3, "all three 3.0s are tied for best");
        assert!(best.iter().all(|c| c.0 == 3.0));

        let last_tied = items.iter().copied().max_by(rank_cmp).unwrap();
        assert_eq!(last_tied.0, best.last().unwrap().0);

        // A better element arriving later clears the ties before it.
        let improving = filter_best([Cut(1.0), Cut(1.0), Cut(2.0)].into_iter());
        assert_eq!(improving.len(), 1);
        assert_eq!(improving[0].0, 2.0);

        assert!(filter_best(std::iter::empty::<Cut>()).is_empty());
    }

    /// `rank_by_objective!` derives the same order the direction states.
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
        crate::rank_by_objective!(Maximized, Minimized);

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

    /// A type ranks by an order of its own and still reports a number. Here
    /// violations come first and cost second, while `evaluate` is a penalized
    /// surrogate that disagrees with that order, and the order is what
    /// `rank_cmp` and `filter_best` follow.
    #[test]
    fn an_order_of_its_own_beside_a_surrogate_number() {
        #[derive(Clone, Copy, Debug)]
        struct Constrained {
            violations: u32,
            cost: f64,
        }
        impl Rankable for Constrained {
            fn is_better_than(&self, other: &Self) -> bool {
                (self.violations, self.cost) < (other.violations, other.cost)
            }
        }
        impl Evaluate for Constrained {
            fn evaluate(&self) -> Evaluable<f64> {
                Evaluable::Minimize(self.cost + 10.0 * f64::from(self.violations))
            }
        }

        let feasible = Constrained {
            violations: 0,
            cost: 100.0,
        };
        let cheap = Constrained {
            violations: 1,
            cost: 1.0,
        };
        assert!(feasible.is_better_than(&cheap));
        assert!(
            !better_by_objective(&feasible, &cheap),
            "the surrogate disagrees"
        );

        let best = filter_best([cheap, feasible, cheap].into_iter());
        assert_eq!(best.len(), 1);
        assert_eq!(best[0].violations, 0);
    }

    /// An order with no number at all ranks by hand and needs no `Evaluate`.
    #[test]
    fn an_order_without_a_number() {
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
