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
/// # Deriving it
///
/// `#[derive(Rankable)]` writes the impl, and every built-in solution and move
/// uses it. With no `rank` attribute the order is the objective, compared by
/// [`better_by_objective`], so the type also implements `Evaluate`.
///
/// ```
/// use optopus::prelude::*;
///
/// #[derive(Rankable)]
/// struct Tour(f64);
///
/// impl Evaluate for Tour {
///     fn evaluate(&self) -> Evaluable<f64> {
///         Evaluable::Minimize(self.0)
///     }
/// }
///
/// assert!(Tour(1.0).is_better_than(&Tour(2.0)));
/// ```
///
/// An order of keys compared one after another, the first that differs
/// deciding, is named with `rank` attributes. `min` ranks the lower key better
/// and `max` the higher one. On fields they are compared in the order the
/// fields are declared.
///
/// ```
/// use optopus::prelude::*;
///
/// #[derive(Rankable)]
/// struct Packing {
///     #[rank(min)]
///     overfull_bins: u32,
///     #[rank(min)]
///     bins: u32,
/// }
///
/// let feasible = Packing { overfull_bins: 0, bins: 9 };
/// let tight = Packing { overfull_bins: 1, bins: 7 };
/// assert!(feasible.is_better_than(&tight));
/// ```
///
/// On the type they are compared in the order written, and a key is a field
/// or a method that takes only `&self`.
///
/// ```
/// use optopus::prelude::*;
///
/// #[derive(Rankable)]
/// #[rank(min = violation(), min = cost)]
/// struct Routes {
///     overload: f64,
///     lateness: f64,
///     cost: f64,
/// }
///
/// impl Routes {
///     fn violation(&self) -> f64 {
///         self.overload + self.lateness
///     }
/// }
///
/// let on_time = Routes { overload: 0.0, lateness: 0.0, cost: 120.0 };
/// let late = Routes { overload: 0.0, lateness: 3.0, cost: 80.0 };
/// assert!(on_time.is_better_than(&late));
/// ```
///
/// The keys are named in one of the two places, never both, so the order is
/// written once.
///
/// ```compile_fail
/// use optopus::prelude::*;
///
/// #[derive(Rankable)]
/// #[rank(min = cost)]
/// struct Twice {
///     #[rank(min)]
///     cost: f64,
/// }
/// ```
///
/// A key reads `self` alone, so an expression that is not a field or a method
/// call with no arguments is refused.
///
/// ```compile_fail
/// use optopus::prelude::*;
///
/// #[derive(Rankable)]
/// #[rank(min = cost * 2.0)]
/// struct Scaled {
///     cost: f64,
/// }
/// ```
///
/// A move ranks the same way, by its own fields. A problem whose solution is
/// ranked by keys gives its moves the keys of the solution each would lead
/// to, since a move ranked by its delta follows the objective and not that
/// order.
pub trait Rankable {
    fn is_better_than(&self, other: &Self) -> bool;
}

/// Whether `a` is better than `b` by their objectives, which is a lower
/// [`minimized`](super::Evaluable::minimized) value, the objective with its
/// direction already applied.
///
/// Strict, so two equal values rank as neither better, which is what
/// [`rank_cmp`] reports as `Equal` and what the tie rules downstream are
/// written against. `#[derive(Rankable)]` with no `rank` attribute implements
/// [`Rankable`] with this, and an impl written by hand can call it too.
#[inline]
pub fn better_by_objective<T: Evaluate>(a: &T, b: &T) -> bool {
    a.evaluate().minimized() < b.evaluate().minimized()
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
///
/// # Deriving it
///
/// Where the distance is the Hamming distance over one field, a `Vec` or
/// anything else that derefs to a slice, `#[derive(Distance)]` writes the impl
/// and the field is marked `#[distance(hamming)]`. The other fields are not
/// read.
///
/// ```
/// use optopus::prelude::*;
///
/// #[derive(Distance)]
/// struct Coloring {
///     #[distance(hamming)]
///     colors: Vec<usize>,
///     colors_used: usize,
/// }
///
/// let a = Coloring { colors: vec![0, 1, 2], colors_used: 3 };
/// let b = Coloring { colors: vec![0, 2, 2], colors_used: 2 };
/// assert_eq!(a.distance(&b), 1);
/// ```
///
/// The kind of distance is named, and `hamming` is the only one.
///
/// ```compile_fail
/// use optopus::prelude::*;
///
/// #[derive(Distance)]
/// struct Tour {
///     #[distance(euclidean)]
///     cities: Vec<usize>,
/// }
/// ```
pub trait Distance {
    fn distance(&self, other: &Self) -> usize;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::trait_defs::{Evaluable, Rankable};

    /// `filter_best` keeps every tie, and `rank_cmp`'s `max_by` keeps the last
    /// of them. `LocalSearch` and `TabuSearch` both rely on the two agreeing,
    /// and on a plateau -- which is where MaxCut spends most of a run -- almost
    /// every candidate is a tie, so a disagreement is not a corner case.
    #[test]
    fn filter_best_keeps_every_tie_and_agrees_with_max_by() {
        #[derive(Clone, Copy, Debug, Rankable, Evaluate)]
        struct Cut(#[objective(maximize)] f64);

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

    /// The derive with no attribute ranks in the order the direction states.
    #[test]
    fn better_means_a_lower_minimized_value() {
        #[derive(Rankable, Evaluate)]
        struct Maximized(#[objective(maximize)] f64);
        #[derive(Rankable, Evaluate)]
        struct Minimized(#[objective(minimize)] f64);

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
