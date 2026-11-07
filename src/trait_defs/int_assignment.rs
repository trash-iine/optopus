use super::{Evaluate, ProblemTrait, RankKeys};
use crate::building_blocks::representation::IntVars;

/// What a change to an [`IntAssignment`] solution does, as the moves of this
/// module price it.
///
/// A move needs two things from it. The change in the solution's
/// [`RankKeys`], which is what the move is compared by, and the change in the
/// raw objective, which is what a search that weighs how much worse a move is
/// reads through [`Evaluate`].
///
/// A solution that ranks by its objective alone has one key, the objective
/// itself, so a plain `f64`, the change in the raw objective, is the delta of
/// `IntAssignment<1>`. A solution ranked by several keys, a violation first
/// and a cost second for one, prices a move as a [`KeyDelta`].
pub trait AssignDelta<const N: usize>: Copy + Send + Sync {
    /// The change in the raw objective, new minus old.
    fn objective(&self) -> f64;

    /// The change in each rank key, new minus old, with the keys oriented as
    /// [`RankKeys`] reports them, lower being better. `sign` is `1.0` on a
    /// minimized objective and `-1.0` on a maximized one.
    fn keys(&self, sign: f64) -> [f64; N];

    /// The change from `before` to `after`, read off the two solutions.
    fn between<S: Evaluate + RankKeys<N>>(before: &S, after: &S) -> Self;
}

/// The change in the raw objective, the delta of a solution whose one rank
/// key is its objective.
impl AssignDelta<1> for f64 {
    #[inline]
    fn objective(&self) -> f64 {
        *self
    }

    #[inline]
    fn keys(&self, sign: f64) -> [f64; 1] {
        [sign * self]
    }

    fn between<S: Evaluate + RankKeys<1>>(before: &S, after: &S) -> Self {
        after.evaluate().raw() - before.evaluate().raw()
    }
}

/// The change a move makes to a solution ranked by `N` keys.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct KeyDelta<const N: usize> {
    /// The change in each rank key, new minus old, oriented as [`RankKeys`]
    /// reports them, lower being better.
    pub keys: [f64; N],
    /// The change in the raw objective, new minus old.
    pub objective: f64,
}

impl<const N: usize> AssignDelta<N> for KeyDelta<N> {
    #[inline]
    fn objective(&self) -> f64 {
        self.objective
    }

    #[inline]
    fn keys(&self, _sign: f64) -> [f64; N] {
        self.keys
    }

    fn between<S: Evaluate + RankKeys<N>>(before: &S, after: &S) -> Self {
        let (b, a) = (before.rank_keys(), after.rank_keys());
        Self {
            keys: std::array::from_fn(|k| a[k] - b[k]),
            objective: after.evaluate().raw() - before.evaluate().raw(),
        }
    }
}

/// A problem whose solutions, of the problem's own type, assign an integer to
/// each variable.
///
/// Reading and writing one value is all the moves of this module need, and
/// the swap and the reversal have defaults built from those two. The solution
/// is yours, so it can keep whatever makes pricing a move cheap, such as a per
/// variable gain updated in [`assign`](Self::assign) and read back in
/// [`assign_delta`](Self::assign_delta).
///
/// The moves compare by the solution's `N` [`RankKeys`], and add their
/// [`Delta`](Self::Delta) to them. A solution that ranks by its objective,
/// with `#[derive(Rankable)]` and no `rankable` attribute, has one key, and
/// its delta is the `f64` change in the objective. A solution ranked by a
/// violation first and a cost second has two, and prices each move as a
/// [`KeyDelta<2>`](KeyDelta).
///
/// [`IntegerProblem`](crate::problem::IntegerProblem) is the shortcut that
/// brings its own solution, and is one of these.
pub trait IntAssignment<const N: usize = 1>:
    ProblemTrait<Solution: Evaluate + RankKeys<N> + Sync> + Sync
{
    /// What a change to a solution does to its keys and objective, `f64` when
    /// the solution ranks by its objective alone.
    type Delta: AssignDelta<N>;

    /// The variables, fixed for the life of the problem.
    fn domains(&self) -> &IntVars;

    /// The value of variable `i` in `sol`.
    fn get(sol: &Self::Solution, i: usize) -> i64;

    /// Sets variable `i` of `sol` to `value`, keeping the solution's
    /// objective, and anything else it caches, up to date.
    fn assign(&self, sol: &mut Self::Solution, i: usize, value: i64);

    /// What setting variable `i` of `sol` to `value` changes.
    ///
    /// The default copies the solution and calls [`assign`](Self::assign) on
    /// the copy, and warns once at runtime that it does.
    fn assign_delta(&self, sol: &Self::Solution, i: usize, value: i64) -> Self::Delta {
        static WARNED: std::sync::OnceLock<()> = std::sync::OnceLock::new();
        WARNED.get_or_init(|| {
            tracing::warn!(
                problem_type = std::any::type_name::<Self>(),
                "Using the default implementation of IntAssignment::assign_delta, \
                 which copies the solution for every candidate move."
            );
        });
        copied_delta(sol, |copy| self.assign(copy, i, value))
    }

    /// [`assign_delta`](Self::assign_delta) for the change that sits at
    /// `slot` when the changes of every variable are laid out in index order,
    /// each variable's other values ascending. A solution that keeps a table
    /// of deltas in that layout is read here, and the default ignores `slot`.
    #[inline]
    fn slot_delta(&self, sol: &Self::Solution, slot: usize, i: usize, value: i64) -> Self::Delta {
        let _ = slot;
        self.assign_delta(sol, i, value)
    }

    /// Exchanges the values of variables `i` and `j`. The default is two
    /// [`assign`](Self::assign) calls.
    fn assign_swap(&self, sol: &mut Self::Solution, i: usize, j: usize) {
        let (a, b) = (Self::get(sol, i), Self::get(sol, j));
        if a != b {
            self.assign(sol, i, b);
            self.assign(sol, j, a);
        }
    }

    /// What exchanging `i` and `j` changes. The default copies the solution,
    /// and warns once at runtime that it does.
    fn assign_swap_delta(&self, sol: &Self::Solution, i: usize, j: usize) -> Self::Delta {
        static WARNED: std::sync::OnceLock<()> = std::sync::OnceLock::new();
        WARNED.get_or_init(|| {
            tracing::warn!(
                problem_type = std::any::type_name::<Self>(),
                "Using the default implementation of IntAssignment::assign_swap_delta, \
                 which copies the solution for every candidate move."
            );
        });
        copied_delta(sol, |copy| self.assign_swap(copy, i, j))
    }

    /// Reverses the values of variables `i..=j`. The default swaps from both
    /// ends inwards.
    fn assign_reverse(&self, sol: &mut Self::Solution, i: usize, j: usize) {
        let (mut a, mut b) = (i, j);
        while a < b {
            self.assign_swap(sol, a, b);
            a += 1;
            b -= 1;
        }
    }

    /// What reversing `i..=j` changes. The default copies the solution, and
    /// warns once at runtime that it does.
    fn assign_reverse_delta(&self, sol: &Self::Solution, i: usize, j: usize) -> Self::Delta {
        static WARNED: std::sync::OnceLock<()> = std::sync::OnceLock::new();
        WARNED.get_or_init(|| {
            tracing::warn!(
                problem_type = std::any::type_name::<Self>(),
                "Using the default implementation of IntAssignment::assign_reverse_delta, \
                 which copies the solution for every candidate move."
            );
        });
        copied_delta(sol, |copy| self.assign_reverse(copy, i, j))
    }
}

/// The change `edit` makes to `sol`, found on a copy. What the defaults above
/// price a move by.
fn copied_delta<const N: usize, S, D>(sol: &S, edit: impl FnOnce(&mut S)) -> D
where
    S: Clone + Evaluate + RankKeys<N>,
    D: AssignDelta<N>,
{
    let mut copy = sol.clone();
    edit(&mut copy);
    D::between(sol, &copy)
}
