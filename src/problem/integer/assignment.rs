use super::problem::{IntVars, raw};
use crate::search_state::{Evaluate, ProblemTrait};

/// A problem whose solutions, of the problem's own type, assign an integer to
/// each variable.
///
/// Reading and writing one value is all the moves of this module need, and
/// the swap and the reversal have defaults built from those two. The solution
/// is yours, so it can keep whatever makes pricing a move cheap, such as a per
/// variable gain updated in [`assign`](Self::assign) and read back in
/// [`assign_delta`](Self::assign_delta).
///
/// [`IntegerProblem`](super::IntegerProblem) is the shortcut that brings its
/// own solution, and is one of these.
pub trait IntAssignment: ProblemTrait<Solution: Evaluate + Sync> + Sync {
    /// The variables, fixed for the life of the problem.
    fn domains(&self) -> &IntVars;

    /// The value of variable `i` in `sol`.
    fn get(sol: &Self::Solution, i: usize) -> i64;

    /// Sets variable `i` of `sol` to `value`, keeping the solution's
    /// objective, and anything else it caches, up to date.
    fn assign(&self, sol: &mut Self::Solution, i: usize, value: i64);

    /// How much the raw objective changes when variable `i` of `sol` is set to
    /// `value`, new minus old.
    ///
    /// The default copies the solution and calls [`assign`](Self::assign) on
    /// the copy, and warns once at runtime that it does.
    fn assign_delta(&self, sol: &Self::Solution, i: usize, value: i64) -> f64 {
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
    fn slot_delta(&self, sol: &Self::Solution, slot: usize, i: usize, value: i64) -> f64 {
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

    /// How much the raw objective changes when `i` and `j` are exchanged. The
    /// default copies the solution, and warns once at runtime that it does.
    fn assign_swap_delta(&self, sol: &Self::Solution, i: usize, j: usize) -> f64 {
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

    /// How much the raw objective changes when `i..=j` is reversed. The
    /// default copies the solution, and warns once at runtime that it does.
    fn assign_reverse_delta(&self, sol: &Self::Solution, i: usize, j: usize) -> f64 {
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
fn copied_delta<S: Clone + Evaluate>(sol: &S, edit: impl FnOnce(&mut S)) -> f64 {
    let mut copy = sol.clone();
    edit(&mut copy);
    raw(copy.evaluate()) - raw(sol.evaluate())
}
