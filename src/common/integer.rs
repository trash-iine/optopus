//! Bounded integer variables, the ranges every integer problem and every node
//! of [`BranchAndBound`](crate::heuristic::BranchAndBound) are stated in.

use std::borrow::Cow;

use crate::error::OptError;
use crate::trait_defs::{BranchSpace, Branchable, ProblemReduction};
use rand::Rng;

/// An integer variable that takes any value in `lower..=upper`.
///
/// A binary variable is the range `0..=1`, see [`IntVar::binary`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct IntVar {
    lower: i64,
    upper: i64,
}

impl IntVar {
    /// A variable ranging over `lower..=upper`.
    ///
    /// # Panics
    ///
    /// If `lower > upper`, or if the range covers every `i64`, whose size does
    /// not fit in a `u64`.
    pub fn new(lower: i64, upper: i64) -> Self {
        assert!(
            lower <= upper,
            "IntVar needs lower <= upper, got [{lower}, {upper}]"
        );
        assert!(
            (i128::from(upper) - i128::from(lower)) < i128::from(u64::MAX),
            "IntVar range [{lower}, {upper}] is too wide"
        );
        Self { lower, upper }
    }

    /// A variable ranging over `0..=1`.
    pub fn binary() -> Self {
        Self::new(0, 1)
    }

    /// The smallest value the variable takes.
    pub fn lower(&self) -> i64 {
        self.lower
    }

    /// The largest value the variable takes.
    pub fn upper(&self) -> i64 {
        self.upper
    }

    /// Whether `value` lies in `lower..=upper`.
    pub fn contains(&self, value: i64) -> bool {
        (self.lower..=self.upper).contains(&value)
    }

    /// How many values other than the current one the variable can change to,
    /// which is `upper - lower` whatever the current value is.
    pub fn num_changes(&self) -> u64 {
        self.upper.abs_diff(self.lower)
    }

    /// The `k`th value in ascending order other than `current`, for `k` below
    /// [`num_changes`](Self::num_changes). The inverse of
    /// [`other_index`](Self::other_index).
    #[inline]
    pub(crate) fn nth_other(&self, k: u64, current: i64) -> i64 {
        let value = self.lower.wrapping_add_unsigned(k);
        if value >= current { value + 1 } else { value }
    }

    /// Where `value` sits among the values other than `current`, in ascending
    /// order. The inverse of [`nth_other`](Self::nth_other).
    #[inline]
    pub(crate) fn other_index(&self, current: i64, value: i64) -> u64 {
        let k = value.abs_diff(self.lower);
        if value > current { k - 1 } else { k }
    }
}

/// Integer variables indexed `0..n`, those of an
/// [`IntegerProblem`](crate::problem::IntegerProblem) or a
/// [`FormulaProblem`](crate::problem::FormulaProblem), and the ranges a node
/// of [`BranchAndBound`](crate::heuristic::BranchAndBound) is stated in,
/// whatever the problem.
///
/// Dereferences to `[IntVar]`. It also keeps the running total of
/// [`IntVar::num_changes`], which is what lets
/// [`IntChangeNeighbor`](crate::problem::IntChangeNeighbor) draw a uniformly
/// random move in `O(log n)`.
#[derive(Clone, Debug)]
pub struct IntVars {
    vars: Vec<IntVar>,
    cum_changes: Vec<u64>,
    all_different: bool,
    /// The number of changes of every variable when all of them have the
    /// same range, so that moving values between variables always keeps them
    /// in range and a change is located by arithmetic alone.
    common_width: Option<u64>,
}

impl IntVars {
    /// # Panics
    ///
    /// If the total number of single variable changes overflows a `u64`.
    pub fn new(vars: Vec<IntVar>) -> Self {
        let mut total = 0u64;
        let cum_changes = vars
            .iter()
            .map(|v| {
                total = total
                    .checked_add(v.num_changes())
                    .expect("the total number of value changes overflows u64");
                total
            })
            .collect();
        let common_width = if vars.windows(2).all(|w| w[0] == w[1]) {
            vars.first().map(IntVar::num_changes)
        } else {
            None
        };
        Self {
            vars,
            cum_changes,
            all_different: false,
            common_width,
        }
    }

    /// `n` variables in `0..=n-1` that take different values, so that a
    /// solution is a permutation of `0..n`. Variable `p` is what sits at
    /// position `p`, the city visited `p`th in a tour for instance.
    ///
    /// A random solution is a uniformly random permutation. Only moves that
    /// keep a permutation apply,
    /// [`IntSwapNeighbor`](crate::problem::IntSwapNeighbor) and
    /// [`IntReverseNeighbor`](crate::problem::IntReverseNeighbor), and
    /// [`IntChangeNeighbor`](crate::problem::IntChangeNeighbor) has no moves
    /// here.
    pub fn permutation(n: usize) -> Self {
        let top = n.saturating_sub(1) as i64;
        Self {
            all_different: true,
            ..Self::new(vec![IntVar::new(0, top); n])
        }
    }

    /// Whether the variables were built by [`permutation`](Self::permutation).
    pub fn is_permutation(&self) -> bool {
        self.all_different
    }

    /// Whether every variable has the same range.
    pub(crate) fn same_range(&self) -> bool {
        self.common_width.is_some() || self.vars.is_empty()
    }

    /// `lower + upper` when every variable ranges over the same two values,
    /// so that the one change of a variable is to that minus its value.
    pub(crate) fn flip_sum(&self) -> Option<i64> {
        (self.common_width == Some(1)).then(|| self.vars[0].lower() + self.vars[0].upper())
    }

    /// How many single variable changes there are in total, the size of every
    /// solution's neighborhood.
    pub fn total_changes(&self) -> u64 {
        self.cum_changes.last().copied().unwrap_or(0)
    }

    /// The variable holding the `r`th change when the changes of every
    /// variable are laid out in index order. `r` must be below
    /// [`total_changes`](Self::total_changes).
    #[inline]
    pub(crate) fn var_of_change(&self, r: u64) -> usize {
        match self.common_width {
            Some(w) => (r / w) as usize,
            None => self.cum_changes.partition_point(|&c| c <= r),
        }
    }

    /// Checks that `values` assigns every variable a value in its range, and
    /// a permutation when the variables are one.
    pub(crate) fn check(&self, values: &[i64]) -> Result<(), OptError> {
        if values.len() != self.len() {
            return Err(OptError::Config(format!(
                "{} values given for {} variables",
                values.len(),
                self.len()
            )));
        }
        if let Some((i, (v, x))) = self
            .iter()
            .zip(values)
            .enumerate()
            .find(|(_, (v, x))| !v.contains(**x))
        {
            return Err(OptError::Config(format!(
                "value {x} of variable {i} lies outside [{}, {}]",
                v.lower(),
                v.upper()
            )));
        }
        if self.is_permutation() {
            let mut seen = vec![false; values.len()];
            if let Some(x) = values
                .iter()
                .find(|&&x| std::mem::replace(&mut seen[x as usize], true))
            {
                return Err(OptError::Config(format!(
                    "value {x} appears twice in a permutation"
                )));
            }
        }
        Ok(())
    }

    /// Every variable drawn uniformly from its range, or a uniformly random
    /// permutation when the variables are one.
    pub(crate) fn random_values(&self, rng: &mut impl Rng) -> Vec<i64> {
        if self.is_permutation() {
            use rand::seq::SliceRandom;
            let mut values: Vec<i64> = (0..self.len() as i64).collect();
            values.shuffle(rng);
            return values;
        }
        self.iter()
            .map(|v| rng.random_range(v.lower()..=v.upper()))
            .collect()
    }

    /// Where the changes of variable `var` start when every variable's
    /// changes are laid out in index order.
    #[inline]
    pub(crate) fn row_start(&self, var: usize) -> usize {
        match self.common_width {
            Some(w) => var * w as usize,
            None => (self.cum_changes[var] - self.vars[var].num_changes()) as usize,
        }
    }

    /// Where the change of `var` from `current` to `value` sits among the
    /// changes laid out by [`row_start`](Self::row_start).
    #[inline]
    pub(crate) fn change_slot(&self, var: usize, current: i64, value: i64) -> usize {
        self.row_start(var) + self.vars[var].other_index(current, value) as usize
    }
}

impl std::ops::Deref for IntVars {
    type Target = [IntVar];
    fn deref(&self) -> &[IntVar] {
        &self.vars
    }
}

impl FromIterator<IntVar> for IntVars {
    fn from_iter<T: IntoIterator<Item = IntVar>>(iter: T) -> Self {
        Self::new(iter.into_iter().collect())
    }
}

/// A node of the search as a map from the whole problem onto the same problem
/// over narrower ranges, so that a heuristic crosses into it and back through
/// [`SearchState::open_reduction`](crate::search_state::SearchState::open_reduction)
/// and [`close_reduction`](crate::search_state::SearchState::close_reduction).
pub struct DomainRestriction<P> {
    target: P,
}

impl<P: Branchable> ProblemReduction for DomainRestriction<P> {
    type Source = P;
    type Target = P;

    fn target(&self) -> &P {
        &self.target
    }

    /// The solution's values, each moved to the nearest end of its narrowed
    /// range when outside it.
    fn project(&self, sol: &P::Solution) -> P::Solution {
        let values = self
            .target
            .domains()
            .iter()
            .enumerate()
            .map(|(i, v)| P::get(sol, i).clamp(v.lower(), v.upper()))
            .collect();
        self.target.solution_from_values(values)
    }

    /// The same values, as a solution of the whole problem.
    fn lift(&self, source: &P, _base: &P::Solution, sol: &P::Solution) -> P::Solution {
        let values = (0..self.target.domains().len())
            .map(|i| P::get(sol, i))
            .collect();
        source.solution_from_values(values)
    }
}

impl<P: Branchable> BranchSpace for P {
    type Node = DomainRestriction<P>;

    fn ranges(&self) -> Cow<'_, IntVars> {
        Cow::Borrowed(self.domains())
    }

    fn value(sol: &P::Solution, i: usize) -> i64 {
        P::get(sol, i)
    }

    fn node(&self, ranges: &IntVars) -> DomainRestriction<P> {
        DomainRestriction {
            target: self.restricted(ranges.clone()),
        }
    }

    fn solution_with(&self, values: Vec<i64>) -> P::Solution {
        self.solution_from_values(values)
    }
}
