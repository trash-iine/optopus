use super::assignment::IntAssignment;
use crate::error::OptError;
use crate::search_state::{Evaluable, Evaluate, ProblemTrait};
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
}

/// The variables of an [`IntegerProblem`], indexed `0..n`.
///
/// Dereferences to `[IntVar]`. It also keeps the running total of
/// [`IntVar::num_changes`], which is what lets
/// [`IntChangeNeighbor`](super::IntChangeNeighbor) draw a uniformly random
/// move in `O(log n)`.
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
    /// keep a permutation apply, [`IntSwapNeighbor`](super::IntSwapNeighbor)
    /// and [`IntReverseNeighbor`](super::IntReverseNeighbor), and
    /// [`IntChangeNeighbor`](super::IntChangeNeighbor) has no moves here.
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
            Some(1) => r as usize,
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
        if self.common_width == Some(1) {
            return var;
        }
        let k = value.abs_diff(self.vars[var].lower()) as usize;
        let k = if value > current { k - 1 } else { k };
        self.row_start(var) + k
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

/// An assignment of a value to every variable of an [`IntegerProblem`],
/// together with its objective.
///
/// The fields are private so that the objective cannot fall out of step with
/// the values. A solution is built by
/// [`new_solution`](ProblemTrait::new_solution) or
/// [`IntegerProblem::solution_from`] and changed only by the moves of this
/// module.
#[derive(Clone, Debug)]
pub struct IntSolution {
    values: Vec<i64>,
    objective: Evaluable<f64>,
}

impl IntSolution {
    /// The value of every variable, in index order.
    pub fn values(&self) -> &[i64] {
        &self.values
    }

    /// The value of variable `i`.
    pub fn value(&self, i: usize) -> i64 {
        self.values[i]
    }

    /// Sets variable `i` and moves the cached objective by `delta`, a change
    /// in the raw objective value.
    pub(crate) fn set(&mut self, i: usize, value: i64, delta: f64) {
        self.values[i] = value;
        self.objective = with_value(self.objective, raw(self.objective) + delta);
    }

    /// Exchanges the values of `i` and `j` and moves the objective by `delta`.
    pub(crate) fn swap(&mut self, i: usize, j: usize, delta: f64) {
        self.values.swap(i, j);
        self.objective = with_value(self.objective, raw(self.objective) + delta);
    }

    /// Reverses the values of `i..=j` and moves the objective by `delta`.
    pub(crate) fn reverse(&mut self, i: usize, j: usize, delta: f64) {
        self.values[i..=j].reverse();
        self.objective = with_value(self.objective, raw(self.objective) + delta);
    }
}

impl Evaluate for IntSolution {
    fn evaluate(&self) -> Evaluable<f64> {
        self.objective
    }
}

impl crate::search_state::Distance for IntSolution {
    /// The number of variables whose values differ.
    fn distance(&self, other: &Self) -> usize {
        crate::common::hamming_distance(&self.values, &other.values)
    }
}

/// The delta of an [`IntegerProblem`] that was not given one. Every move is
/// then priced by evaluating the whole objective.
#[derive(Clone, Copy, Debug, Default)]
pub struct NoDelta;

/// What [`IntegerProblem::with_delta`] takes, a closure
/// `(sol, i, value) -> f64`, or [`NoDelta`].
pub trait ChangeDelta: Sync {
    /// The change of setting variable `i` of `sol` to `value`, or `None` when
    /// none was given.
    fn change_delta(&self, sol: &IntSolution, i: usize, value: i64) -> Option<f64>;
}

impl ChangeDelta for NoDelta {
    #[inline(always)]
    fn change_delta(&self, _: &IntSolution, _: usize, _: i64) -> Option<f64> {
        None
    }
}

impl<D: Fn(&IntSolution, usize, i64) -> f64 + Sync> ChangeDelta for D {
    #[inline(always)]
    fn change_delta(&self, sol: &IntSolution, i: usize, value: i64) -> Option<f64> {
        Some(self(sol, i, value))
    }
}

/// What [`IntegerProblem::with_swap_delta`] and
/// [`IntegerProblem::with_reverse_delta`] take, a closure
/// `(sol, i, j) -> f64`, or [`NoDelta`].
pub trait PairDelta: Sync {
    /// The change of the move on `i` and `j`, or `None` when none was given.
    fn pair_delta(&self, sol: &IntSolution, i: usize, j: usize) -> Option<f64>;
}

impl PairDelta for NoDelta {
    #[inline(always)]
    fn pair_delta(&self, _: &IntSolution, _: usize, _: usize) -> Option<f64> {
        None
    }
}

impl<D: Fn(&IntSolution, usize, usize) -> f64 + Sync> PairDelta for D {
    #[inline(always)]
    fn pair_delta(&self, sol: &IntSolution, i: usize, j: usize) -> Option<f64> {
        Some(self(sol, i, j))
    }
}

/// A problem over bounded integer variables, built from the variables and the
/// objective.
///
/// Nothing needs implementing. [`ProblemTrait`] comes with it, with
/// [`IntSolution`] as the solution and a uniformly random initial value for
/// every variable, or a uniformly random permutation when the variables are
/// [one](IntVars::permutation). So does [`IntAssignment`], and with it every
/// move of this module.
///
/// ```
/// use optopus::prelude::*;
///
/// // Minimize the sum of (x_i - 3)^2 over five variables in 0..=10.
/// let vars: IntVars = (0..5).map(|_| IntVar::new(0, 10)).collect();
/// let prob = IntegerProblem::minimize(vars, |x: &[i64]| {
///     x.iter().map(|&v| ((v - 3) * (v - 3)) as f64).sum()
/// });
///
/// let mut state = SearchState::new_with_seed(&prob, 1);
/// LocalSearch::<IntChangeNeighbor>::new(StopCondition::iterations(100))
///     .run(&mut state)
///     .unwrap();
/// assert_eq!(state.best_solution.values(), &[3; 5]);
/// ```
///
/// Each move asks the problem what it would change. Unless told otherwise the
/// problem copies the values and evaluates the whole objective again, which is
/// correct for any objective, costs a full evaluation per candidate and warns
/// once at runtime. [`with_delta`](Self::with_delta),
/// [`with_swap_delta`](Self::with_swap_delta) and
/// [`with_reverse_delta`](Self::with_reverse_delta) hand it the change
/// computed from what the move touches.
pub struct IntegerProblem<F, D = NoDelta, S = NoDelta, R = NoDelta> {
    vars: IntVars,
    maximize: bool,
    objective: F,
    delta: D,
    swap_delta: S,
    reverse_delta: R,
}

impl<F: Fn(&[i64]) -> f64 + Sync> IntegerProblem<F> {
    /// A problem that minimizes `objective`, `objective(values)` being the
    /// value of the assignment with `values[i]` for variable `i`.
    pub fn minimize(vars: IntVars, objective: F) -> Self {
        Self::with_direction(vars, false, objective)
    }

    /// A problem that maximizes `objective`.
    pub fn maximize(vars: IntVars, objective: F) -> Self {
        Self::with_direction(vars, true, objective)
    }

    fn with_direction(vars: IntVars, maximize: bool, objective: F) -> Self {
        Self {
            vars,
            maximize,
            objective,
            delta: NoDelta,
            swap_delta: NoDelta,
            reverse_delta: NoDelta,
        }
    }
}

impl<F, D, S, R> IntegerProblem<F, D, S, R>
where
    F: Fn(&[i64]) -> f64 + Sync,
{
    /// Prices [`IntChangeNeighbor`](super::IntChangeNeighbor) with
    /// `delta(sol, i, value)`, how much the objective changes when variable
    /// `i` of `sol` is set to `value`, new minus old.
    pub fn with_delta<D2>(self, delta: D2) -> IntegerProblem<F, D2, S, R>
    where
        D2: Fn(&IntSolution, usize, i64) -> f64 + Sync,
    {
        IntegerProblem {
            vars: self.vars,
            maximize: self.maximize,
            objective: self.objective,
            delta,
            swap_delta: self.swap_delta,
            reverse_delta: self.reverse_delta,
        }
    }

    /// Prices [`IntSwapNeighbor`](super::IntSwapNeighbor) with
    /// `swap_delta(sol, i, j)`, how much the objective changes when the values
    /// of variables `i` and `j` are exchanged, new minus old.
    pub fn with_swap_delta<S2>(self, swap_delta: S2) -> IntegerProblem<F, D, S2, R>
    where
        S2: Fn(&IntSolution, usize, usize) -> f64 + Sync,
    {
        IntegerProblem {
            vars: self.vars,
            maximize: self.maximize,
            objective: self.objective,
            delta: self.delta,
            swap_delta,
            reverse_delta: self.reverse_delta,
        }
    }

    /// Prices [`IntReverseNeighbor`](super::IntReverseNeighbor) with
    /// `reverse_delta(sol, i, j)`, how much the objective changes when the
    /// values of variables `i..=j` are reversed, new minus old.
    pub fn with_reverse_delta<R2>(self, reverse_delta: R2) -> IntegerProblem<F, D, S, R2>
    where
        R2: Fn(&IntSolution, usize, usize) -> f64 + Sync,
    {
        IntegerProblem {
            vars: self.vars,
            maximize: self.maximize,
            objective: self.objective,
            delta: self.delta,
            swap_delta: self.swap_delta,
            reverse_delta,
        }
    }

    /// The variables.
    pub fn variables(&self) -> &IntVars {
        &self.vars
    }

    /// The objective of an assignment, with the direction of the problem.
    pub fn objective(&self, values: &[i64]) -> Evaluable<f64> {
        let value = (self.objective)(values);
        if self.maximize {
            Evaluable::Maximize(value)
        } else {
            Evaluable::Minimize(value)
        }
    }

    /// A solution with the given values, for starting a search somewhere other
    /// than a random assignment.
    ///
    /// Fails if the number of values differs from the number of variables, if
    /// a value lies outside its variable's range, or if the variables are a
    /// permutation and a value repeats.
    pub fn solution_from(&self, values: Vec<i64>) -> Result<IntSolution, OptError> {
        self.vars.check(&values)?;
        Ok(self.build(values))
    }

    fn build(&self, values: Vec<i64>) -> IntSolution {
        let objective = self.objective(&values);
        IntSolution { values, objective }
    }

    /// The change `edit` makes, found by evaluating the whole objective on a
    /// copy of the values. What every move falls back on when its delta was
    /// not given. Kept out of line so that the priced path stays small where
    /// a delta was given.
    #[cold]
    #[inline(never)]
    fn full_delta(&self, sol: &IntSolution, move_name: &str, edit: impl FnOnce(&mut [i64])) -> f64 {
        static WARNED: std::sync::OnceLock<()> = std::sync::OnceLock::new();
        WARNED.get_or_init(|| {
            tracing::warn!(
                move_name,
                "An IntegerProblem without a delta for this move evaluates \
                 the whole objective for every candidate. Give it one with \
                 with_delta, with_swap_delta or with_reverse_delta."
            );
        });
        let mut values = sol.values.clone();
        edit(&mut values);
        (self.objective)(&values) - raw(sol.objective)
    }
}

impl<F, D, S, R> ProblemTrait for IntegerProblem<F, D, S, R>
where
    F: Fn(&[i64]) -> f64 + Sync,
{
    type Solution = IntSolution;

    /// Draws every variable uniformly from its range, or a uniformly random
    /// permutation when the variables are one.
    fn new_solution(&self, rng: &mut impl Rng) -> IntSolution {
        self.build(self.vars.random_values(rng))
    }
}

impl<F, D, S, R> IntAssignment for IntegerProblem<F, D, S, R>
where
    F: Fn(&[i64]) -> f64 + Sync,
    D: ChangeDelta,
    S: PairDelta,
    R: PairDelta,
{
    #[inline]
    fn domains(&self) -> &IntVars {
        &self.vars
    }

    #[inline]
    fn get(sol: &IntSolution, i: usize) -> i64 {
        sol.values[i]
    }

    #[inline]
    fn assign(&self, sol: &mut IntSolution, i: usize, value: i64) {
        let delta = self.assign_delta(sol, i, value);
        sol.set(i, value, delta);
    }

    #[inline]
    fn assign_delta(&self, sol: &IntSolution, i: usize, value: i64) -> f64 {
        match self.delta.change_delta(sol, i, value) {
            Some(d) => d,
            None => self.full_delta(sol, "IntChangeNeighbor", |v| v[i] = value),
        }
    }

    #[inline]
    fn assign_swap(&self, sol: &mut IntSolution, i: usize, j: usize) {
        let delta = self.assign_swap_delta(sol, i, j);
        sol.swap(i, j, delta);
    }

    #[inline]
    fn assign_swap_delta(&self, sol: &IntSolution, i: usize, j: usize) -> f64 {
        match self.swap_delta.pair_delta(sol, i, j) {
            Some(d) => d,
            None => self.full_delta(sol, "IntSwapNeighbor", |v| v.swap(i, j)),
        }
    }

    #[inline]
    fn assign_reverse(&self, sol: &mut IntSolution, i: usize, j: usize) {
        let delta = self.assign_reverse_delta(sol, i, j);
        sol.reverse(i, j, delta);
    }

    #[inline]
    fn assign_reverse_delta(&self, sol: &IntSolution, i: usize, j: usize) -> f64 {
        match self.reverse_delta.pair_delta(sol, i, j) {
            Some(d) => d,
            None => self.full_delta(sol, "IntReverseNeighbor", |v| v[i..=j].reverse()),
        }
    }
}

/// The number inside an [`Evaluable`], without its direction.
pub(crate) fn raw(e: Evaluable<f64>) -> f64 {
    match e {
        Evaluable::Maximize(v) | Evaluable::Minimize(v) => v,
    }
}

/// `value` with the direction of `like`.
pub(crate) fn with_value(like: Evaluable<f64>, value: f64) -> Evaluable<f64> {
    match like {
        Evaluable::Maximize(_) => Evaluable::Maximize(value),
        Evaluable::Minimize(_) => Evaluable::Minimize(value),
    }
}
