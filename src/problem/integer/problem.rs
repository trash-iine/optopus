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
        Self { vars, cum_changes }
    }

    /// How many single variable changes there are in total, the size of every
    /// solution's neighborhood.
    pub fn total_changes(&self) -> u64 {
        self.cum_changes.last().copied().unwrap_or(0)
    }

    /// The variable holding the `r`th change when the changes of every
    /// variable are laid out in index order. `r` must be below
    /// [`total_changes`](Self::total_changes).
    pub(crate) fn var_of_change(&self, r: u64) -> usize {
        self.cum_changes.partition_point(|&c| c <= r)
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
/// [`IntegerProblem::solution_from`] and changed only by
/// [`IntChangeNeighbor`](super::IntChangeNeighbor).
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
}

impl Evaluate for IntSolution {
    fn evaluate(&self) -> Evaluable<f64> {
        self.objective
    }
}

/// A problem over bounded integer variables.
///
/// Implementing this is all a problem needs. [`ProblemTrait`] follows from it,
/// with [`IntSolution`] as the solution and a uniformly random initial value
/// for every variable, and [`IntChangeNeighbor`](super::IntChangeNeighbor) is
/// the move that sets one variable to another value in its range. On binary
/// variables that move is a flip.
///
/// ```
/// use optopus::prelude::*;
///
/// /// Minimize the sum of (x_i - 3)^2 over five variables in 0..=10.
/// struct Target { vars: IntVars }
///
/// impl IntegerProblem for Target {
///     fn variables(&self) -> &IntVars {
///         &self.vars
///     }
///     fn objective(&self, values: &[i64]) -> Evaluable<f64> {
///         Evaluable::Minimize(values.iter().map(|&x| ((x - 3) * (x - 3)) as f64).sum())
///     }
/// }
///
/// let prob = Target { vars: (0..5).map(|_| IntVar::new(0, 10)).collect() };
/// let mut state = SearchState::new_with_seed(&prob, 1);
/// LocalSearch::<IntChangeNeighbor>::new(StopCondition::iterations(100))
///     .run(&mut state)
///     .unwrap();
/// assert_eq!(state.best_solution.values(), &[3; 5]);
/// ```
///
/// `Sync` because [`MoveToNeighbor::iter`](crate::search_state::MoveToNeighbor::iter)
/// hands out a `Send` iterator that borrows the problem.
pub trait IntegerProblem: Sync {
    /// The variables, fixed for the life of the problem.
    fn variables(&self) -> &IntVars;

    /// The objective of an assignment, `values[i]` being the value of variable
    /// `i`. Whether it is maximized or minimized is stated here, by the
    /// [`Evaluable`] variant, and must be the same for every assignment.
    fn objective(&self, values: &[i64]) -> Evaluable<f64>;

    /// How much the raw objective changes when variable `i` of `sol` is set to
    /// `value`, new minus old.
    ///
    /// <div class="warning">
    /// The default copies the values and calls
    /// <a href="#tymethod.objective"><code>objective</code></a> on the copy,
    /// which costs a full evaluation for every candidate move. Override it
    /// with the change computed from what variable <code>i</code> touches.
    /// A one shot <code>tracing::warn!</code> is emitted the first time the
    /// default runs in a process.
    /// </div>
    fn delta(&self, sol: &IntSolution, i: usize, value: i64) -> f64 {
        static WARNED: std::sync::OnceLock<()> = std::sync::OnceLock::new();
        WARNED.get_or_init(|| {
            tracing::warn!(
                problem_type = std::any::type_name::<Self>(),
                "Using the default implementation of IntegerProblem::delta, \
                 which evaluates the whole objective for every candidate move. \
                 Override it with an incremental computation for hot-path \
                 heuristics."
            );
        });
        let mut values = sol.values.clone();
        values[i] = value;
        raw(self.objective(&values)) - raw(sol.objective)
    }

    /// A solution with the given values, for starting a search somewhere other
    /// than a random assignment.
    ///
    /// Fails if the number of values differs from the number of variables or a
    /// value lies outside its variable's range.
    fn solution_from(&self, values: Vec<i64>) -> Result<IntSolution, OptError> {
        let vars = self.variables();
        if values.len() != vars.len() {
            return Err(OptError::Config(format!(
                "{} values given for {} variables",
                values.len(),
                vars.len()
            )));
        }
        if let Some((i, (v, x))) = vars
            .iter()
            .zip(&values)
            .enumerate()
            .find(|(_, (v, x))| !v.contains(**x))
        {
            return Err(OptError::Config(format!(
                "value {x} of variable {i} lies outside [{}, {}]",
                v.lower(),
                v.upper()
            )));
        }
        let objective = self.objective(&values);
        Ok(IntSolution { values, objective })
    }
}

impl<P: IntegerProblem> ProblemTrait for P {
    type Solution = IntSolution;

    /// Draws every variable uniformly from its range.
    fn new_solution(&self, rng: &mut impl Rng) -> IntSolution {
        let values: Vec<i64> = self
            .variables()
            .iter()
            .map(|v| rng.random_range(v.lower()..=v.upper()))
            .collect();
        let objective = self.objective(&values);
        IntSolution { values, objective }
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
