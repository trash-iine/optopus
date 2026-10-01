//! What [`BranchAndBound`](crate::heuristic::BranchAndBound) needs of an
//! integer problem: the same problem over narrower ranges, and a bound on what
//! those ranges can hold.

use super::assignment::IntAssignment;
use super::problem::IntVars;
use crate::search_state::Evaluable;
use crate::trait_defs::ProblemReduction;
use rand::SeedableRng;
use rand::rngs::SmallRng;

/// An integer problem that can be rebuilt over narrower ranges.
///
/// [`IntegerProblem`](super::IntegerProblem) and
/// [`FormulaProblem`](super::FormulaProblem) implement it. A problem of your
/// own implements [`restricted`](Self::restricted), and
/// [`solution_from_values`](Self::solution_from_values) has a default.
///
/// A node of the search is the problem with some ranges narrowed, and the
/// heuristic searching the node runs on exactly that, so its moves stay in the
/// node without knowing there is one. The objective must not depend on the
/// ranges, since a node's solutions are compared with the whole problem's.
pub trait Branchable: IntAssignment + Sized {
    /// The same problem with its variables ranging over `vars`, which has as
    /// many variables as [`domains`](IntAssignment::domains) and each range
    /// inside the original one.
    fn restricted(&self, vars: IntVars) -> Self;

    /// The solution assigning `values`, each inside its variable's range.
    ///
    /// A solution crosses between a node and the whole problem by its values,
    /// because what it caches to price moves may be laid out by the ranges.
    /// The default starts from a solution drawn with a fixed seed and assigns
    /// each value that differs, which is correct whatever the solution caches
    /// and costs an [`assign`](IntAssignment::assign) per variable.
    fn solution_from_values(&self, values: Vec<i64>) -> Self::Solution {
        let mut sol = self.new_solution(&mut SmallRng::seed_from_u64(0));
        for (i, value) in values.into_iter().enumerate() {
            if Self::get(&sol, i) != value {
                self.assign(&mut sol, i, value);
            }
        }
        sol
    }
}

/// A bound on the best objective any assignment within some ranges can reach,
/// what [`BranchAndBound`](crate::heuristic::BranchAndBound) prunes a node by.
///
/// The bound carries the direction of the problem. On a maximized problem it
/// is at least the largest objective within the ranges, on a minimized one at
/// most the smallest. A bound that is not one prunes the optimum without an
/// error, so an implementation is worth checking against enumeration on small
/// instances.
///
/// A closure `|prob, vars| -> Evaluable<f64>` is one.
///
/// ```
/// use optopus::prelude::*;
/// use optopus::problem::Relaxation;
///
/// // maximize the sum of three variables in 0..=4. No assignment beats the
/// // sum of the upper ends.
/// let vars: IntVars = (0..3).map(|_| IntVar::new(0, 4)).collect();
/// let prob = FormulaProblem::maximize(vars, Expr::Var(0) + Expr::Var(1) + Expr::Var(2));
/// let mut relaxation = |_: &FormulaProblem, vars: &IntVars| {
///     Evaluable::Maximize(vars.iter().map(|v| v.upper() as f64).sum::<f64>())
/// };
/// let narrowed: IntVars = [IntVar::new(0, 4), IntVar::new(0, 1), IntVar::new(2, 2)].into_iter().collect();
/// assert_eq!(relaxation.bound(&prob, &narrowed).minimized(), -7.0);
/// ```
pub trait Relaxation<P> {
    /// The bound over the assignments with each variable in its range in
    /// `vars`, which are ranges of `prob`'s variables.
    fn bound(&mut self, prob: &P, vars: &IntVars) -> Evaluable<f64>;

    /// The variable to split next, among those whose range in `vars` holds
    /// more than one value. `None`, the default, leaves the choice to the
    /// search.
    fn branch_hint(&mut self, prob: &P, vars: &IntVars) -> Option<usize> {
        let _ = (prob, vars);
        None
    }
}

impl<P, F: FnMut(&P, &IntVars) -> Evaluable<f64>> Relaxation<P> for F {
    fn bound(&mut self, prob: &P, vars: &IntVars) -> Evaluable<f64> {
        self(prob, vars)
    }
}

/// A node of the search as a map from the whole problem onto the same problem
/// over narrower ranges, so that a heuristic crosses into it and back through
/// [`SearchState::open_reduction`](crate::search_state::SearchState::open_reduction)
/// and [`close_reduction`](crate::search_state::SearchState::close_reduction).
pub(crate) struct DomainRestriction<P> {
    pub(crate) target: P,
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
