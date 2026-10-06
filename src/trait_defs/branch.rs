//! What [`BranchAndBound`](crate::heuristic::BranchAndBound) needs of a
//! problem: a node as the problem over narrower ranges, and a bound on what
//! those ranges can hold.
//!
//! A node is stated as the [`IntVars`] ranges of the variables whatever the
//! problem, a binary variable ranging over `0..=1` until it is fixed. Two kinds
//! of problem turn that into something to search.
//!
//! - An integer problem, [`Branchable`], is rebuilt over the narrower ranges,
//!   and its moves stay inside them.
//! - A binary problem, [`FixVariables`](super::FixVariables), folds the
//!   fixed variables into a smaller instance of itself, so its own moves and
//!   heuristics search the node.

use std::borrow::Cow;

use super::{Evaluable, Evaluate, IntAssignment, ProblemReduction, ProblemTrait};
use crate::common::IntVars;
use rand::SeedableRng;
use rand::rngs::SmallRng;

/// What [`BranchAndBound`](crate::heuristic::BranchAndBound) asks of a
/// problem. It is implemented for every [`Branchable`] integer problem, and
/// for MaxCut, Qubo, Sat and VertexCover through
/// [`FixVariables`](super::FixVariables).
///
/// A binary problem of your own implements
/// [`FixVariables`](super::FixVariables), and this trait by handing each
/// method to the functions of the same name in [`crate::common`],
/// [`binary_ranges`](crate::common::binary_ranges),
/// [`binary_value`](crate::common::binary_value),
/// [`binary_node`](crate::common::binary_node) and
/// [`binary_solution_with`](crate::common::binary_solution_with).
pub trait BranchSpace: ProblemTrait<Solution: Evaluate> {
    /// A node, as a map from the whole problem onto what is searched there.
    type Node: ProblemReduction<Source = Self, Target = Self>;

    /// The ranges of the whole problem, the root of the search.
    fn ranges(&self) -> Cow<'_, IntVars>;

    /// The value of variable `i` in `sol`.
    fn value(sol: &Self::Solution, i: usize) -> i64;

    /// The node with the variables in `ranges`, each inside its range in
    /// [`ranges`](Self::ranges) and at least one holding more than one value.
    fn node(&self, ranges: &IntVars) -> Self::Node;

    /// The solution assigning `values`, each inside its variable's range.
    fn solution_with(&self, values: Vec<i64>) -> Self::Solution;
}

/// An integer problem that can be rebuilt over narrower ranges.
///
/// [`IntegerProblem`](crate::problem::IntegerProblem) and
/// [`FormulaProblem`](crate::problem::FormulaProblem) implement it. A problem
/// of your own implements [`restricted`](Self::restricted), and
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
/// use optopus::trait_defs::Relaxation;
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

    /// [`bound`](Self::bound), told the objective of the best solution found
    /// so far. The default ignores it.
    ///
    /// A relaxation that is itself an optimization uses it to stop as soon as
    /// the bound falls to the incumbent, where the node is pruned whatever the
    /// bound would have tightened to, or to aim its steps at it. The value
    /// returned must be a bound whatever `incumbent` is.
    fn bound_against(
        &mut self,
        prob: &P,
        vars: &IntVars,
        incumbent: Evaluable<f64>,
    ) -> Evaluable<f64> {
        let _ = incumbent;
        self.bound(prob, vars)
    }

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
