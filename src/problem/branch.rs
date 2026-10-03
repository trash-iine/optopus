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
//! - A binary problem, [`FixVariables`], folds the fixed variables into a
//!   smaller instance of itself, so its own moves and heuristics search the
//!   node.

use std::borrow::Cow;

use super::integer::{IntAssignment, IntVar, IntVars, raw, with_value};
use crate::common::{BinaryFixing, variable_slots};
use crate::search_state::{Evaluable, Evaluate, ProblemTrait};
use crate::trait_defs::{BinaryProblem, FixVariables, ProblemReduction};
use rand::SeedableRng;
use rand::rngs::SmallRng;

/// What [`BranchAndBound`](crate::heuristic::BranchAndBound) asks of a
/// problem. It is implemented for every [`Branchable`] integer problem, and
/// for MaxCut, Qubo, Sat and VertexCover through [`FixVariables`].
///
/// A binary problem of your own implements [`FixVariables`], and this trait by
/// handing each method to the functions of the same name in this module,
/// [`binary_ranges`], [`binary_value`], [`binary_node`] and
/// [`binary_solution_with`].
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

/// [`BranchSpace::ranges`] of a binary problem, `0..=1` for each variable and
/// `0..=0` for an index that is not one.
pub fn binary_ranges<P: BinaryProblem>(prob: &P) -> Cow<'static, IntVars> {
    let mut ranges = vec![IntVar::new(0, 0); variable_slots(prob)];
    for i in prob.variable_indices() {
        ranges[i] = IntVar::binary();
    }
    Cow::Owned(IntVars::new(ranges))
}

/// [`BranchSpace::value`] of a binary problem.
pub fn binary_value<P: BinaryProblem>(sol: &P::Solution, i: usize) -> i64 {
    i64::from(P::variable(sol, i))
}

/// [`BranchSpace::node`] of a binary problem, the instance with every
/// variable whose range holds one value fixed at it.
pub fn binary_node<P: FixVariables>(prob: &P, ranges: &IntVars) -> BinaryFixing<P> {
    BinaryFixing::new(prob.fix(&fixed_by(ranges)))
}

/// [`BranchSpace::solution_with`] of a binary problem.
pub fn binary_solution_with<P: BinaryProblem>(prob: &P, values: Vec<i64>) -> P::Solution {
    let values: Vec<bool> = values.into_iter().map(|v| v != 0).collect();
    prob.solution_from_assignment(&values)
}

/// The value each variable is held at, where its range holds one.
pub(crate) fn fixed_by(ranges: &IntVars) -> Vec<Option<bool>> {
    ranges
        .iter()
        .map(|r| (r.num_changes() == 0).then(|| r.lower() != 0))
        .collect()
}

macro_rules! binary_branch_space {
    ($($problem:ty),*) => {$(
        impl BranchSpace for $problem {
            type Node = BinaryFixing<$problem>;

            fn ranges(&self) -> Cow<'_, IntVars> {
                binary_ranges(self)
            }

            fn value(sol: &Self::Solution, i: usize) -> i64 {
                binary_value::<$problem>(sol, i)
            }

            fn node(&self, ranges: &IntVars) -> BinaryFixing<$problem> {
                binary_node(self, ranges)
            }

            fn solution_with(&self, values: Vec<i64>) -> Self::Solution {
                binary_solution_with(self, values)
            }
        }
    )*};
}

binary_branch_space!(super::MaxCut, super::Qubo, super::Sat, super::VertexCover);

/// The bound of a binary problem from its folding. The node's fixed variables
/// are folded with [`FixVariables::fix`], and the bound is the folding's offset
/// plus [`FixVariables::trivial_bound`] of the folded instance.
///
/// The bound is only as strong as `trivial_bound`, which on the built-in
/// problems proves optima of a few dozen variables, not of benchmark
/// instances. A stronger bound is given as a closure or a `Relaxation` of its
/// own.
#[derive(Clone, Copy, Debug, Default)]
pub struct BinaryRelaxation;

impl<P: FixVariables> Relaxation<P> for BinaryRelaxation {
    fn bound(&mut self, prob: &P, vars: &IntVars) -> Evaluable<f64> {
        let fixed = prob.fix(&fixed_by(vars));
        let b = fixed.target.trivial_bound();
        with_value(b, raw(b) + fixed.offset)
    }
}

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
