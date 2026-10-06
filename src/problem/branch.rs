//! The glue that lets a binary problem be searched by
//! [`BranchAndBound`](crate::heuristic::BranchAndBound) through
//! [`FixVariables`].

use std::borrow::Cow;

use crate::common::{BinaryFixing, IntVar, IntVars, variable_slots};
use crate::search_state::Evaluable;
use crate::trait_defs::{BinaryProblem, BranchSpace, FixVariables, Relaxation};
use crate::trait_defs::{raw, with_value};

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
