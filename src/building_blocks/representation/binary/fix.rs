//! Fixing variables of a binary problem, and the branch and bound glue built on
//! it.

use std::borrow::Cow;

use crate::building_blocks::{IntVar, IntVars};
use crate::trait_defs::{
    BinaryProblem, Evaluable, FixVariables, FixedVariables, Placed, ProblemReduction, Relaxation,
};

/// The number of variable indices of `prob`, one past the largest, which is
/// how long a solution's assignment is.
pub fn variable_slots<P: BinaryProblem>(prob: &P) -> usize {
    prob.variable_indices().max().map_or(0, |i| i + 1)
}

/// Variables fixed by [`FixVariables::fix`] as a map from the whole problem
/// onto the folded instance, so that a heuristic of the problem crosses into
/// it and back through
/// [`SearchState::open_reduction`](crate::search_state::SearchState::open_reduction)
/// and [`close_reduction`](crate::search_state::SearchState::close_reduction).
pub struct BinaryFixing<P> {
    /// Every `Placed::Free(t)` and the reference name a variable of the
    /// folded instance, which `new` sees to.
    fixed: FixedVariables<P>,
    /// The folded instance's number of variable indices.
    slots: usize,
}

impl<P: FixVariables> BinaryFixing<P> {
    /// `fixed` as a map onto its folded instance.
    ///
    /// A free variable the folding left with no terms is not a variable of
    /// the folded instance, and is held at `false` instead, which changes no
    /// objective.
    pub fn new(mut fixed: FixedVariables<P>) -> Self {
        let slots = variable_slots(&fixed.target);
        let mut present = vec![false; slots];
        for i in fixed.target.variable_indices() {
            present[i] = true;
        }
        let is_variable = |t: usize| present.get(t).copied().unwrap_or(false);
        for placed in &mut fixed.placed {
            if let Placed::Free(t) = *placed
                && !is_variable(t)
            {
                *placed = Placed::Fixed(false);
            }
        }
        fixed.reference = fixed.reference.filter(|&r| is_variable(r));
        Self { fixed, slots }
    }

    /// What the folded instance contributes on top of its own objective.
    pub fn offset(&self) -> f64 {
        self.fixed.offset
    }
}

impl<P: FixVariables> ProblemReduction for BinaryFixing<P> {
    type Source = P;
    type Target = P;

    fn target(&self) -> &P {
        &self.fixed.target
    }

    /// The free variables keep their values, read against a reference
    /// variable held at `false`.
    fn project(&self, sol: &P::Solution) -> P::Solution {
        let mut values = vec![false; self.slots];
        for (i, placed) in self.fixed.placed.iter().enumerate() {
            if let Placed::Free(t) = *placed {
                values[t] = P::variable(sol, i);
            }
        }
        self.fixed.target.solution_from_assignment(&values)
    }

    /// The fixed variables at their values, the free ones read from `sol`.
    fn lift(&self, source: &P, _base: &P::Solution, sol: &P::Solution) -> P::Solution {
        let flip = self.fixed.reference.is_some_and(|r| P::variable(sol, r));
        let values: Vec<bool> = self
            .fixed
            .placed
            .iter()
            .map(|placed| match *placed {
                Placed::Fixed(value) => value,
                Placed::Free(t) => P::variable(sol, t) != flip,
            })
            .collect();
        source.solution_from_assignment(&values)
    }
}

/// [`BranchSpace::ranges`](crate::trait_defs::BranchSpace::ranges) of a
/// binary problem, `0..=1` for each variable and `0..=0` for an index that is
/// not one.
pub fn binary_ranges<P: BinaryProblem>(prob: &P) -> Cow<'static, IntVars> {
    let mut ranges = vec![IntVar::new(0, 0); variable_slots(prob)];
    for i in prob.variable_indices() {
        ranges[i] = IntVar::binary();
    }
    Cow::Owned(IntVars::new(ranges))
}

/// [`BranchSpace::value`](crate::trait_defs::BranchSpace::value) of a binary
/// problem.
pub fn binary_value<P: BinaryProblem>(sol: &P::Solution, i: usize) -> i64 {
    i64::from(P::variable(sol, i))
}

/// [`BranchSpace::node`](crate::trait_defs::BranchSpace::node) of a binary
/// problem, the instance with every variable whose range holds one value
/// fixed at it.
pub fn binary_node<P: FixVariables>(prob: &P, ranges: &IntVars) -> BinaryFixing<P> {
    BinaryFixing::new(prob.fix(&fixed_by(ranges)))
}

/// [`BranchSpace::solution_with`](crate::trait_defs::BranchSpace::solution_with)
/// of a binary problem.
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

/// Implements [`BranchSpace`](crate::trait_defs::BranchSpace) for each binary
/// problem given, every method handed to the function of the same name in
/// [`building_blocks`](crate::building_blocks), such as
/// [`binary_ranges`](crate::building_blocks::binary_ranges).
///
/// A binary problem of your own calls `optopus::binary_branch_space!(MyProblem);`
/// beside its [`FixVariables`](crate::trait_defs::FixVariables) impl, and
/// [`BranchAndBound`](crate::heuristic::BranchAndBound) then searches it. The
/// built-in binary problems call it in their `fix.rs`. A blanket impl over
/// `FixVariables` would overlap the one over
/// [`Branchable`](crate::trait_defs::Branchable), which is why this is a macro.
#[macro_export]
macro_rules! binary_branch_space {
    ($($problem:ty),*) => {$(
        impl $crate::trait_defs::BranchSpace for $problem {
            type Node = $crate::building_blocks::BinaryFixing<$problem>;

            fn ranges(&self) -> ::std::borrow::Cow<'_, $crate::building_blocks::IntVars> {
                $crate::building_blocks::binary_ranges(self)
            }

            fn value(sol: &Self::Solution, i: usize) -> i64 {
                $crate::building_blocks::binary_value::<$problem>(sol, i)
            }

            fn node(
                &self,
                ranges: &$crate::building_blocks::IntVars,
            ) -> $crate::building_blocks::BinaryFixing<$problem> {
                $crate::building_blocks::binary_node(self, ranges)
            }

            fn solution_with(&self, values: Vec<i64>) -> Self::Solution {
                $crate::building_blocks::binary_solution_with(self, values)
            }
        }
    )*};
}

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
        b.with_value(b.raw() + fixed.offset)
    }
}
