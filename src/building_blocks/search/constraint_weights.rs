//! The constraint weights of a weighted search.

use super::IndexSet;
use crate::error::OptError;
use crate::trait_defs::WeightedConstraints;

/// The weights of the constraints of a [`WeightedConstraints`] problem, the
/// set of the constraints the solution violates, and the prices the problem
/// keeps under them.
///
/// A weight changes only through [`raise_violated`](Self::raise_violated) and
/// [`lower_satisfied`](Self::lower_satisfied), and each change reprices
/// through [`WeightedConstraints::reweight`] in the same call. The problem
/// reads the weights and never writes them.
pub struct ConstraintWeights<P: WeightedConstraints> {
    /// `weight[c]` is the weight of constraint `c`.
    weight: Vec<u32>,
    violated: IndexSet,
    prices: P::Prices,
    /// The weight no constraint is raised past.
    cap: u32,
}

impl<P: WeightedConstraints> ConstraintWeights<P> {
    /// Every constraint at weight one, priced for `sol`, with a cap of the
    /// larger of `min_cap` and the number of constraints.
    pub fn new(problem: &P, sol: &P::Solution, min_cap: u32) -> Self {
        let (prices, violated) = problem.unit_prices(sol);
        let n = violated.bound();
        Self {
            weight: vec![1; n],
            violated,
            prices,
            cap: min_cap.max(u32::try_from(n).unwrap_or(u32::MAX)),
        }
    }

    /// The weight of constraint `c`.
    pub fn weight(&self, c: usize) -> u32 {
        self.weight[c]
    }

    /// The weight no constraint is raised past.
    pub fn cap(&self) -> u32 {
        self.cap
    }

    /// How many constraints there are.
    pub fn len(&self) -> usize {
        self.weight.len()
    }

    /// Whether there is no constraint.
    pub fn is_empty(&self) -> bool {
        self.weight.is_empty()
    }

    /// The constraints the solution violates.
    pub fn violated(&self) -> &IndexSet {
        &self.violated
    }

    /// What the problem caches to price a move under the weights.
    pub fn prices(&self) -> &P::Prices {
        &self.prices
    }

    /// Applies `mv` to `sol` through the problem, so the prices follow.
    pub fn apply(
        &mut self,
        problem: &P,
        sol: &mut P::Solution,
        mv: &P::Move,
    ) -> Result<(), OptError> {
        problem.apply_weighted(sol, mv, &self.weight, &mut self.prices, &mut self.violated)
    }

    /// Raises by one the weight of every constraint `sol` violates, up to the
    /// cap.
    pub fn raise_violated(&mut self, problem: &P, sol: &P::Solution) {
        for &c in self.violated.as_slice() {
            if self.weight[c] < self.cap {
                self.weight[c] += 1;
                problem.reweight(sol, &mut self.prices, c, 1);
            }
        }
    }

    /// Lowers by one the weight, where it is above one, of every constraint
    /// `sol` satisfies.
    pub fn lower_satisfied(&mut self, problem: &P, sol: &P::Solution) {
        for c in 0..self.weight.len() {
            if self.weight[c] > 1 && !self.violated.contains(c) {
                self.weight[c] -= 1;
                problem.reweight(sol, &mut self.prices, c, -1);
            }
        }
    }
}
