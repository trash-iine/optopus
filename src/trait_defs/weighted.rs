//! Constraints a search can weight.

use super::{EnabledTabu, MoveToNeighbor, ProblemTrait};
use crate::error::OptError;

/// A problem whose constraints a search can weight.
///
/// A constraint is anything a solution either satisfies or violates, a clause
/// of a formula or an edge a cover has to touch. Each one carries an integer
/// weight that starts at one, and the weighted cost of a solution is its
/// objective plus the weights of the constraints it violates.
/// [`WeightedLocalSearch`](crate::heuristic::WeightedLocalSearch) descends on
/// that cost and raises the weights of what stays violated, so the constraints
/// a search keeps failing come to outweigh the rest.
///
/// The weights live in [`Weights`](Self::Weights), beside whatever the problem
/// caches to price a move under them. The solution is not touched by a change
/// of weight, and it ranks as it always does, so the best solution a weighted
/// search reports is the best by the problem's own order.
///
/// The gain of a move is the drop in the weighted cost, positive when the move
/// lowers it. With every weight at one it counts each constraint the move
/// satisfies or violates once.
pub trait WeightedConstraints: ProblemTrait + Sized {
    /// The move a weighted search makes.
    type Move: MoveToNeighbor<Self> + EnabledTabu;

    /// The weights, with the prices the problem keeps under them.
    type Weights;

    /// Every constraint at weight one, priced for `sol`.
    fn unit_weights(&self, sol: &Self::Solution) -> Self::Weights;

    /// How many constraints the problem has.
    fn constraint_count(&self) -> usize;

    /// The drop in the weighted cost if `mv` were applied to `sol`.
    fn weighted_gain(&self, sol: &Self::Solution, weights: &Self::Weights, mv: &Self::Move) -> f64;

    /// Applies `mv` to `sol` and reprices `weights` for the result.
    ///
    /// The two change together because the prices follow the solution. A move
    /// applied any other way leaves `weights` priced for a solution that is
    /// gone.
    fn apply_weighted(
        &self,
        sol: &mut Self::Solution,
        weights: &mut Self::Weights,
        mv: &Self::Move,
    ) -> Result<(), OptError>;

    /// Raises by one the weight of every constraint `sol` violates, up to
    /// `cap`.
    fn raise_violated(&self, sol: &Self::Solution, weights: &mut Self::Weights, cap: u32);

    /// Lowers by one the weight, where it is above one, of every constraint
    /// `sol` satisfies.
    fn lower_satisfied(&self, sol: &Self::Solution, weights: &mut Self::Weights);

    /// Fills `out` with the moves that lead out of a weighted local optimum.
    ///
    /// While `sol` violates a constraint these are the moves that satisfy one
    /// violated constraint drawn uniformly. Once it violates none they are the
    /// moves that improve the objective, whatever they then violate. `out` is
    /// left empty when there is neither, which ends the search.
    fn escape_moves(
        &self,
        sol: &Self::Solution,
        weights: &Self::Weights,
        rng: &mut rand::rngs::SmallRng,
        out: &mut Vec<Self::Move>,
    );
}
