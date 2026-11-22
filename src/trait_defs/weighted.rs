//! Constraints a search can weight.

use super::{EnabledTabu, MoveToNeighbor, ProblemTrait};
use crate::building_blocks::search::IndexSet;
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
/// The search keeps the weights, the set of violated constraints and when to
/// change a weight. The problem keeps [`Prices`](Self::Prices), whatever it
/// caches to price a move under the weights, and says which moves satisfy a
/// constraint. The solution is not touched by a change of weight, and it ranks
/// as it always does, so the best solution a weighted search reports is the
/// best by the problem's own order.
///
/// The gain of a move is the drop in the weighted cost, positive when the move
/// lowers it. With every weight at one it counts each constraint the move
/// satisfies or violates once.
pub trait WeightedConstraints: ProblemTrait + Sized {
    /// The move a weighted search makes.
    type Move: MoveToNeighbor<Self> + EnabledTabu;

    /// What the problem caches to price a move under the weights.
    type Prices;

    /// The prices for `sol` with every weight at one, and the constraints
    /// `sol` violates.
    ///
    /// The set is over every constraint, so its
    /// [`bound`](IndexSet::bound) is how many there are. A constraint that no
    /// move can satisfy, such as an empty clause, is left out of it.
    fn unit_prices(&self, sol: &Self::Solution) -> (Self::Prices, IndexSet);

    /// The drop in the weighted cost if `mv` were applied to `sol`.
    fn weighted_gain(&self, sol: &Self::Solution, prices: &Self::Prices, mv: &Self::Move) -> f64;

    /// Applies `mv` to `sol`, and reprices `prices` and updates `violated`
    /// for the result. `weight[c]` is the weight of constraint `c`.
    ///
    /// The three change together because the prices follow the solution. A
    /// move applied any other way leaves them priced for a solution that is
    /// gone.
    fn apply_weighted(
        &self,
        sol: &mut Self::Solution,
        mv: &Self::Move,
        weight: &[u32],
        prices: &mut Self::Prices,
        violated: &mut IndexSet,
    ) -> Result<(), OptError>;

    /// Reprices `prices` after the weight of constraint `c` changed by
    /// `delta`, with `sol` unchanged.
    fn reweight(&self, sol: &Self::Solution, prices: &mut Self::Prices, c: usize, delta: i64);

    /// Fills `out` with the moves that satisfy constraint `c`, which `sol`
    /// violates.
    fn repairs(
        &self,
        sol: &Self::Solution,
        prices: &Self::Prices,
        c: usize,
        out: &mut Vec<Self::Move>,
    );

    /// Fills `out` with the moves that lead out of a weighted local optimum
    /// that violates nothing, the moves that work toward a better objective
    /// whatever they then violate. A vertex leaving a cover is one, and so is
    /// a vertex leaving the smallest color class.
    ///
    /// None by default, which ends the search there. That is right for a
    /// problem whose objective is the constraints alone, such as MaxSAT.
    fn objective_escapes(&self, sol: &Self::Solution, out: &mut Vec<Self::Move>) {
        let _ = (sol, out);
    }
}
