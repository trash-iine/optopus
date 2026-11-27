//! Constraints a search can weight.

use super::{EnabledTabu, MoveToNeighbor, ProblemTrait};
use crate::building_blocks::search::IndexSet;
use crate::error::OptError;
use rand::rngs::SmallRng;

/// A problem whose constraints a search can weight.
///
/// A constraint is anything a solution either satisfies or violates, a clause
/// of a formula or an edge a cover has to touch. Each one carries an integer
/// weight that starts at one.
/// [`WeightedLocalSearch`](crate::heuristic::WeightedLocalSearch) prices a
/// move by the weight of the constraints it satisfies less the weight of those
/// it violates, and raises the weights of what stays violated, so the
/// constraints a search keeps failing come to outweigh the rest.
///
/// The objective here is what a solution costs apart from its constraints,
/// such as the size of a cover or the number of colors used. A penalty the
/// problem's own objective adds for a violated constraint is not part of it.
/// A problem whose objective is the constraints alone, such as MaxSAT, keeps
/// the defaults of [`objective`](Self::objective) and
/// [`objective_change`](Self::objective_change), which are zero.
///
/// The search keeps the weights, the set of violated constraints and when to
/// change a weight. The problem keeps [`Prices`](Self::Prices), whatever it
/// caches to price a move under the weights, and says which moves act on a
/// constraint. The solution is not touched by a change of weight, and it ranks
/// as it always does, so the best solution a weighted search reports is the
/// best by the problem's own order.
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

    /// The weight of the constraints `mv` would satisfy less the weight of
    /// those it would violate, if it were applied to `sol`.
    fn weight_gain(&self, sol: &Self::Solution, prices: &Self::Prices, mv: &Self::Move) -> i64;

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

    /// Fills `out` with the tight moves of constraint `c`, the moves of its
    /// variables that leave it as tight as they can.
    ///
    /// For a violated `c` these are the moves that satisfy it. For a
    /// satisfied `c` they are the moves that keep it satisfied while taking
    /// up the slack it has, a vertex leaving an edge both of whose ends are in
    /// a cover, or a true literal turning false in a clause with two or more.
    /// A satisfied constraint with no slack has none.
    fn tight_moves(
        &self,
        sol: &Self::Solution,
        prices: &Self::Prices,
        c: usize,
        out: &mut Vec<Self::Move>,
    );

    /// Fills `out` with the moves that take one step toward a better
    /// objective, whatever they then violate. A vertex leaving a cover is
    /// one, and so is a vertex leaving the smallest color class.
    ///
    /// The search takes the best priced of them when a solution that violates
    /// nothing has no move that keeps it so and betters the objective. None by default,
    /// which ends the search there. That is right for a problem whose
    /// objective is the constraints alone, such as MaxSAT.
    fn objective_escapes(&self, sol: &Self::Solution, out: &mut Vec<Self::Move>) {
        let _ = (sol, out);
    }

    /// The objective of `sol` apart from its constraints, smaller is better.
    /// Zero by default.
    fn objective(&self, sol: &Self::Solution) -> f64 {
        let _ = sol;
        0.0
    }

    /// How much `mv` would change [`objective`](Self::objective) if it were
    /// applied to `sol`. Zero by default.
    fn objective_change(&self, sol: &Self::Solution, mv: &Self::Move) -> f64 {
        let _ = (sol, mv);
        0.0
    }

    /// The solution a restart starts from, given the best one found.
    ///
    /// A fresh [`new_solution`](ProblemTrait::new_solution) by default. A
    /// problem with a uniform crossover does better to take each variable from
    /// `best` or from a fresh solution with even odds, which keeps part of
    /// what the search found.
    fn restart_from(&self, best: &Self::Solution, rng: &mut SmallRng) -> Self::Solution {
        let _ = best;
        self.new_solution(rng)
    }
}
