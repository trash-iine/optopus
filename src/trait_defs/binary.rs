//! Trait for problems over binary variables.

use super::{MoveToNeighbor, ProblemTrait};

/// A problem whose solutions assign a boolean value to each variable and cache
/// a per-variable flip gain.
///
/// MaxCut, QUBO, MaxSAT, [`FormulaProblem`](crate::problem::binary_optimization::FormulaProblem),
/// and Vertex Cover all fit this shape. Implementing this trait gives a problem
/// access to the generic binary-variable machinery in [`crate::common`], such as
/// [`uniform_binary_crossover`](crate::common::uniform_binary_crossover).
pub trait BinaryProblem: ProblemTrait + Sized {
    /// The single-variable flip move for this problem.
    type Flip: MoveToNeighbor<Self>;

    /// Returns an iterator over the indices of the problem's binary variables.
    fn variable_indices(&self) -> impl Iterator<Item = usize> + '_;

    /// Returns the value of variable `i` in `sol`.
    fn variable(sol: &Self::Solution, i: usize) -> bool;

    /// Returns the flip move for variable `i`, carrying the gain cached in `sol`.
    fn flip_move(sol: &Self::Solution, i: usize) -> Self::Flip;
}
