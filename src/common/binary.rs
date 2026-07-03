//! Generic machinery shared by the binary-variable problems
//! (MaxCut / QUBO / MaxSAT / Formula / VertexCover).

use rand::Rng;

use crate::error::OptError;
use crate::trait_defs::{BinaryProblem, MoveToNeighbor};

/// Uniform crossover over binary variables.
///
/// Clones `sol1` and, for each variable where the parents disagree, flips it to
/// `sol2`'s value with 50% probability. Because the flips go through the
/// problem's flip move, the offspring's cached `gain` and objective stay
/// consistent.
///
/// The per-problem `*UniformCrossover` operators all delegate to this function.
pub fn uniform_binary_crossover<P: BinaryProblem>(
    prob: &P,
    sol1: &P::Solution,
    sol2: &P::Solution,
    rng: &mut rand::rngs::SmallRng,
) -> Result<P::Solution, OptError> {
    let mut sol = sol1.clone();
    for i in prob.variable_indices() {
        if P::variable(&sol, i) != P::variable(sol2, i) && rng.random::<bool>() {
            P::flip_move(&sol, i).apply_to_solution(prob, &mut sol)?;
        }
    }
    Ok(sol)
}
