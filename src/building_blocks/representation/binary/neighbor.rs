//! The swap neighborhood every binary problem builds from its flip move.

use rand::Rng;

use crate::error::OptError;
use crate::trait_defs::{BinaryProblem, MoveToNeighbor};

/// Applies a swap move as two sequential flips (`i` then `j`).
///
/// The second flip reads the gain cached after the first, so the interaction
/// term between the two variables is accounted for. The per-problem
/// `*SwapNeighbor::apply_to_solution` impls delegate to this function.
pub fn apply_swap_as_two_flips<P: BinaryProblem>(
    prob: &P,
    sol: &mut P::Solution,
    i: usize,
    j: usize,
) -> Result<(), OptError> {
    P::flip_move(sol, i).apply_to_solution(prob, sol)?;
    P::flip_move(sol, j).apply_to_solution(prob, sol)?;
    Ok(())
}

/// Every pair `(i, j)` with `j < i` whose values differ in `sol`, in the
/// order the swap neighborhoods scan them. The per-problem
/// `*SwapNeighbor::iter` impls delegate to this function.
pub fn differing_pairs<P: BinaryProblem + Sync>(
    prob: &P,
    sol: &P::Solution,
) -> impl Iterator<Item = (usize, usize)> + Send
where
    P::Solution: Sync,
{
    prob.variable_indices().flat_map(move |i| {
        prob.variable_indices()
            .filter(move |&j| j < i && P::variable(sol, i) != P::variable(sol, j))
            .map(move |j| (i, j))
    })
}

/// A uniformly random pair `(i, j)` with `j < i` whose values differ in
/// `sol`, or `None` when every variable holds the same value.
///
/// O(n) and allocation free: one pass counts each side, then one variable is
/// drawn from each side, so every differing pair is hit with equal
/// probability, the distribution of sampling [`differing_pairs`] uniformly.
/// The per-problem `*SwapNeighbor::random_neighbor` impls delegate to this
/// function.
pub fn random_differing_pair<P: BinaryProblem>(
    prob: &P,
    sol: &P::Solution,
    rng: &mut impl Rng,
) -> Option<(usize, usize)> {
    let (mut n_false, mut n_true) = (0, 0);
    for i in prob.variable_indices() {
        if P::variable(sol, i) {
            n_true += 1;
        } else {
            n_false += 1;
        }
    }
    if n_false == 0 || n_true == 0 {
        return None;
    }
    let k_false = rng.random_range(0..n_false);
    let k_true = rng.random_range(0..n_true);
    let a = prob
        .variable_indices()
        .filter(|&i| !P::variable(sol, i))
        .nth(k_false)?;
    let b = prob
        .variable_indices()
        .filter(|&i| P::variable(sol, i))
        .nth(k_true)?;
    Some((a.max(b), a.min(b)))
}
