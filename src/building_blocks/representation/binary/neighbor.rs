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

/// Draws of two variables before [`random_differing_pair`] falls back to its
/// counting pass. A draw lands with the probability that two variables
/// differ, about one half on a balanced assignment.
const PAIR_DRAWS: usize = 16;

/// A uniformly random pair `(i, j)` with `j < i` whose values differ in
/// `sol`, or `None` when every variable holds the same value.
///
/// Two variables are drawn with [`BinaryProblem::random_variable`] and kept
/// if they differ, which is uniform over the differing pairs and O(1) per
/// draw on the built-in problems. When sixteen draws all miss, the
/// values are nearly all equal, and one O(n) pass counts each side and draws
/// one variable from each, uniform too. Either way the distribution is that of
/// sampling [`differing_pairs`] uniformly. The per-problem
/// `*SwapNeighbor::random_neighbor` impls delegate to this function.
pub fn random_differing_pair<P: BinaryProblem>(
    prob: &P,
    sol: &P::Solution,
    rng: &mut impl Rng,
) -> Option<(usize, usize)> {
    for _ in 0..PAIR_DRAWS {
        let a = prob.random_variable(rng)?;
        let b = prob.random_variable(rng)?;
        if P::variable(sol, a) != P::variable(sol, b) {
            return Some((a.max(b), a.min(b)));
        }
    }
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

#[cfg(test)]
mod tests {
    use std::collections::{HashMap, HashSet};

    use rand::SeedableRng;
    use rand::rngs::SmallRng;

    use super::*;
    use crate::building_blocks::instance::Graph;
    use crate::problem::MaxCut;

    /// Both the drawing and the counting path reach exactly the differing
    /// pairs, each about equally often.
    #[test]
    fn random_pairs_are_uniform_over_the_differing_pairs() {
        let mc = MaxCut::new(Graph::from_edges((0..12).map(|i| (i, (i + 1) % 12, 1.0))));
        let mut rng = SmallRng::seed_from_u64(5);
        // Balanced, where the draws land, and one-sided, where they mostly
        // miss and the counting pass answers.
        for ones in [6, 1] {
            let values: Vec<bool> = (0..12).map(|i| i < ones).collect();
            let sol = mc.solution_from_assignment(&values).unwrap();
            let listed: HashSet<_> = differing_pairs(&mc, &sol).collect();
            let mut counts = HashMap::new();
            for _ in 0..400 * listed.len() {
                let pair = random_differing_pair(&mc, &sol, &mut rng).unwrap();
                *counts.entry(pair).or_insert(0usize) += 1;
            }
            assert_eq!(counts.keys().copied().collect::<HashSet<_>>(), listed);
            assert!(
                counts.values().all(|&c| (250..550).contains(&c)),
                "{counts:?}"
            );
        }
        let same = mc.solution_from_assignment(&[false; 12]).unwrap();
        assert!(random_differing_pair(&mc, &same, &mut rng).is_none());
    }
}
