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

/// Hamming distance between two binary assignments of equal length.
///
/// The per-problem [`Distance`](crate::trait_defs::Distance) impls of the
/// binary problems all delegate to this function.
pub fn hamming_distance(a: &[bool], b: &[bool]) -> usize {
    a.iter().zip(b).filter(|(x, y)| x != y).count()
}

/// Shared body for `SubProblemExtractable::lift_solution` when the sub-problem
/// **keeps the parent's variable indices** (MaxCut, VertexCover).
///
/// Clones `sol1`, then for every index in `sub_indices` where the parents
/// disagree (the sub-problem's free variables), flips the clone wherever it
/// differs from `sub_sol`. Fixed variables (parents agree) and free variables
/// absent from `sub_indices` inherit `sol1`. Because the flips go through the
/// problem's flip move, the lifted solution's cached `gain` and objective stay
/// consistent.
pub fn lift_binary_solution<P: BinaryProblem>(
    prob: &P,
    sol1: &P::Solution,
    sol2: &P::Solution,
    sub_sol: &P::Solution,
    sub_indices: impl Iterator<Item = usize>,
) -> P::Solution {
    let mut sol = sol1.clone();
    for i in sub_indices {
        if P::variable(sol1, i) == P::variable(sol2, i) {
            continue; // fixed variable — both parents agree
        }
        if P::variable(&sol, i) != P::variable(sub_sol, i) {
            P::flip_move(&sol, i)
                .apply_to_solution(prob, &mut sol)
                .expect("flip on a valid variable index cannot fail");
        }
    }
    sol
}

/// Shared body for `SubProblemExtractable::lift_solution` when the sub-problem
/// **re-indexes the free variables compactly** (MaxSAT, Formula): the k-th
/// disagreeing variable of the parents corresponds to variable `k` of `sub_sol`.
pub fn lift_compact_binary_solution<P: BinaryProblem>(
    prob: &P,
    sol1: &P::Solution,
    sol2: &P::Solution,
    sub_sol: &P::Solution,
    n_vars: usize,
) -> P::Solution {
    let mut sol = sol1.clone();
    let mut sub_idx = 0;
    for i in 0..n_vars {
        if P::variable(sol1, i) == P::variable(sol2, i) {
            continue; // fixed variable — not part of the sub-problem
        }
        if P::variable(&sol, i) != P::variable(sub_sol, sub_idx) {
            P::flip_move(&sol, i)
                .apply_to_solution(prob, &mut sol)
                .expect("flip on a valid variable index cannot fail");
        }
        sub_idx += 1;
    }
    sol
}

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hamming_distance_counts_disagreements() {
        assert_eq!(hamming_distance(&[], &[]), 0);
        assert_eq!(
            hamming_distance(&[true, false, true], &[true, false, true]),
            0
        );
        assert_eq!(
            hamming_distance(&[true, false, true], &[false, false, false]),
            2
        );
    }

    #[test]
    fn lift_binary_solution_fixes_agreeing_and_lifts_disagreeing_vars() {
        use crate::problem::{MaxCut, MaxCutSolution};

        let mc = MaxCut::from_edges([(0, 1, 1.0), (1, 2, 1.0), (2, 3, 1.0)]);
        // Variables 0 and 3 agree between the parents (fixed); 1 and 2 disagree (free).
        let sol1 = MaxCutSolution::new_from_assignment(&mc, vec![true, false, true, false]);
        let sol2 = MaxCutSolution::new_from_assignment(&mc, vec![true, true, false, false]);
        // The sub-solution proposes values for the free variables.
        let sub_sol = MaxCutSolution::new_from_assignment(&mc, vec![true, true, true, false]);

        let lifted = lift_binary_solution(&mc, &sol1, &sol2, &sub_sol, 0..sub_sol.x.len());

        // Fixed variables keep sol1's values; free variables take sub_sol's.
        assert_eq!(lifted.x, vec![true, true, true, false]);
        // The lifted solution's cached objective must match a from-scratch build.
        let rebuilt = MaxCutSolution::new_from_assignment(&mc, lifted.x.clone());
        assert_eq!(lifted.objective, rebuilt.objective);
        assert_eq!(lifted.gain, rebuilt.gain);
    }
}
