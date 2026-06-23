use rand::Rng;

use crate::search_state::{Crossover, MoveToNeighbor, SubProblemExtractable};

use super::neighbor::QuboFlipNeighbor;
use super::problem::{Qubo, QuboSolution, make_sub_problem_from};

/// Uniform crossover for QUBO.
///
/// For each variable, the value is taken from `sol1` or `sol2`
/// with equal probability.
pub struct QuboUniformCrossover;

impl Crossover<Qubo> for QuboUniformCrossover {
    fn crossover(
        &mut self,
        prob: &Qubo,
        sol1: &QuboSolution,
        sol2: &QuboSolution,
        rng: &mut rand::rngs::SmallRng,
    ) -> Result<QuboSolution, crate::error::OptError> {
        let mut sol = sol1.clone();
        for &i in prob.iter_on_variables() {
            if sol.x[i] != sol2.x[i] && rng.random::<bool>() {
                QuboFlipNeighbor {
                    i,
                    gain: sol.gain[i],
                }
                .apply_to_solution(prob, &mut sol)?;
            }
        }
        Ok(sol)
    }
}

impl SubProblemExtractable for Qubo {
    /// Creates a sub-QUBO for variables that differ between the two parent solutions.
    ///
    /// Fixed variable contributions are folded into the diagonal terms of the sub-QUBO,
    /// using the same approach as the internal `make_sub_problem_from` helper.
    fn extract_sub_problem(&self, sol1: &QuboSolution, sol2: &QuboSolution) -> Qubo {
        make_sub_problem_from(self, &[sol1, sol2])
    }

    /// Lifts the sub-problem solution back to the full solution space.
    ///
    /// - Fixed variables (same value in both parents): inherit from `sol1`.
    /// - Free variables (different value): take from `sub_solution`.
    fn lift_solution(
        &self,
        sol1: &QuboSolution,
        _sol2: &QuboSolution,
        sub_solution: &QuboSolution,
    ) -> QuboSolution {
        let mut sol = sol1.clone();
        for i in sub_solution.iter_on_variables() {
            if sol.x[i] == sub_solution.x[i] {
                continue;
            }
            QuboFlipNeighbor {
                i,
                gain: sol.gain[i],
            }
            .apply_to_solution(self, &mut sol)
            .expect("flipping should never fail");
        }
        sol
    }
}

#[cfg(test)]
mod tests {
    use crate::problem::qubo::{Qubo, QuboSolution};
    use crate::search_state::{Crossover, SubProblemExtractable};
    use rand::SeedableRng;

    use super::QuboUniformCrossover;

    fn make_qubo() -> Qubo {
        let mut qubo = Qubo::new();
        qubo.set_q(0, 1, 1);
        qubo.set_q(1, 2, 2);
        qubo.set_q(0, 2, 3);
        qubo
    }

    fn make_sol(qubo: &Qubo, assignments: &[(usize, bool)]) -> QuboSolution {
        let n = qubo
            .iter_on_variables()
            .copied()
            .max()
            .map(|m| m + 1)
            .unwrap_or(0);
        let mut x = vec![false; n];
        for &(i, v) in assignments {
            x[i] = v;
        }
        QuboSolution::new_from_assignment(qubo, x)
    }

    #[test]
    fn test_uniform_crossover_identical_parents() {
        let qubo = make_qubo();
        let s = make_sol(&qubo, &[(0, false), (1, true), (2, false)]);
        let mut cx = QuboUniformCrossover;
        let mut rng = rand::rngs::SmallRng::seed_from_u64(0);
        let offspring = cx.crossover(&qubo, &s, &s, &mut rng).unwrap();
        assert_eq!(offspring.x, s.x);
        assert_eq!(offspring.objective, s.objective);
    }

    #[test]
    fn test_uniform_crossover_gain_consistency() {
        let qubo = make_qubo();
        let a = make_sol(&qubo, &[(0, false), (1, true), (2, false)]);
        let b = make_sol(&qubo, &[(0, true), (1, false), (2, true)]);
        let mut cx = QuboUniformCrossover;
        let mut rng = rand::rngs::SmallRng::seed_from_u64(0);
        let offspring = cx.crossover(&qubo, &a, &b, &mut rng).unwrap();
        for &i in qubo.iter_on_variables() {
            let g = offspring.gain[i];
            let mut flipped = offspring.x.clone();
            flipped[i] = !flipped[i];
            let expected = qubo.calculate_energy(&flipped) - offspring.objective;
            assert_eq!(g, expected, "gain[{i}] mismatch");
        }
    }

    #[test]
    fn test_extract_sub_problem_size() {
        let qubo = make_qubo();
        let s = make_sol(&qubo, &[(0, false), (1, true), (2, false)]);
        let sub_same = qubo.extract_sub_problem(&s, &s);
        assert_eq!(
            sub_same.num_of_variables(),
            0,
            "identical parents → 0 free variables"
        );

        let all_f = make_sol(&qubo, &[(0, false), (1, false), (2, false)]);
        let all_t = make_sol(&qubo, &[(0, true), (1, true), (2, true)]);
        let sub_diff = qubo.extract_sub_problem(&all_f, &all_t);
        assert_eq!(
            sub_diff.num_of_variables(),
            3,
            "all-different parents → 3 free variables"
        );
    }

    #[test]
    fn test_lift_solution() {
        let qubo = make_qubo();
        // Free: vars 0 and 1 (differ); Fixed: var 2 (same: false)
        let parent_a = make_sol(&qubo, &[(0, false), (1, false), (2, false)]);
        let parent_b = make_sol(&qubo, &[(0, true), (1, true), (2, false)]);
        let sub = qubo.extract_sub_problem(&parent_a, &parent_b);

        // Sub-problem has vars 0 and 1 (original IDs) with Q[0,1]=1
        let sub_sol = make_sol(&sub, &[(0, true), (1, false)]);
        let lifted = qubo.lift_solution(&parent_a, &parent_b, &sub_sol);

        assert_eq!(
            lifted.x[2], parent_a.x[2],
            "fixed var 2 inherits from parent_a"
        );
        assert_eq!(
            lifted.x[0], sub_sol.x[0],
            "free var 0 comes from sub_solution"
        );
        assert_eq!(
            lifted.x[1], sub_sol.x[1],
            "free var 1 comes from sub_solution"
        );

        for &i in qubo.iter_on_variables() {
            let g = lifted.gain[i];
            let mut flipped = lifted.x.clone();
            flipped[i] = !flipped[i];
            let expected = qubo.calculate_energy(&flipped) - lifted.objective;
            assert_eq!(g, expected, "lifted gain[{i}] mismatch");
        }
    }
}
