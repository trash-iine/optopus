use rand::Rng;

use crate::search_state::{Crossover, MoveToNeighbor, SubProblemExtractable};

use super::neighbor::SatFlipNeighbor;
use super::problem::{Sat, SatSolution};

/// Uniform crossover for SAT.
///
/// For each variable, the truth value is taken from `sol1` or `sol2`
/// with equal probability.
pub struct SatUniformCrossover;

impl Crossover<Sat> for SatUniformCrossover {
    fn crossover(
        &mut self,
        prob: &Sat,
        sol1: &SatSolution,
        sol2: &SatSolution,
        rng: &mut rand::rngs::SmallRng,
    ) -> SatSolution {
        let mut sol = sol1.clone();
        for i in 0..prob.n_vars() {
            if sol.x[i] != sol2.x[i] && rng.random::<bool>() {
                SatFlipNeighbor { i, gain: sol.gain[i] }
                    .apply_to_solution(prob, &mut sol)
                    .expect("flipping a variable should never fail");
            }
        }
        sol
    }
}

impl SubProblemExtractable for Sat {
    /// Creates a sub-SAT containing only variables whose truth values differ between
    /// the two parent solutions.
    ///
    /// - Clauses satisfied by a fixed literal are omitted entirely.
    /// - Remaining clauses include only free literals, remapped to 0-indexed sub-problem variables.
    fn extract_sub_problem(
        &self,
        sol1: &SatSolution,
        sol2: &SatSolution,
    ) -> Sat {
        let free_vars: Vec<usize> =
            (0..self.n_vars()).filter(|&i| sol1.x[i] != sol2.x[i]).collect();
        let n_free = free_vars.len();

        // remap[i] = Some(new_idx) for free vars, None for fixed vars
        let mut remap = vec![None::<usize>; self.n_vars()];
        for (new_idx, &old_idx) in free_vars.iter().enumerate() {
            remap[old_idx] = Some(new_idx);
        }

        let fixed_x = &sol1.x;

        let mut sub = Sat::new(n_free);

        for clause in self.all_clauses() {
            // Skip clause if any fixed literal satisfies it
            let satisfied_by_fixed = clause.iter().any(|&lit| {
                let var = lit.unsigned_abs() as usize - 1;
                remap[var].is_none() && fixed_x[var] == (lit > 0)
            });
            if satisfied_by_fixed {
                continue;
            }

            // Keep only free literals, remapped to new variable indices (1-indexed literals)
            let free_lits: Vec<i64> = clause
                .iter()
                .filter_map(|&lit| {
                    let var = lit.unsigned_abs() as usize - 1;
                    remap[var].map(|new_var| {
                        let sign: i64 = if lit > 0 { 1 } else { -1 };
                        sign * (new_var as i64 + 1)
                    })
                })
                .collect();

            if !free_lits.is_empty() {
                sub.add_clause(free_lits);
            }
        }

        sub
    }

    /// Lifts the sub-problem solution back to the full solution space.
    ///
    /// - Fixed variables (same truth value in both parents): inherit from `sol1`.
    /// - Free variables (different truth value): take from `sub_solution`.
    fn lift_solution(
        &self,
        sol1: &SatSolution,
        sol2: &SatSolution,
        sub_solution: &SatSolution,
    ) -> SatSolution {
        let free_vars: Vec<usize> =
            (0..self.n_vars()).filter(|&i| sol1.x[i] != sol2.x[i]).collect();

        let mut sol = sol1.clone();
        for (sub_idx, &orig_idx) in free_vars.iter().enumerate() {
            if sol.x[orig_idx] == sub_solution.x[sub_idx] {
                continue;
            }
            SatFlipNeighbor { i: orig_idx, gain: sol.gain[orig_idx] }
                .apply_to_solution(self, &mut sol)
                .expect("flipping should never fail");
        }
        sol
    }
}

#[cfg(test)]
mod tests {
    use crate::problem::sat::{Sat, SatSolution};
    use crate::search_state::{Crossover, SubProblemExtractable};
    use rand::SeedableRng;

    use super::SatUniformCrossover;

    /// 3-variable SAT: (x1 ∨ x2), (¬x1 ∨ x3), (¬x2 ∨ ¬x3)
    fn make_sat() -> Sat {
        let mut sat = Sat::new(3);
        sat.add_clause([1, 2]);
        sat.add_clause([-1, 3]);
        sat.add_clause([-2, -3]);
        sat
    }

    fn make_sol(sat: &Sat, x: Vec<bool>) -> SatSolution {
        let gain: Vec<i64> = (0..sat.n_vars()).map(|i| sat.calc_gain(&x, i)).collect();
        let n_satisfied = sat.calc_satisfied(&x);
        SatSolution { x, gain, n_satisfied }
    }

    #[test]
    fn test_uniform_crossover_identical_parents() {
        let sat = make_sat();
        let s = make_sol(&sat, vec![true, false, true]);
        let mut cx = SatUniformCrossover;
        let mut rng = rand::rngs::SmallRng::seed_from_u64(0);
        let offspring = cx.crossover(&sat, &s, &s, &mut rng);
        assert_eq!(offspring.x, s.x);
        assert_eq!(offspring.n_satisfied, s.n_satisfied);
    }

    #[test]
    fn test_uniform_crossover_gain_consistency() {
        let sat = make_sat();
        let a = make_sol(&sat, vec![true, false, true]);
        let b = make_sol(&sat, vec![false, true, false]);
        let mut cx = SatUniformCrossover;
        let mut rng = rand::rngs::SmallRng::seed_from_u64(0);
        let offspring = cx.crossover(&sat, &a, &b, &mut rng);
        let n_sat = offspring.n_satisfied as i64;
        for i in 0..sat.n_vars() {
            let mut flipped = offspring.x.clone();
            flipped[i] = !flipped[i];
            let expected = sat.calc_satisfied(&flipped) as i64 - n_sat;
            assert_eq!(offspring.gain[i], expected, "gain[{i}] mismatch");
        }
    }

    #[test]
    fn test_extract_sub_problem_size() {
        let sat = make_sat();
        let s = make_sol(&sat, vec![true, false, true]);
        let sub_same = sat.extract_sub_problem(&s, &s);
        assert_eq!(sub_same.n_vars(), 0, "identical parents → 0 free variables");

        let all_f = make_sol(&sat, vec![false, false, false]);
        let all_t = make_sol(&sat, vec![true, true, true]);
        let sub_diff = sat.extract_sub_problem(&all_f, &all_t);
        assert_eq!(sub_diff.n_vars(), 3, "all-different parents → 3 free variables");
    }

    #[test]
    fn test_lift_solution() {
        let sat = make_sat();
        // Free: vars 1 and 2 (0-indexed); Fixed: var 0 (same: false)
        let parent_a = make_sol(&sat, vec![false, false, false]);
        let parent_b = make_sol(&sat, vec![false, true, true]);
        let sub = sat.extract_sub_problem(&parent_a, &parent_b);

        // Sub-problem: 2 free vars remapped to indices 0 and 1
        let sub_sol = make_sol(&sub, vec![true, false]);
        let lifted = sat.lift_solution(&parent_a, &parent_b, &sub_sol);

        assert_eq!(lifted.x[0], parent_a.x[0], "fixed var 0 inherits from parent_a");
        assert_eq!(lifted.x[1], sub_sol.x[0], "free var 1 (sub idx 0) from sub_solution");
        assert_eq!(lifted.x[2], sub_sol.x[1], "free var 2 (sub idx 1) from sub_solution");
        assert_eq!(lifted.n_satisfied, sat.calc_satisfied(&lifted.x));
    }
}
