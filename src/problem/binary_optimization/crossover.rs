use crate::common::uniform_binary_crossover;
use crate::search_state::{Crossover, MoveToNeighbor, SubProblemExtractable};

use super::neighbor::FormulaFlipNeighbor;
use super::problem::{Constraint, Expr, FormulaProblem, FormulaSolution};

/// Uniform crossover for [`FormulaProblem`].
///
/// For each variable, the value is taken from `sol1` or `sol2`
/// with equal probability.
pub struct FormulaUniformCrossover;

impl Crossover<FormulaProblem> for FormulaUniformCrossover {
    fn crossover(
        &mut self,
        prob: &FormulaProblem,
        sol1: &FormulaSolution,
        sol2: &FormulaSolution,
        rng: &mut rand::rngs::SmallRng,
    ) -> Result<FormulaSolution, crate::error::OptError> {
        uniform_binary_crossover(prob, sol1, sol2, rng)
    }
}

/// Rewrites an [`Expr`] by substituting fixed variable values and remapping free variable indices.
///
/// - `fixed[i] = Some(val)`: variable `i` is fixed; replace `Var(i)` with `Const(val as f64)`.
/// - `fixed[i] = None`: variable `i` is free; remap using `remap[i]`.
fn substitute_and_remap(expr: &Expr, fixed: &[Option<bool>], remap: &[Option<usize>]) -> Expr {
    match expr {
        Expr::Const(c) => Expr::Const(*c),
        Expr::Var(i) => {
            if let Some(val) = fixed[*i] {
                Expr::Const(if val { 1.0 } else { 0.0 })
            } else {
                Expr::Var(remap[*i].expect("variable is neither fixed nor remapped"))
            }
        }
        Expr::Neg(e) => -substitute_and_remap(e, fixed, remap),
        Expr::Add(es) => Expr::Add(
            es.iter()
                .map(|e| substitute_and_remap(e, fixed, remap))
                .collect(),
        ),
        Expr::Mul(es) => Expr::Mul(
            es.iter()
                .map(|e| substitute_and_remap(e, fixed, remap))
                .collect(),
        ),
    }
}

fn substitute_constraint(
    c: &Constraint,
    fixed: &[Option<bool>],
    remap: &[Option<usize>],
) -> Constraint {
    match c {
        Constraint::Comparison {
            lhs,
            rel,
            rhs,
            penalty_weight,
        } => Constraint::Comparison {
            lhs: substitute_and_remap(lhs, fixed, remap),
            rel: rel.clone(),
            rhs: substitute_and_remap(rhs, fixed, remap),
            penalty_weight: *penalty_weight,
        },
        Constraint::Clamp {
            expr,
            lo,
            hi,
            penalty_weight,
        } => Constraint::Clamp {
            expr: substitute_and_remap(expr, fixed, remap),
            lo: *lo,
            hi: *hi,
            penalty_weight: *penalty_weight,
        },
    }
}

impl SubProblemExtractable for FormulaProblem {
    /// Creates a sub-[`FormulaProblem`] for variables whose values differ between the two parents.
    ///
    /// Fixed variable values (from `sol1`) are substituted into the objective and constraints.
    /// Free variable indices are remapped to `0..n_free`.
    fn extract_sub_problem(
        &self,
        sol1: &FormulaSolution,
        sol2: &FormulaSolution,
    ) -> FormulaProblem {
        let free_vars: Vec<usize> = (0..self.n_vars)
            .filter(|&i| sol1.x[i] != sol2.x[i])
            .collect();
        let n_free = free_vars.len();

        let mut fixed = vec![None::<bool>; self.n_vars];
        let mut remap = vec![None::<usize>; self.n_vars];
        for (new_idx, &old_idx) in free_vars.iter().enumerate() {
            remap[old_idx] = Some(new_idx);
        }
        for i in 0..self.n_vars {
            if remap[i].is_none() {
                fixed[i] = Some(sol1.x[i]);
            }
        }

        let new_obj = substitute_and_remap(&self.objective, &fixed, &remap);
        let new_constraints = self
            .constraints
            .iter()
            .map(|c| substitute_constraint(c, &fixed, &remap))
            .collect();

        FormulaProblem::new(n_free, new_obj, self.direction.clone(), new_constraints)
    }

    /// Lifts the sub-problem solution back to the full solution space.
    ///
    /// - Fixed variables (same value in both parents): inherit from `sol1`.
    /// - Free variables (different value): take from `sub_solution`.
    fn lift_solution(
        &self,
        sol1: &FormulaSolution,
        sol2: &FormulaSolution,
        sub_solution: &FormulaSolution,
    ) -> FormulaSolution {
        let free_vars: Vec<usize> = (0..self.n_vars)
            .filter(|&i| sol1.x[i] != sol2.x[i])
            .collect();

        let mut sol = sol1.clone();
        for (sub_idx, &orig_idx) in free_vars.iter().enumerate() {
            if sol.x[orig_idx] == sub_solution.x[sub_idx] {
                continue;
            }
            FormulaFlipNeighbor {
                i: orig_idx,
                gain: sol.gain[orig_idx],
            }
            .apply_to_solution(self, &mut sol)
            .expect("flipping should never fail");
        }
        sol
    }
}

#[cfg(test)]
mod tests {
    use crate::problem::binary_optimization::problem::{
        Expr, FormulaProblem, FormulaSolution, OptDirection, Value,
    };
    use crate::search_state::{Crossover, SubProblemExtractable};
    use rand::SeedableRng;

    use super::FormulaUniformCrossover;

    /// 3-variable problem: maximize x[0] + x[1] + x[2] (no constraints)
    fn make_prob() -> FormulaProblem {
        let obj = Expr::Add(vec![Expr::Var(0), Expr::Var(1), Expr::Var(2)]);
        FormulaProblem::new(3, obj, OptDirection::Maximize, vec![])
    }

    fn make_sol(prob: &FormulaProblem, x: Vec<bool>) -> FormulaSolution {
        let score = prob.eval_score(&x);
        let constraint_vals = prob.eval_constraint_vals(&x);
        let gain: Vec<Value> = (0..prob.n_vars)
            .map(|i| prob.calc_gain_fast(&x, &constraint_vals, i))
            .collect();
        FormulaSolution {
            x,
            gain,
            score,
            constraint_vals,
        }
    }

    #[test]
    fn test_uniform_crossover_identical_parents() {
        let prob = make_prob();
        let s = make_sol(&prob, vec![true, false, true]);
        let mut cx = FormulaUniformCrossover;
        let mut rng = rand::rngs::SmallRng::seed_from_u64(0);
        let offspring = cx.crossover(&prob, &s, &s, &mut rng).unwrap();
        assert_eq!(offspring.x, s.x);
        assert!((offspring.score - s.score).abs() < 1e-9);
    }

    #[test]
    fn test_uniform_crossover_gain_consistency() {
        let prob = make_prob();
        let a = make_sol(&prob, vec![true, false, true]);
        let b = make_sol(&prob, vec![false, true, false]);
        let mut cx = FormulaUniformCrossover;
        let mut rng = rand::rngs::SmallRng::seed_from_u64(0);
        let offspring = cx.crossover(&prob, &a, &b, &mut rng).unwrap();
        for i in 0..prob.n_vars {
            let mut flipped = offspring.x.clone();
            flipped[i] = !flipped[i];
            let expected = prob.eval_score(&flipped) - offspring.score;
            assert!(
                (offspring.gain[i] - expected).abs() < 1e-9,
                "gain[{i}] mismatch"
            );
        }
    }

    #[test]
    fn test_extract_sub_problem_size() {
        let prob = make_prob();
        let s = make_sol(&prob, vec![true, false, true]);
        let sub_same = prob.extract_sub_problem(&s, &s);
        assert_eq!(sub_same.n_vars, 0, "identical parents → 0 free variables");

        let all_f = make_sol(&prob, vec![false, false, false]);
        let all_t = make_sol(&prob, vec![true, true, true]);
        let sub_diff = prob.extract_sub_problem(&all_f, &all_t);
        assert_eq!(
            sub_diff.n_vars, 3,
            "all-different parents → 3 free variables"
        );
    }

    #[test]
    fn test_lift_solution() {
        let prob = make_prob();
        // Free: vars 1 and 2 (differ); Fixed: var 0 (same: false)
        let parent_a = make_sol(&prob, vec![false, false, false]);
        let parent_b = make_sol(&prob, vec![false, true, true]);
        let sub = prob.extract_sub_problem(&parent_a, &parent_b);

        // Sub-problem: 2 free vars remapped to indices 0 and 1
        let sub_sol = make_sol(&sub, vec![true, false]);
        let lifted = prob.lift_solution(&parent_a, &parent_b, &sub_sol);

        assert_eq!(
            lifted.x[0], parent_a.x[0],
            "fixed var 0 inherits from parent_a"
        );
        assert_eq!(
            lifted.x[1], sub_sol.x[0],
            "free var 1 (sub idx 0) from sub_solution"
        );
        assert_eq!(
            lifted.x[2], sub_sol.x[1],
            "free var 2 (sub idx 1) from sub_solution"
        );

        let expected_score = prob.eval_score(&lifted.x);
        assert!(
            (lifted.score - expected_score).abs() < 1e-9,
            "lifted score mismatch"
        );
        for i in 0..prob.n_vars {
            let mut flipped = lifted.x.clone();
            flipped[i] = !flipped[i];
            let expected_gain = prob.eval_score(&flipped) - lifted.score;
            assert!(
                (lifted.gain[i] - expected_gain).abs() < 1e-9,
                "lifted gain[{i}] mismatch"
            );
        }
    }
}
