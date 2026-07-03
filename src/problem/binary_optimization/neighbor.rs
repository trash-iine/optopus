use super::problem::{FormulaProblem, FormulaSolution, Value};
use crate::{
    common::{VarTabuMap, add_var_to_tabu, is_var_enabled},
    error::OptError,
    search_state::{EnabledTabu, Evaluable, Evaluate, MoveToNeighbor, Rankable},
};

/// A flip move that toggles a single variable `i`.
///
/// `gain` is the change in score after the flip (positive = improvement).
#[derive(Debug, Clone)]
pub struct FormulaFlipNeighbor {
    /// Index of the variable to flip.
    pub i: usize,
    /// Change in score when this variable is flipped (positive = improvement).
    pub gain: Value,
}

impl Rankable for FormulaFlipNeighbor {
    fn is_better_than(&self, other: &Self) -> bool {
        self.gain > other.gain
    }
}

impl Evaluate for FormulaFlipNeighbor {
    fn evaluate(&self) -> Evaluable<f64> {
        Evaluable::Maximize(self.gain)
    }
}

impl EnabledTabu for FormulaFlipNeighbor {
    type TabuMap = VarTabuMap;

    fn is_move_enabled(&self, tabu_map: &Self::TabuMap, iteration: u64) -> bool {
        is_var_enabled(tabu_map, self.i, iteration)
    }

    fn add_to_tabu_map(
        &self,
        tabu_map: &mut Self::TabuMap,
        iteration: u64,
        tabu_tenure: (u64, u64),
    ) {
        add_var_to_tabu(tabu_map, self.i, iteration, tabu_tenure);
    }
}

impl MoveToNeighbor<FormulaProblem> for FormulaFlipNeighbor {
    fn apply_to_solution(
        &self,
        prob: &FormulaProblem,
        sol: &mut FormulaSolution,
    ) -> Result<(), OptError> {
        // Update constraint_vals BEFORE flipping (eval_poly_delta uses current x[i] value)
        for (cv, poly) in sol
            .constraint_vals
            .iter_mut()
            .zip(prob.constraint_polys.iter())
        {
            *cv += prob.eval_poly_delta(poly, &sol.x, self.i);
        }

        sol.x[self.i] = !sol.x[self.i];
        sol.score += self.gain;
        sol.gain[self.i] = -self.gain;

        // Only recompute gains for variables that share a monomial with `self.i`.
        // Variables with no shared monomial are guaranteed to be unaffected.
        for &j in &prob.interaction_neighbors[self.i] {
            sol.gain[j] = prob.calc_gain_fast(&sol.x, &sol.constraint_vals, j);
        }
        Ok(())
    }

    fn iter(prob: &FormulaProblem, sol: &FormulaSolution) -> impl Iterator<Item = Self> + Send {
        let n = prob.n_vars;
        (0..n).map(move |i| FormulaFlipNeighbor {
            i,
            gain: sol.gain[i],
        })
    }

    fn move_to_be_better_than(
        &self,
        _: &FormulaProblem,
        src: &FormulaSolution,
        other: &FormulaSolution,
    ) -> bool {
        src.score + self.gain > other.score
    }
}

/// A swap move that simultaneously flips variables `i` and `j`.
///
/// `gain` is the combined change in score (positive = improvement).
#[derive(Debug, Clone)]
pub struct FormulaSwapNeighbor {
    pub i: usize,
    pub j: usize,
    /// Combined change in score (positive = improvement).
    pub gain: Value,
}

impl Rankable for FormulaSwapNeighbor {
    fn is_better_than(&self, other: &Self) -> bool {
        self.gain > other.gain
    }
}

impl Evaluate for FormulaSwapNeighbor {
    fn evaluate(&self) -> Evaluable<f64> {
        Evaluable::Maximize(self.gain)
    }
}

impl EnabledTabu for FormulaSwapNeighbor {
    type TabuMap = VarTabuMap;

    fn is_move_enabled(&self, tabu_map: &Self::TabuMap, iteration: u64) -> bool {
        is_var_enabled(tabu_map, self.i, iteration) && is_var_enabled(tabu_map, self.j, iteration)
    }

    fn add_to_tabu_map(
        &self,
        tabu_map: &mut Self::TabuMap,
        iteration: u64,
        tabu_tenure: (u64, u64),
    ) {
        add_var_to_tabu(tabu_map, self.i, iteration, tabu_tenure);
        add_var_to_tabu(tabu_map, self.j, iteration, tabu_tenure);
    }
}

impl MoveToNeighbor<FormulaProblem> for FormulaSwapNeighbor {
    fn apply_to_iteration(&self, iter: u64) -> u64 {
        iter + 2
    }

    fn apply_to_solution(
        &self,
        prob: &FormulaProblem,
        sol: &mut FormulaSolution,
    ) -> Result<(), OptError> {
        let flip_i = FormulaFlipNeighbor {
            i: self.i,
            gain: sol.gain[self.i],
        };
        flip_i.apply_to_solution(prob, sol)?;

        let flip_j = FormulaFlipNeighbor {
            i: self.j,
            gain: sol.gain[self.j],
        };
        flip_j.apply_to_solution(prob, sol)?;
        Ok(())
    }

    fn iter(prob: &FormulaProblem, sol: &FormulaSolution) -> impl Iterator<Item = Self> + Send {
        let n = prob.n_vars;
        let mut x = sol.x.clone();
        let mut cv = sol.constraint_vals.clone();

        let mut items = Vec::with_capacity(n * (n - 1) / 2);
        for i in 0..n {
            // Pre-compute constraint deltas for flipping i (needed for undo as well)
            let delta_i: Vec<Value> = prob
                .constraint_polys
                .iter()
                .map(|poly| prob.eval_poly_delta(poly, &x, i))
                .collect();
            let gain_i = prob.calc_gain_fast(&x, &cv, i);

            // Apply virtual flip of i
            for (v, &d) in cv.iter_mut().zip(delta_i.iter()) {
                *v += d;
            }
            x[i] = !x[i];

            for j in (i + 1)..n {
                let gain_j = prob.calc_gain_fast(&x, &cv, j);
                items.push(FormulaSwapNeighbor {
                    i,
                    j,
                    gain: gain_i + gain_j,
                });
            }

            // Restore virtual flip of i
            x[i] = !x[i];
            for (v, &d) in cv.iter_mut().zip(delta_i.iter()) {
                *v -= d;
            }
        }
        items.into_iter()
    }

    fn move_to_be_better_than(
        &self,
        _: &FormulaProblem,
        src: &FormulaSolution,
        other: &FormulaSolution,
    ) -> bool {
        src.score + self.gain > other.score
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::problem::binary_optimization::problem::{
        Constraint, ConstraintRel, Expr, FormulaProblem, OptDirection,
    };
    use crate::search_state::SearchState;

    fn make_problem() -> FormulaProblem {
        let objective = Expr::Add(vec![
            Expr::Var(0),
            Expr::Mul(vec![Expr::Const(2.0), Expr::Var(1)]),
            Expr::Mul(vec![Expr::Const(3.0), Expr::Var(2)]),
        ]);
        let constraint = Constraint::Comparison {
            lhs: Expr::Add(vec![Expr::Var(0), Expr::Var(1), Expr::Var(2)]),
            rel: ConstraintRel::Le,
            rhs: Expr::Const(2.0),
            penalty_weight: 10.0,
        };
        FormulaProblem::new(3, objective, OptDirection::Maximize, vec![constraint])
    }

    fn make_solution(prob: &FormulaProblem, x: Vec<bool>) -> FormulaSolution {
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
    fn test_search_state_new() {
        let prob = make_problem();
        let _state = SearchState::new(&prob);
    }

    #[test]
    fn test_flip_gain_matches_score_delta() {
        let prob = make_problem();
        let sol = make_solution(&prob, vec![false, true, false]);

        for neighbor in FormulaFlipNeighbor::iter(&prob, &sol) {
            let mut x2 = sol.x.clone();
            x2[neighbor.i] = !x2[neighbor.i];
            let expected = prob.eval_score(&x2) - sol.score;
            assert!(
                (neighbor.gain - expected).abs() < 1e-9,
                "flip {}: gain={} expected={}",
                neighbor.i,
                neighbor.gain,
                expected
            );
        }
    }

    #[test]
    fn test_flip_apply_consistency() {
        let prob = make_problem();
        let sol = make_solution(&prob, vec![false, true, false]);

        for neighbor in FormulaFlipNeighbor::iter(&prob, &sol) {
            let mut s = sol.clone();
            neighbor.apply_to_solution(&prob, &mut s).unwrap();

            let expected_score = prob.eval_score(&s.x);
            assert!(
                (s.score - expected_score).abs() < 1e-9,
                "flip {}: score={} expected={}",
                neighbor.i,
                s.score,
                expected_score
            );

            for i in 0..prob.n_vars {
                let expected_gain = prob.calc_gain(&s.x, i);
                assert!(
                    (s.gain[i] - expected_gain).abs() < 1e-9,
                    "flip {}: gain[{}]={} expected={}",
                    neighbor.i,
                    i,
                    s.gain[i],
                    expected_gain
                );
            }
        }
    }

    #[test]
    fn test_swap_gain_matches_score_delta() {
        let prob = make_problem();
        let sol = make_solution(&prob, vec![false, true, false]);

        for neighbor in FormulaSwapNeighbor::iter(&prob, &sol) {
            let mut x2 = sol.x.clone();
            x2[neighbor.i] = !x2[neighbor.i];
            x2[neighbor.j] = !x2[neighbor.j];
            let expected = prob.eval_score(&x2) - sol.score;
            assert!(
                (neighbor.gain - expected).abs() < 1e-9,
                "swap ({},{}): gain={} expected={}",
                neighbor.i,
                neighbor.j,
                neighbor.gain,
                expected
            );
        }
    }

    #[test]
    fn test_swap_apply_consistency() {
        let prob = make_problem();
        let sol = make_solution(&prob, vec![false, true, false]);

        for neighbor in FormulaSwapNeighbor::iter(&prob, &sol) {
            let mut s = sol.clone();
            neighbor.apply_to_solution(&prob, &mut s).unwrap();

            let expected_score = prob.eval_score(&s.x);
            assert!(
                (s.score - expected_score).abs() < 1e-9,
                "swap ({},{}): score={} expected={}",
                neighbor.i,
                neighbor.j,
                s.score,
                expected_score
            );

            for i in 0..prob.n_vars {
                let expected_gain = prob.calc_gain(&s.x, i);
                assert!(
                    (s.gain[i] - expected_gain).abs() < 1e-9,
                    "swap ({},{}): gain[{}]={} expected={}",
                    neighbor.i,
                    neighbor.j,
                    i,
                    s.gain[i],
                    expected_gain
                );
            }
        }
    }
}
