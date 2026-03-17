use std::collections::HashMap;

use super::definition::{FormulaProblem, FormulaSolution, Value};
use crate::search_state::{EnabledTabu, Evaluable, MoveToNeigbor, Rankable};

// ---------------------------------------------------------------------------
// Flip 近傍 (1-opt: 変数を 1 つフリップ)
// ---------------------------------------------------------------------------

/// 変数 i を 1 つフリップする近傍
#[derive(Debug, Clone)]
pub struct FormulaFlipNeighbor {
    pub i: usize,
    /// スコアの変化量 (正 = 改善)
    pub gain: Value,
}

impl Rankable for FormulaFlipNeighbor {
    fn is_better_than(&self, other: &Self) -> bool {
        self.gain > other.gain
    }
}

impl Evaluable<f64> for FormulaFlipNeighbor {
    /// SA 用: 正 = 悪化量
    fn evaluate(&self) -> f64 {
        -self.gain
    }
}

impl EnabledTabu for FormulaFlipNeighbor {
    type TabuMap = HashMap<usize, u64>;

    fn is_move_enabled(&self, tabu_map: &Self::TabuMap, iteration: u64) -> bool {
        tabu_map.get(&self.i).map_or(true, |&t| iteration > t)
    }

    fn add_to_tabu_map(
        &self,
        tabu_map: &mut Self::TabuMap,
        iteration: u64,
        tabu_tenure: (u64, u64),
    ) {
        let d = rand::random_range(tabu_tenure.0..=tabu_tenure.1);
        tabu_map.insert(self.i, iteration + d);
    }
}

impl MoveToNeigbor<FormulaProblem> for FormulaFlipNeighbor {
    fn apply_to_solution(&self, prob: &FormulaProblem, sol: &mut FormulaSolution) {
        sol.x[self.i] = !sol.x[self.i];
        sol.score += self.gain;
        sol.gain[self.i] = -self.gain;

        // i 以外の全変数のゲインを再計算
        for j in 0..prob.n_vars {
            if j != self.i {
                sol.gain[j] = prob.calc_gain(&sol.x, j);
            }
        }
    }

    fn iter(prob: &FormulaProblem, sol: &FormulaSolution) -> impl Iterator<Item = Self> + Send {
        let n = prob.n_vars;
        let gain = sol.gain.clone();
        (0..n).map(move |i| FormulaFlipNeighbor { i, gain: gain[i] })
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

// ---------------------------------------------------------------------------
// Swap 近傍 (2-opt: 変数 2 つを同時にフリップ)
// ---------------------------------------------------------------------------

/// 変数 i と j を同時にフリップする近傍
#[derive(Debug, Clone)]
pub struct FormulaSwapNeighbor {
    pub i: usize,
    pub j: usize,
    /// スコアの変化量 (正 = 改善)
    pub gain: Value,
}

impl Rankable for FormulaSwapNeighbor {
    fn is_better_than(&self, other: &Self) -> bool {
        self.gain > other.gain
    }
}

impl Evaluable<f64> for FormulaSwapNeighbor {
    fn evaluate(&self) -> f64 {
        -self.gain
    }
}

impl EnabledTabu for FormulaSwapNeighbor {
    type TabuMap = HashMap<usize, u64>;

    fn is_move_enabled(&self, tabu_map: &Self::TabuMap, iteration: u64) -> bool {
        let ok_i = tabu_map.get(&self.i).map_or(true, |&t| iteration > t);
        let ok_j = tabu_map.get(&self.j).map_or(true, |&t| iteration > t);
        ok_i && ok_j
    }

    fn add_to_tabu_map(
        &self,
        tabu_map: &mut Self::TabuMap,
        iteration: u64,
        tabu_tenure: (u64, u64),
    ) {
        let d = rand::random_range(tabu_tenure.0..=tabu_tenure.1);
        tabu_map.insert(self.i, iteration + d);
        let d = rand::random_range(tabu_tenure.0..=tabu_tenure.1);
        tabu_map.insert(self.j, iteration + d);
    }
}

impl MoveToNeigbor<FormulaProblem> for FormulaSwapNeighbor {
    fn apply_to_iteration(&self, iter: u64) -> u64 {
        iter + 2
    }

    fn apply_to_solution(&self, prob: &FormulaProblem, sol: &mut FormulaSolution) {
        let flip_i = FormulaFlipNeighbor { i: self.i, gain: sol.gain[self.i] };
        flip_i.apply_to_solution(prob, sol);

        let flip_j = FormulaFlipNeighbor { i: self.j, gain: sol.gain[self.j] };
        flip_j.apply_to_solution(prob, sol);
    }

    fn iter(prob: &FormulaProblem, sol: &FormulaSolution) -> impl Iterator<Item = Self> + Send {
        let n = prob.n_vars;
        let x = sol.x.clone();
        let gain = sol.gain.clone();

        let mut items = Vec::with_capacity(n * n / 2);
        for i in 0..n {
            for j in (i + 1)..n {
                let gain_j_after_flip_i = prob.calc_gain_with_virtual_flip(&x, i, j);
                let swap_gain = gain[i] + gain_j_after_flip_i;
                items.push(FormulaSwapNeighbor { i, j, gain: swap_gain });
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

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::problem::formula::definition::{
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
        let gain: Vec<Value> = (0..prob.n_vars).map(|i| prob.calc_gain(&x, i)).collect();
        FormulaSolution { x, gain, score }
    }

    #[test]
    fn test_search_state_new() {
        let prob = make_problem();
        let _state = SearchState::new(&prob, rand::rng());
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
                neighbor.i, neighbor.gain, expected
            );
        }
    }

    #[test]
    fn test_flip_apply_consistency() {
        let prob = make_problem();
        let sol = make_solution(&prob, vec![false, true, false]);

        for neighbor in FormulaFlipNeighbor::iter(&prob, &sol) {
            let mut s = sol.clone();
            neighbor.apply_to_solution(&prob, &mut s);

            let expected_score = prob.eval_score(&s.x);
            assert!(
                (s.score - expected_score).abs() < 1e-9,
                "flip {}: score={} expected={}",
                neighbor.i, s.score, expected_score
            );

            for i in 0..prob.n_vars {
                let expected_gain = prob.calc_gain(&s.x, i);
                assert!(
                    (s.gain[i] - expected_gain).abs() < 1e-9,
                    "flip {}: gain[{}]={} expected={}",
                    neighbor.i, i, s.gain[i], expected_gain
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
                neighbor.i, neighbor.j, neighbor.gain, expected
            );
        }
    }

    #[test]
    fn test_swap_apply_consistency() {
        let prob = make_problem();
        let sol = make_solution(&prob, vec![false, true, false]);

        for neighbor in FormulaSwapNeighbor::iter(&prob, &sol) {
            let mut s = sol.clone();
            neighbor.apply_to_solution(&prob, &mut s);

            let expected_score = prob.eval_score(&s.x);
            assert!(
                (s.score - expected_score).abs() < 1e-9,
                "swap ({},{}): score={} expected={}",
                neighbor.i, neighbor.j, s.score, expected_score
            );

            for i in 0..prob.n_vars {
                let expected_gain = prob.calc_gain(&s.x, i);
                assert!(
                    (s.gain[i] - expected_gain).abs() < 1e-9,
                    "swap ({},{}): gain[{}]={} expected={}",
                    neighbor.i, neighbor.j, i, s.gain[i], expected_gain
                );
            }
        }
    }
}