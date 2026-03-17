use std::ops::{Add, Mul, Neg, Sub};

use crate::search_state::{ProblemTrait, Rankable};

pub type Value = f64;

/// 数式の AST ノード
#[derive(Debug, Clone)]
pub enum Expr {
    /// 定数
    Const(Value),
    /// 2値変数 (0-indexed)
    Var(usize),
    /// 符号反転
    Neg(Box<Expr>),
    /// 加算 (可変長)
    Add(Vec<Expr>),
    /// 乗算 (可変長; 2値変数間では AND に相当)
    Mul(Vec<Expr>),
}

// ---------------------------------------------------------------------------
// 演算子オーバーロード
// ---------------------------------------------------------------------------

impl Neg for Expr {
    type Output = Expr;
    fn neg(self) -> Expr {
        Expr::Neg(Box::new(self))
    }
}

impl Add for Expr {
    type Output = Expr;
    fn add(self, rhs: Expr) -> Expr {
        Expr::Add(vec![self, rhs])
    }
}

impl Sub for Expr {
    type Output = Expr;
    fn sub(self, rhs: Expr) -> Expr {
        Expr::Add(vec![self, -rhs])
    }
}

impl Mul for Expr {
    type Output = Expr;
    fn mul(self, rhs: Expr) -> Expr {
        Expr::Mul(vec![self, rhs])
    }
}

impl Add<Value> for Expr {
    type Output = Expr;
    fn add(self, rhs: Value) -> Expr {
        Expr::Add(vec![self, Expr::Const(rhs)])
    }
}

impl Sub<Value> for Expr {
    type Output = Expr;
    fn sub(self, rhs: Value) -> Expr {
        Expr::Add(vec![self, Expr::Const(-rhs)])
    }
}

impl Mul<Value> for Expr {
    type Output = Expr;
    fn mul(self, rhs: Value) -> Expr {
        Expr::Mul(vec![self, Expr::Const(rhs)])
    }
}

impl Add<Expr> for Value {
    type Output = Expr;
    fn add(self, rhs: Expr) -> Expr {
        Expr::Add(vec![Expr::Const(self), rhs])
    }
}

impl Sub<Expr> for Value {
    type Output = Expr;
    fn sub(self, rhs: Expr) -> Expr {
        Expr::Add(vec![Expr::Const(self), -rhs])
    }
}

impl Mul<Expr> for Value {
    type Output = Expr;
    fn mul(self, rhs: Expr) -> Expr {
        Expr::Mul(vec![Expr::Const(self), rhs])
    }
}

/// 制約の比較演算子
#[derive(Debug, Clone)]
pub enum ConstraintRel {
    Lt,
    Gt,
    Le,
    Ge,
    Eq,
}

/// 制約
#[derive(Debug, Clone)]
pub enum Constraint {
    /// lhs rel rhs の形式
    Comparison {
        lhs: Expr,
        rel: ConstraintRel,
        rhs: Expr,
        penalty_weight: Value,
    },
    /// lo <= expr <= hi の形式
    Clamp {
        expr: Expr,
        lo: Value,
        hi: Value,
        penalty_weight: Value,
    },
}

/// 最適化の方向
#[derive(Debug, Clone)]
pub enum OptDirection {
    Maximize,
    Minimize,
}

/// 数式形式の問題
#[derive(Debug, Clone)]
pub struct FormulaProblem {
    pub n_vars: usize,
    pub objective: Expr,
    pub direction: OptDirection,
    pub constraints: Vec<Constraint>,
}

/// 数式問題の解
#[derive(Debug, Clone)]
pub struct FormulaSolution {
    /// 変数の割り当て (x[i] ∈ {false=0, true=1})
    pub x: Vec<bool>,
    /// gain[i] = x[i] をフリップしたときのスコア変化量 (正 = 改善)
    pub gain: Vec<Value>,
    /// 現在のスコア (方向・ペナルティ込み; 常に大きい方が良い)
    pub score: Value,
}

impl Rankable for FormulaSolution {
    fn is_better_than(&self, other: &Self) -> bool {
        self.score > other.score
    }
}

// ---------------------------------------------------------------------------
// 式の評価
// ---------------------------------------------------------------------------

pub fn eval_expr(expr: &Expr, x: &[bool]) -> Value {
    match expr {
        Expr::Const(c) => *c,
        Expr::Var(i) => {
            if x[*i] {
                1.0
            } else {
                0.0
            }
        }
        Expr::Neg(e) => -eval_expr(e, x),
        Expr::Add(es) => es.iter().map(|e| eval_expr(e, x)).sum(),
        Expr::Mul(es) => es.iter().map(|e| eval_expr(e, x)).product(),
    }
}

// ---------------------------------------------------------------------------
// 制約ペナルティの評価
// ---------------------------------------------------------------------------

const STRICT_EPSILON: Value = 1e-9;

fn eval_constraint_penalty(c: &Constraint, x: &[bool]) -> Value {
    match c {
        Constraint::Comparison {
            lhs,
            rel,
            rhs,
            penalty_weight,
        } => {
            let l = eval_expr(lhs, x);
            let r = eval_expr(rhs, x);
            let violation = match rel {
                ConstraintRel::Lt => (l - r + STRICT_EPSILON).max(0.0),
                ConstraintRel::Gt => (r - l + STRICT_EPSILON).max(0.0),
                ConstraintRel::Le => (l - r).max(0.0),
                ConstraintRel::Ge => (r - l).max(0.0),
                ConstraintRel::Eq => (l - r).abs(),
            };
            violation * penalty_weight
        }
        Constraint::Clamp {
            expr,
            lo,
            hi,
            penalty_weight,
        } => {
            let v = eval_expr(expr, x);
            let violation = (lo - v).max(0.0) + (v - hi).max(0.0);
            violation * penalty_weight
        }
    }
}

// ---------------------------------------------------------------------------
// FormulaProblem のメソッド
// ---------------------------------------------------------------------------

impl FormulaProblem {
    pub fn new(
        n_vars: usize,
        objective: Expr,
        direction: OptDirection,
        constraints: Vec<Constraint>,
    ) -> Self {
        Self {
            n_vars,
            objective,
            direction,
            constraints,
        }
    }

    /// 目的関数値を計算する
    pub fn eval_objective(&self, x: &[bool]) -> Value {
        eval_expr(&self.objective, x)
    }

    /// 全制約のペナルティ合計を計算する
    pub fn eval_penalty(&self, x: &[bool]) -> Value {
        self.constraints
            .iter()
            .map(|c| eval_constraint_penalty(c, x))
            .sum()
    }

    /// スコアを計算する (方向・ペナルティ込み; 常に大きい方が良い)
    pub fn eval_score(&self, x: &[bool]) -> Value {
        let obj = self.eval_objective(x);
        let penalty = self.eval_penalty(x);
        match self.direction {
            OptDirection::Maximize => obj - penalty,
            OptDirection::Minimize => -obj - penalty,
        }
    }

    /// 変数 i をフリップしたときのスコア変化量を計算する
    pub fn calc_gain(&self, x: &[bool], i: usize) -> Value {
        let current = self.eval_score(x);
        let mut xf = x.to_vec();
        xf[i] = !xf[i];
        self.eval_score(&xf) - current
    }

    /// 変数 flipped をフリップした状態で変数 j のゲインを計算する (x は変更しない)
    pub fn calc_gain_with_virtual_flip(&self, x: &[bool], flipped: usize, j: usize) -> Value {
        let mut xf = x.to_vec();
        xf[flipped] = !xf[flipped];
        self.calc_gain(&xf, j)
    }
}

impl ProblemTrait for FormulaProblem {
    type Solution = FormulaSolution;

    fn new_solution(&self, rng: &mut impl rand::Rng) -> FormulaSolution {
        let x: Vec<bool> = (0..self.n_vars).map(|_| rng.random_bool(0.5)).collect();
        let score = self.eval_score(&x);
        let gain: Vec<Value> = (0..self.n_vars).map(|i| self.calc_gain(&x, i)).collect();
        FormulaSolution { x, gain, score }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// maximize x[0] + 2*x[1] + 3*x[2]  s.t. x[0] + x[1] + x[2] <= 2
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

    #[test]
    fn test_eval_score_feasible() {
        let prob = make_problem();
        // x = [0, 1, 1]: obj = 5, sum = 2 <= 2, penalty = 0, score = 5
        let x = vec![false, true, true];
        assert!((prob.eval_score(&x) - 5.0).abs() < 1e-9);
    }

    #[test]
    fn test_eval_score_infeasible() {
        let prob = make_problem();
        // x = [1, 1, 1]: obj = 6, sum = 3 > 2, violation = 1, penalty = 10, score = -4
        let x = vec![true, true, true];
        assert!((prob.eval_score(&x) - (-4.0)).abs() < 1e-9);
    }

    #[test]
    fn test_calc_gain_consistency() {
        let prob = make_problem();
        let x = vec![false, true, false];
        let score = prob.eval_score(&x);
        for i in 0..3 {
            let mut xf = x.clone();
            xf[i] = !xf[i];
            let expected = prob.eval_score(&xf) - score;
            assert!(
                (prob.calc_gain(&x, i) - expected).abs() < 1e-9,
                "gain[{}] mismatch",
                i
            );
        }
    }

    #[test]
    fn test_minimize_direction() {
        let objective = Expr::Add(vec![Expr::Var(0), Expr::Var(1)]);
        let prob = FormulaProblem::new(2, objective, OptDirection::Minimize, vec![]);
        // minimize: score = -obj; x=[0,0] → score=0, x=[1,1] → score=-2
        assert!((prob.eval_score(&[false, false]) - 0.0).abs() < 1e-9);
        assert!((prob.eval_score(&[true, true]) - (-2.0)).abs() < 1e-9);
        // x=[0,0] is better (larger score)
        let s0 = FormulaSolution {
            x: vec![false, false],
            gain: vec![0.0; 2],
            score: 0.0,
        };
        let s1 = FormulaSolution {
            x: vec![true, true],
            gain: vec![0.0; 2],
            score: -2.0,
        };
        assert!(s0.is_better_than(&s1));
    }

    #[test]
    fn test_clamp_constraint() {
        // clamp constraint: 1 <= x[0] + x[1] + x[2] <= 2, weight=10
        let prob = FormulaProblem::new(
            3,
            Expr::Const(0.0),
            OptDirection::Maximize,
            vec![Constraint::Clamp {
                expr: Expr::Add(vec![Expr::Var(0), Expr::Var(1), Expr::Var(2)]),
                lo: 1.0,
                hi: 2.0,
                penalty_weight: 10.0,
            }],
        );
        // sum=0 → penalty = (1-0)*10 = 10
        assert!((prob.eval_penalty(&[false, false, false]) - 10.0).abs() < 1e-9);
        // sum=1 → no penalty
        assert!((prob.eval_penalty(&[true, false, false]) - 0.0).abs() < 1e-9);
        // sum=3 → penalty = (3-2)*10 = 10
        assert!((prob.eval_penalty(&[true, true, true]) - 10.0).abs() < 1e-9);
    }
}
