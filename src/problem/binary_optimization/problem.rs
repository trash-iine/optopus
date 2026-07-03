use std::ops::{Add, Div, Mul, Neg, Sub};

use crate::search_state::{Distance, ProblemTrait, Rankable};

/// Numeric value type used in formula expressions.
pub type Value = f64;

/// An arithmetic expression AST node over binary variables.
#[derive(Debug, Clone)]
pub enum Expr {
    /// A numeric constant.
    Const(Value),
    /// A binary variable (0-indexed); evaluates to 0.0 or 1.0.
    Var(usize),
    /// Negation of an expression.
    Neg(Box<Expr>),
    /// Sum of multiple expressions.
    Add(Vec<Expr>),
    /// Product of multiple expressions (equivalent to AND for binary variables).
    Mul(Vec<Expr>),
}

impl Neg for Expr {
    type Output = Expr;
    fn neg(self) -> Expr {
        match self {
            Expr::Neg(inner) => *inner,
            Expr::Const(c) => Expr::Const(-c),
            other => Expr::Neg(Box::new(other)),
        }
    }
}

impl Add for Expr {
    type Output = Expr;
    fn add(self, rhs: Expr) -> Expr {
        let mut terms = match self {
            Expr::Add(v) => v,
            other => vec![other],
        };
        match rhs {
            Expr::Add(v) => terms.extend(v),
            other => terms.push(other),
        }
        Expr::Add(terms)
    }
}

impl Sub for Expr {
    type Output = Expr;
    fn sub(self, rhs: Expr) -> Expr {
        self + (-rhs)
    }
}

impl Mul for Expr {
    type Output = Expr;
    fn mul(self, rhs: Expr) -> Expr {
        let mut factors = match self {
            Expr::Mul(v) => v,
            other => vec![other],
        };
        match rhs {
            Expr::Mul(v) => factors.extend(v),
            other => factors.push(other),
        }
        Expr::Mul(factors)
    }
}

impl Add<Value> for Expr {
    type Output = Expr;
    fn add(self, rhs: Value) -> Expr {
        let mut terms = match self {
            Expr::Add(v) => v,
            other => vec![other],
        };
        terms.push(Expr::Const(rhs));
        Expr::Add(terms)
    }
}

impl Sub<Value> for Expr {
    type Output = Expr;
    fn sub(self, rhs: Value) -> Expr {
        self + (-rhs)
    }
}

impl Mul<Value> for Expr {
    type Output = Expr;
    fn mul(self, rhs: Value) -> Expr {
        let mut factors = match self {
            Expr::Mul(v) => v,
            other => vec![other],
        };
        factors.push(Expr::Const(rhs));
        Expr::Mul(factors)
    }
}

impl Add<Expr> for Value {
    type Output = Expr;
    fn add(self, rhs: Expr) -> Expr {
        let mut terms = vec![Expr::Const(self)];
        match rhs {
            Expr::Add(v) => terms.extend(v),
            other => terms.push(other),
        }
        Expr::Add(terms)
    }
}

impl Sub<Expr> for Value {
    type Output = Expr;
    fn sub(self, rhs: Expr) -> Expr {
        self + (-rhs)
    }
}

impl Mul<Expr> for Value {
    type Output = Expr;
    fn mul(self, rhs: Expr) -> Expr {
        let mut factors = vec![Expr::Const(self)];
        match rhs {
            Expr::Mul(v) => factors.extend(v),
            Expr::Const(c) => factors[0] = Expr::Const(self * c),
            other => factors.push(other),
        }
        Expr::Mul(factors)
    }
}
impl Div for Expr {
    type Output = Expr;
    fn div(self, rhs: Expr) -> Expr {
        match rhs {
            Expr::Const(c) => self * Expr::Const(1.0 / c),
            other => panic!(
                "division by non-constant expressions is not supported: {:?}",
                other
            ),
        }
    }
}
impl Div<Value> for Expr {
    type Output = Expr;
    fn div(self, rhs: Value) -> Expr {
        self * Expr::Const(1.0 / rhs)
    }
}

/// Comparison operator for [`Constraint::Comparison`].
#[derive(Debug, Clone)]
pub enum ConstraintRel {
    Lt,
    Gt,
    Le,
    Ge,
    Eq,
}

/// A penalty-weighted constraint on the binary variables.
#[derive(Debug, Clone)]
pub enum Constraint {
    /// `lhs rel rhs` form: penalty = violation * `penalty_weight`.
    Comparison {
        lhs: Expr,
        rel: ConstraintRel,
        rhs: Expr,
        penalty_weight: Value,
    },
    /// `lo <= expr <= hi` form: penalty = amount outside the range * `penalty_weight`.
    Clamp {
        expr: Expr,
        lo: Value,
        hi: Value,
        penalty_weight: Value,
    },
}

/// Optimization direction for a [`FormulaProblem`].
#[derive(Debug, Clone)]
pub enum OptDirection {
    Maximize,
    Minimize,
}

/// A single monomial: `coeff * product(x[v] for v in vars)`.
/// For binary variables x^2 = x, so `vars` is sorted and deduplicated.
#[derive(Debug, Clone)]
pub(super) struct Monomial {
    pub(super) coeff: Value,
    pub(super) vars: Vec<usize>,
}

/// A flat polynomial compiled from an [`Expr`], with a per-variable term index
/// enabling O(d) delta computation (d = number of terms containing a given variable).
#[derive(Debug, Clone)]
pub(super) struct CompiledPoly {
    pub(super) terms: Vec<Monomial>,
    /// `var_terms[i]` = indices into `terms` of monomials that contain variable `i`.
    pub(super) var_terms: Vec<Vec<usize>>,
}

fn compile_expr(expr: &Expr) -> Vec<Monomial> {
    match expr {
        Expr::Const(c) => vec![Monomial {
            coeff: *c,
            vars: vec![],
        }],
        Expr::Var(i) => vec![Monomial {
            coeff: 1.0,
            vars: vec![*i],
        }],
        Expr::Neg(e) => compile_expr(e)
            .into_iter()
            .map(|m| Monomial {
                coeff: -m.coeff,
                vars: m.vars,
            })
            .collect(),
        Expr::Add(es) => es.iter().flat_map(compile_expr).collect(),
        Expr::Mul(es) => es.iter().fold(
            vec![Monomial {
                coeff: 1.0,
                vars: vec![],
            }],
            |acc, e| {
                let sub = compile_expr(e);
                acc.into_iter()
                    .flat_map(|m1| {
                        sub.iter().map(move |m2| {
                            let coeff = m1.coeff * m2.coeff;
                            let mut vars: Vec<usize> =
                                m1.vars.iter().chain(m2.vars.iter()).copied().collect();
                            vars.sort_unstable();
                            vars.dedup(); // x^2 = x for binary variables
                            Monomial { coeff, vars }
                        })
                    })
                    .collect()
            },
        ),
    }
}

fn build_var_terms(terms: &[Monomial], n_vars: usize) -> Vec<Vec<usize>> {
    let mut vt = vec![vec![]; n_vars];
    for (t, m) in terms.iter().enumerate() {
        for &v in &m.vars {
            if v < n_vars {
                vt[v].push(t);
            }
        }
    }
    vt
}

fn compile_poly(expr: &Expr, n_vars: usize) -> CompiledPoly {
    let terms = compile_expr(expr);
    let var_terms = build_var_terms(&terms, n_vars);
    CompiledPoly { terms, var_terms }
}

/// Compile the expression whose value is tracked in `constraint_vals`:
/// - Comparison: tracks `eval(lhs - rhs, x)`
/// - Clamp: tracks `eval(expr, x)`
fn compile_constraint_expr(c: &Constraint, n_vars: usize) -> CompiledPoly {
    match c {
        Constraint::Comparison { lhs, rhs, .. } => {
            let mut terms = compile_expr(lhs);
            terms.extend(compile_expr(rhs).into_iter().map(|m| Monomial {
                coeff: -m.coeff,
                vars: m.vars,
            }));
            let var_terms = build_var_terms(&terms, n_vars);
            CompiledPoly { terms, var_terms }
        }
        Constraint::Clamp { expr, .. } => compile_poly(expr, n_vars),
    }
}

fn eval_mono(m: &Monomial, x: &[bool]) -> Value {
    m.vars
        .iter()
        .fold(m.coeff, |acc, &v| acc * if x[v] { 1.0 } else { 0.0 })
}

/// Compute penalty from a pre-evaluated constraint expression value.
/// - Comparison: `val = eval(lhs - rhs, x)`
/// - Clamp:      `val = eval(expr, x)`
fn constraint_penalty_from_val(c: &Constraint, val: Value) -> Value {
    match c {
        Constraint::Comparison {
            rel,
            penalty_weight,
            ..
        } => {
            let violation = match rel {
                ConstraintRel::Lt => (val + STRICT_EPSILON).max(0.0),
                ConstraintRel::Gt => (-val + STRICT_EPSILON).max(0.0),
                ConstraintRel::Le => val.max(0.0),
                ConstraintRel::Ge => (-val).max(0.0),
                ConstraintRel::Eq => val.abs(),
            };
            violation * penalty_weight
        }
        Constraint::Clamp {
            lo,
            hi,
            penalty_weight,
            ..
        } => ((lo - val).max(0.0) + (val - hi).max(0.0)) * penalty_weight,
    }
}

/// For each variable `i`, the set of variables `j` (j ≠ i) whose gain may change when `i`
/// is flipped.
///
/// Two sources of dependency:
/// 1. **Objective**: `i` and `j` share a monomial → flipping `i` changes j's objective delta.
/// 2. **Constraint**: `i` and `j` both appear in the same constraint expression
///    (even in separate monomials) → flipping `i` shifts the base `cv[c]`, and because
///    the penalty function `max(0,·)` / `abs(·)` is non-linear, j's penalty delta can
///    change even when the expression delta for `j` itself is unaffected.
fn build_interaction_neighbors(
    obj_poly: &CompiledPoly,
    constraint_polys: &[CompiledPoly],
    n_vars: usize,
) -> Vec<Vec<usize>> {
    use std::collections::HashSet;
    let mut nbrs: Vec<HashSet<usize>> = (0..n_vars).map(|_| HashSet::new()).collect();

    // Objective: only variables that share a monomial are neighbors.
    for m in &obj_poly.terms {
        for (a, &va) in m.vars.iter().enumerate() {
            for &vb in m.vars.iter().skip(a + 1) {
                nbrs[va].insert(vb);
                nbrs[vb].insert(va);
            }
        }
    }

    // Constraints: all pairs of variables that appear anywhere in the same constraint
    // expression are neighbors (non-linear penalty makes the base value matter).
    for poly in constraint_polys {
        let vars_in_constraint: Vec<usize> = poly
            .terms
            .iter()
            .flat_map(|m| m.vars.iter().copied())
            .collect::<HashSet<usize>>()
            .into_iter()
            .collect();
        for (a, &va) in vars_in_constraint.iter().enumerate() {
            for &vb in vars_in_constraint.iter().skip(a + 1) {
                nbrs[va].insert(vb);
                nbrs[vb].insert(va);
            }
        }
    }

    nbrs.into_iter().map(|s| s.into_iter().collect()).collect()
}

/// A binary optimization problem defined by an arithmetic expression.
///
/// The score is always maximized internally:
/// - For [`OptDirection::Maximize`]: `score = objective − penalty`
/// - For [`OptDirection::Minimize`]: `score = −objective − penalty`
#[derive(Debug, Clone)]
pub struct FormulaProblem {
    pub n_vars: usize,
    pub objective: Expr,
    pub direction: OptDirection,
    pub constraints: Vec<Constraint>,
    // Compiled polynomial representations — built once at construction time.
    pub(super) obj_poly: CompiledPoly,
    pub(super) constraint_polys: Vec<CompiledPoly>,
    /// For each variable `i`, the variables whose gain may change when `i` is flipped.
    /// Computed once at construction; used by `FormulaFlipNeighbor::apply_to_solution`
    /// to skip variables that cannot be affected.
    pub(super) interaction_neighbors: Vec<Vec<usize>>,
}

/// A solution to a [`FormulaProblem`].
///
/// - `x` — variable assignment (`x[i] ∈ {false=0, true=1}`)
/// - `gain` — change in score when flipping each variable (positive = improvement)
/// - `score` — current combined score (objective adjusted for direction, minus penalty; higher is better)
#[derive(Debug, Clone)]
pub struct FormulaSolution {
    pub x: Vec<bool>,
    pub gain: Vec<Value>,
    pub score: Value,
    /// Current value of each constraint's tracked expression (used for O(d) gain computation).
    /// - Comparison constraint c: `constraint_vals[c] = eval(lhs - rhs, x)`
    /// - Clamp constraint c:      `constraint_vals[c] = eval(expr, x)`
    pub(crate) constraint_vals: Vec<Value>,
}

impl Rankable for FormulaSolution {
    fn is_better_than(&self, other: &Self) -> bool {
        self.score > other.score
    }
}

impl Distance for FormulaSolution {
    fn distance(&self, other: &Self) -> usize {
        self.x
            .iter()
            .zip(other.x.iter())
            .filter(|(a, b)| a != b)
            .count()
    }
}

// ---------------------------------------------------------------------------
// Expression evaluation (kept for eval_score / eval_objective / eval_penalty)
// ---------------------------------------------------------------------------

/// Evaluates an expression with the given variable assignment.
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

impl FormulaProblem {
    pub fn new(
        n_vars: usize,
        objective: Expr,
        direction: OptDirection,
        constraints: Vec<Constraint>,
    ) -> Self {
        let obj_poly = compile_poly(&objective, n_vars);
        let constraint_polys: Vec<CompiledPoly> = constraints
            .iter()
            .map(|c| compile_constraint_expr(c, n_vars))
            .collect();
        let interaction_neighbors =
            build_interaction_neighbors(&obj_poly, &constraint_polys, n_vars);
        Self {
            n_vars,
            objective,
            direction,
            constraints,
            obj_poly,
            constraint_polys,
            interaction_neighbors,
        }
    }

    /// Evaluates the raw objective value for the given assignment.
    pub fn eval_objective(&self, x: &[bool]) -> Value {
        eval_expr(&self.objective, x)
    }

    /// Evaluates the total constraint penalty for the given assignment.
    pub fn eval_penalty(&self, x: &[bool]) -> Value {
        self.constraints
            .iter()
            .map(|c| eval_constraint_penalty(c, x))
            .sum()
    }

    /// Evaluates the combined score for the given assignment (higher is always better).
    ///
    /// - Maximize: `score = objective − penalty`
    /// - Minimize: `score = −objective − penalty`
    pub fn eval_score(&self, x: &[bool]) -> Value {
        let obj = self.eval_objective(x);
        let penalty = self.eval_penalty(x);
        match self.direction {
            OptDirection::Maximize => obj - penalty,
            OptDirection::Minimize => -obj - penalty,
        }
    }

    /// Evaluates the tracked value of each constraint's expression for the given assignment.
    /// Used to initialize `FormulaSolution::constraint_vals`.
    pub(crate) fn eval_constraint_vals(&self, x: &[bool]) -> Vec<Value> {
        self.constraint_polys
            .iter()
            .map(|p| p.terms.iter().map(|m| eval_mono(m, x)).sum())
            .collect()
    }

    /// Computes `eval(poly, x_with_i_flipped) − eval(poly, x)` in O(d) time,
    /// where d = number of terms in `poly` containing variable `i`.
    pub(super) fn eval_poly_delta(&self, poly: &CompiledPoly, x: &[bool], i: usize) -> Value {
        // Turning on (x[i]=false→true): sign=+1; turning off (x[i]=true→false): sign=−1.
        let sign: Value = if x[i] { -1.0 } else { 1.0 };
        poly.var_terms[i]
            .iter()
            .map(|&t| {
                let m = &poly.terms[t];
                let rest: Value = m
                    .vars
                    .iter()
                    .filter(|&&v| v != i)
                    .fold(1.0, |acc, &v| acc * if x[v] { 1.0 } else { 0.0 });
                m.coeff * sign * rest
            })
            .sum()
    }

    /// Computes the gain when flipping variable `i` in O(d) time, using pre-computed
    /// `constraint_vals` to avoid full expression re-evaluation.
    pub(crate) fn calc_gain_fast(&self, x: &[bool], constraint_vals: &[Value], i: usize) -> Value {
        let direction_sign: Value = match self.direction {
            OptDirection::Maximize => 1.0,
            OptDirection::Minimize => -1.0,
        };
        let delta_obj = self.eval_poly_delta(&self.obj_poly, x, i);
        let delta_penalty: Value = self
            .constraints
            .iter()
            .zip(constraint_vals.iter())
            .zip(self.constraint_polys.iter())
            .map(|((c, &cv), poly)| {
                let dcv = self.eval_poly_delta(poly, x, i);
                constraint_penalty_from_val(c, cv + dcv) - constraint_penalty_from_val(c, cv)
            })
            .sum();
        direction_sign * delta_obj - delta_penalty
    }

    /// Calculates the change in score when variable `i` is flipped.
    pub fn calc_gain(&self, x: &[bool], i: usize) -> Value {
        let cv = self.eval_constraint_vals(x);
        self.calc_gain_fast(x, &cv, i)
    }

    /// Calculates the gain of variable `j` assuming variable `flipped` has been
    /// virtually flipped (without modifying `x`).
    pub fn calc_gain_with_virtual_flip(&self, x: &[bool], flipped: usize, j: usize) -> Value {
        let mut cv = self.eval_constraint_vals(x);
        for (v, poly) in cv.iter_mut().zip(self.constraint_polys.iter()) {
            *v += self.eval_poly_delta(poly, x, flipped);
        }
        let mut xm = x.to_vec();
        xm[flipped] = !xm[flipped];
        self.calc_gain_fast(&xm, &cv, j)
    }
}

impl ProblemTrait for FormulaProblem {
    type Solution = FormulaSolution;

    fn new_solution(&self, rng: &mut impl rand::Rng) -> FormulaSolution {
        let x: Vec<bool> = (0..self.n_vars).map(|_| rng.random_bool(0.5)).collect();
        let score = self.eval_score(&x);
        let constraint_vals = self.eval_constraint_vals(&x);
        let gain: Vec<Value> = (0..self.n_vars)
            .map(|i| self.calc_gain_fast(&x, &constraint_vals, i))
            .collect();
        FormulaSolution {
            x,
            gain,
            score,
            constraint_vals,
        }
    }
}

impl crate::trait_defs::BinaryProblem for FormulaProblem {
    type Flip = super::FormulaFlipNeighbor;

    fn variable_indices(&self) -> impl Iterator<Item = usize> + '_ {
        0..self.n_vars
    }

    fn variable(sol: &FormulaSolution, i: usize) -> bool {
        sol.x[i]
    }

    fn flip_move(sol: &FormulaSolution, i: usize) -> Self::Flip {
        super::FormulaFlipNeighbor {
            i,
            gain: sol.gain[i],
        }
    }
}

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
            constraint_vals: vec![],
        };
        let s1 = FormulaSolution {
            x: vec![true, true],
            gain: vec![0.0; 2],
            score: -2.0,
            constraint_vals: vec![],
        };
        assert!(s0.is_better_than(&s1));
    }

    #[test]
    fn test_neg_simplification() {
        // Neg(Const(c)) → Const(-c)
        let e = -Expr::Const(3.0);
        assert!(matches!(e, Expr::Const(c) if (c + 3.0).abs() < 1e-12));
        // Neg(Neg(e)) → e
        let e2 = -(-Expr::Var(0));
        assert!(matches!(e2, Expr::Var(0)));
        // Neg(Var) stays as Neg
        let e3 = -Expr::Var(1);
        assert!(matches!(e3, Expr::Neg(_)));
    }

    #[test]
    fn test_add_flattening() {
        // (a + b) + c + d should produce a single flat Add with 4 children
        let e = Expr::Var(0) + Expr::Var(1) + Expr::Var(2) + Expr::Var(3);
        match e {
            Expr::Add(v) => assert_eq!(v.len(), 4),
            _ => panic!("expected flat Add"),
        }
    }

    #[test]
    fn test_mul_flattening() {
        // (a * b) * c should produce a single flat Mul with 3 children
        let e = Expr::Var(0) * Expr::Var(1) * Expr::Var(2);
        match e {
            Expr::Mul(v) => assert_eq!(v.len(), 3),
            _ => panic!("expected flat Mul"),
        }
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
