//! Problems whose objective and constraints are arithmetic expressions over
//! integer variables, with every delta derived from the expressions.

use std::ops::{Add, Div, Mul, Neg, Sub};

use super::assignment::IntAssignment;
use super::problem::{IntVars, raw};
use crate::error::OptError;
use crate::search_state::{Distance, Evaluable, Evaluate, ProblemTrait, SubProblemExtractable};
use rand::Rng;

/// An arithmetic expression over integer variables.
///
/// Built by overloading `+ - * /` rather than by naming variants directly.
///
/// ```
/// use optopus::prelude::*;
///
/// // linear combination 2*x[0] + x[1] - 3
/// let linear = 2.0 * Expr::Var(0) + Expr::Var(1) - 3.0;
///
/// // a product, which on binary variables is an AND
/// let and_of_two = Expr::Var(0) * Expr::Var(1);
/// ```
#[derive(Debug, Clone)]
pub enum Expr {
    /// A numeric constant.
    Const(f64),
    /// Variable `i`, which evaluates to its value.
    Var(usize),
    /// Negation of an expression.
    Neg(Box<Expr>),
    /// Sum of expressions.
    Add(Vec<Expr>),
    /// Product of expressions.
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

impl Add<f64> for Expr {
    type Output = Expr;
    fn add(self, rhs: f64) -> Expr {
        let mut terms = match self {
            Expr::Add(v) => v,
            other => vec![other],
        };
        terms.push(Expr::Const(rhs));
        Expr::Add(terms)
    }
}

impl Sub<f64> for Expr {
    type Output = Expr;
    fn sub(self, rhs: f64) -> Expr {
        self + (-rhs)
    }
}

impl Mul<f64> for Expr {
    type Output = Expr;
    fn mul(self, rhs: f64) -> Expr {
        let mut factors = match self {
            Expr::Mul(v) => v,
            other => vec![other],
        };
        factors.push(Expr::Const(rhs));
        Expr::Mul(factors)
    }
}

impl Add<Expr> for f64 {
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

impl Sub<Expr> for f64 {
    type Output = Expr;
    fn sub(self, rhs: Expr) -> Expr {
        self + (-rhs)
    }
}

impl Mul<Expr> for f64 {
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
            other => panic!("division by non-constant expressions is not supported: {other:?}"),
        }
    }
}

impl Div<f64> for Expr {
    type Output = Expr;
    fn div(self, rhs: f64) -> Expr {
        self * Expr::Const(1.0 / rhs)
    }
}

impl Expr {
    /// The value of the expression, reading variable `i` as `values[i]`.
    pub fn eval(&self, values: &[i64]) -> f64 {
        match self {
            Expr::Const(c) => *c,
            Expr::Var(i) => values[*i] as f64,
            Expr::Neg(e) => -e.eval(values),
            Expr::Add(es) => es.iter().map(|e| e.eval(values)).sum(),
            Expr::Mul(es) => es.iter().map(|e| e.eval(values)).product(),
        }
    }

    fn max_var(&self) -> Option<usize> {
        match self {
            Expr::Const(_) => None,
            Expr::Var(i) => Some(*i),
            Expr::Neg(e) => e.max_var(),
            Expr::Add(es) | Expr::Mul(es) => es.iter().filter_map(Expr::max_var).max(),
        }
    }

    /// The expression with each fixed variable replaced by its value and each
    /// free one renumbered.
    fn substitute(&self, fixed: &[Option<i64>], remap: &[Option<usize>]) -> Expr {
        match self {
            Expr::Const(c) => Expr::Const(*c),
            Expr::Var(i) => match fixed[*i] {
                Some(v) => Expr::Const(v as f64),
                None => Expr::Var(remap[*i].expect("a variable is either fixed or free")),
            },
            Expr::Neg(e) => -e.substitute(fixed, remap),
            Expr::Add(es) => Expr::Add(es.iter().map(|e| e.substitute(fixed, remap)).collect()),
            Expr::Mul(es) => Expr::Mul(es.iter().map(|e| e.substitute(fixed, remap)).collect()),
        }
    }
}

/// Comparison operator for [`Constraint::Comparison`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConstraintRel {
    Lt,
    Gt,
    Le,
    Ge,
    Eq,
}

/// A penalty-weighted constraint on the variables.
///
/// ```
/// use optopus::prelude::*;
///
/// // x[0] + x[1] + x[2] <= 2, penalized at weight 10.0 per unit of violation
/// let constraint = Constraint::Comparison {
///     lhs: Expr::Var(0) + Expr::Var(1) + Expr::Var(2),
///     rel: ConstraintRel::Le,
///     rhs: Expr::Const(2.0),
///     penalty_weight: 10.0,
/// };
/// ```
#[derive(Debug, Clone)]
pub enum Constraint {
    /// `lhs rel rhs`, charging the violation times `penalty_weight`.
    Comparison {
        lhs: Expr,
        rel: ConstraintRel,
        rhs: Expr,
        penalty_weight: f64,
    },
    /// `lo <= expr <= hi`, charging the distance outside the range times
    /// `penalty_weight`.
    Clamp {
        expr: Expr,
        lo: f64,
        hi: f64,
        penalty_weight: f64,
    },
}

impl Constraint {
    /// The expression whose value decides the penalty, `lhs - rhs` or `expr`.
    fn tracked(&self) -> Expr {
        match self {
            Constraint::Comparison { lhs, rhs, .. } => lhs.clone() - rhs.clone(),
            Constraint::Clamp { expr, .. } => expr.clone(),
        }
    }

    /// The penalty when the tracked expression has value `val`.
    fn penalty(&self, val: f64) -> f64 {
        match self {
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

    fn penalty_at(&self, values: &[i64]) -> f64 {
        let val = match self {
            Constraint::Comparison { lhs, rhs, .. } => lhs.eval(values) - rhs.eval(values),
            Constraint::Clamp { expr, .. } => expr.eval(values),
        };
        self.penalty(val)
    }

    fn substitute(&self, fixed: &[Option<i64>], remap: &[Option<usize>]) -> Constraint {
        match self {
            Constraint::Comparison {
                lhs,
                rel,
                rhs,
                penalty_weight,
            } => Constraint::Comparison {
                lhs: lhs.substitute(fixed, remap),
                rel: *rel,
                rhs: rhs.substitute(fixed, remap),
                penalty_weight: *penalty_weight,
            },
            Constraint::Clamp {
                expr,
                lo,
                hi,
                penalty_weight,
            } => Constraint::Clamp {
                expr: expr.substitute(fixed, remap),
                lo: *lo,
                hi: *hi,
                penalty_weight: *penalty_weight,
            },
        }
    }

    fn max_var(&self) -> Option<usize> {
        match self {
            Constraint::Comparison { lhs, rhs, .. } => lhs.max_var().max(rhs.max_var()),
            Constraint::Clamp { expr, .. } => expr.max_var(),
        }
    }
}

/// How much a strict relation charges at equality, so that a tie is never
/// free.
const STRICT_EPSILON: f64 = 1e-9;

/// `coeff` times the product of `x[v]^power` over `vars`, sorted by variable.
#[derive(Debug, Clone)]
struct Monomial {
    coeff: f64,
    vars: Vec<(usize, u32)>,
}

/// An expression flattened into monomials, with the monomials each variable
/// appears in and its power there, so that a change to one variable is priced
/// from those alone.
#[derive(Debug, Clone)]
struct Poly {
    terms: Vec<Monomial>,
    var_terms: Vec<Vec<(usize, u32)>>,
}

/// `x^q` for `q >= 1`, by repeated multiplication. `powi` would be folded into
/// a call to the runtime's `__powidf2` even for `q == 1`, which is almost every
/// power a formula has, and that call was most of the time a move took.
#[inline]
fn power(x: i64, q: u32) -> f64 {
    let x = x as f64;
    let mut r = x;
    for _ in 1..q {
        r *= x;
    }
    r
}

fn monomials(expr: &Expr) -> Vec<Monomial> {
    match expr {
        Expr::Const(c) => vec![Monomial {
            coeff: *c,
            vars: vec![],
        }],
        Expr::Var(i) => vec![Monomial {
            coeff: 1.0,
            vars: vec![(*i, 1)],
        }],
        Expr::Neg(e) => monomials(e)
            .into_iter()
            .map(|m| Monomial {
                coeff: -m.coeff,
                vars: m.vars,
            })
            .collect(),
        Expr::Add(es) => es.iter().flat_map(monomials).collect(),
        Expr::Mul(es) => es.iter().fold(
            vec![Monomial {
                coeff: 1.0,
                vars: vec![],
            }],
            |acc, e| {
                let sub = monomials(e);
                acc.into_iter()
                    .flat_map(|m1| {
                        sub.iter().map(move |m2| Monomial {
                            coeff: m1.coeff * m2.coeff,
                            vars: multiply(&m1.vars, &m2.vars),
                        })
                    })
                    .collect()
            },
        ),
    }
}

/// The product of two sorted variable lists, adding the powers of a variable
/// that appears in both.
fn multiply(a: &[(usize, u32)], b: &[(usize, u32)]) -> Vec<(usize, u32)> {
    let mut out: Vec<(usize, u32)> = a.iter().chain(b).copied().collect();
    out.sort_unstable_by_key(|&(v, _)| v);
    let mut merged: Vec<(usize, u32)> = Vec::with_capacity(out.len());
    for (v, q) in out {
        match merged.last_mut() {
            Some((last, p)) if *last == v => *p += q,
            _ => merged.push((v, q)),
        }
    }
    merged
}

impl Poly {
    fn compile(expr: &Expr, n: usize) -> Self {
        let terms = monomials(expr);
        let mut var_terms = vec![vec![]; n];
        for (t, m) in terms.iter().enumerate() {
            for &(v, q) in &m.vars {
                var_terms[v].push((t, q));
            }
        }
        Self { terms, var_terms }
    }

    fn eval(&self, x: &[i64]) -> f64 {
        self.terms
            .iter()
            .map(|m| {
                m.vars
                    .iter()
                    .fold(m.coeff, |acc, &(v, q)| acc * power(x[v], q))
            })
            .sum()
    }

    /// How much the polynomial changes when `x[i]` becomes `b`.
    #[inline]
    fn change_delta(&self, x: &[i64], i: usize, b: i64) -> f64 {
        let a = x[i];
        self.var_terms[i]
            .iter()
            .map(|&(t, p)| {
                let m = &self.terms[t];
                let rest = m
                    .vars
                    .iter()
                    .filter(|&&(v, _)| v != i)
                    .fold(1.0, |acc, &(v, q)| acc * power(x[v], q));
                let step = if p == 1 {
                    (b - a) as f64
                } else {
                    power(b, p) - power(a, p)
                };
                m.coeff * step * rest
            })
            .sum()
    }

    /// How much the polynomial changes when `x[i]` and `x[j]` are exchanged.
    fn swap_delta(&self, x: &[i64], i: usize, j: usize) -> f64 {
        let swapped = |v: usize| {
            if v == i {
                x[j]
            } else if v == j {
                x[i]
            } else {
                x[v]
            }
        };
        let touched = self.var_terms[i].iter().chain(
            self.var_terms[j]
                .iter()
                .filter(|&&(t, _)| self.terms[t].vars.iter().all(|&(v, _)| v != i)),
        );
        touched
            .map(|&(t, _)| {
                let m = &self.terms[t];
                let after = m
                    .vars
                    .iter()
                    .fold(1.0, |acc, &(v, q)| acc * power(swapped(v), q));
                let before = m.vars.iter().fold(1.0, |acc, &(v, q)| acc * power(x[v], q));
                m.coeff * (after - before)
            })
            .sum()
    }
}

/// A problem whose objective is an [`Expr`] over integer variables, with any
/// number of penalty-weighted [`Constraint`]s.
///
/// The expressions are compiled into monomials when the problem is built, so a
/// change to one variable is priced from the monomials it appears in, and each
/// solution keeps the delta of every change it can make. A move is read from
/// that table and applying one refreshes the rows it can have touched.
///
/// ```
/// use optopus::prelude::*;
///
/// // maximize x[0] + 2*x[1] + 3*x[2] over binary x, with x[0] + x[1] + x[2] <= 2
/// let vars: IntVars = (0..3).map(|_| IntVar::binary()).collect();
/// let objective = Expr::Var(0) + 2.0 * Expr::Var(1) + 3.0 * Expr::Var(2);
/// let prob = FormulaProblem::maximize(vars, objective).with_constraint(Constraint::Comparison {
///     lhs: Expr::Var(0) + Expr::Var(1) + Expr::Var(2),
///     rel: ConstraintRel::Le,
///     rhs: Expr::Const(2.0),
///     penalty_weight: 10.0,
/// });
///
/// let mut state = SearchState::new_with_seed(&prob, 1);
/// LocalSearch::<IntChangeNeighbor>::new(StopCondition::iterations(100))
///     .run(&mut state)
///     .unwrap();
/// assert_eq!(prob.eval_penalty(state.best_solution.values()), 0.0);
/// ```
#[derive(Debug, Clone)]
pub struct FormulaProblem {
    vars: IntVars,
    maximize: bool,
    objective: Expr,
    constraints: Vec<Constraint>,
    objective_poly: Poly,
    constraint_polys: Vec<Poly>,
    /// For each variable `i`, the variables whose deltas can change when `i`
    /// changes. Those sharing a monomial of the objective, and those appearing
    /// in any constraint with `i`, since a penalty is not linear in the value
    /// of its expression.
    interaction_neighbors: Vec<Vec<usize>>,
    /// For each variable, the constraints whose expressions read it, in
    /// ascending order. The others do not move when it changes.
    var_constraints: Vec<Vec<usize>>,
}

impl FormulaProblem {
    /// A problem that minimizes `objective`.
    ///
    /// # Panics
    ///
    /// If `objective` reads a variable outside `vars`.
    pub fn minimize(vars: IntVars, objective: Expr) -> Self {
        Self::build(vars, false, objective, vec![])
    }

    /// A problem that maximizes `objective`.
    ///
    /// # Panics
    ///
    /// If `objective` reads a variable outside `vars`.
    pub fn maximize(vars: IntVars, objective: Expr) -> Self {
        Self::build(vars, true, objective, vec![])
    }

    /// The problem with `constraint` added. Its penalty is subtracted from a
    /// maximized objective and added to a minimized one.
    ///
    /// # Panics
    ///
    /// If `constraint` reads a variable outside the variables.
    pub fn with_constraint(self, constraint: Constraint) -> Self {
        let mut constraints = self.constraints;
        constraints.push(constraint);
        Self::build(self.vars, self.maximize, self.objective, constraints)
    }

    fn build(vars: IntVars, maximize: bool, objective: Expr, constraints: Vec<Constraint>) -> Self {
        let n = vars.len();
        let top = objective
            .max_var()
            .into_iter()
            .chain(constraints.iter().filter_map(Constraint::max_var))
            .max();
        if let Some(v) = top {
            assert!(v < n, "the formula reads variable {v} of {n}");
        }
        let objective_poly = Poly::compile(&objective, n);
        let constraint_polys: Vec<Poly> = constraints
            .iter()
            .map(|c| Poly::compile(&c.tracked(), n))
            .collect();
        let interaction_neighbors = interaction_neighbors(&objective_poly, &constraint_polys, n);
        let mut var_constraints = vec![vec![]; n];
        for (c, poly) in constraint_polys.iter().enumerate() {
            for (v, terms) in poly.var_terms.iter().enumerate() {
                if !terms.is_empty() {
                    var_constraints[v].push(c);
                }
            }
        }
        Self {
            vars,
            maximize,
            objective,
            constraints,
            objective_poly,
            constraint_polys,
            interaction_neighbors,
            var_constraints,
        }
    }

    /// The variables.
    pub fn variables(&self) -> &IntVars {
        &self.vars
    }

    /// The objective expression's value, before any penalty.
    pub fn eval_objective(&self, values: &[i64]) -> f64 {
        self.objective.eval(values)
    }

    /// The sum of the constraints' penalties.
    pub fn eval_penalty(&self, values: &[i64]) -> f64 {
        self.constraints.iter().map(|c| c.penalty_at(values)).sum()
    }

    /// The penalized objective, with the direction of the problem. This is
    /// what a solution reports and what the search ranks by.
    pub fn objective(&self, values: &[i64]) -> Evaluable<f64> {
        let obj = self.eval_objective(values);
        let penalty = self.eval_penalty(values);
        if self.maximize {
            Evaluable::Maximize(obj - penalty)
        } else {
            Evaluable::Minimize(obj + penalty)
        }
    }

    /// A solution with the given values.
    ///
    /// Fails if the number of values differs from the number of variables, if
    /// a value lies outside its variable's range, or if the variables are a
    /// permutation and a value repeats.
    pub fn solution_from(&self, values: Vec<i64>) -> Result<FormulaSolution, OptError> {
        self.vars.check(&values)?;
        Ok(self.build_solution(values))
    }

    fn build_solution(&self, values: Vec<i64>) -> FormulaSolution {
        let objective = self.objective(&values);
        let constraint_vals = self
            .constraint_polys
            .iter()
            .map(|p| p.eval(&values))
            .collect();
        let mut sol = FormulaSolution {
            values,
            objective,
            constraint_vals,
            deltas: vec![0.0; self.vars.total_changes() as usize],
        };
        for var in 0..sol.values.len() {
            self.fill_row(&mut sol, var);
        }
        sol
    }

    /// The change of setting `x[i]` to `b`, penalties included, as the
    /// difference of the raw objective.
    #[inline]
    fn change_delta(&self, x: &[i64], cv: &[f64], i: usize, b: i64) -> f64 {
        self.priced(
            self.objective_poly.change_delta(x, i, b),
            cv,
            self.var_constraints[i].iter().copied(),
            |p| p.change_delta(x, i, b),
        )
    }

    /// The change of the raw objective when the objective expression moves by
    /// `d_obj` and each constraint in `touched` moves by `d` of its
    /// polynomial, from the values `cv`.
    #[inline]
    fn priced(
        &self,
        d_obj: f64,
        cv: &[f64],
        touched: impl Iterator<Item = usize>,
        d: impl Fn(&Poly) -> f64,
    ) -> f64 {
        let d_pen: f64 = touched
            .map(|c| {
                let (con, v) = (&self.constraints[c], cv[c]);
                con.penalty(v + d(&self.constraint_polys[c])) - con.penalty(v)
            })
            .sum();
        if self.maximize {
            d_obj - d_pen
        } else {
            d_obj + d_pen
        }
    }

    /// Recomputes the cached delta of every change of variable `var`.
    fn fill_row(&self, sol: &mut FormulaSolution, var: usize) {
        let v = self.vars[var];
        let start = self.vars.row_start(var);
        let cur = sol.values[var];
        for k in 0..v.num_changes() {
            sol.deltas[start + k as usize] =
                self.change_delta(&sol.values, &sol.constraint_vals, var, v.nth_other(k, cur));
        }
    }
}

/// For each variable, the other variables whose deltas can change with it.
fn interaction_neighbors(objective: &Poly, constraints: &[Poly], n: usize) -> Vec<Vec<usize>> {
    use std::collections::BTreeSet;
    let mut nbrs: Vec<BTreeSet<usize>> = vec![BTreeSet::new(); n];
    let mut link = |vars: &[usize]| {
        for (a, &va) in vars.iter().enumerate() {
            for &vb in &vars[a + 1..] {
                if va != vb {
                    nbrs[va].insert(vb);
                    nbrs[vb].insert(va);
                }
            }
        }
    };
    for m in &objective.terms {
        let vars: Vec<usize> = m.vars.iter().map(|&(v, _)| v).collect();
        link(&vars);
    }
    for poly in constraints {
        let vars: BTreeSet<usize> = poly
            .terms
            .iter()
            .flat_map(|m| m.vars.iter().map(|&(v, _)| v))
            .collect();
        link(&vars.into_iter().collect::<Vec<_>>());
    }
    nbrs.into_iter().map(|s| s.into_iter().collect()).collect()
}

/// A solution of a [`FormulaProblem`].
#[derive(Debug, Clone)]
pub struct FormulaSolution {
    values: Vec<i64>,
    objective: Evaluable<f64>,
    /// The value of each constraint's expression, `lhs - rhs` or `expr`.
    constraint_vals: Vec<f64>,
    /// The raw objective's change for every change of a variable, laid out
    /// like `IntVars::row_start`.
    deltas: Vec<f64>,
}

impl FormulaSolution {
    /// The value of every variable, in index order.
    pub fn values(&self) -> &[i64] {
        &self.values
    }
}

impl Evaluate for FormulaSolution {
    /// The penalized objective, with the direction of the problem.
    fn evaluate(&self) -> Evaluable<f64> {
        self.objective
    }
}

impl Distance for FormulaSolution {
    /// The number of variables whose values differ.
    fn distance(&self, other: &Self) -> usize {
        crate::common::hamming_distance(&self.values, &other.values)
    }
}

impl ProblemTrait for FormulaProblem {
    type Solution = FormulaSolution;

    /// Draws every variable uniformly from its range, or a uniformly random
    /// permutation when the variables are one.
    fn new_solution(&self, rng: &mut impl Rng) -> FormulaSolution {
        self.build_solution(self.vars.random_values(rng))
    }
}

impl IntAssignment for FormulaProblem {
    #[inline]
    fn domains(&self) -> &IntVars {
        &self.vars
    }

    #[inline]
    fn get(sol: &FormulaSolution, i: usize) -> i64 {
        sol.values[i]
    }

    /// Moves the constraint values and the objective, then refreshes the
    /// deltas of `i` and of every variable that interacts with it.
    fn assign(&self, sol: &mut FormulaSolution, i: usize, value: i64) {
        let delta = self.assign_delta(sol, i, value);
        for &c in &self.var_constraints[i] {
            sol.constraint_vals[c] += self.constraint_polys[c].change_delta(&sol.values, i, value);
        }
        sol.values[i] = value;
        sol.objective = super::problem::with_value(sol.objective, raw(sol.objective) + delta);
        if self.vars[i].num_changes() == 1 {
            // A binary variable's one change is to go back, which undoes this.
            sol.deltas[self.vars.row_start(i)] = -delta;
        } else {
            self.fill_row(sol, i);
        }
        for &j in &self.interaction_neighbors[i] {
            self.fill_row(sol, j);
        }
    }

    #[inline]
    fn assign_delta(&self, sol: &FormulaSolution, i: usize, value: i64) -> f64 {
        sol.deltas[self.vars.change_slot(i, sol.values[i], value)]
    }

    #[inline]
    fn slot_delta(&self, sol: &FormulaSolution, slot: usize, _: usize, _: i64) -> f64 {
        sol.deltas[slot]
    }

    /// Priced from the monomials and constraints reading `i` or `j`.
    fn assign_swap_delta(&self, sol: &FormulaSolution, i: usize, j: usize) -> f64 {
        let x = &sol.values;
        self.priced(
            self.objective_poly.swap_delta(x, i, j),
            &sol.constraint_vals,
            0..self.constraints.len(),
            |p| p.swap_delta(x, i, j),
        )
    }
}

impl SubProblemExtractable for FormulaProblem {
    /// The problem over the variables whose values differ between the parents,
    /// with every other variable fixed at its value in `sol1` and substituted
    /// into the objective and the constraints.
    ///
    /// # Panics
    ///
    /// If the variables are a [permutation](IntVars::permutation), since
    /// fixing some positions of a permutation does not leave one.
    fn extract_sub_problem(&self, sol1: &FormulaSolution, sol2: &FormulaSolution) -> Self {
        assert!(
            !self.vars.is_permutation(),
            "a sub-problem of a permutation is not a permutation"
        );
        let n = self.vars.len();
        let mut fixed = vec![None; n];
        let mut remap = vec![None; n];
        let mut free = Vec::new();
        for i in 0..n {
            if sol1.values[i] == sol2.values[i] {
                fixed[i] = Some(sol1.values[i]);
            } else {
                remap[i] = Some(free.len());
                free.push(self.vars[i]);
            }
        }
        Self::build(
            IntVars::new(free),
            self.maximize,
            self.objective.substitute(&fixed, &remap),
            self.constraints
                .iter()
                .map(|c| c.substitute(&fixed, &remap))
                .collect(),
        )
    }

    /// The full solution taking the free variables from `sub_solution` and
    /// the rest from `sol1`.
    fn lift_solution(
        &self,
        sol1: &FormulaSolution,
        sol2: &FormulaSolution,
        sub_solution: &FormulaSolution,
    ) -> FormulaSolution {
        let mut free = sub_solution.values.iter();
        let values = sol1
            .values
            .iter()
            .zip(&sol2.values)
            .map(|(&a, &b)| {
                if a == b {
                    a
                } else {
                    *free
                        .next()
                        .expect("one sub-problem value per free variable")
                }
            })
            .collect();
        self.build_solution(values)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::heuristic::{Heuristic, StopCondition, TabuSearch};
    use crate::problem::{IntChangeNeighbor, IntSwapNeighbor, IntVar};
    use crate::search_state::{MoveToNeighbor, Rankable, SearchState};
    use rand::SeedableRng;
    use rand::rngs::SmallRng;

    fn binary(n: usize) -> IntVars {
        (0..n).map(|_| IntVar::binary()).collect()
    }

    /// maximize x[0] + 2*x[1] + 3*x[2]  s.t. x[0] + x[1] + x[2] <= 2
    fn make_problem() -> FormulaProblem {
        let objective = Expr::Var(0) + 2.0 * Expr::Var(1) + 3.0 * Expr::Var(2);
        FormulaProblem::maximize(binary(3), objective).with_constraint(Constraint::Comparison {
            lhs: Expr::Var(0) + Expr::Var(1) + Expr::Var(2),
            rel: ConstraintRel::Le,
            rhs: Expr::Const(2.0),
            penalty_weight: 10.0,
        })
    }

    fn raw_of(prob: &FormulaProblem, x: &[i64]) -> f64 {
        raw(prob.objective(x))
    }

    #[test]
    fn a_feasible_assignment_scores_its_objective() {
        // obj = 5, sum = 2 <= 2, no penalty
        assert!((raw_of(&make_problem(), &[0, 1, 1]) - 5.0).abs() < 1e-9);
    }

    #[test]
    fn an_infeasible_assignment_pays_the_penalty() {
        // obj = 6, sum = 3 > 2, violation 1 at weight 10
        assert!((raw_of(&make_problem(), &[1, 1, 1]) - (-4.0)).abs() < 1e-9);
    }

    #[test]
    fn a_minimized_formula_adds_its_penalty_and_ranks_lower_as_better() {
        let prob = FormulaProblem::minimize(binary(2), Expr::Var(0) + Expr::Var(1))
            .with_constraint(Constraint::Comparison {
                lhs: Expr::Var(0),
                rel: ConstraintRel::Ge,
                rhs: Expr::Const(1.0),
                penalty_weight: 5.0,
            });
        assert!(matches!(prob.objective(&[0, 0]), Evaluable::Minimize(v) if v == 5.0));
        let a = prob.solution_from(vec![1, 0]).unwrap();
        let b = prob.solution_from(vec![1, 1]).unwrap();
        assert!(a.is_better_than(&b));
    }

    /// The operator impls flatten nested `Add` and `Mul` as they build, and
    /// the compilation folds either shape. So the shape of the tree is not a
    /// contract, what is a contract is that the two shapes score the same.
    #[test]
    fn a_flattened_expression_scores_the_same_as_a_nested_one() {
        let nested = Expr::Add(vec![
            Expr::Add(vec![
                Expr::Var(0),
                Expr::Mul(vec![Expr::Const(2.0), Expr::Var(1)]),
            ]),
            Expr::Mul(vec![
                Expr::Mul(vec![Expr::Const(3.0), Expr::Var(2)]),
                Expr::Const(1.0),
            ]),
            Expr::Neg(Box::new(Expr::Neg(Box::new(Expr::Var(0))))),
        ]);
        let flat = Expr::Var(0) + 2.0 * Expr::Var(1) + 3.0 * Expr::Var(2) + Expr::Var(0);
        let (a, b) = (
            FormulaProblem::maximize(binary(3), nested),
            FormulaProblem::maximize(binary(3), flat),
        );
        for i in 0..8i64 {
            let x: Vec<i64> = (0..3).map(|k| i >> k & 1).collect();
            assert!((raw_of(&a, &x) - raw_of(&b, &x)).abs() < 1e-9, "{x:?}");
        }
    }

    /// `Lt` and `Gt` charge `STRICT_EPSILON` at equality, where `Le` and `Ge`
    /// charge nothing, so a tie is never free under a strict relation.
    #[test]
    fn the_strict_relations_charge_at_equality_and_the_loose_ones_do_not() {
        let with = |rel| {
            FormulaProblem::maximize(binary(2), Expr::Const(0.0)).with_constraint(
                Constraint::Comparison {
                    lhs: Expr::Var(0) + Expr::Var(1),
                    rel,
                    rhs: Expr::Const(1.0),
                    penalty_weight: 10.0,
                },
            )
        };
        let (below, equal, above) = ([0, 0], [1, 0], [1, 1]);
        assert_eq!(with(ConstraintRel::Le).eval_penalty(&equal), 0.0);
        assert!(with(ConstraintRel::Lt).eval_penalty(&equal) > 0.0);
        assert_eq!(with(ConstraintRel::Ge).eval_penalty(&equal), 0.0);
        assert!(with(ConstraintRel::Gt).eval_penalty(&equal) > 0.0);
        assert_eq!(with(ConstraintRel::Eq).eval_penalty(&equal), 0.0);
        for x in [&below, &above] {
            let (lt, le) = (
                with(ConstraintRel::Lt).eval_penalty(x),
                with(ConstraintRel::Le).eval_penalty(x),
            );
            assert!((lt - le).abs() < 1e-6, "{x:?}");
        }
        assert!((with(ConstraintRel::Eq).eval_penalty(&below) - 10.0).abs() < 1e-9);
        assert!((with(ConstraintRel::Eq).eval_penalty(&above) - 10.0).abs() < 1e-9);
    }

    #[test]
    fn a_clamp_charges_the_distance_outside_its_range() {
        let prob = FormulaProblem::maximize(binary(3), Expr::Const(0.0)).with_constraint(
            Constraint::Clamp {
                expr: Expr::Var(0) + Expr::Var(1) + Expr::Var(2),
                lo: 1.0,
                hi: 2.0,
                penalty_weight: 10.0,
            },
        );
        assert!((prob.eval_penalty(&[0, 0, 0]) - 10.0).abs() < 1e-9);
        assert!((prob.eval_penalty(&[1, 0, 0]) - 0.0).abs() < 1e-9);
        assert!((prob.eval_penalty(&[1, 1, 1]) - 10.0).abs() < 1e-9);
    }

    #[test]
    #[should_panic(expected = "division by non-constant expressions is not supported")]
    fn dividing_by_a_variable_panics() {
        let _ = Expr::Var(0) / Expr::Var(1);
    }

    #[test]
    #[should_panic(expected = "the formula reads variable 3 of 3")]
    fn a_variable_outside_the_domains_panics() {
        let _ = FormulaProblem::maximize(binary(3), Expr::Var(3));
    }

    /// Integer variables with powers, products and a constraint, so that
    /// every path of the delta computation runs on values other than 0 and 1.
    fn integer_problem() -> FormulaProblem {
        let vars: IntVars = [(-2, 3), (0, 4), (1, 5), (-3, 0)]
            .into_iter()
            .map(|(l, u)| IntVar::new(l, u))
            .collect();
        let (x0, x1, x2, x3) = (Expr::Var(0), Expr::Var(1), Expr::Var(2), Expr::Var(3));
        let objective = x0.clone() * x0.clone() * x1.clone() - 3.0 * x1.clone() * x2.clone()
            + x2.clone() * x2.clone() * x2.clone()
            + 0.5 * x3.clone()
            - x0.clone() * x3.clone();
        FormulaProblem::minimize(vars, objective).with_constraint(Constraint::Clamp {
            expr: x0 + x1 + x2 * 2.0 + x3,
            lo: 2.0,
            hi: 7.0,
            penalty_weight: 4.0,
        })
    }

    /// The change from `x` to `y` as a cost, lower being better.
    fn cost(prob: &FormulaProblem, x: &[i64], y: &[i64]) -> f64 {
        prob.objective(y).minimized() - prob.objective(x).minimized()
    }

    fn check_table(prob: &FormulaProblem, sol: &FormulaSolution) {
        assert!((raw(sol.evaluate()) - raw_of(prob, sol.values())).abs() < 1e-9);
        for m in IntChangeNeighbor::iter(prob, sol) {
            let mut x = sol.values().to_vec();
            x[m.var] = m.value;
            let want = cost(prob, sol.values(), &x);
            assert!((m.cost - want).abs() < 1e-9, "{m:?} want {want}");
        }
    }

    #[test]
    fn integer_deltas_stay_exact_along_random_moves() {
        let prob = integer_problem();
        let mut sol = prob.new_solution(&mut SmallRng::seed_from_u64(1));
        let mut r = SmallRng::seed_from_u64(2);
        let mut swaps = 0;
        for step in 0..300 {
            check_table(&prob, &sol);
            for m in IntSwapNeighbor::iter(&prob, &sol) {
                let mut x = sol.values().to_vec();
                x.swap(m.i, m.j);
                let want = cost(&prob, sol.values(), &x);
                assert!((m.cost - want).abs() < 1e-9, "{m:?} want {want}");
            }
            // The ranges differ, so some states have no swap that stays in range.
            let swap = (step % 3 == 0)
                .then(|| IntSwapNeighbor::random_neighbor(&prob, &sol, &mut r))
                .flatten();
            if let Some(m) = swap {
                m.apply_to_solution(&prob, &mut sol).unwrap();
                swaps += 1;
            } else {
                let m = IntChangeNeighbor::random_neighbor(&prob, &sol, &mut r).unwrap();
                m.apply_to_solution(&prob, &mut sol).unwrap();
            }
            for (v, &x) in prob.variables().iter().zip(sol.values()) {
                assert!(v.contains(x));
            }
        }
        assert!(swaps > 0, "the walk applied no swap");
    }

    #[test]
    fn binary_deltas_stay_exact_along_random_moves() {
        let prob = make_problem();
        let mut sol = prob.new_solution(&mut SmallRng::seed_from_u64(3));
        let mut r = SmallRng::seed_from_u64(4);
        for _ in 0..100 {
            check_table(&prob, &sol);
            let m = IntChangeNeighbor::random_neighbor(&prob, &sol, &mut r).unwrap();
            m.apply_to_solution(&prob, &mut sol).unwrap();
        }
    }

    #[test]
    fn tabu_search_finds_the_constrained_optimum() {
        let prob = make_problem();
        let mut state = SearchState::new_with_seed(&prob, 1);
        TabuSearch::<IntChangeNeighbor>::new(StopCondition::iterations(100), (1, 1))
            .run(&mut state)
            .unwrap();
        assert_eq!(state.best_solution.values(), &[0, 1, 1]);
    }

    #[test]
    fn extracting_keeps_only_the_variables_the_parents_disagree_on() {
        let prob = make_problem();
        let s = prob.solution_from(vec![1, 0, 1]).unwrap();
        assert_eq!(prob.extract_sub_problem(&s, &s).variables().len(), 0);
        let (a, b) = (
            prob.solution_from(vec![0, 0, 0]).unwrap(),
            prob.solution_from(vec![1, 1, 1]).unwrap(),
        );
        assert_eq!(prob.extract_sub_problem(&a, &b).variables().len(), 3);
    }

    #[test]
    fn a_lifted_solution_prices_like_one_built_from_scratch() {
        let prob = integer_problem();
        let a = prob.solution_from(vec![1, 2, 3, -1]).unwrap();
        let b = prob.solution_from(vec![1, 4, 3, -3]).unwrap();
        let sub = prob.extract_sub_problem(&a, &b);
        assert_eq!(sub.variables().len(), 2);
        let sub_sol = sub.solution_from(vec![0, -2]).unwrap();
        // The sub-problem's objective is the full one with the fixed values in.
        assert!((raw(sub_sol.evaluate()) - raw_of(&prob, &[1, 0, 3, -2])).abs() < 1e-9);
        let lifted = prob.lift_solution(&a, &b, &sub_sol);
        assert_eq!(lifted.values(), &[1, 0, 3, -2]);
        check_table(&prob, &lifted);
    }
}
