//! The eigenvalue bound of MaxCut, certified in floating point.

use super::problem::MaxCut;
use crate::building_blocks::{IntVars, binary::fixed_by};
use crate::trait_defs::{Evaluable, FixVariables, Relaxation};

/// The eigenvalue bound of Delorme and Poljak, a [`Relaxation`] of MaxCut much
/// tighter than the positive weights
/// [`BinaryRelaxation`](crate::building_blocks::BinaryRelaxation) sums.
///
/// With `L` the weighted Laplacian of a graph on `n` vertices, a cut of sides
/// `x ∈ {±1}ⁿ` weighs `¼ xᵀLx`, and since `xᵀ diag(u) x = Σ u` for any `u`,
///
/// ```text
/// cut(x) ≤ (n/4) λ_max(L + diag(u)) − ¼ Σ u      for every u.
/// ```
///
/// The node's fixed vertices are folded first, by
/// [`FixVariables::fix`](crate::trait_defs::FixVariables::fix), and the bound
/// is the folding's offset plus this bound of the folded graph.
///
/// # How the incumbent is used
///
/// The correction `u` is improved by subgradient steps, sized by Polyak's rule
/// with the incumbent standing in for the unknown optimum. The steps stop as
/// soon as the estimate falls to the incumbent, where the node is pruned
/// however much further it would have fallen. A node that cannot be pruned
/// gets the bound tightened to the iteration limit, since that bound is what
/// orders the open nodes and what [`dual_bound`](crate::heuristic::BranchAndBound::dual_bound)
/// reports.
///
/// # Certification
///
/// An iterative eigenvalue is never above the true one, so the estimate is
/// not a bound on its own. The value returned rests on a floating-point
/// Cholesky factorization of `sI − L − diag(u)` running to completion with
/// the safety shift of Rump, "Verification of positive definiteness", BIT 46
/// (2006), Corollary 2.4 and Algorithm 1, which proves `λ_max ≤ s` with every
/// rounding error accounted for. Where it cannot be certified, or the folded
/// graph has more than [`with_certify_limit`](Self::with_certify_limit)
/// vertices, the positive weights are returned instead. When every weight is
/// an integer the bound is rounded down.
///
/// The factorization is dense, so its cost grows with the cube of the folded
/// graph's size.
///
/// ```
/// use optopus::prelude::*;
/// use optopus::trait_defs::{BranchSpace, Relaxation};
///
/// // a 5-cycle cuts at most 4 of its edges, and the eigenvalue bound shows
/// // it where the positive weights say 5
/// let prob = MaxCut::new(Graph::from_edges((0..5).map(|i| (i, (i + 1) % 5, 1.0))));
/// let root = prob.ranges().into_owned();
/// assert_eq!(BinaryRelaxation.bound(&prob, &root).minimized(), -5.0);
/// assert_eq!(EigenvalueRelaxation::new().bound(&prob, &root).minimized(), -4.0);
/// ```
#[derive(Clone, Debug)]
pub struct EigenvalueRelaxation {
    max_iterations: usize,
    lanczos_steps: usize,
    certify_limit: usize,
}

impl Default for EigenvalueRelaxation {
    fn default() -> Self {
        Self::new()
    }
}

impl EigenvalueRelaxation {
    /// Subgradient steps per bound, unless stopped earlier.
    pub const MAX_ITERATIONS: usize = 100;
    /// Lanczos steps per eigenvalue estimate.
    pub const LANCZOS_STEPS: usize = 30;
    /// The largest folded graph the bound is certified on.
    pub const CERTIFY_LIMIT: usize = 3000;

    /// The relaxation with the default limits.
    pub fn new() -> Self {
        Self {
            max_iterations: Self::MAX_ITERATIONS,
            lanczos_steps: Self::LANCZOS_STEPS,
            certify_limit: Self::CERTIFY_LIMIT,
        }
    }

    /// At most `n` subgradient steps per bound.
    pub fn with_max_iterations(mut self, n: usize) -> Self {
        self.max_iterations = n;
        self
    }

    /// `n` Lanczos steps per eigenvalue estimate.
    pub fn with_lanczos_steps(mut self, n: usize) -> Self {
        self.lanczos_steps = n.max(2);
        self
    }

    /// Certify only folded graphs of at most `n` vertices, and fall back to the
    /// positive weights above that.
    pub fn with_certify_limit(mut self, n: usize) -> Self {
        self.certify_limit = n;
        self
    }

    /// The bound on the cut of the assignments in `vars`, given the best cut
    /// found so far if any.
    fn cut_bound(&self, prob: &MaxCut, vars: &IntVars, incumbent: Option<f64>) -> f64 {
        let fixed = prob.fix(&fixed_by(vars));
        let positive = fixed.offset + fixed.target.trivial_bound().raw();
        let lap = Laplacian::of(&fixed.target);
        if lap.n == 0 || lap.n > self.certify_limit {
            return positive;
        }
        let integral = fixed.offset.fract() == 0.0 && lap.weights_integral;
        // The folded cut the bound has to fall below for the node to go.
        let prune_below = incumbent.map(|lb| {
            let goal = lb - fixed.offset;
            if integral { goal + 1.0 } else { goal }
        });
        let best = self.subgradient(&lap, prune_below);
        // A certified bound is above the estimate, so an estimate already at
        // the positive weights cannot improve on them.
        if fixed.offset + best.value >= positive {
            return positive;
        }
        let Some(certified) = certify(&lap, &best) else {
            return positive;
        };
        let bound = (fixed.offset + certified).min(positive);
        if integral { bound.floor() } else { bound }
    }

    /// Subgradient descent on the correction `u`, with Polyak steps aimed at
    /// `prune_below` when there is one.
    fn subgradient(&self, lap: &Laplacian, prune_below: Option<f64>) -> Estimate {
        let n = lap.n as f64;
        let value_at = |theta: f64, u: &[f64]| 0.25 * n * theta - 0.25 * u.iter().sum::<f64>();
        let mean = lap.diag.iter().sum::<f64>() / n;
        let mut u: Vec<f64> = lap.diag.iter().map(|d| mean - d).collect();
        let start: Vec<f64> = (0..lap.n).map(start_component).collect();
        let (mut theta, mut v) = lanczos_top(lap, &u, &start, self.lanczos_steps);
        let mut value = value_at(theta, &u);
        let mut best = Estimate {
            value,
            theta,
            u: u.clone(),
            v: v.clone(),
        };
        let mut mu = 2.0;
        let mut since_improved = 0;
        for _ in 1..self.max_iterations {
            if mu < MIN_STEP_SCALE || prune_below.is_some_and(|stop| best.value < stop) {
                break;
            }
            let g: Vec<f64> = v.iter().map(|x| 0.25 * n * (x * x - 1.0 / n)).collect();
            let g2: f64 = g.iter().map(|x| x * x).sum();
            if g2 < 1e-30 {
                break;
            }
            // Aim at the incumbent, but no lower than a little below the best
            // estimate: a target far below the bound makes Polyak's steps
            // overshoot, and the node is not pruned anyway.
            let own = best.value - 0.05 * best.value.abs() - 1.0;
            let aim = match prune_below {
                Some(stop) if stop < value => stop.max(own),
                _ => own,
            };
            let step = mu * (value - aim) / g2;
            for (ui, gi) in u.iter_mut().zip(&g) {
                *ui -= step * gi;
            }
            (theta, v) = lanczos_top(lap, &u, &v, self.lanczos_steps);
            value = value_at(theta, &u);
            if value < best.value - 1e-9 * best.value.abs() {
                best = Estimate {
                    value,
                    theta,
                    u: u.clone(),
                    v: v.clone(),
                };
                since_improved = 0;
            } else {
                since_improved += 1;
                if since_improved >= PATIENCE {
                    mu /= 2.0;
                    since_improved = 0;
                }
            }
        }
        best
    }
}

impl Relaxation<MaxCut> for EigenvalueRelaxation {
    fn bound(&mut self, prob: &MaxCut, vars: &IntVars) -> Evaluable<f64> {
        Evaluable::Maximize(self.cut_bound(prob, vars, None))
    }

    fn bound_against(
        &mut self,
        prob: &MaxCut,
        vars: &IntVars,
        incumbent: Evaluable<f64>,
    ) -> Evaluable<f64> {
        Evaluable::Maximize(self.cut_bound(prob, vars, Some(-incumbent.minimized())))
    }
}

/// Steps without improvement before the step size halves.
const PATIENCE: usize = 5;
/// The step size scale below which the descent has stalled.
const MIN_STEP_SCALE: f64 = 1e-4;

/// The best point of the descent.
struct Estimate {
    /// `(n/4) θ − ¼ Σ u`, the bound were `θ` the true largest eigenvalue.
    value: f64,
    theta: f64,
    u: Vec<f64>,
    v: Vec<f64>,
}

/// A fixed, irregular start for Lanczos, which must not be orthogonal to the
/// top eigenvector. The all-ones vector would be, being an eigenvector of `L`.
fn start_component(i: usize) -> f64 {
    let h = (i as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    (h >> 11) as f64 / (1u64 << 53) as f64 - 0.5
}

/// The Laplacian of a graph, over the vertices that have an edge, numbered in
/// order.
struct Laplacian {
    n: usize,
    /// Per vertex, `(neighbour, weight)`.
    adj: Vec<Vec<(usize, f64)>>,
    /// The weighted degrees, `L_ii`.
    diag: Vec<f64>,
    weights_integral: bool,
}

impl Laplacian {
    fn of(prob: &MaxCut) -> Self {
        let graph = &prob.graph;
        let mut index = vec![usize::MAX; graph.len()];
        for (k, &v) in graph.iter_on_vertices().enumerate() {
            index[v] = k;
        }
        let n = graph.num_vertices();
        let mut adj = vec![vec![]; n];
        let mut weights_integral = true;
        for &v in graph.iter_on_vertices() {
            for &(w, weight) in graph.neighbors(v) {
                if w != v {
                    let weight = f64::from(weight);
                    weights_integral &= weight.fract() == 0.0;
                    adj[index[v]].push((index[w], weight));
                }
            }
        }
        let diag = adj
            .iter()
            .map(|row| row.iter().map(|e| e.1).sum())
            .collect();
        Self {
            n,
            adj,
            diag,
            weights_integral,
        }
    }

    /// `(L + diag(u)) x`.
    fn apply(&self, u: &[f64], x: &[f64], y: &mut [f64]) {
        for i in 0..self.n {
            let off: f64 = self.adj[i].iter().map(|&(j, w)| w * x[j]).sum();
            y[i] = (self.diag[i] + u[i]) * x[i] - off;
        }
    }
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn normalize(x: &mut [f64]) {
    let norm = dot(x, x).sqrt();
    if norm > 0.0 {
        x.iter_mut().for_each(|v| *v /= norm);
    }
}

/// The largest Ritz value of `L + diag(u)` after `steps` Lanczos steps from
/// `start`, with full reorthogonalization, and its unit Ritz vector. The value
/// is at most the largest eigenvalue.
fn lanczos_top(lap: &Laplacian, u: &[f64], start: &[f64], steps: usize) -> (f64, Vec<f64>) {
    let n = lap.n;
    let steps = steps.min(n);
    let mut q = start.to_vec();
    normalize(&mut q);
    let mut basis: Vec<Vec<f64>> = vec![q];
    let mut alpha = vec![];
    let mut beta: Vec<f64> = vec![];
    let mut w = vec![0.0; n];
    for j in 0..steps {
        lap.apply(u, &basis[j], &mut w);
        let a = dot(&basis[j], &w);
        alpha.push(a);
        for _ in 0..2 {
            for b in &basis {
                let c = dot(b, &w);
                w.iter_mut().zip(b).for_each(|(x, y)| *x -= c * y);
            }
        }
        let norm = dot(&w, &w).sqrt();
        if j + 1 == steps || norm <= 1e-12 * (a.abs() + 1.0) {
            break;
        }
        beta.push(norm);
        basis.push(w.iter().map(|x| x / norm).collect());
    }
    let m = alpha.len();
    let mut t = vec![0.0; m * m];
    for i in 0..m {
        t[i * m + i] = alpha[i];
        if i + 1 < m {
            t[i * m + i + 1] = beta[i];
            t[(i + 1) * m + i] = beta[i];
        }
    }
    let (values, vectors) = symmetric_eigen(t, m);
    let top = (0..m)
        .max_by(|&a, &b| values[a].total_cmp(&values[b]))
        .expect("at least one Lanczos step");
    let mut ritz = vec![0.0; n];
    for (k, b) in basis.iter().take(m).enumerate() {
        let y = vectors[k * m + top];
        ritz.iter_mut().zip(b).for_each(|(r, x)| *r += y * x);
    }
    normalize(&mut ritz);
    (values[top], ritz)
}

/// The eigenvalues of a small dense symmetric matrix, and its eigenvectors as
/// the columns of a row-major matrix, by cyclic Jacobi rotations.
fn symmetric_eigen(mut a: Vec<f64>, n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut v = vec![0.0; n * n];
    for i in 0..n {
        v[i * n + i] = 1.0;
    }
    for _ in 0..100 {
        let off: f64 = (0..n)
            .flat_map(|p| (p + 1..n).map(move |q| (p, q)))
            .map(|(p, q)| a[p * n + q] * a[p * n + q])
            .sum();
        let scale: f64 = (0..n).map(|i| a[i * n + i] * a[i * n + i]).sum::<f64>() + off;
        if off <= 1e-30 * scale {
            break;
        }
        for p in 0..n {
            for q in p + 1..n {
                let apq = a[p * n + q];
                if apq == 0.0 {
                    continue;
                }
                let theta = (a[q * n + q] - a[p * n + p]) / (2.0 * apq);
                let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;
                for k in 0..n {
                    let (akp, akq) = (a[k * n + p], a[k * n + q]);
                    a[k * n + p] = c * akp - s * akq;
                    a[k * n + q] = s * akp + c * akq;
                }
                for k in 0..n {
                    let (apk, aqk) = (a[p * n + k], a[q * n + k]);
                    a[p * n + k] = c * apk - s * aqk;
                    a[q * n + k] = s * apk + c * aqk;
                }
                for k in 0..n {
                    let (vkp, vkq) = (v[k * n + p], v[k * n + q]);
                    v[k * n + p] = c * vkp - s * vkq;
                    v[k * n + q] = s * vkp + c * vkq;
                }
            }
        }
    }
    ((0..n).map(|i| a[i * n + i]).collect(), v)
}

/// A certified bound on the folded cut from the estimate, or `None` when the
/// certification fails.
///
/// `s` starts at the estimated eigenvalue plus its residual and grows tenfold
/// on each failure. `λ_max(L + diag(u)) ≤ s` once `sI − L − diag(u)` is shown
/// positive definite, and the bound is `(n/4) s − ¼ Σ u` rounded up.
fn certify(lap: &Laplacian, est: &Estimate) -> Option<f64> {
    let mut r = vec![0.0; lap.n];
    lap.apply(&est.u, &est.v, &mut r);
    let residual = r
        .iter()
        .zip(&est.v)
        .map(|(y, x)| (y - est.theta * x).powi(2))
        .sum::<f64>()
        .sqrt();
    let mut margin = residual.max(1e-9 * (1.0 + est.theta.abs()));
    for _ in 0..4 {
        let s = est.theta + margin;
        if is_positive_definite_shifted(lap, &est.u, s) {
            return Some(cut_bound_from(lap.n, s, &est.u));
        }
        margin *= 10.0;
    }
    None
}

const EPS: f64 = f64::EPSILON / 2.0;

/// `(n/4) s − ¼ Σ u`, rounded up: the rounding of the sum is bounded by
/// `n eps Σ|u|`, and a relative `1e-12` covers the few operations around it.
fn cut_bound_from(n: usize, s: f64, u: &[f64]) -> f64 {
    let sum: f64 = u.iter().sum();
    let abs: f64 = u.iter().map(|x| x.abs()).sum();
    let nf = n as f64;
    let value = 0.25 * nf * s - 0.25 * sum + 0.25 * 2.0 * nf * EPS * abs;
    value + 1e-12 * (0.25 * nf * s.abs() + 0.25 * sum.abs() + 1.0)
}

/// Whether `sI − L − diag(u)` is proven positive definite, by Rump's
/// criterion (BIT 46, 2006, Corollary 2.4, bound II of Section 3, as in
/// Algorithm 1 without the scaling and reordering).
///
/// The matrix is formed in floating point. Its off-diagonal entries are the
/// weights, exactly. Each diagonal entry is computed with a rounding error
/// at most `e_i`, which is taken off together with Rump's shift `c`, so the
/// matrix factorized has every diagonal entry at most the exact one less `c`.
fn is_positive_definite_shifted(lap: &Laplacian, u: &[f64], s: f64) -> bool {
    let n = lap.n;
    let eta = f64::from_bits(1);
    let mut a = vec![0.0; n * n];
    let mut diag_up = vec![0.0; n];
    let mut err = vec![0.0; n];
    for i in 0..n {
        for &(j, w) in &lap.adj[i] {
            a[i * n + j] = w;
        }
        let d = (s - u[i]) - lap.diag[i];
        // The weighted degree carries at most deg·eps·Σ|w| of rounding, the two
        // subtractions eps of their operands each. Doubled for the products.
        let deg = lap.adj[i].len() as f64;
        let abs_row: f64 = lap.adj[i].iter().map(|e| e.1.abs()).sum();
        err[i] = 2.0 * (deg + 4.0) * EPS * (s.abs() + u[i].abs() + abs_row);
        if d - err[i] <= 0.0 {
            return false;
        }
        a[i * n + i] = d;
        diag_up[i] = (d + err[i]) * (1.0 + 4.0 * EPS);
    }
    // Rump's bound II: t_j counts the entries above the diagonal in column j's
    // envelope.
    let mut dd = 0.0;
    let mut max_diag: f64 = 0.0;
    for (j, &up) in diag_up.iter().enumerate() {
        let first = lap.adj[j]
            .iter()
            .map(|e| e.0)
            .filter(|&i| i < j)
            .min()
            .unwrap_or(j);
        let t = (j - first) as f64;
        let alpha = (t + 3.0) * EPS;
        let d = (alpha * up).sqrt() / (1.0 - 3.0 * EPS);
        dd += d * d;
        max_diag = max_diag.max(up);
    }
    let nf = n as f64;
    let c0 = (3.0 * nf * (2.0 * nf + max_diag) / (1.0 - 3.0 * EPS)) * eta;
    let c = 2.0 * ((dd / (1.0 - (nf + 2.0) * EPS) + c0) / (1.0 - 2.0 * EPS));
    let phi = EPS * (1.0 + 2.0 * EPS);
    for i in 0..n {
        let shifted = a[i * n + i] - (c + err[i]) * (1.0 + 2.0 * EPS);
        if shifted <= 0.0 {
            return false;
        }
        a[i * n + i] = shifted - shifted * phi;
    }
    cholesky_completes(&mut a, n)
}

/// Whether the floating-point Cholesky factorization of the symmetric matrix
/// `a` runs to completion, every pivot positive. `a` is overwritten by the
/// factor in its lower triangle.
fn cholesky_completes(a: &mut [f64], n: usize) -> bool {
    for j in 0..n {
        let row_j = &a[j * n..j * n + j];
        let pivot = a[j * n + j] - dot(row_j, row_j);
        if pivot <= 0.0 || !pivot.is_finite() {
            return false;
        }
        let pivot = pivot.sqrt();
        a[j * n + j] = pivot;
        let (upper, lower) = a.split_at_mut((j + 1) * n);
        let row_j = &upper[j * n..j * n + j];
        for row_i in lower.chunks_exact_mut(n) {
            let x = row_i[j] - dot(&row_i[..j], row_j);
            row_i[j] = x / pivot;
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::building_blocks::Graph;
    use rand::rngs::SmallRng;
    use rand::{Rng, SeedableRng};

    fn random_graph(rng: &mut SmallRng, n: usize) -> MaxCut {
        let mut edges = vec![];
        for i in 0..n {
            for j in i + 1..n {
                if rng.random_bool(0.4) {
                    edges.push((i, j, rng.random_range(-3..=5) as f32));
                }
            }
        }
        MaxCut::new(Graph::from_edges(edges))
    }

    /// `L + diag(u)` as a dense row-major matrix.
    fn dense(lap: &Laplacian, u: &[f64]) -> Vec<f64> {
        let n = lap.n;
        let mut m = vec![0.0; n * n];
        for i in 0..n {
            m[i * n + i] = lap.diag[i] + u[i];
            for &(j, w) in &lap.adj[i] {
                m[i * n + j] = -w;
            }
        }
        m
    }

    #[test]
    fn jacobi_diagonalizes_a_known_matrix() {
        // [[2,1],[1,2]] has eigenvalues 1 and 3
        let (values, vectors) = symmetric_eigen(vec![2.0, 1.0, 1.0, 2.0], 2);
        let mut sorted = values.clone();
        sorted.sort_by(f64::total_cmp);
        assert!((sorted[0] - 1.0).abs() < 1e-12 && (sorted[1] - 3.0).abs() < 1e-12);
        let top = if values[0] > values[1] { 0 } else { 1 };
        assert!((vectors[top].abs() - vectors[2 + top].abs()).abs() < 1e-12);
    }

    #[test]
    fn lanczos_finds_the_largest_eigenvalue() {
        let mut rng = SmallRng::seed_from_u64(0);
        for _ in 0..20 {
            let lap = Laplacian::of(&random_graph(&mut rng, 14));
            let u: Vec<f64> = (0..lap.n).map(|_| rng.random_range(-2.0..2.0)).collect();
            let (values, _) = symmetric_eigen(dense(&lap, &u), lap.n);
            let exact = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let start: Vec<f64> = (0..lap.n).map(start_component).collect();
            let (theta, _) = lanczos_top(&lap, &u, &start, lap.n);
            assert!((theta - exact).abs() < 1e-8, "{theta} against {exact}");
        }
    }

    #[test]
    fn the_certificate_holds_above_and_fails_below_the_top_eigenvalue() {
        let mut rng = SmallRng::seed_from_u64(1);
        for _ in 0..20 {
            let lap = Laplacian::of(&random_graph(&mut rng, 14));
            let u: Vec<f64> = (0..lap.n).map(|_| rng.random_range(-2.0..2.0)).collect();
            let (values, _) = symmetric_eigen(dense(&lap, &u), lap.n);
            let top = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            assert!(is_positive_definite_shifted(&lap, &u, top + 1e-6));
            assert!(!is_positive_definite_shifted(&lap, &u, top - 1e-6));
        }
    }
}
