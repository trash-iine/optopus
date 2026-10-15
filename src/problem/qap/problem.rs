use std::sync::OnceLock;

use rand::seq::SliceRandom;

use crate::building_blocks::representation::check_permutation;
use crate::error::OptError;
use crate::search_state::{Distance, Evaluable, Evaluate, ProblemTrait};

/// A solution to the quadratic assignment problem.
#[derive(Debug, Clone)]
pub struct QapSolution {
    /// `assignment[i]` is the location facility `i` is placed at, a
    /// permutation of `0..n`.
    pub assignment: Vec<usize>,
    /// The cost `Σ_i Σ_j a[i][j] · b[assignment[i]][assignment[j]]`.
    pub objective: i64,
    /// The swap deltas of every pair, built by the first full scan of the
    /// swap neighborhood and kept up to date by each applied swap. A search
    /// that never scans the whole neighborhood never builds it.
    pub(super) deltas: OnceLock<SwapDeltas>,
}

/// The swap delta of every pair of facilities for one assignment.
///
/// A full scan reads each pair in O(1) from it, and an applied swap brings it
/// up to date in O(n²), where pricing every pair anew is O(n³).
#[derive(Debug, Clone)]
pub(super) struct SwapDeltas {
    /// The assignment the deltas are for. A table whose assignment is not the
    /// solution's is never read.
    assignment: Vec<usize>,
    /// `delta[i * n + j]`, for `i < j`.
    delta: Vec<i64>,
}

impl Evaluate for QapSolution {
    /// The QAP minimizes its cost.
    fn evaluate(&self) -> Evaluable<f64> {
        Evaluable::Minimize(self.objective as f64)
    }
}

impl Distance for QapSolution {
    /// The number of facilities placed at different locations.
    fn distance(&self, other: &Self) -> usize {
        crate::building_blocks::representation::hamming_distance(
            &self.assignment,
            &other.assignment,
        )
    }
}

/// A quadratic assignment instance in the Koopmans and Beckmann form.
///
/// `n` facilities are placed at `n` locations, one each. Matrix `a` holds what
/// passes between every pair of facilities and `b` the distance between every
/// pair of locations, and the cost of an assignment `p` is
/// `Σ_i Σ_j a[i][j] · b[p(i)][p(j)]`, which is minimized. Neither matrix needs
/// to be symmetric. This is the order QAPLIB lists its two matrices in.
#[derive(Debug, Clone)]
pub struct Qap {
    /// Instance name, from the file stem or given by the caller.
    pub name: String,
    /// Number of facilities, and of locations.
    pub n: usize,
    /// Row-major `n × n` facility matrix.
    a: Vec<i64>,
    /// Row-major `n × n` location matrix.
    b: Vec<i64>,
}

impl Qap {
    /// Creates an instance from the two matrices.
    ///
    /// # Panics
    ///
    /// Panics if either matrix is not `n × n` for the same `n`.
    ///
    /// # Examples
    ///
    /// ```
    /// use optopus::prelude::*;
    ///
    /// // Facilities 0 and 1 exchange 5, locations 0 and 2 are far apart.
    /// let qap = Qap::new(
    ///     "tiny",
    ///     vec![vec![0, 5, 0], vec![5, 0, 1], vec![0, 1, 0]],
    ///     vec![vec![0, 1, 9], vec![1, 0, 2], vec![9, 2, 0]],
    /// );
    /// assert_eq!(qap.cost(&[0, 1, 2]), 2 * (5 * 1 + 1 * 2));
    /// ```
    pub fn new(name: impl Into<String>, a: Vec<Vec<i64>>, b: Vec<Vec<i64>>) -> Self {
        let n = a.len();
        assert!(
            b.len() == n && a.iter().chain(&b).all(|row| row.len() == n),
            "both matrices have to be n × n"
        );
        Self {
            name: name.into(),
            n,
            a: a.into_iter().flatten().collect(),
            b: b.into_iter().flatten().collect(),
        }
    }

    /// Loads a QAPLIB instance, `n` followed by the `n × n` matrices `a` and
    /// `b`, all whitespace separated. Line breaks inside a matrix row are
    /// allowed, as QAPLIB uses them for wide instances.
    pub fn load_file(path: impl AsRef<std::path::Path>) -> Result<Self, OptError> {
        use crate::building_blocks::instance::InstanceLines;

        let path = path.as_ref();
        let mut tokens = InstanceLines::open(path)?.into_tokens()?;
        let n: usize = tokens.next("n")?;
        let mut read_matrix = |name: &str| -> Result<Vec<Vec<i64>>, OptError> {
            (0..n)
                .map(|i| {
                    (0..n)
                        .map(|j| tokens.next(&format!("{name}[{i}][{j}]")))
                        .collect()
                })
                .collect()
        };
        let a = read_matrix("a")?;
        let b = read_matrix("b")?;
        tokens.expect_end()?;

        let name = path.file_stem().and_then(|s| s.to_str()).unwrap_or("qap");
        Ok(Self::new(name, a, b))
    }

    /// `a[i][j]`, between facilities `i` and `j`.
    #[inline]
    pub fn a(&self, i: usize, j: usize) -> i64 {
        self.a[i * self.n + j]
    }

    /// `b[k][l]`, between locations `k` and `l`.
    #[inline]
    pub fn b(&self, k: usize, l: usize) -> i64 {
        self.b[k * self.n + l]
    }

    /// The cost of an assignment, in O(n²).
    pub fn cost(&self, assignment: &[usize]) -> i64 {
        let mut total = 0;
        for (i, &pi) in assignment.iter().enumerate() {
            for (j, &pj) in assignment.iter().enumerate() {
                total += self.a(i, j) * self.b(pi, pj);
            }
        }
        total
    }

    /// What exchanging the locations of facilities `r` and `s` changes the
    /// cost by, in O(n).
    ///
    /// The general formula, for matrices that need not be symmetric, from
    /// Taillard's robust tabu search.
    pub fn swap_delta(&self, p: &[usize], r: usize, s: usize) -> i64 {
        let (pr, ps) = (p[r], p[s]);
        let mut delta = self.a(r, r) * (self.b(ps, ps) - self.b(pr, pr))
            + self.a(r, s) * (self.b(ps, pr) - self.b(pr, ps))
            + self.a(s, r) * (self.b(pr, ps) - self.b(ps, pr))
            + self.a(s, s) * (self.b(pr, pr) - self.b(ps, ps));
        for (k, &pk) in p.iter().enumerate() {
            if k == r || k == s {
                continue;
            }
            delta += self.a(k, r) * (self.b(pk, ps) - self.b(pk, pr))
                + self.a(k, s) * (self.b(pk, pr) - self.b(pk, ps))
                + self.a(r, k) * (self.b(ps, pk) - self.b(pr, pk))
                + self.a(s, k) * (self.b(pr, pk) - self.b(ps, pk));
        }
        delta
    }

    /// The swap deltas of `sol` indexed `i * n + j` for `i < j`, built on the
    /// first call, or `None` if `sol.assignment` was changed other than by a
    /// swap move since they were built.
    pub(super) fn swap_deltas<'s>(&self, sol: &'s QapSolution) -> Option<&'s [i64]> {
        let n = self.n;
        let table = sol.deltas.get_or_init(|| {
            let p = &sol.assignment;
            let mut delta = vec![0; n * n];
            for i in 0..n {
                for j in i + 1..n {
                    delta[i * n + j] = self.swap_delta(p, i, j);
                }
            }
            SwapDeltas {
                assignment: p.clone(),
                delta,
            }
        });
        (table.assignment == sol.assignment).then_some(&table.delta)
    }

    /// Exchanges facilities `r` and `s` in `sol`, its cost moving by `gain`,
    /// and brings the swap deltas along if they were built.
    ///
    /// A pair `(u, v)` that shares no facility with the swap changes by
    /// `(a[r][u] - a[r][v] - a[s][u] + a[s][v]) X + (a[u][r] - a[v][r] - a[u][s] + a[v][s]) Y`,
    /// where `X = b[ps][pv] - b[ps][pu] - b[pr][pv] + b[pr][pu]` and
    /// `Y = b[pv][ps] - b[pu][ps] - b[pv][pr] + b[pu][pr]` read the locations
    /// before the swap, Taillard's O(1) update. The 2n pairs that share one
    /// are priced anew.
    pub(super) fn apply_swap(&self, sol: &mut QapSolution, r: usize, s: usize, gain: i64) {
        let table = sol.deltas.take().filter(|t| t.assignment == sol.assignment);
        let (pr, ps) = (sol.assignment[r], sol.assignment[s]);
        sol.assignment.swap(r, s);
        sol.objective += gain;
        let Some(mut table) = table else {
            return;
        };
        let n = self.n;
        let p = &sol.assignment;
        for u in 0..n {
            for v in u + 1..n {
                table.delta[u * n + v] = if u == r || u == s || v == r || v == s {
                    self.swap_delta(p, u, v)
                } else {
                    let (pu, pv) = (p[u], p[v]);
                    let x = self.b(ps, pv) - self.b(ps, pu) - self.b(pr, pv) + self.b(pr, pu);
                    let y = self.b(pv, ps) - self.b(pu, ps) - self.b(pv, pr) + self.b(pu, pr);
                    table.delta[u * n + v]
                        + (self.a(r, u) - self.a(r, v) - self.a(s, u) + self.a(s, v)) * x
                        + (self.a(u, r) - self.a(v, r) - self.a(u, s) + self.a(v, s)) * y
                };
            }
        }
        table.assignment.clone_from(p);
        sol.deltas = OnceLock::from(table);
    }

    /// Wraps an assignment as a solution, computing its cost, for starting a
    /// search somewhere other than a random assignment.
    ///
    /// Fails if `assignment` is not a permutation of `0..n`.
    pub fn solution_from_assignment(
        &self,
        assignment: Vec<usize>,
    ) -> Result<QapSolution, OptError> {
        check_permutation(&assignment, self.n, "location")?;
        let objective = self.cost(&assignment);
        Ok(QapSolution {
            assignment,
            objective,
            deltas: OnceLock::new(),
        })
    }
}

impl ProblemTrait for Qap {
    type Solution = QapSolution;

    /// A uniformly random assignment.
    fn new_solution(&self, rng: &mut impl rand::Rng) -> QapSolution {
        let mut assignment: Vec<usize> = (0..self.n).collect();
        assignment.shuffle(rng);
        self.solution_from_assignment(assignment)
            .expect("a shuffle of 0..n is a permutation")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::SmallRng;
    use rand::{Rng, SeedableRng};

    fn random_instance(n: usize, seed: u64) -> Qap {
        let mut rng = SmallRng::seed_from_u64(seed);
        let mut matrix = || -> Vec<Vec<i64>> {
            (0..n)
                .map(|_| (0..n).map(|_| rng.random_range(-5..20)).collect())
                .collect()
        };
        let (a, b) = (matrix(), matrix());
        Qap::new("random", a, b)
    }

    /// The O(n) delta agrees with recomputing the cost, on asymmetric
    /// matrices with non-zero diagonals, for every pair.
    #[test]
    fn swap_delta_matches_a_full_recompute() {
        let qap = random_instance(9, 1);
        let mut rng = SmallRng::seed_from_u64(2);
        for _ in 0..10 {
            let sol = qap.new_solution(&mut rng);
            for r in 0..qap.n {
                for s in 0..qap.n {
                    if r == s {
                        continue;
                    }
                    let mut swapped = sol.assignment.clone();
                    swapped.swap(r, s);
                    assert_eq!(
                        qap.swap_delta(&sol.assignment, r, s),
                        qap.cost(&swapped) - sol.objective
                    );
                }
            }
        }
    }

    #[test]
    fn load_file_reads_qaplib_layout() {
        let dir = std::env::temp_dir().join("optopus_qap_load");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("tiny.dat");
        // A row of b wrapped over two lines, as wide QAPLIB files do.
        std::fs::write(&path, "2\n\n0 3\n3 0\n\n0\n7\n7 0\n").unwrap();
        let qap = Qap::load_file(&path).unwrap();
        assert_eq!((qap.n, qap.name.as_str()), (2, "tiny"));
        assert_eq!(qap.cost(&[0, 1]), 2 * 3 * 7);

        std::fs::write(&path, "2\n0 3\n3 0\n0 7\n7\n").unwrap();
        assert!(Qap::load_file(&path).is_err());
        std::fs::write(&path, "2\n0 3\n3 0\n0 7\n7 0\n1\n").unwrap();
        assert!(Qap::load_file(&path).is_err());
    }
}
