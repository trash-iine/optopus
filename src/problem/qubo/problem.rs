use std::collections::HashSet;
use std::fs::File;
use std::io::{BufRead, BufReader};

use crate::search_state::{ProblemTrait, Rankable};

/// Integer coefficient type used in the Q matrix.
pub type Coefficient = i32;

/// A solution to the QUBO problem.
///
/// - `x` — variable assignment (`x[i] = true` means variable `i` is set to 1)
/// - `gain` — incremental energy change for each variable flip (`gain[i] < 0` means flipping `i` improves the objective)
/// - `objective` — current energy value (minimized)
#[derive(Debug, Clone)]
pub struct QuboSolution {
    pub x: Vec<bool>,
    pub gain: Vec<Coefficient>,
    pub objective: Coefficient,
}

impl Rankable for QuboSolution {
    fn is_better_than(&self, other: &Self) -> bool {
        self.objective < other.objective
    }
}

impl QuboSolution {
    /// Iterates over all variable indices in the solution.
    pub fn iter_on_variables(&self) -> impl Iterator<Item = usize> + '_ {
        0..self.x.len()
    }
}

/// QUBO problem instance.
///
/// Stores the Q matrix as a symmetric sparse adjacency list.
/// Diagonal entries `Q[i][i]` represent linear coefficients and are stored
/// in `adj[i]` as `(i, Q[i][i])`.
#[derive(Debug, Clone)]
pub struct Qubo {
    /// adj[i] = list of (j, Q[i][j]) for all j in Q[i], including diagonal (i, Q[i][i]).
    adj: Vec<Vec<(usize, Coefficient)>>,
    /// Sorted list of variable IDs that appear in the problem.
    variables: Vec<usize>,
}

impl Qubo {
    pub fn new() -> Qubo {
        Qubo {
            adj: vec![],
            variables: vec![],
        }
    }

    /// Sets (or overwrites) Q[i][j] = Q[j][i] = v.
    /// For diagonal entries (i == j), stores only once.
    pub fn set_q(&mut self, i: usize, j: usize, v: Coefficient) {
        let n = i.max(j) + 1;
        if self.adj.len() < n {
            self.adj.resize_with(n, Vec::new);
        }
        self.set_directed(i, j, v);
        if i != j {
            self.set_directed(j, i, v);
        }
        for &var in &[i, j] {
            if let Err(pos) = self.variables.binary_search(&var) {
                self.variables.insert(pos, var);
            }
        }
    }

    fn set_directed(&mut self, from: usize, to: usize, v: Coefficient) {
        if let Some(entry) = self.adj[from].iter_mut().find(|(k, _)| *k == to) {
            entry.1 = v;
        } else {
            self.adj[from].push((to, v));
        }
    }

    pub fn get_q(&self, i: usize, j: usize) -> Option<Coefficient> {
        if i < self.adj.len() {
            self.adj[i].iter().find(|(k, _)| *k == j).map(|(_, v)| *v)
        } else {
            None
        }
    }

    pub fn iter_on_variables(&self) -> impl Iterator<Item = &usize> {
        self.variables.iter()
    }

    pub fn iter_on_adjacency(&self, i: usize) -> std::slice::Iter<'_, (usize, Coefficient)> {
        if i < self.adj.len() {
            self.adj[i].iter()
        } else {
            [].iter()
        }
    }

    pub fn num_of_variables(&self) -> usize {
        self.variables.len()
    }

    pub fn load_file_as_max_cut(filename: &str) -> Result<Self, crate::error::OptError> {
        use crate::error::OptError;

        let err = |line: usize, detail: String| OptError::FileLoad {
            path: filename.to_string(),
            line,
            detail,
        };

        let file = File::open(filename).map_err(|e| err(0, format!("failed to open file: {e}")))?;
        let reader = BufReader::new(file);
        let mut line_iter = reader.lines();
        let (n, _) = {
            let line = line_iter
                .next()
                .ok_or_else(|| {
                    err(
                        1,
                        "file is empty, expected header 'N M' (MaxCut format for QUBO conversion)"
                            .into(),
                    )
                })?
                .map_err(|e| err(1, format!("failed to read header line: {e}")))?;
            let mut iter = line.split_whitespace();
            let n = iter
                .next()
                .ok_or_else(|| err(1, "expected header 'N M', but line is empty".into()))?
                .parse::<usize>()
                .map_err(|e| err(1, format!("failed to parse variable count N: {e}")))?;
            let m = iter
                .next()
                .ok_or_else(|| {
                    err(
                        1,
                        "expected header 'N M', but entry count M is missing".into(),
                    )
                })?
                .parse::<usize>()
                .map_err(|e| err(1, format!("failed to parse entry count M: {e}")))?;
            (n, m)
        };

        let mut qubo = Qubo {
            adj: vec![vec![]; n],
            variables: (0..n).collect(),
        };

        let mut line_num = 1;
        while let Some(result) = line_iter.next() {
            line_num += 1;
            let line = result.map_err(|e| err(line_num, format!("failed to read line: {e}")))?;
            if line.trim().is_empty() {
                continue;
            }
            let mut iter = line.split_whitespace();
            let i = iter
                .next()
                .ok_or_else(|| {
                    err(
                        line_num,
                        "expected entry 'i j v', but index i is missing".into(),
                    )
                })?
                .parse::<usize>()
                .map_err(|e| err(line_num, format!("failed to parse index i: {e}")))?;
            if i == 0 {
                return Err(err(line_num, "index i must be >= 1 (1-indexed)".into()));
            }
            let i = i - 1;
            let j = iter
                .next()
                .ok_or_else(|| {
                    err(
                        line_num,
                        "expected entry 'i j v', but index j is missing".into(),
                    )
                })?
                .parse::<usize>()
                .map_err(|e| err(line_num, format!("failed to parse index j: {e}")))?;
            if j == 0 {
                return Err(err(line_num, "index j must be >= 1 (1-indexed)".into()));
            }
            let j = j - 1;
            let v = iter
                .next()
                .ok_or_else(|| {
                    err(
                        line_num,
                        "expected entry 'i j v', but coefficient v is missing".into(),
                    )
                })?
                .parse::<i32>()
                .map_err(|e| err(line_num, format!("failed to parse coefficient v: {e}")))?;

            // MaxCut → QUBO transformation:
            // edge (i,j,w) contributes: Q[i][j] += 2w, Q[i][i] -= w, Q[j][j] -= w
            if let Some(old_v) = qubo.get_q(i, j) {
                qubo.set_directed(i, j, old_v + 2 * v);
                qubo.set_directed(j, i, old_v + 2 * v);
            } else {
                qubo.adj[i].push((j, 2 * v));
                qubo.adj[j].push((i, 2 * v));
            }

            if let Some(old_v) = qubo.get_q(i, i) {
                qubo.set_directed(i, i, old_v - v);
            } else {
                qubo.adj[i].push((i, -v));
            }

            if let Some(old_v) = qubo.get_q(j, j) {
                qubo.set_directed(j, j, old_v - v);
            } else {
                qubo.adj[j].push((j, -v));
            }
        }

        Ok(qubo)
    }

    /// Calculates the change in energy when variable `i` is flipped.
    ///
    /// Returns `gain` such that `E(x') = E(x) + gain` where `x'` is `x` with variable `i` flipped.
    /// A negative value indicates an improvement (energy decrease).
    pub fn calculate_gain(&self, x: &[bool], i: usize) -> Coefficient {
        let mut gain = 0;
        for &(j, q) in self.iter_on_adjacency(i) {
            if i == j {
                gain += q;
            } else if x[j] {
                gain += q;
            }
        }
        if x[i] { -gain } else { gain }
    }

    /// Calculates the total energy `E(x) = Σ Q[i][j] * x[i] * x[j]` for the given assignment.
    pub fn calculate_energy(&self, x: &[bool]) -> Coefficient {
        let mut energy = 0;
        for &i in self.iter_on_variables() {
            if !x[i] {
                continue;
            }
            for &(j, q) in self.iter_on_adjacency(i) {
                if i < j {
                    continue;
                }
                if i == j {
                    energy += q;
                } else if x[j] {
                    energy += q;
                }
            }
        }
        energy
    }
}

impl ProblemTrait for Qubo {
    type Solution = QuboSolution;

    fn new_solution(&self, rng: &mut impl rand::Rng) -> Self::Solution {
        let n = self.adj.len();
        let mut x = vec![false; n];
        for &i in self.iter_on_variables() {
            x[i] = rng.random_bool(0.5);
        }
        let mut gain = vec![0; n];
        for &i in self.iter_on_variables() {
            gain[i] = self.calculate_gain(&x, i);
        }
        let objective = self.calculate_energy(&x);
        QuboSolution { x, gain, objective }
    }
}

pub fn calc_xor_of_solutions(sol1: &QuboSolution, sol2: &QuboSolution) -> usize {
    sol1.x
        .iter()
        .zip(sol2.x.iter())
        .filter(|(a, b)| a != b)
        .count()
}

pub fn make_sub_problem_from(qubo: &Qubo, parents: &[&QuboSolution]) -> Qubo {
    let mut ind_set = HashSet::new();
    for &ind in qubo.iter_on_variables() {
        let base = parents[0].x[ind];
        let is_free = parents[1..].iter().any(|p| p.x[ind] != base);
        if is_free {
            ind_set.insert(ind);
        }
    }

    let mut sub_qubo = Qubo::new();

    for &ind in ind_set.iter() {
        for &(j, v) in qubo.iter_on_adjacency(ind) {
            if ind_set.contains(&j) {
                if ind < j {
                    sub_qubo.set_q(ind, j, v);
                } else if ind == j {
                    if let Some(old_v) = sub_qubo.get_q(ind, ind) {
                        sub_qubo.set_q(ind, ind, old_v + v);
                    } else {
                        sub_qubo.set_q(ind, ind, v);
                    }
                }
            } else if j < parents[0].x.len() && parents[0].x[j] {
                if let Some(old_v) = sub_qubo.get_q(ind, ind) {
                    sub_qubo.set_q(ind, ind, old_v + v);
                } else {
                    sub_qubo.set_q(ind, ind, v);
                }
            }
        }
    }

    sub_qubo
}

#[cfg(test)]
mod qubo_tests {
    use super::*;

    #[test]
    fn test_qubo() {
        let mut qubo = Qubo::new();
        qubo.set_q(0, 1, 1);
        qubo.set_q(1, 2, 2);
        qubo.set_q(2, 0, 3);
        assert!(qubo.get_q(0, 1).is_some_and(|v| v == 1));
        assert!(qubo.get_q(1, 2).is_some_and(|v| v == 2));
        assert!(qubo.get_q(0, 2).is_some_and(|v| v == 3));
        assert_eq!(qubo.num_of_variables(), 3);
    }

    #[test]
    fn test_load_file_as_max_cut() {
        let qubo_result = Qubo::load_file_as_max_cut("data/max_cut/test_data.txt");
        assert!(qubo_result.is_ok());
        let qubo = qubo_result.unwrap();
        assert!(qubo.get_q(0, 1).is_some_and(|v| v == 2));
        assert!(qubo.get_q(1, 2).is_some_and(|v| v == -6));
        assert!(qubo.get_q(0, 0).is_some_and(|v| v == -1));
        assert!(qubo.get_q(1, 1).is_some_and(|v| v == 2));
        assert!(qubo.get_q(2, 2).is_some_and(|v| v == 3));
        assert_eq!(qubo.num_of_variables(), 3);
    }

    #[test]
    fn test_calc_energy() {
        let mut qubo = Qubo::new();
        qubo.set_q(0, 1, 1);
        qubo.set_q(1, 2, 2);
        qubo.set_q(2, 0, 3);
        qubo.set_q(2, 2, 1);
        {
            let x = vec![true, false, true];
            assert_eq!(qubo.calculate_energy(&x), 4);
        }
        {
            let x = vec![false, true, false];
            assert_eq!(qubo.calculate_energy(&x), 0);
        }
        {
            let x = vec![true, true, true];
            assert_eq!(qubo.calculate_energy(&x), 7);
        }
        {
            let x = vec![false, false, false];
            assert_eq!(qubo.calculate_energy(&x), 0);
        }
    }

    #[test]
    fn test_calc_gain() {
        let mut qubo = Qubo::new();
        qubo.set_q(0, 1, 1);
        qubo.set_q(1, 2, 2);
        qubo.set_q(2, 0, 3);
        qubo.set_q(2, 2, 1);
        {
            let x = vec![true, false, true];
            assert_eq!(qubo.calculate_energy(&x), 4);
            for i in 0..3 {
                let mut flipped = x.clone();
                flipped[i] = !flipped[i];
                assert_eq!(
                    qubo.calculate_gain(&x, i) + qubo.calculate_energy(&x),
                    qubo.calculate_energy(&flipped)
                );
            }
        }
        {
            let x = vec![false, true, false];
            assert_eq!(qubo.calculate_energy(&x), 0);
            for i in 0..3 {
                let mut flipped = x.clone();
                flipped[i] = !flipped[i];
                assert_eq!(
                    qubo.calculate_gain(&x, i) + qubo.calculate_energy(&x),
                    qubo.calculate_energy(&flipped)
                );
            }
        }
    }
}
