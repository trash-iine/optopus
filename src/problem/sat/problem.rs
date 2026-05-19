use std::fs::File;
use std::io::{BufRead, BufReader};

use crate::search_state::{Distance, ProblemTrait, Rankable};

/// A solution to the MaxSAT problem.
///
/// - `x` — variable assignment (`x[i]` is the truth value of variable `i+1`, 0-indexed)
/// - `gain` — incremental change in satisfied-clause count when flipping each variable
///   (`gain[i] > 0` means flipping variable `i` increases the number of satisfied clauses)
/// - `n_satisfied` — number of currently satisfied clauses
#[derive(Debug, Clone)]
pub struct SatSolution {
    pub x: Vec<bool>,
    pub gain: Vec<i64>,
    pub n_satisfied: usize,
}

impl Rankable for SatSolution {
    // MaxSAT: more satisfied clauses is better
    fn is_better_than(&self, other: &Self) -> bool {
        self.n_satisfied > other.n_satisfied
    }
}

impl Distance for SatSolution {
    fn distance(&self, other: &Self) -> usize {
        self.x
            .iter()
            .zip(other.x.iter())
            .filter(|(a, b)| a != b)
            .count()
    }
}

/// MaxSAT problem instance in DIMACS CNF format.
///
/// Variables are 1-indexed in the DIMACS format; internally, `clauses_per_var[i]`
/// stores the indices of clauses that contain variable `i+1` (0-indexed).
#[derive(Debug, Clone)]
pub struct Sat {
    /// Number of variables.
    n_vars: usize,
    /// All clauses (literals are signed integers, variables are 1-indexed).
    clauses: Vec<Vec<i64>>,
    /// `clauses_per_var[i]` = indices of clauses containing variable `i+1` (0-indexed).
    clauses_per_var: Vec<Vec<usize>>,
}

impl Sat {
    pub fn new(n_vars: usize) -> Self {
        Self {
            n_vars,
            clauses: Vec::new(),
            clauses_per_var: vec![vec![]; n_vars],
        }
    }

    pub fn n_vars(&self) -> usize {
        self.n_vars
    }

    pub fn n_clauses(&self) -> usize {
        self.clauses.len()
    }

    /// Adds a clause. Literals are signed integers (1-indexed variables).
    pub fn add_clause(&mut self, literals: impl IntoIterator<Item = i64>) {
        let clause_idx = self.clauses.len();
        let clause: Vec<i64> = literals.into_iter().collect();

        for &lit in &clause {
            let var_idx = lit.unsigned_abs() as usize - 1;
            if var_idx < self.n_vars {
                self.clauses_per_var[var_idx].push(clause_idx);
            }
        }
        self.clauses.push(clause);
    }

    /// Returns the literals of clause `idx`.
    pub fn clause(&self, idx: usize) -> &[i64] {
        &self.clauses[idx]
    }

    /// Returns an iterator over all clauses.
    pub fn all_clauses(&self) -> impl Iterator<Item = &[i64]> {
        self.clauses.iter().map(|c| c.as_slice())
    }

    /// Returns the indices of clauses that contain variable `i` (0-indexed).
    pub fn clause_indices_of_var(&self, i: usize) -> &[usize] {
        &self.clauses_per_var[i]
    }

    /// Returns an iterator over the literal slices of clauses that contain variable `i` (0-indexed).
    pub fn clauses_of_var(&self, i: usize) -> impl Iterator<Item = &[i64]> {
        self.clauses_per_var[i]
            .iter()
            .map(|&idx| self.clauses[idx].as_slice())
    }

    /// Returns `true` if the given clause is satisfied under assignment `x`.
    pub fn is_clause_satisfied(clause: &[i64], x: &[bool]) -> bool {
        clause.iter().any(|&lit| {
            let var = lit.unsigned_abs() as usize - 1;
            x[var] == (lit > 0)
        })
    }

    /// Counts the number of satisfied clauses under assignment `x`.
    pub fn calc_satisfied(&self, x: &[bool]) -> usize {
        self.clauses
            .iter()
            .filter(|c| Self::is_clause_satisfied(c, x))
            .count()
    }

    /// Calculates the change in satisfied-clause count when variable `i` (0-indexed) is flipped.
    ///
    /// Returns `gain` such that `calc_satisfied(x') = calc_satisfied(x) + gain`
    /// where `x'` is `x` with variable `i` flipped.
    pub fn calc_gain(&self, x: &[bool], i: usize) -> i64 {
        let mut gain = 0i64;
        for &clause_idx in &self.clauses_per_var[i] {
            let clause = &self.clauses[clause_idx];

            let i_lit = *clause
                .iter()
                .find(|&&lit| lit.unsigned_abs() as usize - 1 == i)
                .expect("clauses_per_var is inconsistent");
            let i_lit_sat = x[i] == (i_lit > 0);

            // check if any literal other than i satisfies the clause
            let other_sat = clause.iter().any(|&lit| {
                let var = lit.unsigned_abs() as usize - 1;
                if var == i {
                    return false;
                }
                x[var] == (lit > 0)
            });

            let was_sat = i_lit_sat || other_sat;
            let will_be_sat = !i_lit_sat || other_sat; // after flipping i

            gain += will_be_sat as i64 - was_sat as i64;
        }
        gain
    }

    /// Calculates the gain of variable `j` (0-indexed) assuming variable `flipped` has been
    /// virtually flipped (without actually modifying `x`).
    ///
    /// Used for efficiently computing swap-neighbor gain values.
    pub fn calc_gain_with_virtual_flip(&self, x: &[bool], flipped: usize, j: usize) -> i64 {
        let mut gain = 0i64;
        for &clause_idx in &self.clauses_per_var[j] {
            let clause = &self.clauses[clause_idx];

            let j_lit = *clause
                .iter()
                .find(|&&lit| lit.unsigned_abs() as usize - 1 == j)
                .expect("clauses_per_var is inconsistent");
            let j_lit_sat = x[j] == (j_lit > 0);

            // use the virtually flipped value for the `flipped` variable
            let other_sat = clause.iter().any(|&lit| {
                let var = lit.unsigned_abs() as usize - 1;
                if var == j {
                    return false;
                }
                let x_var = if var == flipped { !x[var] } else { x[var] };
                x_var == (lit > 0)
            });

            let was_sat = j_lit_sat || other_sat;
            let will_be_sat = !j_lit_sat || other_sat;

            gain += will_be_sat as i64 - was_sat as i64;
        }
        gain
    }

    /// Loads a MaxSAT instance from a DIMACS CNF file.
    pub fn load_file(filename: &str) -> Result<Self, crate::error::OptError> {
        use crate::error::OptError;

        let err = |line: usize, detail: String| OptError::FileLoad {
            path: filename.to_string(),
            line,
            detail,
        };

        let file = File::open(filename)
            .map_err(|e| err(0, format!("failed to open file: {e}")))?;
        let reader = BufReader::new(file);

        let mut n_clauses = 0usize;
        let mut sat = None::<Self>;

        for (idx, result) in reader.lines().enumerate() {
            let line_num = idx + 1;
            let line = result
                .map_err(|e| err(line_num, format!("failed to read line: {e}")))?;
            let trimmed = line.trim();
            if trimmed.is_empty() || trimmed.starts_with('c') || trimmed.starts_with('%') {
                continue; // skip comments and SATLIB '%' terminator
            }
            if trimmed.starts_with('p') {
                // "p cnf <n_vars> <n_clauses>"
                let mut iter = trimmed.split_whitespace();
                iter.next(); // "p"
                iter.next(); // "cnf"
                let n_vars: usize = iter
                    .next()
                    .ok_or_else(|| err(line_num, "expected header 'p cnf <n_vars> <n_clauses>', but n_vars is missing".into()))?
                    .parse()
                    .map_err(|e| err(line_num, format!("failed to parse n_vars in header 'p cnf <n_vars> <n_clauses>': {e}")))?;
                n_clauses = iter
                    .next()
                    .ok_or_else(|| err(line_num, "expected header 'p cnf <n_vars> <n_clauses>', but n_clauses is missing".into()))?
                    .parse()
                    .map_err(|e| err(line_num, format!("failed to parse n_clauses in header 'p cnf <n_vars> <n_clauses>': {e}")))?;
                sat = Some(Self::new(n_vars));
                continue;
            }
            let s = sat.as_mut().ok_or_else(|| {
                err(line_num, "clause data found before header 'p cnf <n_vars> <n_clauses>'".into())
            })?;
            // clause line: space-separated literals terminated by 0
            let literals: Vec<i64> = trimmed
                .split_whitespace()
                .map(|t| t.parse::<i64>())
                .collect::<Result<_, _>>()
                .map_err(|e| err(line_num, format!("failed to parse literal in clause: {e}")))?;
            let clause: Vec<i64> = literals.into_iter().take_while(|&v| v != 0).collect();
            if !clause.is_empty() {
                s.add_clause(clause);
            }
        }

        let s = sat.ok_or_else(|| {
            err(0, "file is empty or contains no header 'p cnf <n_vars> <n_clauses>'".into())
        })?;
        if s.n_clauses() != n_clauses {
            return Err(err(
                0,
                format!(
                    "clause count mismatch: header declares {} clauses, but {} were found",
                    n_clauses,
                    s.n_clauses()
                ),
            ));
        }
        Ok(s)
    }
}

impl ProblemTrait for Sat {
    type Solution = SatSolution;

    fn new_solution(&self, rng: &mut impl rand::Rng) -> SatSolution {
        let x: Vec<bool> = (0..self.n_vars).map(|_| rng.random_bool(0.5)).collect();
        let gain: Vec<i64> = (0..self.n_vars).map(|i| self.calc_gain(&x, i)).collect();
        let n_satisfied = self.calc_satisfied(&x);
        SatSolution {
            x,
            gain,
            n_satisfied,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 3-variable, 3-clause SAT instance: (x1 ∨ x2), (¬x1 ∨ x3), (¬x2 ∨ ¬x3)
    fn make_sat() -> Sat {
        let mut sat = Sat::new(3);
        sat.add_clause([1, 2]); // x1 ∨ x2
        sat.add_clause([-1, 3]); // ¬x1 ∨ x3
        sat.add_clause([-2, -3]); // ¬x2 ∨ ¬x3
        sat
    }

    #[test]
    fn test_calc_satisfied() {
        let sat = make_sat();
        // x = [true, false, true]: (T∨F)=T, (F∨T)=T, (T∨F)=T → all satisfied
        assert_eq!(sat.calc_satisfied(&[true, false, true]), 3);
        // x = [false, false, false]: (F∨F)=F, (T∨F)=T, (T∨T)=T → 2 satisfied
        assert_eq!(sat.calc_satisfied(&[false, false, false]), 2);
    }

    #[test]
    fn test_calc_gain_matches_delta() {
        let sat = make_sat();
        let x = vec![true, false, true];
        let n_sat = sat.calc_satisfied(&x);
        for i in 0..3 {
            let mut x2 = x.clone();
            x2[i] = !x2[i];
            let expected_delta = sat.calc_satisfied(&x2) as i64 - n_sat as i64;
            assert_eq!(sat.calc_gain(&x, i), expected_delta, "gain[{}] mismatch", i);
        }
    }

    #[test]
    fn test_n_clauses() {
        let sat = make_sat();
        assert_eq!(sat.n_clauses(), 3);
        assert_eq!(sat.n_vars(), 3);
    }
}
