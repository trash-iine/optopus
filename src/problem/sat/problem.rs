use crate::search_state::{Distance, ProblemTrait, Rankable};

fn insert_unique_sorted(v: &mut Vec<usize>, x: usize) {
    if let Err(pos) = v.binary_search(&x) {
        v.insert(pos, x);
    }
}

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
        crate::common::hamming_distance(&self.x, &other.x)
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
    /// Inverted index: `clauses_per_var[i]` lists every clause index that
    /// references variable `i` (0-indexed, sign-agnostic).
    ///
    /// # Note
    ///
    /// This field is maintained only by [`Sat::add_clause`].
    /// If `clauses_per_var[i]` contains `c`, then `clauses[c]` contains some
    /// literal `lit` with `lit.unsigned_abs() as usize - 1 == i`.
    ///
    /// Mutating `clauses` directly without updating this index breaks the
    /// invariant and triggers an `unreachable!()` in [`Sat::calc_gain`] /
    /// [`Sat::calc_gain_with_virtual_flip`].
    clauses_per_var: Vec<Vec<usize>>,
    /// For each variable `i`, the sorted, deduplicated set of other variables
    /// that share at least one clause with `i`. Used in flip-move incremental
    /// updates to avoid recomputing this set on every iteration.
    var_neighbors: Vec<Vec<usize>>,
}

impl Sat {
    pub fn new(n_vars: usize) -> Self {
        Self {
            n_vars,
            clauses: Vec::new(),
            clauses_per_var: vec![vec![]; n_vars],
            var_neighbors: vec![vec![]; n_vars],
        }
    }

    pub fn n_vars(&self) -> usize {
        self.n_vars
    }

    pub fn n_clauses(&self) -> usize {
        self.clauses.len()
    }

    /// Adds a clause. Literals are signed integers (1-indexed variables).
    ///
    /// # Panics
    ///
    /// Panics if a literal is `0` or its variable exceeds `n_vars`; such a
    /// literal would otherwise cause an out-of-bounds access during search.
    pub fn add_clause(&mut self, literals: impl IntoIterator<Item = i64>) {
        let clause_idx = self.clauses.len();
        let clause: Vec<i64> = literals.into_iter().collect();

        // Collect the in-clause variable indices once to set up both
        // `clauses_per_var` and the pairwise `var_neighbors` entries.
        let mut vars: Vec<usize> = clause
            .iter()
            .map(|&lit| {
                assert!(
                    lit != 0 && lit.unsigned_abs() as usize <= self.n_vars,
                    "literal {lit} is out of range for a problem with {} variables",
                    self.n_vars
                );
                lit.unsigned_abs() as usize - 1
            })
            .collect();
        vars.sort_unstable();
        vars.dedup();

        for &v in &vars {
            self.clauses_per_var[v].push(clause_idx);
        }
        for (a_idx, &a) in vars.iter().enumerate() {
            for &b in &vars[a_idx + 1..] {
                insert_unique_sorted(&mut self.var_neighbors[a], b);
                insert_unique_sorted(&mut self.var_neighbors[b], a);
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

    /// Returns the sorted, deduplicated set of variables that share at least
    /// one clause with variable `i` (0-indexed), excluding `i` itself.
    ///
    /// This is the precomputed set of variables whose gain may change when
    /// variable `i` is flipped, used by [`super::neighbor::SatFlipNeighbor`].
    pub fn var_neighbors(&self, i: usize) -> &[usize] {
        &self.var_neighbors[i]
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

            let i_lit = match clause
                .iter()
                .find(|&&lit| lit.unsigned_abs() as usize - 1 == i)
            {
                Some(&lit) => lit,
                None => unreachable!(
                    "Sat::clauses_per_var invariant broken: variable {i} listed \
                     as in clause {clause_idx} but no literal there references \
                     it. This is a library bug — please report."
                ),
            };
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

            let j_lit = match clause
                .iter()
                .find(|&&lit| lit.unsigned_abs() as usize - 1 == j)
            {
                Some(&lit) => lit,
                None => unreachable!(
                    "Sat::clauses_per_var invariant broken: variable {j} listed \
                     as in clause {clause_idx} but no literal there references \
                     it. This is a library bug — please report."
                ),
            };
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
    pub fn load_file(path: impl AsRef<std::path::Path>) -> Result<Self, crate::error::OptError> {
        use crate::common::InstanceLines;

        let mut lines = InstanceLines::open(path)?;

        let mut n_clauses = 0usize;
        let mut sat = None::<Self>;

        while let Some(line) = lines.next_line()? {
            let trimmed = line.trim();
            if trimmed.is_empty() || trimmed.starts_with('c') || trimmed.starts_with('%') {
                continue; // skip comments and SATLIB '%' terminator
            }
            if trimmed.starts_with('p') {
                // "p cnf <n_vars> <n_clauses>"
                let mut tokens = trimmed.split_whitespace();
                tokens.next(); // "p"
                tokens.next(); // "cnf"
                let n_vars: usize = lines
                    .parse_next(&mut tokens, "n_vars in header 'p cnf <n_vars> <n_clauses>'")?;
                n_clauses = lines.parse_next(
                    &mut tokens,
                    "n_clauses in header 'p cnf <n_vars> <n_clauses>'",
                )?;
                sat = Some(Self::new(n_vars));
                continue;
            }
            let s = sat.as_mut().ok_or_else(|| {
                lines.err("clause data found before header 'p cnf <n_vars> <n_clauses>'")
            })?;
            // clause line: space-separated literals terminated by 0
            let literals: Vec<i64> = trimmed
                .split_whitespace()
                .map(|t| t.parse::<i64>())
                .collect::<Result<_, _>>()
                .map_err(|e| lines.err(format!("failed to parse literal in clause: {e}")))?;
            let clause: Vec<i64> = literals.into_iter().take_while(|&v| v != 0).collect();
            if let Some(&lit) = clause
                .iter()
                .find(|&&lit| lit.unsigned_abs() as usize > s.n_vars)
            {
                return Err(lines.err(format!(
                    "literal {lit} exceeds n_vars = {} declared in the header",
                    s.n_vars
                )));
            }
            if !clause.is_empty() {
                s.add_clause(clause);
            }
        }

        let s = sat.ok_or_else(|| {
            lines.err_at(
                0,
                "file is empty or contains no header 'p cnf <n_vars> <n_clauses>'",
            )
        })?;
        if s.n_clauses() != n_clauses {
            return Err(lines.err_at(
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

impl crate::trait_defs::BinaryProblem for Sat {
    type Flip = super::SatFlipNeighbor;

    fn variable_indices(&self) -> impl Iterator<Item = usize> + '_ {
        0..self.n_vars
    }

    fn variable(sol: &SatSolution, i: usize) -> bool {
        sol.x[i]
    }

    fn flip_move(sol: &SatSolution, i: usize) -> Self::Flip {
        super::SatFlipNeighbor {
            i,
            gain: sol.gain[i],
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

    #[test]
    #[should_panic(expected = "out of range")]
    fn test_add_clause_rejects_out_of_range_literal() {
        let mut sat = Sat::new(3);
        sat.add_clause([1, 4]);
    }

    #[test]
    #[should_panic(expected = "out of range")]
    fn test_add_clause_rejects_zero_literal() {
        let mut sat = Sat::new(3);
        sat.add_clause([1, 0]);
    }

    #[test]
    fn test_load_file_rejects_literal_exceeding_n_vars() {
        use std::io::Write;
        let mut path = std::env::temp_dir();
        path.push(format!(
            "optopus_sat_oob_{}_{}.cnf",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        {
            let mut f = std::fs::File::create(&path).unwrap();
            writeln!(f, "p cnf 3 2").unwrap();
            writeln!(f, "1 -2 0").unwrap();
            writeln!(f, "2 -4 0").unwrap(); // variable 4 > n_vars = 3
        }
        let result = Sat::load_file(&path);
        let _ = std::fs::remove_file(&path);
        let err = result.expect_err("literal exceeding n_vars must be rejected");
        assert!(err.to_string().contains("exceeds n_vars"), "{err}");
    }

    #[test]
    fn test_var_neighbors_construction() {
        // (x1 ∨ x2): pairs (0,1)
        // (¬x1 ∨ x3): pairs (0,2)
        // (¬x2 ∨ ¬x3): pairs (1,2)
        let sat = make_sat();
        assert_eq!(sat.var_neighbors(0), &[1, 2]);
        assert_eq!(sat.var_neighbors(1), &[0, 2]);
        assert_eq!(sat.var_neighbors(2), &[0, 1]);

        // Variables appearing together in multiple clauses are not duplicated.
        let mut sat2 = Sat::new(3);
        sat2.add_clause([1, 2]);
        sat2.add_clause([1, 2, 3]);
        sat2.add_clause([-1, -2]);
        assert_eq!(sat2.var_neighbors(0), &[1, 2]);
        assert_eq!(sat2.var_neighbors(1), &[0, 2]);
        assert_eq!(sat2.var_neighbors(2), &[0, 1]);

        // A variable not co-occurring with anything has an empty neighbor list.
        let mut sat3 = Sat::new(3);
        sat3.add_clause([1]);
        sat3.add_clause([-2, 3]);
        assert_eq!(sat3.var_neighbors(0), &[] as &[usize]);
        assert_eq!(sat3.var_neighbors(1), &[2]);
        assert_eq!(sat3.var_neighbors(2), &[1]);
    }
}
