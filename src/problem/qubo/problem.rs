use std::collections::HashSet;
use std::fs::File;
use std::io::{BufRead, BufReader};

use crate::search_state::{Distance, ProblemTrait, Rankable};

/// Integer coefficient type used in the Q matrix.
pub type Coefficient = i32;

/// Shared zero constant returned by `Index` for non-existent entries.
static ZERO_COEFF: Coefficient = 0;

/// A solution to the QUBO problem.
///
/// # Core fields
///
/// - [`x`](Self::x) — variable assignment (`x[i] = true` means variable `i` is set to 1)
/// - [`gain`](Self::gain) — incremental energy change for each variable flip (`gain[i] < 0` means flipping `i` improves the objective)
/// - [`objective`](Self::objective) — current energy value (minimized)
///
/// These three fields are all you need to inspect results and build custom logic.
///
/// # Advanced: negative-gain index
///
/// An optional index tracks which variables currently have negative gain (i.e. improving
/// moves). Call [`enable_negative_gain_index`](Self::enable_negative_gain_index) to activate it.
/// Standard heuristics ([`LocalSearch`](crate::heuristic::LocalSearch),
/// [`TabuSearch`](crate::heuristic::TabuSearch),
/// [`SimulatedAnnealing`](crate::heuristic::SimulatedAnnealing), etc.)
/// do **not** require this index — it is a performance optimization for problem-specific
/// algorithms.
///
/// # Examples
///
/// ```
/// use optopus::prelude::*;
///
/// let qubo = Qubo::from_entries([(0, 1, 1), (1, 2, 2), (0, 2, 3)]);
/// let mut state = SearchState::new(&qubo);
/// LocalSearch::<QuboFlipNeighbor>::new(StopCondition::iterations(1000))
///     .run(&mut state).unwrap();
///
/// let sol = &state.best_solution;
/// println!("objective = {}", sol.objective);
/// println!("x = {:?}", sol.x);
/// ```
#[derive(Debug, Clone)]
pub struct QuboSolution {
    pub x: Vec<bool>,
    pub gain: Vec<Coefficient>,
    pub objective: Coefficient,
    /// Advanced: whether the `negative_gain` index is enabled.
    /// When `false`, `update_negative_gain_membership` is a no-op.
    /// See [`enable_negative_gain_index`](Self::enable_negative_gain_index).
    pub(crate) negative_gain_enabled: bool,
    /// Advanced: unordered list of variables `v` with `gain[v] < 0`.
    /// Only maintained when `negative_gain_enabled` is `true`.
    /// Not needed for standard heuristic use.
    pub(crate) negative_gain: Vec<usize>,
    /// Advanced: inverse index for O(1) membership updates.
    /// `negative_gain_pos[v]` = position of `v` in `negative_gain`, or `-1` if absent.
    pub(crate) negative_gain_pos: Vec<i32>,
}

impl Rankable for QuboSolution {
    fn is_better_than(&self, other: &Self) -> bool {
        self.objective < other.objective
    }
}

impl Distance for QuboSolution {
    fn distance(&self, other: &Self) -> usize {
        calc_xor_of_solutions(self, other)
    }
}

impl QuboSolution {
    /// Iterates over all variable indices in the solution.
    pub fn iter_on_variables(&self) -> impl Iterator<Item = usize> + '_ {
        0..self.x.len()
    }

    /// Builds a [`QuboSolution`] from pre-computed components.
    ///
    /// The resulting solution is fully functional for all standard heuristics.
    /// The advanced `negative_gain` index is not initialised; see
    /// [`enable_negative_gain_index`](Self::enable_negative_gain_index) if you need it.
    ///
    /// Prefer [`new_from_assignment`](Self::new_from_assignment) for constructing solutions
    /// from a variable assignment — it computes `gain` and `objective` automatically.
    pub(crate) fn new_from_parts(x: Vec<bool>, gain: Vec<Coefficient>, objective: Coefficient) -> Self {
        Self {
            x,
            gain,
            objective,
            negative_gain_enabled: false,
            negative_gain: Vec::new(),
            negative_gain_pos: Vec::new(),
        }
    }

    /// Creates a [`QuboSolution`] from a variable assignment, computing gain and objective automatically.
    ///
    /// # Examples
    ///
    /// ```
    /// use optopus::prelude::*;
    ///
    /// let qubo = Qubo::from_entries([(0, 1, 1), (0, 2, 2), (1, 2, 3)]);
    /// let sol = QuboSolution::new_from_assignment(&qubo, vec![true, false, true]);
    /// assert_eq!(sol.objective, 2);  // Q[0][2] = 2
    /// ```
    pub fn new_from_assignment(qubo: &Qubo, x: Vec<bool>) -> Self {
        let n = qubo.len();
        let mut gain = vec![0; n];
        for &i in qubo.iter_on_variables() {
            gain[i] = qubo.calculate_gain(&x, i);
        }
        let objective = qubo.calculate_energy(&x);
        Self::new_from_parts(x, gain, objective)
    }

    /// **Advanced.** Enables the `negative_gain` index, building it from the current
    /// `gain` vector.
    ///
    /// Most users do **not** need to call this method. Standard heuristics
    /// ([`LocalSearch`](crate::heuristic::LocalSearch),
    /// [`TabuSearch`](crate::heuristic::TabuSearch),
    /// [`SimulatedAnnealing`](crate::heuristic::SimulatedAnnealing), etc.)
    /// work correctly without it.
    ///
    /// This index is useful for problem-specific algorithms that need to iterate
    /// only over variables with negative gain, reducing the inner-loop cost from
    /// O(n) to O(|improving moves|).
    ///
    /// Once enabled, the index is maintained incrementally by
    /// [`QuboFlipNeighbor::apply_to_solution`](super::QuboFlipNeighbor).
    ///
    /// If already enabled, this is a no-op. O(n).
    ///
    /// # Examples
    ///
    /// ```
    /// use optopus::prelude::*;
    ///
    /// let qubo = Qubo::from_entries([(0, 1, 1), (1, 2, 1), (0, 2, 1)]);
    /// let mut state = SearchState::new(&qubo);
    /// state.solution.enable_negative_gain_index();
    /// ```
    pub fn enable_negative_gain_index(&mut self) {
        if self.negative_gain_enabled {
            return;
        }
        self.negative_gain_enabled = true;
        let n = self.gain.len();
        self.negative_gain.clear();
        self.negative_gain_pos = vec![-1i32; n];
        for (v, &g) in self.gain.iter().enumerate() {
            if g < 0 {
                self.negative_gain_pos[v] = self.negative_gain.len() as i32;
                self.negative_gain.push(v);
            }
        }
    }

    /// Records that variable `v`'s gain is changing from `self.gain[v]` to `new_gain`.
    /// Updates membership of `v` in the `negative_gain` index (does **not** write
    /// `self.gain[v]` — the caller is expected to do that).
    ///
    /// No-op when the index is not enabled.
    #[inline]
    pub(crate) fn update_negative_gain_membership(&mut self, v: usize, new_gain: Coefficient) {
        if !self.negative_gain_enabled {
            return;
        }
        let was_negative = self.gain[v] < 0;
        let is_negative = new_gain < 0;
        if was_negative == is_negative {
            return;
        }
        if is_negative {
            self.negative_gain_pos[v] = self.negative_gain.len() as i32;
            self.negative_gain.push(v);
        } else {
            let pos = self.negative_gain_pos[v] as usize;
            let last = *self.negative_gain.last().expect("negative_gain non-empty");
            self.negative_gain.swap_remove(pos);
            if last != v {
                self.negative_gain_pos[last] = pos as i32;
            }
            self.negative_gain_pos[v] = -1;
        }
    }
}

/// QUBO problem instance.
///
/// Stores the Q matrix as a symmetric sparse adjacency list.
/// Diagonal entries `Q[i][i]` represent linear coefficients and are stored
/// in `adj[i]` as `(i, Q[i][i])`.
///
/// # Construction
///
/// ```
/// use optopus::problem::Qubo;
///
/// // From entry list
/// let qubo = Qubo::from_entries([(0, 1, 1), (1, 2, 2)]);
///
/// // Incremental
/// let mut qubo = Qubo::new();
/// qubo.set_q(0, 1, 1);
///
/// // Read coefficient via Index
/// assert_eq!(qubo[(0, 1)], 1);
/// assert_eq!(qubo[(8, 9)], 0);  // non-existent → 0
/// ```
///
/// # Optimization direction
///
/// Minimization: A solution with a lower `objective` is better.
#[derive(Debug, Clone)]
pub struct Qubo {
    /// adj[i] = list of (j, Q[i][j]) for all j in Q[i], including diagonal (i, Q[i][i]).
    adj: Vec<Vec<(usize, Coefficient)>>,
    /// Sorted list of variable IDs that appear in the problem.
    pub(super) variables: Vec<usize>,
}

impl Qubo {
    /// Creates a new empty [`Qubo`].
    ///
    /// # Examples
    ///
    /// ```
    /// let qubo = optopus::problem::Qubo::new();
    /// ```
    #[allow(clippy::new_without_default)]
    pub fn new() -> Qubo {
        Qubo {
            adj: vec![],
            variables: vec![],
        }
    }

    /// Creates a [`Qubo`] from an iterator of `(i, j, coefficient)` entries.
    ///
    /// Duplicate entries are overwritten (last occurrence wins).
    ///
    /// # Examples
    ///
    /// ```
    /// let qubo = optopus::problem::Qubo::from_entries([
    ///     (0, 1, 1),
    ///     (0, 2, 2),
    ///     (1, 2, 3),
    /// ]);
    /// assert_eq!(qubo[(0, 1)], 1);
    /// assert_eq!(qubo.num_entries(), 3);
    /// ```
    pub fn from_entries(
        entries: impl IntoIterator<Item = (usize, usize, Coefficient)>,
    ) -> Self {
        let mut qubo = Self::new();
        for (i, j, v) in entries {
            qubo.set_q(i, j, v);
        }
        qubo
    }

    /// Returns the internal dimension (`max_variable_id + 1`).
    ///
    /// This may be larger than the number of variables with entries.
    /// For the count of variables that participate in entries, use
    /// [`num_of_variables`](Self::num_of_variables).
    ///
    /// # Examples
    ///
    /// ```
    /// let mut qubo = optopus::problem::Qubo::new();
    /// assert_eq!(qubo.len(), 0);
    ///
    /// qubo.set_q(0, 1, 1);
    /// assert_eq!(qubo.len(), 2);
    /// ```
    pub fn len(&self) -> usize {
        self.adj.len()
    }

    /// Returns the number of entries in the upper triangle of the Q matrix
    /// (including diagonal).
    ///
    /// Each symmetric pair `(i, j)` with `i < j` is counted once.
    /// Diagonal entries `(i, i)` are counted once.
    ///
    /// # Examples
    ///
    /// ```
    /// let qubo = optopus::problem::Qubo::from_entries([
    ///     (0, 1, 1), (0, 2, 2), (1, 2, 3), (0, 0, 5),
    /// ]);
    /// assert_eq!(qubo.num_entries(), 4);  // 3 off-diagonal + 1 diagonal
    /// ```
    pub fn num_entries(&self) -> usize {
        self.adj
            .iter()
            .enumerate()
            .map(|(i, neighbors)| neighbors.iter().filter(|&&(j, _)| j >= i).count())
            .sum()
    }

    /// Returns `true` if the QUBO has no variables.
    ///
    /// # Examples
    ///
    /// ```
    /// let qubo = optopus::problem::Qubo::new();
    /// assert!(qubo.is_empty());
    /// ```
    pub fn is_empty(&self) -> bool {
        self.variables.is_empty()
    }

    /// Sets (or overwrites) `Q[i][j] = Q[j][i] = v`.
    /// For diagonal entries (`i == j`), stores only once.
    ///
    /// # Examples
    ///
    /// ```
    /// let mut qubo = optopus::problem::Qubo::new();
    /// qubo.set_q(0, 1, 5);
    /// qubo.set_q(0, 1, 3);
    /// assert_eq!(qubo[(0, 1)], 3);  // overwritten, not 8
    /// ```
    pub fn set_q(&mut self, i: usize, j: usize, v: Coefficient) {
        self.ensure_capacity(i.max(j) + 1);
        self.set_directed(i, j, v);
        if i != j {
            self.set_directed(j, i, v);
        }
        self.ensure_variable(i);
        self.ensure_variable(j);
    }

    /// Adds (or accumulates) the coefficient `v` on entry `(i, j)`.
    ///
    /// If the entry already exists, `v` is **added** to the current value.
    /// To overwrite instead, use [`set_q`](Self::set_q).
    ///
    /// # Examples
    ///
    /// ```
    /// let mut qubo = optopus::problem::Qubo::new();
    /// qubo.add_q(0, 1, 1);
    /// qubo.add_q(0, 1, 2);
    /// assert_eq!(qubo[(0, 1)], 3);  // 1 + 2
    /// ```
    pub fn add_q(&mut self, i: usize, j: usize, v: Coefficient) {
        self.ensure_capacity(i.max(j) + 1);
        self.add_directed(i, j, v);
        if i != j {
            self.add_directed(j, i, v);
        }
        self.ensure_variable(i);
        self.ensure_variable(j);
    }

    fn ensure_capacity(&mut self, n: usize) {
        if self.adj.len() < n {
            self.adj.resize_with(n, Vec::new);
        }
    }

    fn ensure_variable(&mut self, v: usize) {
        if let Err(pos) = self.variables.binary_search(&v) {
            self.variables.insert(pos, v);
        }
    }

    fn set_directed(&mut self, from: usize, to: usize, v: Coefficient) {
        if let Some(entry) = self.adj[from].iter_mut().find(|(k, _)| *k == to) {
            entry.1 = v;
        } else {
            self.adj[from].push((to, v));
        }
    }

    fn add_directed(&mut self, from: usize, to: usize, v: Coefficient) {
        if let Some(entry) = self.adj[from].iter_mut().find(|(k, _)| *k == to) {
            entry.1 += v;
        } else {
            self.adj[from].push((to, v));
        }
    }

    /// Gets the coefficient `Q[i][j]`, returning `0` if no such entry exists.
    ///
    /// You can also use the [`Index`](std::ops::Index) syntax: `qubo[(i, j)]`.
    ///
    /// # Examples
    ///
    /// ```
    /// let mut qubo = optopus::problem::Qubo::new();
    /// qubo.set_q(0, 1, 1);
    ///
    /// assert_eq!(qubo.get_q(0, 1), 1);
    /// assert_eq!(qubo[(0, 1)], 1);   // equivalent
    /// assert_eq!(qubo.get_q(5, 6), 0);  // non-existent
    /// ```
    pub fn get_q(&self, i: usize, j: usize) -> Coefficient {
        if i < self.adj.len() {
            self.adj[i]
                .iter()
                .find(|(k, _)| *k == j)
                .map(|(_, v)| *v)
                .unwrap_or(0)
        } else {
            0
        }
    }

    /// Returns `true` if an entry for `(i, j)` exists in the Q matrix.
    ///
    /// # Examples
    ///
    /// ```
    /// let qubo = optopus::problem::Qubo::from_entries([(0, 1, 1)]);
    /// assert!(qubo.has_entry(0, 1));
    /// assert!(qubo.has_entry(1, 0));  // symmetric
    /// assert!(!qubo.has_entry(0, 2));
    /// ```
    pub fn has_entry(&self, i: usize, j: usize) -> bool {
        i < self.adj.len() && self.adj[i].iter().any(|(k, _)| *k == j)
    }

    /// Returns an iterator visiting all variables that have at least one entry.
    pub fn iter_on_variables(&self) -> impl Iterator<Item = &usize> {
        self.variables.iter()
    }

    /// Returns an iterator over `(variable_id, coefficient)` pairs for variable `i`.
    pub fn iter_on_adjacency(&self, i: usize) -> std::slice::Iter<'_, (usize, Coefficient)> {
        if i < self.adj.len() {
            self.adj[i].iter()
        } else {
            [].iter()
        }
    }

    /// Returns an iterator over `(variable_id, coefficient)` pairs for variable `i`.
    ///
    /// This is an alias for [`iter_on_adjacency`](Self::iter_on_adjacency).
    ///
    /// # Examples
    ///
    /// ```
    /// let qubo = optopus::problem::Qubo::from_entries([(0, 1, 1), (0, 2, 2)]);
    /// let coeffs: Vec<i32> = qubo.neighbors(0).map(|&(_, v)| v).collect();
    /// assert_eq!(coeffs, vec![1, 2]);
    /// ```
    pub fn neighbors(&self, i: usize) -> std::slice::Iter<'_, (usize, Coefficient)> {
        self.iter_on_adjacency(i)
    }

    /// Returns the degree (number of adjacent entries) of variable `i`.
    ///
    /// Includes the diagonal entry `Q[i][i]` if present.
    ///
    /// # Examples
    ///
    /// ```
    /// let qubo = optopus::problem::Qubo::from_entries([(0, 1, 1), (0, 2, 1), (1, 2, 1)]);
    /// assert_eq!(qubo.degree(0), 2);
    /// assert_eq!(qubo.degree(99), 0);  // out of bounds
    /// ```
    pub fn degree(&self, i: usize) -> usize {
        if i < self.adj.len() {
            self.adj[i].len()
        } else {
            0
        }
    }

    /// Returns an iterator over all entries as `(i, j, coefficient)` with `i <= j`.
    ///
    /// Each symmetric pair appears exactly once. Diagonal entries appear as `(i, i, v)`.
    ///
    /// # Examples
    ///
    /// ```
    /// let qubo = optopus::problem::Qubo::from_entries([(0, 1, 1), (0, 2, 2)]);
    /// let entries: Vec<_> = qubo.entries().collect();
    /// assert_eq!(entries.len(), 2);
    /// ```
    pub fn entries(&self) -> impl Iterator<Item = (usize, usize, Coefficient)> + '_ {
        self.adj.iter().enumerate().flat_map(|(i, neighbors)| {
            neighbors
                .iter()
                .filter(move |&&(j, _)| j >= i)
                .map(move |&(j, v)| (i, j, v))
        })
    }

    /// Returns the number of variables that have at least one entry.
    pub fn num_of_variables(&self) -> usize {
        self.variables.len()
    }

    /// Loads a QUBO problem from a MaxCut-format file, applying the MaxCut→QUBO
    /// transformation.
    ///
    /// # File format
    ///
    /// The expected format is one header line followed by edge lines (**1-indexed** vertices):
    ///
    /// ```text
    /// N M
    /// i j w
    /// i j w
    /// ...
    /// ```
    ///
    /// Each edge `(i, j, w)` contributes: `Q[i][j] += 2w`, `Q[i][i] -= w`, `Q[j][j] -= w`.
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
        for result in line_iter {
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
            qubo.add_directed(i, j, 2 * v);
            qubo.add_directed(j, i, 2 * v);
            qubo.add_directed(i, i, -v);
            qubo.add_directed(j, j, -v);
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
            if i == j || x[j] {
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
                if i == j || x[j] {
                    energy += q;
                }
            }
        }
        energy
    }
}

impl std::ops::Index<(usize, usize)> for Qubo {
    type Output = Coefficient;

    /// Returns the coefficient `Q[i][j]`, or `&0` if no such entry exists.
    ///
    /// # Examples
    ///
    /// ```
    /// let qubo = optopus::problem::Qubo::from_entries([(0, 1, 3)]);
    /// assert_eq!(qubo[(0, 1)], 3);
    /// assert_eq!(qubo[(1, 0)], 3);  // symmetric
    /// assert_eq!(qubo[(0, 2)], 0);  // non-existent entry
    /// ```
    fn index(&self, (i, j): (usize, usize)) -> &Coefficient {
        if i < self.adj.len()
            && let Some((_, v)) = self.adj[i].iter().find(|(k, _)| *k == j)
        {
            return v;
        }
        &ZERO_COEFF
    }
}

/// Displays a summary: `Qubo(variables: N, entries: M)` or `Qubo(empty)`.
///
/// # Examples
///
/// ```
/// let qubo = optopus::problem::Qubo::from_entries([(0, 1, 1), (1, 2, 2)]);
/// assert_eq!(format!("{qubo}"), "Qubo(variables: 3, entries: 2)");
///
/// let empty = optopus::problem::Qubo::new();
/// assert_eq!(format!("{empty}"), "Qubo(empty)");
/// ```
impl std::fmt::Display for Qubo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.variables.is_empty() {
            write!(f, "Qubo(empty)")
        } else {
            write!(
                f,
                "Qubo(variables: {}, entries: {})",
                self.num_of_variables(),
                self.num_entries(),
            )
        }
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
        QuboSolution::new_from_assignment(self, x)
    }
}

#[allow(dead_code)]
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
                    sub_qubo.add_q(ind, ind, v);
                }
            } else if j < parents[0].x.len() && parents[0].x[j] {
                sub_qubo.add_q(ind, ind, v);
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
        assert_eq!(qubo.get_q(0, 1), 1);
        assert_eq!(qubo.get_q(1, 2), 2);
        assert_eq!(qubo.get_q(0, 2), 3);
        assert_eq!(qubo.num_of_variables(), 3);
    }

    #[test]
    fn test_load_file_as_max_cut() {
        let qubo_result = Qubo::load_file_as_max_cut("data/max_cut/test_data.txt");
        assert!(qubo_result.is_ok());
        let qubo = qubo_result.unwrap();
        assert_eq!(qubo.get_q(0, 1), 2);
        assert_eq!(qubo.get_q(1, 2), -6);
        assert_eq!(qubo.get_q(0, 0), -1);
        assert_eq!(qubo.get_q(1, 1), 2);
        assert_eq!(qubo.get_q(2, 2), 3);
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

    #[test]
    fn test_from_entries() {
        let qubo = Qubo::from_entries([(0, 1, 1), (0, 2, 2), (1, 2, 3)]);
        assert_eq!(qubo[(0, 1)], 1);
        assert_eq!(qubo[(0, 2)], 2);
        assert_eq!(qubo[(1, 2)], 3);
        assert_eq!(qubo.num_of_variables(), 3);
    }

    #[test]
    fn test_from_entries_duplicate_last_wins() {
        let qubo = Qubo::from_entries([(0, 1, 1), (0, 1, 5)]);
        assert_eq!(qubo[(0, 1)], 5);
    }

    #[test]
    fn test_add_q_accumulation() {
        let mut qubo = Qubo::new();
        qubo.add_q(0, 1, 1);
        qubo.add_q(0, 1, 2);
        assert_eq!(qubo[(0, 1)], 3); // 1 + 2
        assert_eq!(qubo[(1, 0)], 3); // symmetric
    }

    #[test]
    fn test_add_q_and_set_q_interaction() {
        let mut qubo = Qubo::new();
        qubo.set_q(0, 1, 5);
        qubo.add_q(0, 1, 2);
        assert_eq!(qubo[(0, 1)], 7); // 5 + 2

        qubo.set_q(0, 1, 1); // overwrite back
        assert_eq!(qubo[(0, 1)], 1);
    }

    #[test]
    fn test_set_q_overwrites() {
        let mut qubo = Qubo::new();
        qubo.set_q(0, 1, 5);
        assert_eq!(qubo[(0, 1)], 5);

        qubo.set_q(0, 1, 3);
        assert_eq!(qubo[(0, 1)], 3); // overwritten, not 8
        assert_eq!(qubo[(1, 0)], 3); // symmetric
    }

    #[test]
    fn test_len() {
        let mut qubo = Qubo::new();
        assert_eq!(qubo.len(), 0);

        qubo.set_q(0, 1, 1);
        assert_eq!(qubo.len(), 2);

        qubo.set_q(0, 5, 1);
        assert_eq!(qubo.len(), 6);
    }

    #[test]
    fn test_num_entries() {
        let qubo = Qubo::from_entries([(0, 1, 1), (0, 2, 2), (1, 2, 3)]);
        assert_eq!(qubo.num_entries(), 3); // 3 off-diagonal pairs

        let qubo = Qubo::from_entries([(0, 1, 1), (0, 0, 5)]);
        assert_eq!(qubo.num_entries(), 2); // 1 off-diagonal + 1 diagonal
    }

    #[test]
    fn test_is_empty() {
        let qubo = Qubo::new();
        assert!(qubo.is_empty());

        let qubo = Qubo::from_entries([(0, 1, 1)]);
        assert!(!qubo.is_empty());
    }

    #[test]
    fn test_neighbors() {
        let qubo = Qubo::from_entries([(0, 1, 1), (0, 2, 2)]);
        let coeffs: Vec<Coefficient> = qubo.neighbors(0).map(|&(_, v)| v).collect();
        assert_eq!(coeffs, vec![1, 2]);
    }

    #[test]
    fn test_degree() {
        let qubo = Qubo::from_entries([(0, 1, 1), (0, 2, 1), (1, 2, 1)]);
        assert_eq!(qubo.degree(0), 2);
        assert_eq!(qubo.degree(1), 2);
        assert_eq!(qubo.degree(2), 2);
        assert_eq!(qubo.degree(99), 0); // out of bounds
    }

    #[test]
    fn test_entries_iterator() {
        let qubo = Qubo::from_entries([(0, 1, 1), (0, 2, 2), (1, 2, 3)]);
        let mut entries: Vec<_> = qubo.entries().collect();
        entries.sort_by_key(|&(i, j, _)| (i, j));
        assert_eq!(entries, vec![(0, 1, 1), (0, 2, 2), (1, 2, 3)]);
    }

    #[test]
    fn test_has_entry() {
        let qubo = Qubo::from_entries([(0, 1, 1)]);
        assert!(qubo.has_entry(0, 1));
        assert!(qubo.has_entry(1, 0)); // symmetric
        assert!(!qubo.has_entry(0, 2));
        assert!(!qubo.has_entry(5, 6)); // out of bounds
    }

    #[test]
    fn test_index_existing() {
        let qubo = Qubo::from_entries([(0, 1, 3), (1, 2, 7)]);
        assert_eq!(qubo[(0, 1)], 3);
        assert_eq!(qubo[(1, 0)], 3);
        assert_eq!(qubo[(1, 2)], 7);
    }

    #[test]
    fn test_index_missing() {
        let qubo = Qubo::from_entries([(0, 1, 1)]);
        assert_eq!(qubo[(0, 2)], 0);
        assert_eq!(qubo[(5, 6)], 0); // out of bounds
    }

    #[test]
    fn test_get_q_default_zero() {
        let qubo = Qubo::from_entries([(0, 1, 1)]);
        assert_eq!(qubo.get_q(0, 1), 1);
        assert_eq!(qubo.get_q(0, 2), 0);
        assert_eq!(qubo.get_q(99, 100), 0);
    }

    #[test]
    fn test_display_empty() {
        let qubo = Qubo::new();
        assert_eq!(format!("{qubo}"), "Qubo(empty)");
    }

    #[test]
    fn test_display_nonempty() {
        let qubo = Qubo::from_entries([(0, 1, 1), (0, 2, 1), (1, 2, 1)]);
        assert_eq!(format!("{qubo}"), "Qubo(variables: 3, entries: 3)");
    }

    #[test]
    fn test_new_from_assignment() {
        let qubo = Qubo::from_entries([(0, 1, 1), (1, 2, 2), (0, 2, 3)]);
        let sol = QuboSolution::new_from_assignment(&qubo, vec![true, false, true]);
        assert_eq!(sol.objective, qubo.calculate_energy(&sol.x));
        for &i in qubo.iter_on_variables() {
            assert_eq!(sol.gain[i], qubo.calculate_gain(&sol.x, i));
        }
    }

    #[test]
    fn test_new_from_parts() {
        let qubo = Qubo::from_entries([(0, 1, 1), (1, 2, 2), (0, 2, 3)]);
        let x = vec![true, false, true];
        let gain: Vec<Coefficient> = (0..3).map(|i| qubo.calculate_gain(&x, i)).collect();
        let objective = qubo.calculate_energy(&x);
        let sol = QuboSolution::new_from_parts(x.clone(), gain.clone(), objective);
        assert_eq!(sol.x, x);
        assert_eq!(sol.gain, gain);
        assert_eq!(sol.objective, objective);
    }

    fn make_qubo_and_solution() -> (Qubo, QuboSolution) {
        let qubo = Qubo::from_entries([(0, 1, 1), (1, 2, 1), (0, 2, 1)]);
        let n = qubo.len();
        let x = vec![false; n];
        let sol = QuboSolution::new_from_assignment(&qubo, x);
        (qubo, sol)
    }

    #[test]
    fn test_from_parts_index_disabled_by_default() {
        let (_qubo, sol) = make_qubo_and_solution();
        assert!(
            !sol.negative_gain_enabled,
            "index should be disabled after construction"
        );
        assert!(sol.negative_gain.is_empty());
        assert!(sol.negative_gain_pos.is_empty());
    }

    #[test]
    fn test_enable_negative_gain_index_builds_correctly() {
        // All variables false → gain[i] = sum of Q[i][j] for all j with x[j]=true = 0
        // Since QUBO gain for all-false is all positive (flipping adds energy),
        // negative_gain should be empty.
        let (_qubo, mut sol) = make_qubo_and_solution();
        sol.enable_negative_gain_index();

        assert!(sol.negative_gain_enabled);
        // All-false assignment on all-positive Q: gains are non-negative → no negative gains
        for &v in &sol.negative_gain {
            assert!(sol.gain[v] < 0, "all entries in negative_gain must have gain < 0");
        }
    }

    #[test]
    fn test_enable_negative_gain_index_with_negative_gains() {
        // Use a QUBO where some gains are negative:
        // All-true assignment with Q[i][j] all positive → flipping any variable
        // removes energy, so gain is negative.
        let qubo = Qubo::from_entries([(0, 1, 1), (1, 2, 1), (0, 2, 1)]);
        let x = vec![true, true, true];
        let mut sol = QuboSolution::new_from_assignment(&qubo, x);

        // gain[i] for all-true: flipping removes all edges involving i
        // gain[0] = -(Q[0][1] + Q[0][2]) = -2
        // gain[1] = -(Q[0][1] + Q[1][2]) = -2
        // gain[2] = -(Q[0][2] + Q[1][2]) = -2
        sol.enable_negative_gain_index();

        assert!(sol.negative_gain_enabled);
        let mut listed = sol.negative_gain.clone();
        listed.sort();
        assert_eq!(listed, vec![0, 1, 2], "all variables should have gain < 0");

        // Verify inverse index consistency
        for &v in &sol.negative_gain {
            let pos = sol.negative_gain_pos[v] as usize;
            assert_eq!(sol.negative_gain[pos], v);
        }
    }

    #[test]
    fn test_enable_negative_gain_index_excludes_non_negative() {
        // All-false on positive Q: all gains >= 0, no negative gains
        let qubo = Qubo::from_entries([(0, 1, 1), (1, 2, 1), (0, 2, 1)]);
        let x = vec![false, false, false];
        let mut sol = QuboSolution::new_from_assignment(&qubo, x);
        sol.enable_negative_gain_index();

        assert!(
            sol.negative_gain.is_empty(),
            "no variable has gain < 0; negative_gain must be empty"
        );
        let n = sol.gain.len();
        assert_eq!(sol.negative_gain_pos.len(), n);
        for v in 0..n {
            assert_eq!(sol.negative_gain_pos[v], -1);
        }
    }

    #[test]
    fn test_enable_negative_gain_index_idempotent() {
        let qubo = Qubo::from_entries([(0, 1, 1), (1, 2, 1), (0, 2, 1)]);
        let mut sol = QuboSolution::new_from_assignment(&qubo, vec![true, true, true]);
        sol.enable_negative_gain_index();
        let ng_after_first = sol.negative_gain.clone();
        let ng_pos_after_first = sol.negative_gain_pos.clone();

        sol.enable_negative_gain_index();

        assert_eq!(sol.negative_gain, ng_after_first);
        assert_eq!(sol.negative_gain_pos, ng_pos_after_first);
    }

    #[test]
    fn test_update_negative_gain_membership_noop_when_disabled() {
        let (_qubo, mut sol) = make_qubo_and_solution();
        sol.update_negative_gain_membership(0, -5);
        assert!(sol.negative_gain.is_empty());
        assert!(sol.negative_gain_pos.is_empty());
    }

    #[test]
    fn test_update_negative_gain_membership_maintains_index() {
        let qubo = Qubo::from_entries([(0, 1, 1), (1, 2, 1), (0, 2, 1)]);
        let mut sol = QuboSolution::new_from_assignment(&qubo, vec![true, true, true]);
        // All gains are -2 (negative)
        sol.enable_negative_gain_index();
        assert_eq!(sol.negative_gain.len(), 3);

        // Simulate: variable 0's gain changes to +1 (non-negative → should leave the index)
        sol.update_negative_gain_membership(0, 1);
        sol.gain[0] = 1;
        assert!(
            !sol.negative_gain.contains(&0),
            "variable 0 must leave negative_gain when gain becomes non-negative"
        );
        assert_eq!(sol.negative_gain_pos[0], -1);
        assert_eq!(sol.negative_gain.len(), 2);

        // Verify inverse consistency for remaining variables
        for &v in &sol.negative_gain {
            let pos = sol.negative_gain_pos[v] as usize;
            assert_eq!(sol.negative_gain[pos], v);
        }

        // Simulate: variable 0's gain changes back to -3 (should re-enter the index)
        sol.update_negative_gain_membership(0, -3);
        sol.gain[0] = -3;
        assert!(sol.negative_gain.contains(&0));
        assert_eq!(sol.negative_gain.len(), 3);
        for &v in &sol.negative_gain {
            let pos = sol.negative_gain_pos[v] as usize;
            assert_eq!(sol.negative_gain[pos], v);
        }
    }

    #[test]
    fn test_negative_gain_index_consistent_after_flip() {
        use crate::problem::qubo::QuboFlipNeighbor;
        use crate::search_state::MoveToNeighbor;

        let qubo = Qubo::from_entries([(0, 1, 1), (1, 2, 1), (0, 2, 1)]);
        let mut sol = QuboSolution::new_from_assignment(&qubo, vec![true, true, true]);
        sol.enable_negative_gain_index();
        assert_eq!(sol.negative_gain.len(), 3);

        let flip = QuboFlipNeighbor {
            i: 1,
            gain: sol.gain[1],
        };
        flip.apply_to_solution(&qubo, &mut sol).unwrap();

        // Verify the index matches the actual gain values
        for &v in qubo.iter_on_variables() {
            let in_index = sol.negative_gain.contains(&v);
            let has_negative_gain = sol.gain[v] < 0;
            assert_eq!(
                in_index, has_negative_gain,
                "variable {v}: in_index={in_index} but gain={} (negative={})",
                sol.gain[v], has_negative_gain
            );
        }

        // Verify inverse index consistency
        for &v in &sol.negative_gain {
            let pos = sol.negative_gain_pos[v] as usize;
            assert_eq!(sol.negative_gain[pos], v);
        }
    }

    #[test]
    fn test_clone_preserves_negative_gain_index() {
        let qubo = Qubo::from_entries([(0, 1, 1), (1, 2, 1), (0, 2, 1)]);
        let mut sol = QuboSolution::new_from_assignment(&qubo, vec![true, true, true]);
        sol.enable_negative_gain_index();

        let cloned = sol.clone();

        assert!(cloned.negative_gain_enabled);
        let mut orig_sorted = sol.negative_gain.clone();
        orig_sorted.sort();
        let mut clone_sorted = cloned.negative_gain.clone();
        clone_sorted.sort();
        assert_eq!(orig_sorted, clone_sorted);
        assert_eq!(cloned.negative_gain_pos, sol.negative_gain_pos);
    }

    #[test]
    fn test_clone_preserves_disabled_state() {
        let (_qubo, sol) = make_qubo_and_solution();
        assert!(!sol.negative_gain_enabled);

        let cloned = sol.clone();
        assert!(!cloned.negative_gain_enabled);
        assert!(cloned.negative_gain.is_empty());
        assert!(cloned.negative_gain_pos.is_empty());
    }
}
