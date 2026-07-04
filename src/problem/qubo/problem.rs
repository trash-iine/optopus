use std::collections::HashSet;

use crate::common::GainIndex;
use crate::search_state::{Distance, ProblemTrait, Rankable};
use crate::trait_defs::BinaryProblem;

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
    /// Advanced: index of variables `v` with `gain[v] < 0`, maintained
    /// incrementally once enabled. Not needed for standard heuristic use.
    /// See [`enable_negative_gain_index`](Self::enable_negative_gain_index).
    pub(crate) negative_gain: GainIndex,
}

impl Rankable for QuboSolution {
    fn is_better_than(&self, other: &Self) -> bool {
        self.objective < other.objective
    }
}

impl Distance for QuboSolution {
    fn distance(&self, other: &Self) -> usize {
        crate::common::hamming_distance(&self.x, &other.x)
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
    /// The advanced `negative_gain` index is not initialized; see
    /// [`enable_negative_gain_index`](Self::enable_negative_gain_index) if you need it.
    ///
    /// Prefer [`new_from_assignment`](Self::new_from_assignment) for constructing solutions
    /// from a variable assignment — it computes `gain` and `objective` automatically.
    pub(crate) fn new_from_parts(
        x: Vec<bool>,
        gain: Vec<Coefficient>,
        objective: Coefficient,
    ) -> Self {
        Self {
            x,
            gain,
            objective,
            negative_gain: GainIndex::default(),
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
        self.negative_gain.enable(&self.gain, |&g| g < 0);
    }

    /// Records that variable `v`'s gain is changing from `self.gain[v]` to `new_gain`.
    /// Updates membership of `v` in the `negative_gain` index (does **not** write
    /// `self.gain[v]` — the caller is expected to do that).
    ///
    /// No-op when the index is not enabled.
    #[inline]
    pub(crate) fn update_negative_gain_membership(&mut self, v: usize, new_gain: Coefficient) {
        self.negative_gain.update(v, self.gain[v] < 0, new_gain < 0);
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
    pub fn from_entries(entries: impl IntoIterator<Item = (usize, usize, Coefficient)>) -> Self {
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

    /// Loads a QUBO problem from a sparse Q-matrix file.
    ///
    /// # File format
    ///
    /// One header line followed by Q-matrix entries (**1-indexed**):
    ///
    /// ```text
    /// N M
    /// i j v
    /// i j v
    /// ...
    /// ```
    ///
    /// Each entry sets `Q[i][j] = Q[j][i] = v`. When `i == j`, `v` is the linear
    /// (diagonal) coefficient. Duplicate entries follow [`set_q`](Self::set_q)
    /// semantics: the last write wins.
    pub fn load_file(path: impl AsRef<std::path::Path>) -> Result<Self, crate::error::OptError> {
        use crate::common::InstanceLines;

        let mut lines = InstanceLines::open(path)?;
        let header = lines
            .next_line()?
            .ok_or_else(|| lines.err("file is empty, expected header 'N M'"))?;
        let mut tokens = header.split_whitespace();
        let n: usize = lines.parse_next(&mut tokens, "variable count N in header 'N M'")?;
        let _m: usize = lines.parse_next(&mut tokens, "entry count M in header 'N M'")?;

        let mut qubo = Qubo {
            adj: vec![vec![]; n],
            variables: (0..n).collect(),
        };

        while let Some(line) = lines.next_data_line()? {
            let mut tokens = line.split_whitespace();
            let i: usize = lines.parse_next(&mut tokens, "index i in entry 'i j v'")?;
            if i == 0 {
                return Err(lines.err("index i must be >= 1 (1-indexed)"));
            }
            let j: usize = lines.parse_next(&mut tokens, "index j in entry 'i j v'")?;
            if j == 0 {
                return Err(lines.err("index j must be >= 1 (1-indexed)"));
            }
            let v: i32 = lines.parse_next(&mut tokens, "coefficient v in entry 'i j v'")?;
            qubo.set_q(i - 1, j - 1, v);
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

impl BinaryProblem for Qubo {
    type Flip = super::QuboFlipNeighbor;

    fn variable_indices(&self) -> impl Iterator<Item = usize> + '_ {
        self.iter_on_variables().copied()
    }

    fn variable(sol: &QuboSolution, i: usize) -> bool {
        sol.x[i]
    }

    fn flip_move(sol: &QuboSolution, i: usize) -> Self::Flip {
        super::QuboFlipNeighbor {
            i,
            gain: sol.gain[i],
        }
    }
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
    fn test_load_file() {
        let qubo = Qubo::load_file("data/instances/qubo/test_data.txt").unwrap();
        assert_eq!(qubo.get_q(0, 1), 2);
        assert_eq!(qubo.get_q(1, 2), -6);
        assert_eq!(qubo.get_q(0, 0), -1);
        assert_eq!(qubo.get_q(1, 1), 0);
        assert_eq!(qubo.get_q(2, 2), 3);
        assert_eq!(qubo.get_q(2, 1), -6); // symmetric
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
            !sol.negative_gain.is_enabled(),
            "index should be disabled after construction"
        );
        assert!(sol.negative_gain.is_empty());
    }

    #[test]
    fn test_enable_negative_gain_index_builds_correctly() {
        // All variables false → gain[i] = sum of Q[i][j] for all j with x[j]=true = 0
        // Since QUBO gain for all-false is all positive (flipping adds energy),
        // negative_gain should be empty.
        let (_qubo, mut sol) = make_qubo_and_solution();
        sol.enable_negative_gain_index();

        assert!(sol.negative_gain.is_enabled());
        // All-false assignment on all-positive Q: gains are non-negative → no negative gains
        for &v in sol.negative_gain.as_slice() {
            assert!(
                sol.gain[v] < 0,
                "all entries in negative_gain must have gain < 0"
            );
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

        assert!(sol.negative_gain.is_enabled());
        let mut listed = sol.negative_gain.as_slice().to_vec();
        listed.sort();
        assert_eq!(listed, vec![0, 1, 2], "all variables should have gain < 0");
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
        for v in 0..sol.gain.len() {
            assert!(!sol.negative_gain.contains(v));
        }
    }

    #[test]
    fn test_enable_negative_gain_index_idempotent() {
        let qubo = Qubo::from_entries([(0, 1, 1), (1, 2, 1), (0, 2, 1)]);
        let mut sol = QuboSolution::new_from_assignment(&qubo, vec![true, true, true]);
        sol.enable_negative_gain_index();
        let ng_after_first = sol.negative_gain.as_slice().to_vec();

        sol.enable_negative_gain_index();

        assert_eq!(sol.negative_gain.as_slice(), ng_after_first);
    }

    #[test]
    fn test_update_negative_gain_membership_noop_when_disabled() {
        let (_qubo, mut sol) = make_qubo_and_solution();
        sol.update_negative_gain_membership(0, -5);
        assert!(sol.negative_gain.is_empty());
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
            !sol.negative_gain.contains(0),
            "variable 0 must leave negative_gain when gain becomes non-negative"
        );
        assert_eq!(sol.negative_gain.len(), 2);

        // Simulate: variable 0's gain changes back to -3 (should re-enter the index)
        sol.update_negative_gain_membership(0, -3);
        sol.gain[0] = -3;
        assert!(sol.negative_gain.contains(0));
        assert_eq!(sol.negative_gain.len(), 3);
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
            let in_index = sol.negative_gain.contains(v);
            let has_negative_gain = sol.gain[v] < 0;
            assert_eq!(
                in_index, has_negative_gain,
                "variable {v}: in_index={in_index} but gain={} (negative={})",
                sol.gain[v], has_negative_gain
            );
        }
    }

    #[test]
    fn test_clone_preserves_negative_gain_index() {
        let qubo = Qubo::from_entries([(0, 1, 1), (1, 2, 1), (0, 2, 1)]);
        let mut sol = QuboSolution::new_from_assignment(&qubo, vec![true, true, true]);
        sol.enable_negative_gain_index();

        let cloned = sol.clone();

        assert!(cloned.negative_gain.is_enabled());
        let mut orig_sorted = sol.negative_gain.as_slice().to_vec();
        orig_sorted.sort();
        let mut clone_sorted = cloned.negative_gain.as_slice().to_vec();
        clone_sorted.sort();
        assert_eq!(orig_sorted, clone_sorted);
    }

    #[test]
    fn test_clone_preserves_disabled_state() {
        let (_qubo, sol) = make_qubo_and_solution();
        assert!(!sol.negative_gain.is_enabled());

        let cloned = sol.clone();
        assert!(!cloned.negative_gain.is_enabled());
        assert!(cloned.negative_gain.is_empty());
    }
}
