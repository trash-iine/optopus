use crate::common::{GainIndex, Graph};
use crate::search_state::{Distance, ProblemTrait, Rankable};
use crate::trait_defs::BinaryProblem;

/// The MaxCut problem instance — an undirected weighted graph.
///
/// MaxCut seeks a partition of vertices into two sets that maximizes the total
/// weight of edges crossing the partition.
///
/// # Graph construction
///
/// ```
/// use optopus::problem::MaxCut;
/// use optopus::common::Graph;
///
/// // From edge list
/// let mc = MaxCut::from_edges([(0, 1, 1.0), (1, 2, 2.0)]);
///
/// // From a Graph
/// let mut g = Graph::new();
/// g.set_weight(0, 1, 1.0);
/// let mc = MaxCut::new(g);
///
/// // Read weight via Graph's Index
/// assert_eq!(mc.graph[(0, 1)], 1.0);
/// assert_eq!(mc.graph[(8, 9)], 0.0);  // non-existent → 0.0
/// ```
///
/// # Optimization direction
///
/// Maximization: A solution with a higher `objective` is better.
#[derive(Debug, Clone)]
pub struct MaxCut {
    /// The underlying graph.
    pub graph: Graph,
}

/// A solution for the MaxCut problem.
///
/// # Core fields
///
/// - [`cut`](Self::cut) — partition assignment (`cut[i]` is the side of vertex `i`)
/// - [`gain`](Self::gain) — per-vertex flip gain (`gain[i]` = change in cut weight when `i` is flipped; positive = improvement)
/// - [`objective`](Self::objective) — total weight of edges crossing the cut
///
/// These three fields are all you need to inspect results and build custom logic.
///
/// # Advanced: positive-gain index
///
/// An optional index tracks which vertices currently have positive gain (i.e. improving
/// moves). Call [`enable_positive_gain_index`](Self::enable_positive_gain_index) to activate it.
/// Standard heuristics ([`LocalSearch`](crate::heuristic::LocalSearch),
/// [`TabuSearch`](crate::heuristic::TabuSearch),
/// [`SimulatedAnnealing`](crate::heuristic::SimulatedAnnealing), etc.)
/// do **not** require this index — it is a performance optimization for problem-specific
/// algorithms such as [`BreakoutLocalSearchForMaxCut`](crate::heuristic::BreakoutLocalSearchForMaxCut).
///
/// # Examples
///
/// ```
/// use optopus::prelude::*;
///
/// let mc = MaxCut::from_edges([(0, 1, 1.0), (1, 2, 2.0), (0, 2, 3.0)]);
/// let mut state = SearchState::new(&mc);
/// LocalSearch::<MaxCutFlipNeighbor>::new(StopCondition::iterations(1000))
///     .run(&mut state).unwrap();
///
/// let sol = &state.best_solution;
/// // sol.objective — the cut weight
/// // sol.cut[i]   — which side vertex i is on
/// // sol.gain[i]  — how much flipping vertex i would change the objective
/// ```
#[derive(Debug, Clone)]
pub struct MaxCutSolution {
    /// The cut assignment for each vertex: `cut[i]` is the side of vertex `i`.
    /// Sized to `max_vertex_id + 1`; only indices in `MaxCut::graph.vertices` are meaningful.
    pub cut: Vec<bool>,
    /// The gain of flipping each vertex: `gain[i]` = change in cut weight when flipping `i`.
    /// Sized to `max_vertex_id + 1`.
    pub gain: Vec<f32>,
    /// The total weight of edges crossing the cut.
    pub objective: f32,
    /// Advanced: index of vertices `v` with `gain[v] > 0`, maintained
    /// incrementally once enabled. Not needed for standard heuristic use.
    /// See [`enable_positive_gain_index`](Self::enable_positive_gain_index).
    pub(crate) positive_gain: GainIndex,
}

impl Rankable for MaxCutSolution {
    fn is_better_than(&self, other: &Self) -> bool {
        self.objective > other.objective
    }
}

impl Distance for MaxCutSolution {
    fn distance(&self, other: &Self) -> usize {
        self.cut
            .iter()
            .zip(other.cut.iter())
            .filter(|(a, b)| a != b)
            .count()
    }
}

impl MaxCutSolution {
    /// Returns an iterator over all vertex indices `0..cut.len()`.
    ///
    /// # Examples
    ///
    /// ```
    /// use optopus::prelude::*;
    ///
    /// let mc = MaxCut::from_edges([(0, 1, 1.0), (1, 2, 1.0)]);
    /// let state = SearchState::new(&mc);
    /// for v in state.solution.iter_on_vertices() {
    ///     println!("vertex {v}: side={}", state.solution.cut[v]);
    /// }
    /// ```
    pub fn iter_on_vertices(&self) -> impl Iterator<Item = usize> + '_ {
        0..self.cut.len()
    }

    /// Builds a [`MaxCutSolution`] from pre-computed components.
    ///
    /// The resulting solution is fully functional for all standard heuristics.
    /// The advanced `positive_gain` index is not initialised; see
    /// [`enable_positive_gain_index`](Self::enable_positive_gain_index) if you need it.
    ///
    /// Prefer [`new_from_cut`](Self::new_from_cut) for constructing solutions from
    /// a cut assignment — it computes `gain` and `objective` automatically.
    pub(crate) fn new_from_parts(cut: Vec<bool>, gain: Vec<f32>, objective: f32) -> Self {
        Self {
            cut,
            gain,
            objective,
            positive_gain: GainIndex::default(),
        }
    }

    /// Creates a [`MaxCutSolution`] from a cut assignment, computing gain and objective automatically.
    ///
    /// # Examples
    ///
    /// ```
    /// use optopus::prelude::*;
    ///
    /// let mc = MaxCut::from_edges([(0, 1, 1.0), (0, 2, 2.0), (1, 2, 3.0)]);
    /// let sol = MaxCutSolution::new_from_cut(&mc, vec![true, false, false]);
    /// assert_eq!(sol.objective, 3.0);  // edges (0,1)=1.0 + (0,2)=2.0
    /// ```
    pub fn new_from_cut(mc: &MaxCut, cut: Vec<bool>) -> Self {
        let n = mc.graph.len();
        let mut gain = vec![0.0; n];
        for &i in mc.graph.iter_on_vertices() {
            gain[i] = mc.calculate_gain(&cut, i);
        }
        let objective = mc.calculate_cut_size(&cut);
        Self::new_from_parts(cut, gain, objective)
    }

    /// **Advanced.** Enables the `positive_gain` index, building it from the current
    /// `gain` vector.
    ///
    /// Most users do **not** need to call this method. Standard heuristics
    /// ([`LocalSearch`](crate::heuristic::LocalSearch),
    /// [`TabuSearch`](crate::heuristic::TabuSearch),
    /// [`SimulatedAnnealing`](crate::heuristic::SimulatedAnnealing), etc.)
    /// work correctly without it.
    ///
    /// This index is useful for problem-specific algorithms (such as
    /// [`BreakoutLocalSearchForMaxCut`](crate::heuristic::BreakoutLocalSearchForMaxCut))
    /// that need to iterate only over vertices with positive gain, reducing the
    /// inner-loop cost from O(n) to O(|improving moves|).
    ///
    /// Once enabled, the index is maintained incrementally by
    /// [`MaxCutFlipNeighbor::apply_to_solution`](super::MaxCutFlipNeighbor).
    ///
    /// If already enabled, this is a no-op. O(n).
    ///
    /// # Examples
    ///
    /// ```
    /// use optopus::prelude::*;
    ///
    /// let mc = MaxCut::from_edges([(0, 1, 1.0), (1, 2, 1.0), (0, 2, 1.0)]);
    /// let mut state = SearchState::new(&mc);
    /// state.solution.enable_positive_gain_index();
    /// // Now positive_gain tracks vertices with gain > 0
    /// ```
    pub fn enable_positive_gain_index(&mut self) {
        self.positive_gain.enable(&self.gain, |&g| g > 0.0);
    }

    /// Records that vertex `v`'s gain is changing from `self.gain[v]` to `new_gain`.
    /// Updates membership of `v` in the `positive_gain` index (does **not** write
    /// `self.gain[v]` — the caller is expected to do that).
    ///
    /// No-op when the index is not enabled.
    #[inline]
    pub(crate) fn update_positive_gain_membership(&mut self, v: usize, new_gain: f32) {
        self.positive_gain
            .update(v, self.gain[v] > 0.0, new_gain > 0.0);
    }
}

impl MaxCut {
    /// Creates a [`MaxCut`] from a [`Graph`].
    ///
    /// # Examples
    ///
    /// ```
    /// use optopus::common::Graph;
    /// use optopus::problem::MaxCut;
    ///
    /// let mc = MaxCut::new(Graph::from_edges([(0, 1, 1.0), (1, 2, 2.0)]));
    /// assert_eq!(mc.graph.num_edges(), 2);
    /// ```
    pub fn new(graph: Graph) -> Self {
        Self { graph }
    }

    /// Creates a [`MaxCut`] from an iterator of `(i, j, weight)` edges.
    ///
    /// Duplicate edges are overwritten (last occurrence wins).
    ///
    /// # Examples
    ///
    /// ```
    /// let mc = optopus::problem::MaxCut::from_edges([
    ///     (0, 1, 1.0),
    ///     (0, 2, 2.0),
    ///     (1, 2, 3.0),
    /// ]);
    /// assert_eq!(mc.graph[(0, 1)], 1.0);
    /// assert_eq!(mc.graph.num_edges(), 3);
    /// ```
    pub fn from_edges(edges: impl IntoIterator<Item = (usize, usize, f32)>) -> Self {
        Self::new(Graph::from_edges(edges))
    }

    /// Loads a [`MaxCut`] instance from a file in the `N M / i j w` format.
    ///
    /// Wraps [`Graph::load_from_file`] and constructs the problem with
    /// [`MaxCut::new`].
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use optopus::problem::MaxCut;
    ///
    /// let mc = MaxCut::load_file("data/instances/max_cut/G1").unwrap();
    /// ```
    pub fn load_file(path: impl AsRef<std::path::Path>) -> Result<Self, crate::error::OptError> {
        Graph::load_from_file(path).map(Self::new)
    }

    /// Calculates the total weight of edges crossing the partition defined by `cut`.
    ///
    /// `cut[i]` is the side of vertex `i`. An edge `(i, j)` is crossing when
    /// `cut[i] != cut[j]`.
    ///
    /// # Examples
    ///
    /// ```
    /// let mc = optopus::problem::MaxCut::from_edges([
    ///     (0, 1, 1.0), (0, 2, 2.0), (1, 2, 3.0),
    /// ]);
    /// let cut = vec![true, false, false];     // vertex 0 on one side, 1 and 2 on the other
    /// assert_eq!(mc.calculate_cut_size(&cut), 3.0);  // edges (0,1)=1.0 + (0,2)=2.0
    /// ```
    pub fn calculate_cut_size(&self, cut: &[bool]) -> f32 {
        let mut ret = 0.0;
        for &i in self.graph.iter_on_vertices() {
            let bi = cut[i];
            for &(j, w) in self.graph.iter_on_adjacency(i) {
                if bi ^ cut[j] {
                    ret += w;
                }
            }
        }
        ret / 2.0
    }

    /// Calculates the gain of flipping vertex `i` given the current cut assignment.
    ///
    /// A positive return value means flipping vertex `i` would **improve** the cut.
    ///
    /// # Examples
    ///
    /// ```
    /// let mc = optopus::problem::MaxCut::from_edges([
    ///     (0, 1, 1.0), (0, 2, 2.0), (1, 2, 3.0),
    /// ]);
    /// let cut = vec![false, false, false];  // all on the same side
    /// assert_eq!(mc.calculate_gain(&cut, 0), 3.0);  // flipping 0 crosses edges (0,1)+(0,2)
    /// assert_eq!(mc.calculate_gain(&cut, 1), 4.0);  // flipping 1 crosses edges (0,1)+(1,2)
    /// assert_eq!(mc.calculate_gain(&cut, 2), 5.0);  // flipping 2 crosses edges (0,2)+(1,2)
    /// ```
    pub fn calculate_gain(&self, cut: &[bool], i: usize) -> f32 {
        let bi = cut[i];
        self.graph
            .iter_on_adjacency(i)
            .map(|&(j, w)| if bi ^ cut[j] { -w } else { w })
            .sum()
    }
}

/// Displays a summary of the graph: `MaxCut(vertices: N, edges: M)` or `MaxCut(empty)`.
///
/// # Examples
///
/// ```
/// let mc = optopus::problem::MaxCut::from_edges([(0, 1, 1.0), (1, 2, 2.0)]);
/// assert_eq!(format!("{mc}"), "MaxCut(vertices: 3, edges: 2)");
///
/// let empty = optopus::problem::MaxCut::new(optopus::common::Graph::new());
/// assert_eq!(format!("{empty}"), "MaxCut(empty)");
/// ```
impl std::fmt::Display for MaxCut {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.graph.is_empty() {
            write!(f, "MaxCut(empty)")
        } else {
            write!(
                f,
                "MaxCut(vertices: {}, edges: {})",
                self.graph.num_vertices(),
                self.graph.num_edges(),
            )
        }
    }
}

impl ProblemTrait for MaxCut {
    type Solution = MaxCutSolution;
    fn new_solution(&self, rng: &mut impl rand::Rng) -> Self::Solution {
        let n = self.graph.len();
        let mut cut = vec![false; n];
        for &i in self.graph.iter_on_vertices() {
            cut[i] = rng.random_bool(0.5);
        }
        MaxCutSolution::new_from_cut(self, cut)
    }
}

impl BinaryProblem for MaxCut {
    type Flip = super::MaxCutFlipNeighbor;

    fn variable_indices(&self) -> impl Iterator<Item = usize> + '_ {
        self.graph.iter_on_vertices().copied()
    }

    fn variable(sol: &MaxCutSolution, i: usize) -> bool {
        sol.cut[i]
    }

    fn flip_move(sol: &MaxCutSolution, i: usize) -> Self::Flip {
        super::MaxCutFlipNeighbor {
            i,
            gain: sol.gain[i],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_blank_graph() {
        let mc = MaxCut::new(Graph::new());
        assert_eq!(mc.graph.len(), 0);
        assert!(mc.graph.is_empty());
    }

    #[test]
    fn test_load_file_roundtrip() {
        use std::io::Write;
        let mut path = std::env::temp_dir();
        path.push(format!(
            "optopus_maxcut_{}_{}.txt",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        {
            let mut f = std::fs::File::create(&path).unwrap();
            // 3-vertex triangle with unit weights, 1-indexed
            writeln!(f, "3 3").unwrap();
            writeln!(f, "1 2 1").unwrap();
            writeln!(f, "1 3 1").unwrap();
            writeln!(f, "2 3 1").unwrap();
        }
        let mc = MaxCut::load_file(&path).expect("load_file should succeed");
        assert_eq!(mc.graph.num_vertices(), 3);
        assert_eq!(mc.graph.num_edges(), 3);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_set_and_get_weight() {
        let mut mc = MaxCut::new(Graph::new());
        mc.graph.add_weight(0, 1, 1.0);
        mc.graph.add_weight(0, 2, 1.0);
        mc.graph.add_weight(0, 1, 2.0);

        assert_eq!(mc.graph.len(), 3);

        assert_eq!(mc.graph.get_weight(0, 1), 3.0);
        assert_eq!(mc.graph.get_weight(0, 2), 1.0);
    }

    #[test]
    fn test_calculate_cut_size() {
        let mut mc = MaxCut::new(Graph::new());
        mc.graph.add_weight(0, 1, 1.0);
        mc.graph.add_weight(0, 2, 2.0);
        mc.graph.add_weight(1, 2, 3.0);

        {
            let cut = vec![false, false, false];
            assert_eq!(mc.calculate_cut_size(&cut), 0.0);
        }
        {
            let cut = vec![true, false, false];
            assert_eq!(mc.calculate_cut_size(&cut), 3.0);
        }
        {
            let cut = vec![true, false, true];
            assert_eq!(mc.calculate_cut_size(&cut), 4.0);
        }
        {
            let cut = vec![true, true, false];
            assert_eq!(mc.calculate_cut_size(&cut), 5.0);
        }
    }

    #[test]
    fn test_calculate_gain_list() {
        let mut mc = MaxCut::new(Graph::new());
        mc.graph.add_weight(0, 1, 1.0);
        mc.graph.add_weight(0, 2, 2.0);
        mc.graph.add_weight(1, 2, 3.0);

        let cut = vec![false, false, false];
        assert_eq!(mc.calculate_gain(&cut, 0), 3.0);
        assert_eq!(mc.calculate_gain(&cut, 1), 4.0);
        assert_eq!(mc.calculate_gain(&cut, 2), 5.0);
    }

    // ---------------------------------------------------------------------------
    // Helper: 3-vertex triangle (unit weights), all-false cut.
    // gain[v] = total edge weight of v = 2.0 for each vertex (positive for all).
    // ---------------------------------------------------------------------------
    fn make_triangle_solution() -> (MaxCut, MaxCutSolution) {
        let mut mc = MaxCut::new(Graph::new());
        mc.graph.add_weight(0, 1, 1.0);
        mc.graph.add_weight(1, 2, 1.0);
        mc.graph.add_weight(0, 2, 1.0);
        let n = mc.graph.len(); // 3
        let cut = vec![false; n];
        let gain: Vec<f32> = (0..n).map(|v| mc.calculate_gain(&cut, v)).collect();
        let objective = mc.calculate_cut_size(&cut);
        let sol = MaxCutSolution::new_from_parts(cut, gain, objective);
        (mc, sol)
    }

    // 1. from_parts() creates a solution with the positive_gain index disabled.
    #[test]
    fn test_from_parts_index_disabled_by_default() {
        let (_mc, sol) = make_triangle_solution();
        assert!(
            !sol.positive_gain.is_enabled(),
            "index should be disabled after from_parts"
        );
        assert!(sol.positive_gain.is_empty(), "positive_gain must be empty");
    }

    // 2. enable_positive_gain_index() correctly builds the index from gain[].
    //    All-false cut on a unit-weight triangle: gain[v] = 2.0 > 0 for every vertex,
    //    so all three vertices must appear in positive_gain.
    #[test]
    fn test_enable_positive_gain_index_builds_correctly() {
        let (_mc, mut sol) = make_triangle_solution();
        sol.enable_positive_gain_index();

        assert!(
            sol.positive_gain.is_enabled(),
            "index should be enabled after call"
        );
        // All gains are 2.0 > 0, so all three vertices appear.
        let mut listed = sol.positive_gain.as_slice().to_vec();
        listed.sort();
        assert_eq!(
            listed,
            vec![0, 1, 2],
            "all vertices should be in positive_gain"
        );
    }

    // Also verify the case where some vertices have non-positive gain.
    // Triangle, cut = [true, false, false]: gain[0] = -2 (negative), gain[1] = 0 (zero), gain[2] = 0 (zero).
    // Actually: cut [true, false, false] — edges 0-1 (crossing, w=1) and 0-2 (crossing, w=1) and 1-2 (not crossing, w=1).
    // gain[0] = -(1+1) = -2  (both edges cross, flipping 0 removes both)
    // gain[1] = 1 - 1 = 0    (edge 0-1 crosses (+1), edge 1-2 not crossing (-1))
    // gain[2] = 1 - 1 = 0    (edge 0-2 crosses (+1), edge 1-2 not crossing (-1))
    // Only vertices with gain > 0 should be in positive_gain; none here.
    #[test]
    fn test_enable_positive_gain_index_excludes_non_positive() {
        let mut mc = MaxCut::new(Graph::new());
        mc.graph.add_weight(0, 1, 1.0);
        mc.graph.add_weight(1, 2, 1.0);
        mc.graph.add_weight(0, 2, 1.0);
        let n = mc.graph.len();
        let cut = vec![true, false, false];
        let gain: Vec<f32> = (0..n).map(|v| mc.calculate_gain(&cut, v)).collect();
        let objective = mc.calculate_cut_size(&cut);
        let mut sol = MaxCutSolution::new_from_parts(cut, gain, objective);

        sol.enable_positive_gain_index();

        // gain[0]=-2, gain[1]=0, gain[2]=0 — none are strictly positive.
        assert!(
            sol.positive_gain.is_empty(),
            "no vertex has gain > 0; positive_gain must be empty"
        );
        for v in 0..n {
            assert!(
                !sol.positive_gain.contains(v),
                "vertex {v} must be absent from the index"
            );
        }
    }

    // 3. enable_positive_gain_index() is idempotent: calling it twice must not
    //    corrupt the index (no duplicates, consistent inverse).
    #[test]
    fn test_enable_positive_gain_index_idempotent() {
        let (_mc, mut sol) = make_triangle_solution();
        sol.enable_positive_gain_index();
        let pg_after_first = sol.positive_gain.as_slice().to_vec();

        sol.enable_positive_gain_index();

        assert_eq!(
            sol.positive_gain.as_slice(),
            pg_after_first,
            "second call must not change positive_gain"
        );
    }

    // 4. update_positive_gain_membership() is a no-op when the index is disabled.
    #[test]
    fn test_update_positive_gain_membership_noop_when_disabled() {
        let (_mc, mut sol) = make_triangle_solution();
        // Index not enabled; calling the update must not populate it.
        sol.update_positive_gain_membership(0, 5.0);
        assert!(
            sol.positive_gain.is_empty(),
            "positive_gain must stay empty when index is disabled"
        );
    }

    // 5. After enabling, update_positive_gain_membership() correctly maintains
    //    the index through manual gain changes that simulate a flip:
    //    move vertex 0 from positive_gain to absent (new_gain <= 0),
    //    and then back to positive (new_gain > 0).
    #[test]
    fn test_update_positive_gain_membership_maintains_index() {
        let (_mc, mut sol) = make_triangle_solution();
        // Enable: all three vertices are in positive_gain (gain = 2.0 each).
        sol.enable_positive_gain_index();
        assert_eq!(sol.positive_gain.len(), 3);

        // Simulate: vertex 0's gain changes to -1.0 (negative → should leave the index).
        sol.update_positive_gain_membership(0, -1.0);
        sol.gain[0] = -1.0;
        assert!(
            !sol.positive_gain.contains(0),
            "vertex 0 must leave positive_gain when gain becomes negative"
        );
        assert_eq!(sol.positive_gain.len(), 2);

        // Simulate: vertex 0's gain changes back to 3.0 (should re-enter the index).
        sol.update_positive_gain_membership(0, 3.0);
        sol.gain[0] = 3.0;
        assert!(
            sol.positive_gain.contains(0),
            "vertex 0 must re-enter positive_gain when gain becomes positive"
        );
        assert_eq!(sol.positive_gain.len(), 3);
    }

    // 5b. Verify that the positive_gain index stays consistent after applying a real
    //     flip move through the neighbor machinery (uses MaxCutFlipNeighbor).
    //     Triangle, all-false: flip vertex 1 (gain=2.0). After the flip:
    //     cut = [F, T, F], objective = 2.0.
    //     New gains: gain[1] = -2.0, gain[0] = 0.0, gain[2] = 0.0.
    //     None are strictly positive → positive_gain must be empty.
    #[test]
    fn test_positive_gain_index_consistent_after_flip() {
        use crate::problem::max_cut::MaxCutFlipNeighbor;
        use crate::search_state::MoveToNeighbor;

        let (mc, mut sol) = make_triangle_solution();
        sol.enable_positive_gain_index();
        assert_eq!(
            sol.positive_gain.len(),
            3,
            "all vertices start as improving"
        );

        let flip = MaxCutFlipNeighbor {
            i: 1,
            gain: sol.gain[1],
        };
        flip.apply_to_solution(&mc, &mut sol).unwrap();

        // After flipping vertex 1: gain[1] = -2.0, gain[0] = 0.0, gain[2] = 0.0.
        // None strictly positive → index must be empty.
        assert!(
            sol.positive_gain.is_empty(),
            "positive_gain must be empty after flip makes all gains non-positive"
        );
        for v in 0..sol.gain.len() {
            assert!(
                !sol.positive_gain.contains(v),
                "vertex {v} must be absent when index is empty"
            );
        }
    }

    // 6. Clone of an enabled solution preserves the enabled state and the
    //    index content intact.
    #[test]
    fn test_clone_preserves_positive_gain_index() {
        let (_mc, mut sol) = make_triangle_solution();
        sol.enable_positive_gain_index();

        let cloned = sol.clone();

        assert!(
            cloned.positive_gain.is_enabled(),
            "clone must inherit the enabled index"
        );
        let mut orig_sorted = sol.positive_gain.as_slice().to_vec();
        orig_sorted.sort();
        let mut clone_sorted = cloned.positive_gain.as_slice().to_vec();
        clone_sorted.sort();
        assert_eq!(
            orig_sorted, clone_sorted,
            "clone must have the same positive_gain contents"
        );
    }

    // 6b. Clone of a disabled solution preserves the disabled state.
    #[test]
    fn test_clone_preserves_disabled_state() {
        let (_mc, sol) = make_triangle_solution();
        assert!(!sol.positive_gain.is_enabled());

        let cloned = sol.clone();
        assert!(
            !cloned.positive_gain.is_enabled(),
            "clone must preserve the disabled index"
        );
        assert!(cloned.positive_gain.is_empty());
    }

    #[test]
    fn test_set_weight_overwrites() {
        let mut mc = MaxCut::new(Graph::new());
        mc.graph.set_weight(0, 1, 5.0);
        assert_eq!(mc.graph[(0, 1)], 5.0);

        mc.graph.set_weight(0, 1, 3.0);
        assert_eq!(mc.graph[(0, 1)], 3.0); // overwritten, not 8.0
        assert_eq!(mc.graph[(1, 0)], 3.0); // symmetric
    }

    #[test]
    fn test_set_weight_and_add_weight_interaction() {
        let mut mc = MaxCut::new(Graph::new());
        mc.graph.set_weight(0, 1, 5.0);
        mc.graph.add_weight(0, 1, 2.0);
        assert_eq!(mc.graph[(0, 1)], 7.0); // 5.0 + 2.0

        mc.graph.set_weight(0, 1, 1.0); // overwrite back
        assert_eq!(mc.graph[(0, 1)], 1.0);
    }

    #[test]
    fn test_index_existing_edge() {
        let mc = MaxCut::from_edges([(0, 1, 3.0), (1, 2, 7.0)]);
        assert_eq!(mc.graph[(0, 1)], 3.0);
        assert_eq!(mc.graph[(1, 0)], 3.0);
        assert_eq!(mc.graph[(1, 2)], 7.0);
    }

    #[test]
    fn test_index_missing_edge() {
        let mc = MaxCut::from_edges([(0, 1, 1.0)]);
        assert_eq!(mc.graph[(0, 2)], 0.0);
        assert_eq!(mc.graph[(5, 6)], 0.0); // out of bounds
    }

    #[test]
    fn test_num_vertices_and_edges() {
        let mc = MaxCut::from_edges([(0, 1, 1.0), (0, 2, 1.0), (1, 2, 1.0)]);
        assert_eq!(mc.graph.num_vertices(), 3);
        assert_eq!(mc.graph.num_edges(), 3);
    }

    #[test]
    fn test_is_empty() {
        let mc = MaxCut::new(Graph::new());
        assert!(mc.graph.is_empty());

        let mc = MaxCut::from_edges([(0, 1, 1.0)]);
        assert!(!mc.graph.is_empty());
    }

    #[test]
    fn test_from_edges() {
        let mc = MaxCut::from_edges([(0, 1, 1.0), (0, 2, 2.0), (1, 2, 3.0)]);
        assert_eq!(mc.graph[(0, 1)], 1.0);
        assert_eq!(mc.graph[(0, 2)], 2.0);
        assert_eq!(mc.graph[(1, 2)], 3.0);
        assert_eq!(mc.graph.num_edges(), 3);
    }

    #[test]
    fn test_from_edges_duplicate_last_wins() {
        let mc = MaxCut::from_edges([(0, 1, 1.0), (0, 1, 5.0)]);
        assert_eq!(mc.graph[(0, 1)], 5.0);
    }

    #[test]
    fn test_edges_iterator() {
        let mc = MaxCut::from_edges([(0, 1, 1.0), (0, 2, 2.0), (1, 2, 3.0)]);
        let mut edges: Vec<_> = mc.graph.edges().collect();
        edges.sort_by_key(|&(i, j, _)| (i, j));
        assert_eq!(edges, vec![(0, 1, 1.0), (0, 2, 2.0), (1, 2, 3.0)]);
    }

    #[test]
    fn test_degree() {
        let mc = MaxCut::from_edges([(0, 1, 1.0), (0, 2, 1.0), (1, 2, 1.0)]);
        assert_eq!(mc.graph.degree(0), 2);
        assert_eq!(mc.graph.degree(1), 2);
        assert_eq!(mc.graph.degree(2), 2);
        assert_eq!(mc.graph.degree(99), 0); // out of bounds
    }

    #[test]
    fn test_display_empty() {
        let mc = MaxCut::new(Graph::new());
        assert_eq!(format!("{mc}"), "MaxCut(empty)");
    }

    #[test]
    fn test_display_nonempty() {
        let mc = MaxCut::from_edges([(0, 1, 1.0), (0, 2, 1.0), (1, 2, 1.0)]);
        assert_eq!(format!("{mc}"), "MaxCut(vertices: 3, edges: 3)");
    }
}
