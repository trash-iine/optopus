use crate::common::Graph;
use crate::search_state::{Distance, ProblemTrait, Rankable};
use crate::trait_defs::{BinaryProblem, Evaluable, Evaluate};

/// The MaxCut problem instance, an undirected weighted graph.
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
/// - [`x`](Self::x), partition assignment (`x[i]` is the side of vertex `i`)
/// - [`gain`](Self::gain), per-vertex flip gain (`gain[i]` = change in cut weight when `i` is flipped; positive = improvement)
/// - [`objective`](Self::objective), total weight of edges crossing the cut
///
/// These three fields are all you need to inspect results and build custom logic.
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
/// // sol.objective, the cut weight
/// // sol.x[i], which side vertex i is on
/// // sol.gain[i], how much flipping vertex i would change the objective
/// ```
#[derive(Debug, Clone)]
pub struct MaxCutSolution {
    /// The cut assignment for each vertex: `x[i]` is the side of vertex `i`.
    /// Sized to `max_vertex_id + 1`; only indices in `MaxCut::graph.vertices` are meaningful.
    pub x: Vec<bool>,
    /// The gain of flipping each vertex: `gain[i]` = change in cut weight when flipping `i`.
    /// Sized to `max_vertex_id + 1`.
    pub gain: Vec<f32>,
    /// The total weight of edges crossing the cut.
    pub objective: f32,
}

impl Evaluate for MaxCutSolution {
    /// MaxCut maximizes the cut weight.
    fn evaluate(&self) -> Evaluable<f64> {
        Evaluable::Maximize(self.objective as f64)
    }
}

impl Rankable for MaxCutSolution {
    fn is_better_than(&self, other: &Self) -> bool {
        self.objective > other.objective
    }
}

impl Distance for MaxCutSolution {
    fn distance(&self, other: &Self) -> usize {
        crate::common::hamming_distance(&self.x, &other.x)
    }
}

impl MaxCutSolution {
    /// Returns an iterator over all vertex indices `0..x.len()`.
    ///
    /// # Examples
    ///
    /// ```
    /// use optopus::prelude::*;
    ///
    /// let mc = MaxCut::from_edges([(0, 1, 1.0), (1, 2, 1.0)]);
    /// let state = SearchState::new(&mc);
    /// for v in state.solution.iter_on_vertices() {
    ///     println!("vertex {v}: side={}", state.solution.x[v]);
    /// }
    /// ```
    pub fn iter_on_vertices(&self) -> impl Iterator<Item = usize> + '_ {
        0..self.x.len()
    }

    /// Builds a [`MaxCutSolution`] from pre-computed components.
    ///
    /// The resulting solution is fully functional for all standard heuristics.
    ///
    /// Prefer [`new_from_assignment`](Self::new_from_assignment) for constructing solutions from
    /// a cut assignment, it computes `gain` and `objective` automatically.
    pub(crate) fn new_from_parts(x: Vec<bool>, gain: Vec<f32>, objective: f32) -> Self {
        Self { x, gain, objective }
    }

    /// Creates a [`MaxCutSolution`] from a cut assignment, computing gain and objective automatically.
    ///
    /// # Examples
    ///
    /// ```
    /// use optopus::prelude::*;
    ///
    /// let mc = MaxCut::from_edges([(0, 1, 1.0), (0, 2, 2.0), (1, 2, 3.0)]);
    /// let sol = MaxCutSolution::new_from_assignment(&mc, vec![true, false, false]);
    /// assert_eq!(sol.objective, 3.0);  // edges (0,1)=1.0 + (0,2)=2.0
    /// ```
    pub fn new_from_assignment(mc: &MaxCut, cut: Vec<bool>) -> Self {
        let n = mc.graph.len();
        let mut gain = vec![0.0; n];
        for &i in mc.graph.iter_on_vertices() {
            gain[i] = mc.calculate_gain(&cut, i);
        }
        let objective = mc.calculate_cut_size(&cut);
        Self::new_from_parts(cut, gain, objective)
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
    /// A positive return value means flipping vertex `i` would improve the cut.
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
        MaxCutSolution::new_from_assignment(self, cut)
    }
}

impl BinaryProblem for MaxCut {
    type Flip = super::MaxCutFlipNeighbor;

    fn variable_indices(&self) -> impl Iterator<Item = usize> + '_ {
        self.graph.iter_on_vertices().copied()
    }

    fn variable(sol: &MaxCutSolution, i: usize) -> bool {
        sol.x[i]
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
        path.push(format!("optopus_maxcut_{}.txt", std::process::id()));
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
