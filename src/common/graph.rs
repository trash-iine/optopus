//! An undirected weighted graph stored as sorted adjacency lists.

/// Shared zero constant returned by `Index` for non-existent edges.
static ZERO_WEIGHT: f32 = 0.0;

/// An undirected weighted graph stored as sorted adjacency lists.
///
/// Each vertex `i` has a list of `(neighbor_id, weight)` pairs sorted by
/// neighbor ID, giving O(log(degree)) lookup for `has_edge` and `get_weight`.
///
/// # Construction
///
/// ```
/// use optopus::common::Graph;
///
/// // From edge list
/// let g = Graph::from_edges([(0, 1, 1.0), (1, 2, 2.0)]);
///
/// // Incremental
/// let mut g = Graph::new();
/// g.set_weight(0, 1, 1.0);
///
/// // Read weight via Index
/// assert_eq!(g[(0, 1)], 1.0);
/// assert_eq!(g[(8, 9)], 0.0);  // non-existent -> 0.0
/// ```
#[derive(Debug, Clone)]
pub struct Graph {
    /// `adj[i]` = sorted list of `(j, weight)` for all neighbours of vertex `i`.
    adj: Vec<Vec<(usize, f32)>>,
    /// Sorted list of vertex IDs that appear in the graph (used by `iter_on_vertices`).
    pub vertices: Vec<usize>,
}

impl Graph {
    /// Creates a new empty graph.
    ///
    /// # Examples
    ///
    /// ```
    /// let g = optopus::common::Graph::new();
    /// assert!(g.is_empty());
    /// ```
    pub fn new() -> Self {
        Self {
            adj: vec![],
            vertices: vec![],
        }
    }

    /// Creates a graph from an iterator of `(i, j, weight)` edges.
    ///
    /// Duplicate edges are overwritten (last occurrence wins).
    ///
    /// # Examples
    ///
    /// ```
    /// let g = optopus::common::Graph::from_edges([
    ///     (0, 1, 1.0),
    ///     (0, 2, 2.0),
    ///     (1, 2, 3.0),
    /// ]);
    /// assert_eq!(g[(0, 1)], 1.0);
    /// assert_eq!(g.num_edges(), 3);
    /// ```
    pub fn from_edges(edges: impl IntoIterator<Item = (usize, usize, f32)>) -> Self {
        let mut g = Self::new();
        for (i, j, w) in edges {
            g.set_weight(i, j, w);
        }
        g
    }

    /// Returns the internal dimension of the graph (`max_vertex_id + 1`).
    ///
    /// This may be larger than the number of vertices with edges.
    /// For the count of vertices that participate in edges, use
    /// [`num_vertices`](Self::num_vertices).
    ///
    /// # Examples
    ///
    /// ```
    /// let mut g = optopus::common::Graph::new();
    /// assert_eq!(g.len(), 0);
    ///
    /// g.add_weight(0, 1, 1.0);
    /// g.add_weight(0, 2, 1.0);
    /// assert_eq!(g.len(), 3);
    /// ```
    pub fn len(&self) -> usize {
        self.adj.len()
    }

    /// Returns the number of vertices that have at least one edge.
    ///
    /// # Examples
    ///
    /// ```
    /// let g = optopus::common::Graph::from_edges([(0, 1, 1.0), (0, 2, 1.0)]);
    /// assert_eq!(g.num_vertices(), 3);
    /// ```
    pub fn num_vertices(&self) -> usize {
        self.vertices.len()
    }

    /// Returns the number of undirected edges in the graph.
    ///
    /// # Examples
    ///
    /// ```
    /// let g = optopus::common::Graph::from_edges([(0, 1, 1.0), (0, 2, 1.0), (1, 2, 1.0)]);
    /// assert_eq!(g.num_edges(), 3);
    /// ```
    pub fn num_edges(&self) -> usize {
        self.adj.iter().map(|a| a.len()).sum::<usize>() / 2
    }

    /// Returns `true` if the graph has no edges.
    ///
    /// # Examples
    ///
    /// ```
    /// let g = optopus::common::Graph::new();
    /// assert!(g.is_empty());
    /// ```
    pub fn is_empty(&self) -> bool {
        self.vertices.is_empty()
    }

    /// Returns an iterator visiting all vertices that have at least one edge.
    ///
    /// # Examples
    ///
    /// ```
    /// let mut g = optopus::common::Graph::new();
    /// g.add_weight(0, 1, 1.0);
    /// g.add_weight(0, 2, 1.0);
    ///
    /// for i in g.iter_on_vertices() {
    ///    println!("{}", i); // 0, 1, 2
    /// }
    /// ```
    pub fn iter_on_vertices(&self) -> impl Iterator<Item = &usize> {
        self.vertices.iter()
    }

    /// Returns an iterator over `(neighbour_id, weight)` pairs for vertex `i`.
    ///
    /// # Examples
    ///
    /// ```
    /// let g = optopus::common::Graph::from_edges([(0, 1, 1.0), (0, 2, 2.0)]);
    /// for &(j, w) in g.iter_on_adjacency(0) {
    ///     println!("{} {}", j, w);
    /// }
    /// ```
    pub fn iter_on_adjacency(&self, i: usize) -> std::slice::Iter<'_, (usize, f32)> {
        if i < self.adj.len() {
            self.adj[i].iter()
        } else {
            [].iter()
        }
    }

    /// Returns an iterator over `(neighbour_id, weight)` pairs for vertex `i`.
    ///
    /// This is an alias for [`iter_on_adjacency`](Self::iter_on_adjacency).
    pub fn neighbors(&self, i: usize) -> std::slice::Iter<'_, (usize, f32)> {
        self.iter_on_adjacency(i)
    }

    /// Returns the degree (number of adjacent edges) of vertex `i`.
    ///
    /// # Examples
    ///
    /// ```
    /// let g = optopus::common::Graph::from_edges([(0, 1, 1.0), (0, 2, 1.0), (1, 2, 1.0)]);
    /// assert_eq!(g.degree(0), 2);
    /// assert_eq!(g.degree(1), 2);
    /// ```
    pub fn degree(&self, i: usize) -> usize {
        if i < self.adj.len() {
            self.adj[i].len()
        } else {
            0
        }
    }

    /// Returns an iterator over all edges as `(i, j, weight)` with `i < j`.
    ///
    /// Each undirected edge appears exactly once.
    ///
    /// # Examples
    ///
    /// ```
    /// let g = optopus::common::Graph::from_edges([(0, 1, 1.0), (0, 2, 2.0)]);
    /// let edges: Vec<_> = g.edges().collect();
    /// assert_eq!(edges.len(), 2);
    /// ```
    pub fn edges(&self) -> impl Iterator<Item = (usize, usize, f32)> + '_ {
        self.adj.iter().enumerate().flat_map(|(i, neighbors)| {
            neighbors
                .iter()
                .filter(move |&&(j, _)| j > i)
                .map(move |&(j, w)| (i, j, w))
        })
    }

    /// Adds (or accumulates) the weight `w` on edge `(i, j)`.
    ///
    /// If the edge already exists, `w` is **added** to the current weight.
    /// To overwrite instead, use [`set_weight`](Self::set_weight).
    ///
    /// # Examples
    ///
    /// ```
    /// let mut g = optopus::common::Graph::new();
    /// g.add_weight(0, 1, 1.0);
    /// g.add_weight(0, 2, 1.0);
    /// g.add_weight(0, 1, 2.0);
    /// assert_eq!(g[(0, 1)], 3.0); // 1.0 + 2.0
    /// ```
    pub fn add_weight(&mut self, i: usize, j: usize, w: f32) {
        self.ensure_capacity(i.max(j) + 1);
        self.add_directed(i, j, w);
        self.add_directed(j, i, w);
        self.ensure_vertex(i);
        self.ensure_vertex(j);
    }

    /// Sets (overwrites) the weight of edge `(i, j)` to `w`.
    ///
    /// Unlike [`add_weight`](Self::add_weight), this replaces the existing weight
    /// rather than accumulating.
    ///
    /// # Examples
    ///
    /// ```
    /// let mut g = optopus::common::Graph::new();
    /// g.set_weight(0, 1, 5.0);
    /// g.set_weight(0, 1, 3.0);
    /// assert_eq!(g[(0, 1)], 3.0); // overwritten, not 8.0
    /// ```
    pub fn set_weight(&mut self, i: usize, j: usize, w: f32) {
        self.ensure_capacity(i.max(j) + 1);
        self.set_directed(i, j, w);
        if i != j {
            self.set_directed(j, i, w);
        }
        self.ensure_vertex(i);
        self.ensure_vertex(j);
    }

    /// Adds an unweighted edge `(i, j)` with weight 1.0.
    ///
    /// If the edge already exists, this is a no-op.
    ///
    /// # Examples
    ///
    /// ```
    /// let mut g = optopus::common::Graph::new();
    /// g.add_edge(0, 1);
    /// assert_eq!(g[(0, 1)], 1.0);
    /// assert!(g.has_edge(0, 1));
    /// ```
    pub fn add_edge(&mut self, i: usize, j: usize) {
        self.ensure_capacity(i.max(j) + 1);
        self.add_directed_dedup(i, j);
        self.add_directed_dedup(j, i);
        self.ensure_vertex(i);
        self.ensure_vertex(j);
    }

    /// Gets the weight of edge `(i, j)`, returning `0.0` if no such edge exists.
    ///
    /// You can also use the [`Index`](std::ops::Index) syntax: `g[(i, j)]`.
    ///
    /// # Examples
    ///
    /// ```
    /// let mut g = optopus::common::Graph::new();
    /// g.set_weight(0, 1, 1.0);
    /// assert_eq!(g.get_weight(0, 1), 1.0);
    /// assert_eq!(g[(0, 1)], 1.0);   // equivalent
    /// ```
    pub fn get_weight(&self, i: usize, j: usize) -> f32 {
        if i < self.adj.len() {
            self.adj[i]
                .binary_search_by_key(&j, |&(v, _)| v)
                .ok()
                .map(|idx| self.adj[i][idx].1)
                .unwrap_or(0.0)
        } else {
            0.0
        }
    }

    /// Returns `true` if an edge between `i` and `j` exists in the graph.
    ///
    /// # Examples
    ///
    /// ```
    /// let g = optopus::common::Graph::from_edges([(0, 1, 1.0)]);
    /// assert!(g.has_edge(0, 1));
    /// assert!(g.has_edge(1, 0));  // symmetric
    /// assert!(!g.has_edge(0, 2));
    /// ```
    pub fn has_edge(&self, i: usize, j: usize) -> bool {
        i < self.adj.len()
            && self.adj[i]
                .binary_search_by_key(&j, |&(v, _)| v)
                .is_ok()
    }

    /// Loads a graph from a file.
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
    /// - `N` -- number of vertices, `M` -- number of edges
    /// - Each edge line: vertex `i`, vertex `j`, weight `w` (space-separated)
    /// - Weight is optional and defaults to 1.0 if absent
    /// - Vertices are 1-indexed in the file and automatically converted to 0-indexed internally
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use optopus::common::Graph;
    ///
    /// let g = Graph::load_from_file("data/max_cut/G1").unwrap();
    /// println!("{g}");
    /// ```
    pub fn load_from_file(filename: &str) -> Result<Self, crate::error::OptError> {
        use crate::error::OptError;
        use std::io::BufRead;

        let err = |line: usize, detail: String| OptError::FileLoad {
            path: filename.to_string(),
            line,
            detail,
        };

        let file = std::fs::File::open(filename)
            .map_err(|e| err(0, format!("failed to open file: {e}")))?;
        let reader = std::io::BufReader::new(file);
        let mut line_iter = reader.lines();

        // parse the number of vertices and edges
        let (n, _) = {
            let line = line_iter
                .next()
                .ok_or_else(|| err(1, "file is empty, expected header 'N M'".into()))?
                .map_err(|e| err(1, format!("failed to read header line: {e}")))?;
            let mut iter = line.split_whitespace();
            let n = iter
                .next()
                .ok_or_else(|| err(1, "expected header 'N M', but line is empty".into()))?
                .parse::<usize>()
                .map_err(|e| err(1, format!("failed to parse vertex count N: {e}")))?;
            let m = iter
                .next()
                .ok_or_else(|| {
                    err(
                        1,
                        "expected header 'N M', but edge count M is missing".into(),
                    )
                })?
                .parse::<usize>()
                .map_err(|e| err(1, format!("failed to parse edge count M: {e}")))?;
            (n, m)
        };

        let mut g = Graph {
            adj: vec![vec![]; n],
            vertices: Vec::new(),
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
                        "expected edge 'i j [w]', but vertex i is missing".into(),
                    )
                })?
                .parse::<usize>()
                .map_err(|e| err(line_num, format!("failed to parse vertex i: {e}")))?;
            if i == 0 {
                return Err(err(
                    line_num,
                    "vertex index i must be >= 1 (1-indexed)".into(),
                ));
            }
            let i = i - 1;
            let j = iter
                .next()
                .ok_or_else(|| {
                    err(
                        line_num,
                        "expected edge 'i j [w]', but vertex j is missing".into(),
                    )
                })?
                .parse::<usize>()
                .map_err(|e| err(line_num, format!("failed to parse vertex j: {e}")))?;
            if j == 0 {
                return Err(err(
                    line_num,
                    "vertex index j must be >= 1 (1-indexed)".into(),
                ));
            }
            let j = j - 1;
            let w = iter
                .next()
                .and_then(|s| s.parse::<f32>().ok())
                .unwrap_or(1.0);
            // File-loaded instances never have duplicate edges, so push directly.
            g.adj[i].push((j, w));
            g.adj[j].push((i, w));
        }

        // Sort adjacency lists for binary search
        for neighbors in &mut g.adj {
            neighbors.sort_by_key(|&(v, _)| v);
        }

        g.vertices = (0..n).filter(|&i| !g.adj[i].is_empty()).collect();
        Ok(g)
    }

    fn ensure_capacity(&mut self, n: usize) {
        if self.adj.len() < n {
            self.adj.resize_with(n, Vec::new);
        }
    }

    fn ensure_vertex(&mut self, v: usize) {
        if let Err(pos) = self.vertices.binary_search(&v) {
            self.vertices.insert(pos, v);
        }
    }

    fn add_directed(&mut self, from: usize, to: usize, w: f32) {
        match self.adj[from].binary_search_by_key(&to, |&(v, _)| v) {
            Ok(idx) => self.adj[from][idx].1 += w,
            Err(idx) => self.adj[from].insert(idx, (to, w)),
        }
    }

    fn set_directed(&mut self, from: usize, to: usize, w: f32) {
        match self.adj[from].binary_search_by_key(&to, |&(v, _)| v) {
            Ok(idx) => self.adj[from][idx].1 = w,
            Err(idx) => self.adj[from].insert(idx, (to, w)),
        }
    }

    /// Insert edge (from, to) with weight 1.0 if not already present.
    fn add_directed_dedup(&mut self, from: usize, to: usize) {
        if let Err(idx) = self.adj[from].binary_search_by_key(&to, |&(v, _)| v) {
            self.adj[from].insert(idx, (to, 1.0));
        }
    }
}

impl std::ops::Index<(usize, usize)> for Graph {
    type Output = f32;

    /// Returns the weight of edge `(i, j)`, or `&0.0` if no such edge exists.
    ///
    /// # Examples
    ///
    /// ```
    /// let g = optopus::common::Graph::from_edges([(0, 1, 3.0)]);
    /// assert_eq!(g[(0, 1)], 3.0);
    /// assert_eq!(g[(1, 0)], 3.0); // symmetric
    /// assert_eq!(g[(0, 2)], 0.0); // non-existent edge
    /// ```
    fn index(&self, (i, j): (usize, usize)) -> &f32 {
        if i < self.adj.len() {
            if let Ok(idx) = self.adj[i].binary_search_by_key(&j, |&(v, _)| v) {
                return &self.adj[i][idx].1;
            }
        }
        &ZERO_WEIGHT
    }
}

/// Displays a summary of the graph: `Graph(vertices: N, edges: M)` or `Graph(empty)`.
///
/// # Examples
///
/// ```
/// let g = optopus::common::Graph::from_edges([(0, 1, 1.0), (1, 2, 2.0)]);
/// assert_eq!(format!("{g}"), "Graph(vertices: 3, edges: 2)");
///
/// let empty = optopus::common::Graph::new();
/// assert_eq!(format!("{empty}"), "Graph(empty)");
/// ```
impl std::fmt::Display for Graph {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.vertices.is_empty() {
            write!(f, "Graph(empty)")
        } else {
            write!(
                f,
                "Graph(vertices: {}, edges: {})",
                self.num_vertices(),
                self.num_edges(),
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_blank_graph() {
        let g = Graph::new();
        assert_eq!(g.len(), 0);
        assert!(g.is_empty());
    }

    #[test]
    fn test_set_and_get_weight() {
        let mut g = Graph::new();
        g.add_weight(0, 1, 1.0);
        g.add_weight(0, 2, 1.0);
        g.add_weight(0, 1, 2.0);

        assert_eq!(g.len(), 3);

        assert_eq!(g.get_weight(0, 1), 3.0);
        assert_eq!(g.get_weight(0, 2), 1.0);
    }

    #[test]
    fn test_set_weight_overwrites() {
        let mut g = Graph::new();
        g.set_weight(0, 1, 5.0);
        assert_eq!(g[(0, 1)], 5.0);

        g.set_weight(0, 1, 3.0);
        assert_eq!(g[(0, 1)], 3.0); // overwritten, not 8.0
        assert_eq!(g[(1, 0)], 3.0); // symmetric
    }

    #[test]
    fn test_set_weight_and_add_weight_interaction() {
        let mut g = Graph::new();
        g.set_weight(0, 1, 5.0);
        g.add_weight(0, 1, 2.0);
        assert_eq!(g[(0, 1)], 7.0); // 5.0 + 2.0

        g.set_weight(0, 1, 1.0); // overwrite back
        assert_eq!(g[(0, 1)], 1.0);
    }

    #[test]
    fn test_add_edge_dedup() {
        let mut g = Graph::new();
        g.add_edge(0, 1);
        g.add_edge(0, 1); // duplicate, should be no-op
        assert_eq!(g[(0, 1)], 1.0);
        assert_eq!(g.num_edges(), 1);
    }

    #[test]
    fn test_index_existing_edge() {
        let g = Graph::from_edges([(0, 1, 3.0), (1, 2, 7.0)]);
        assert_eq!(g[(0, 1)], 3.0);
        assert_eq!(g[(1, 0)], 3.0);
        assert_eq!(g[(1, 2)], 7.0);
    }

    #[test]
    fn test_index_missing_edge() {
        let g = Graph::from_edges([(0, 1, 1.0)]);
        assert_eq!(g[(0, 2)], 0.0);
        assert_eq!(g[(5, 6)], 0.0); // out of bounds
    }

    #[test]
    fn test_num_vertices_and_edges() {
        let g = Graph::from_edges([(0, 1, 1.0), (0, 2, 1.0), (1, 2, 1.0)]);
        assert_eq!(g.num_vertices(), 3);
        assert_eq!(g.num_edges(), 3);
    }

    #[test]
    fn test_is_empty() {
        let g = Graph::new();
        assert!(g.is_empty());

        let g = Graph::from_edges([(0, 1, 1.0)]);
        assert!(!g.is_empty());
    }

    #[test]
    fn test_from_edges() {
        let g = Graph::from_edges([(0, 1, 1.0), (0, 2, 2.0), (1, 2, 3.0)]);
        assert_eq!(g[(0, 1)], 1.0);
        assert_eq!(g[(0, 2)], 2.0);
        assert_eq!(g[(1, 2)], 3.0);
        assert_eq!(g.num_edges(), 3);
    }

    #[test]
    fn test_from_edges_duplicate_last_wins() {
        let g = Graph::from_edges([(0, 1, 1.0), (0, 1, 5.0)]);
        assert_eq!(g[(0, 1)], 5.0);
    }

    #[test]
    fn test_edges_iterator() {
        let g = Graph::from_edges([(0, 1, 1.0), (0, 2, 2.0), (1, 2, 3.0)]);
        let mut edges: Vec<_> = g.edges().collect();
        edges.sort_by_key(|&(i, j, _)| (i, j));
        assert_eq!(edges, vec![(0, 1, 1.0), (0, 2, 2.0), (1, 2, 3.0)]);
    }

    #[test]
    fn test_degree() {
        let g = Graph::from_edges([(0, 1, 1.0), (0, 2, 1.0), (1, 2, 1.0)]);
        assert_eq!(g.degree(0), 2);
        assert_eq!(g.degree(1), 2);
        assert_eq!(g.degree(2), 2);
        assert_eq!(g.degree(99), 0); // out of bounds
    }

    #[test]
    fn test_has_edge() {
        let g = Graph::from_edges([(0, 1, 1.0)]);
        assert!(g.has_edge(0, 1));
        assert!(g.has_edge(1, 0)); // symmetric
        assert!(!g.has_edge(0, 2));
    }

    #[test]
    fn test_display_empty() {
        let g = Graph::new();
        assert_eq!(format!("{g}"), "Graph(empty)");
    }

    #[test]
    fn test_display_nonempty() {
        let g = Graph::from_edges([(0, 1, 1.0), (0, 2, 1.0), (1, 2, 1.0)]);
        assert_eq!(format!("{g}"), "Graph(vertices: 3, edges: 3)");
    }
}
