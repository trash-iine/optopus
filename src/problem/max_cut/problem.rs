use crate::search_state::{ProblemTrait, Rankable};

/// The MaxCut problem.
///
/// Given an undirected weighted graph, MaxCut seeks a partition of the vertices into
/// two sets that maximizes the total weight of edges crossing the partition.
pub struct MaxCut {
    /// `adj[i]` = list of `(j, weight)` for all neighbours of vertex `i`.
    adj: Vec<Vec<(usize, f32)>>,
    /// Sorted list of vertex IDs that appear in the graph (used by `iter_on_vertices`).
    vertices: Vec<usize>,
}

/// A solution for the MaxCut problem.
#[derive(Debug, Clone)]
pub struct MaxCutSolution {
    /// The cut assignment for each vertex: `cut[i]` is the side of vertex `i`.
    /// Sized to `max_vertex_id + 1`; only indices in `MaxCut::vertices` are meaningful.
    pub cut: Vec<bool>,
    /// The gain of flipping each vertex: `gain[i]` = change in cut weight when flipping `i`.
    /// Sized to `max_vertex_id + 1`.
    pub gain: Vec<f32>,
    /// The total weight of edges crossing the cut.
    pub objective: f32,
}

impl Rankable for MaxCutSolution {
    fn is_better_than(&self, other: &Self) -> bool {
        self.objective > other.objective
    }
}

impl MaxCutSolution {
    /// Iterates over all vertices in the solution.
    pub fn iter_on_vertices(&self) -> impl Iterator<Item = usize> + '_ {
        0..self.cut.len()
    }
}

impl MaxCut {
    /// Creates a new empty [`MaxCut`].
    ///
    /// # Examples
    ///
    /// ```
    /// let mc = optopus::problem::MaxCut::new();
    /// ```
    pub fn new() -> Self {
        Self {
            adj: vec![],
            vertices: vec![],
        }
    }

    /// Returns the number of vertices in the graph (`max_vertex_id + 1`).
    ///
    /// # Examples
    ///
    /// ```
    /// let mut mc = optopus::problem::MaxCut::new();
    /// assert_eq!(mc.len(), 0);
    ///
    /// mc.add_weight(0, 1, 1.0);
    /// mc.add_weight(0, 2, 1.0);
    /// assert_eq!(mc.len(), 3);
    /// ```
    pub fn len(&self) -> usize {
        self.adj.len()
    }

    /// Returns an iterator visiting all vertices that have at least one edge.
    ///
    /// # Examples
    ///
    /// ```
    /// let mut mc = optopus::problem::MaxCut::new();
    /// mc.add_weight(0, 1, 1.0);
    /// mc.add_weight(0, 2, 1.0);
    ///
    /// for i in mc.iter_on_vertices() {
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
    /// let mut mc = optopus::problem::MaxCut::new();
    /// mc.add_weight(0, 1, 1.0);
    /// mc.add_weight(0, 2, 1.0);
    /// mc.add_weight(1, 2, 1.0);
    /// for &(j, w) in mc.iter_on_adjacency(0) {
    ///     println!("{} {}", j, w); // 1 1.0, 2 1.0
    /// }
    /// ```
    pub fn iter_on_adjacency(&self, i: usize) -> std::slice::Iter<'_, (usize, f32)> {
        if i < self.adj.len() {
            self.adj[i].iter()
        } else {
            [].iter()
        }
    }

    /// Adds (or accumulates) the weight `w` on edge `(i, j)`.
    ///
    /// # Examples
    ///
    /// ```
    /// let mut mc = optopus::problem::MaxCut::new();
    /// mc.add_weight(0, 1, 1.0);
    /// mc.add_weight(0, 2, 1.0);
    /// mc.add_weight(0, 1, 2.0);
    /// ```
    pub fn add_weight(&mut self, i: usize, j: usize, w: f32) {
        let n = i.max(j) + 1;
        if self.adj.len() < n {
            self.adj.resize_with(n, Vec::new);
        }
        self.add_directed(i, j, w);
        self.add_directed(j, i, w);
        // Update the sorted vertex list for any newly seen vertices.
        for &v in &[i, j] {
            if let Err(pos) = self.vertices.binary_search(&v) {
                self.vertices.insert(pos, v);
            }
        }
    }

    fn add_directed(&mut self, from: usize, to: usize, w: f32) {
        if let Some(entry) = self.adj[from].iter_mut().find(|(v, _)| *v == to) {
            entry.1 += w;
        } else {
            self.adj[from].push((to, w));
        }
    }

    /// Gets the weight of edge `(i, j)`, returning `0.0` if no such edge exists.
    ///
    /// # Examples
    ///
    /// ```
    /// let mut mc = optopus::problem::MaxCut::new();
    /// mc.add_weight(0, 1, 1.0);
    /// mc.add_weight(0, 2, 1.0);
    ///
    /// assert_eq!(mc.get_weight(0, 1), 1.0);
    ///
    /// mc.add_weight(0, 1, 2.0); // allows to add weight to existing edge
    /// assert_eq!(mc.get_weight(0, 1), 3.0);
    /// ```
    pub fn get_weight(&self, i: usize, j: usize) -> f32 {
        if i < self.adj.len() {
            self.adj[i]
                .iter()
                .find(|(v, _)| *v == j)
                .map(|(_, w)| *w)
                .unwrap_or(0.0)
        } else {
            0.0
        }
    }

    /// Returns `true` if there is a non-zero-weight edge between `i` and `j`.
    ///
    /// # Examples
    ///
    /// ```
    /// let mut mc = optopus::problem::MaxCut::new();
    /// mc.add_weight(0, 1, 1.0);
    /// assert!(mc.has_edge(0, 1));
    /// assert!(!mc.has_edge(0, 2));
    /// ```
    pub fn has_edge(&self, i: usize, j: usize) -> bool {
        i < self.adj.len() && self.adj[i].iter().any(|(v, _)| *v == j)
    }

    /// Calculates the cut size of the given assignment slice (indexed by vertex ID).
    pub fn calculate_cut_size(&self, cut: &[bool]) -> f32 {
        let mut ret = 0.0;
        for &i in &self.vertices {
            let bi = cut[i];
            for &(j, w) in &self.adj[i] {
                if bi ^ cut[j] {
                    ret += w;
                }
            }
        }
        ret / 2.0
    }

    /// Loads a MaxCut problem instance from a file.
    /// The file format should be as follows:
    /// ```skip
    /// N M
    /// i j w
    /// i j w
    /// ...
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

        let mut mc = MaxCut {
            adj: vec![vec![]; n],
            vertices: (0..n).collect(),
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
                        "expected edge 'i j w', but vertex i is missing".into(),
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
                        "expected edge 'i j w', but vertex j is missing".into(),
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
                .ok_or_else(|| {
                    err(
                        line_num,
                        "expected edge 'i j w', but weight w is missing".into(),
                    )
                })?
                .parse::<f32>()
                .map_err(|e| err(line_num, format!("failed to parse edge weight w: {e}")))?;
            // File-loaded instances never have duplicate edges, so push directly.
            mc.adj[i].push((j, w));
            mc.adj[j].push((i, w));
        }

        Ok(mc)
    }

    /// Calculates the gain of flipping vertex `i` given the current cut assignment slice.
    pub fn calculate_gain(&self, cut: &[bool], i: usize) -> f32 {
        let bi = cut[i];
        self.adj[i]
            .iter()
            .map(|&(j, w)| if bi ^ cut[j] { -w } else { w })
            .sum()
    }
}

impl ProblemTrait for MaxCut {
    type Solution = MaxCutSolution;
    fn new_solution(&self, rng: &mut impl rand::Rng) -> Self::Solution {
        let n = self.adj.len();
        let mut cut = vec![false; n];
        for &i in &self.vertices {
            cut[i] = rng.random_bool(0.5);
        }
        let mut gain = vec![0.0; n];
        for &i in &self.vertices {
            gain[i] = self.calculate_gain(&cut, i);
        }
        let objective = self.calculate_cut_size(&cut);
        MaxCutSolution {
            cut,
            gain,
            objective,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_blank_graph() {
        let mc = MaxCut::new();
        assert_eq!(mc.len(), 0);
    }

    #[test]
    fn test_set_and_get_weight() {
        let mut mc = MaxCut::new();
        mc.add_weight(0, 1, 1.0);
        mc.add_weight(0, 2, 1.0);
        mc.add_weight(0, 1, 2.0);

        assert_eq!(mc.len(), 3);

        assert_eq!(mc.get_weight(0, 1), 3.0);
        assert_eq!(mc.get_weight(0, 2), 1.0);
    }

    #[test]
    fn test_calculate_cut_size() {
        let mut mc = MaxCut::new();
        mc.add_weight(0, 1, 1.0);
        mc.add_weight(0, 2, 2.0);
        mc.add_weight(1, 2, 3.0);

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
        let mut mc = MaxCut::new();
        mc.add_weight(0, 1, 1.0);
        mc.add_weight(0, 2, 2.0);
        mc.add_weight(1, 2, 3.0);

        let cut = vec![false, false, false];
        assert_eq!(mc.calculate_gain(&cut, 0), 3.0);
        assert_eq!(mc.calculate_gain(&cut, 1), 4.0);
        assert_eq!(mc.calculate_gain(&cut, 2), 5.0);
    }
}
