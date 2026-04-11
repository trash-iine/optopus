use crate::search_state::{ProblemTrait, Rankable};

/// The MaxCut problem.
///
/// Given an undirected weighted graph, MaxCut seeks a partition of the vertices into
/// two sets that maximizes the total weight of edges crossing the partition.
pub struct MaxCut {
    /// `adj[i]` = list of `(j, weight)` for all neighbours of vertex `i`.
    adj: Vec<Vec<(usize, f32)>>,
    /// Sorted list of vertex IDs that appear in the graph (used by `iter_on_vertices`).
    pub(super) vertices: Vec<usize>,
}

/// A solution for the MaxCut problem.
///
/// `Clone` preserves the full `positive_gain` index state: a cloned solution
/// with the index enabled does not need a rebuild.
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
    /// Whether the `positive_gain` index is enabled.
    /// When `false`, `update_positive_gain_membership` is a no-op.
    pub(crate) positive_gain_enabled: bool,
    /// Unordered list of vertices `v` with `gain[v] > 0`. Only maintained when
    /// `positive_gain_enabled` is `true`. Call [`enable_positive_gain_index`](Self::enable_positive_gain_index)
    /// to activate.
    pub(crate) positive_gain: Vec<usize>,
    /// Inverse index: `positive_gain_pos[v]` = position of `v` in `positive_gain`,
    /// or `-1` if `v` is not currently in the list.
    pub(crate) positive_gain_pos: Vec<i32>,
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

    /// Builds a [`MaxCutSolution`] from pre-computed components.
    ///
    /// The `positive_gain` index is **not** initialised; call
    /// [`enable_positive_gain_index`](Self::enable_positive_gain_index) to activate it.
    pub fn new(cut: Vec<bool>, gain: Vec<f32>, objective: f32) -> Self {
        Self {
            cut,
            gain,
            objective,
            positive_gain_enabled: false,
            positive_gain: Vec::new(),
            positive_gain_pos: Vec::new(),
        }
    }

    /// Enables the `positive_gain` index, building it from the current `gain` vector.
    ///
    /// If already enabled, this is a no-op. O(n).
    pub fn enable_positive_gain_index(&mut self) {
        if self.positive_gain_enabled {
            return;
        }
        self.positive_gain_enabled = true;
        let n = self.gain.len();
        self.positive_gain.clear();
        self.positive_gain_pos = vec![-1i32; n];
        for (v, &g) in self.gain.iter().enumerate() {
            if g > 0.0 {
                self.positive_gain_pos[v] = self.positive_gain.len() as i32;
                self.positive_gain.push(v);
            }
        }
    }

    /// Records that vertex `v`'s gain is changing from `self.gain[v]` to `new_gain`.
    /// Updates membership of `v` in the `positive_gain` index (does **not** write
    /// `self.gain[v]` — the caller is expected to do that).
    ///
    /// No-op when the index is not enabled.
    #[inline]
    pub(crate) fn update_positive_gain_membership(&mut self, v: usize, new_gain: f32) {
        if !self.positive_gain_enabled {
            return;
        }
        let was_positive = self.gain[v] > 0.0;
        let is_positive = new_gain > 0.0;
        if was_positive == is_positive {
            return;
        }
        if is_positive {
            self.positive_gain_pos[v] = self.positive_gain.len() as i32;
            self.positive_gain.push(v);
        } else {
            let pos = self.positive_gain_pos[v] as usize;
            let last = *self.positive_gain.last().expect("positive_gain non-empty");
            self.positive_gain.swap_remove(pos);
            if last != v {
                self.positive_gain_pos[last] = pos as i32;
            }
            self.positive_gain_pos[v] = -1;
        }
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
        MaxCutSolution::new(cut, gain, objective)
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

    // ---------------------------------------------------------------------------
    // Helper: 3-vertex triangle (unit weights), all-false cut.
    // gain[v] = total edge weight of v = 2.0 for each vertex (positive for all).
    // ---------------------------------------------------------------------------
    fn make_triangle_solution() -> (MaxCut, MaxCutSolution) {
        let mut mc = MaxCut::new();
        mc.add_weight(0, 1, 1.0);
        mc.add_weight(1, 2, 1.0);
        mc.add_weight(0, 2, 1.0);
        let n = mc.len(); // 3
        let cut = vec![false; n];
        let gain: Vec<f32> = (0..n).map(|v| mc.calculate_gain(&cut, v)).collect();
        let objective = mc.calculate_cut_size(&cut);
        let sol = MaxCutSolution::new(cut, gain, objective);
        (mc, sol)
    }

    // 1. from_parts() creates a solution with positive_gain_enabled == false
    //    and empty positive_gain / positive_gain_pos vecs.
    #[test]
    fn test_from_parts_index_disabled_by_default() {
        let (_mc, sol) = make_triangle_solution();
        // The index must not be built yet.
        assert!(
            !sol.positive_gain_enabled,
            "index should be disabled after from_parts"
        );
        assert!(
            sol.positive_gain.is_empty(),
            "positive_gain vec must be empty"
        );
        assert!(
            sol.positive_gain_pos.is_empty(),
            "positive_gain_pos vec must be empty"
        );
    }

    // 2. enable_positive_gain_index() correctly builds the index from gain[].
    //    All-false cut on a unit-weight triangle: gain[v] = 2.0 > 0 for every vertex,
    //    so all three vertices must appear in positive_gain.
    #[test]
    fn test_enable_positive_gain_index_builds_correctly() {
        let (_mc, mut sol) = make_triangle_solution();
        sol.enable_positive_gain_index();

        assert!(
            sol.positive_gain_enabled,
            "index should be enabled after call"
        );
        // All gains are 2.0 > 0, so all three vertices appear.
        let mut listed = sol.positive_gain.clone();
        listed.sort();
        assert_eq!(
            listed,
            vec![0, 1, 2],
            "all vertices should be in positive_gain"
        );
        // Inverse index must be consistent.
        for &v in &sol.positive_gain {
            let pos = sol.positive_gain_pos[v] as usize;
            assert_eq!(
                sol.positive_gain[pos], v,
                "positive_gain_pos[{v}] must point back to {v}"
            );
        }
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
        let mut mc = MaxCut::new();
        mc.add_weight(0, 1, 1.0);
        mc.add_weight(1, 2, 1.0);
        mc.add_weight(0, 2, 1.0);
        let n = mc.len();
        let cut = vec![true, false, false];
        let gain: Vec<f32> = (0..n).map(|v| mc.calculate_gain(&cut, v)).collect();
        let objective = mc.calculate_cut_size(&cut);
        let mut sol = MaxCutSolution::new(cut, gain, objective);

        sol.enable_positive_gain_index();

        // gain[0]=-2, gain[1]=0, gain[2]=0 — none are strictly positive.
        assert!(
            sol.positive_gain.is_empty(),
            "no vertex has gain > 0; positive_gain must be empty"
        );
        // positive_gain_pos vec must be length n and all -1.
        assert_eq!(sol.positive_gain_pos.len(), n);
        for v in 0..n {
            assert_eq!(
                sol.positive_gain_pos[v], -1,
                "positive_gain_pos[{v}] must be -1 when vertex is absent"
            );
        }
    }

    // 3. enable_positive_gain_index() is idempotent: calling it twice must not
    //    corrupt the index (no duplicates, consistent inverse).
    #[test]
    fn test_enable_positive_gain_index_idempotent() {
        let (_mc, mut sol) = make_triangle_solution();
        sol.enable_positive_gain_index();
        let pg_after_first = sol.positive_gain.clone();
        let pg_pos_after_first = sol.positive_gain_pos.clone();

        sol.enable_positive_gain_index();

        assert_eq!(
            sol.positive_gain, pg_after_first,
            "second call must not change positive_gain"
        );
        assert_eq!(
            sol.positive_gain_pos, pg_pos_after_first,
            "second call must not change positive_gain_pos"
        );
    }

    // 4. update_positive_gain_membership() is a no-op when the index is disabled.
    //    positive_gain and positive_gain_pos must remain empty even after the call.
    #[test]
    fn test_update_positive_gain_membership_noop_when_disabled() {
        let (_mc, mut sol) = make_triangle_solution();
        // Index not enabled; calling the update must not populate the vecs.
        sol.update_positive_gain_membership(0, 5.0);
        assert!(
            sol.positive_gain.is_empty(),
            "positive_gain must stay empty when index is disabled"
        );
        assert!(
            sol.positive_gain_pos.is_empty(),
            "positive_gain_pos must stay empty when index is disabled"
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
            !sol.positive_gain.contains(&0),
            "vertex 0 must leave positive_gain when gain becomes negative"
        );
        assert_eq!(
            sol.positive_gain_pos[0], -1,
            "inverse index must be -1 for absent vertex"
        );
        assert_eq!(sol.positive_gain.len(), 2);

        // Verify inverse consistency for remaining vertices.
        for &v in &sol.positive_gain {
            let pos = sol.positive_gain_pos[v] as usize;
            assert_eq!(
                sol.positive_gain[pos], v,
                "inverse must be consistent after removal"
            );
        }

        // Simulate: vertex 0's gain changes back to 3.0 (should re-enter the index).
        sol.update_positive_gain_membership(0, 3.0);
        sol.gain[0] = 3.0;
        assert!(
            sol.positive_gain.contains(&0),
            "vertex 0 must re-enter positive_gain when gain becomes positive"
        );
        assert_eq!(sol.positive_gain.len(), 3);
        for &v in &sol.positive_gain {
            let pos = sol.positive_gain_pos[v] as usize;
            assert_eq!(
                sol.positive_gain[pos], v,
                "inverse must be consistent after re-insertion"
            );
        }
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
        // Verify the inverse index is all -1.
        for v in 0..sol.positive_gain_pos.len() {
            assert_eq!(
                sol.positive_gain_pos[v], -1,
                "positive_gain_pos[{v}] must be -1 when index is empty"
            );
        }
    }

    // 6. Clone of an enabled solution preserves positive_gain_enabled == true
    //    and the index content intact.
    #[test]
    fn test_clone_preserves_positive_gain_index() {
        let (_mc, mut sol) = make_triangle_solution();
        sol.enable_positive_gain_index();

        let cloned = sol.clone();

        assert!(
            cloned.positive_gain_enabled,
            "clone must inherit positive_gain_enabled == true"
        );
        let mut orig_sorted = sol.positive_gain.clone();
        orig_sorted.sort();
        let mut clone_sorted = cloned.positive_gain.clone();
        clone_sorted.sort();
        assert_eq!(
            orig_sorted, clone_sorted,
            "clone must have the same positive_gain contents"
        );
        assert_eq!(
            cloned.positive_gain_pos, sol.positive_gain_pos,
            "clone must have the same positive_gain_pos contents"
        );
    }

    // 6b. Clone of a disabled solution preserves positive_gain_enabled == false.
    #[test]
    fn test_clone_preserves_disabled_state() {
        let (_mc, sol) = make_triangle_solution();
        assert!(!sol.positive_gain_enabled);

        let cloned = sol.clone();
        assert!(
            !cloned.positive_gain_enabled,
            "clone must preserve positive_gain_enabled == false"
        );
        assert!(cloned.positive_gain.is_empty());
        assert!(cloned.positive_gain_pos.is_empty());
    }
}
