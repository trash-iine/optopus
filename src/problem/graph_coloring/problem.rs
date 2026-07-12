use crate::common::Graph;
use crate::search_state::{Distance, ProblemTrait, Rankable};
use rand::Rng;

/// The Graph Coloring problem in a self-contained, penalty-augmented form.
///
/// Every vertex is assigned a color in `0..k`, where the palette size `k` is
/// derived from the graph (`max_degree + 1`, Brooks' bound, so a proper
/// coloring always exists). The objective, **minimized**, is
///
/// ```text
/// objective = colors_used + penalty_weight * conflicts
/// ```
///
/// where `conflicts` is the number of edges whose endpoints share a color and
/// `penalty_weight = n + 1`. With this weight, removing any one conflict always
/// beats any change in the number of colors used, so the global optimum is a
/// proper coloring that uses as few colors as possible (same trick as
/// [`crate::problem::VertexCover`]).
pub struct GraphColoring {
    /// The underlying graph.
    pub graph: Graph,
    /// Palette size (number of available colors).
    pub k: usize,
    /// Penalty applied to each conflicting edge (`n + 1`).
    pub penalty_weight: i64,
}

/// A solution for the Graph Coloring problem.
#[derive(Debug, Clone)]
pub struct GraphColoringSolution {
    /// `colors[v]` = color assigned to vertex `v`, in `0..k`.
    pub colors: Vec<usize>,
    /// Flat `n * k` matrix: `gamma[v * k + c]` = number of neighbors of `v`
    /// currently colored `c` (the TabuCol Γ matrix; drives O(1) gain deltas).
    pub(crate) gamma: Vec<u32>,
    /// `class_size[c]` = number of vertices currently colored `c`.
    pub(crate) class_size: Vec<usize>,
    /// Number of non-empty color classes.
    pub colors_used: usize,
    /// Number of edges whose endpoints share a color.
    pub conflicts: usize,
    /// Penalty-augmented objective: `colors_used + penalty_weight * conflicts`.
    pub objective: i64,
}

// === Rankable trait: solution comparison (minimize) ===
impl Rankable for GraphColoringSolution {
    fn is_better_than(&self, other: &Self) -> bool {
        self.objective < other.objective
    }
}

// === Distance trait: Hamming distance over color assignments ===
impl Distance for GraphColoringSolution {
    fn distance(&self, other: &Self) -> usize {
        self.colors
            .iter()
            .zip(&other.colors)
            .filter(|(a, b)| a != b)
            .count()
    }
}

impl GraphColoring {
    /// Creates a [`GraphColoring`] from a [`Graph`], deriving the palette size
    /// as `max_degree + 1` (minimum 1).
    pub fn new(graph: Graph) -> Self {
        let max_degree = (0..graph.len()).map(|v| graph.degree(v)).max().unwrap_or(0);
        let k = (max_degree + 1).max(1);
        Self::with_palette(graph, k)
    }

    /// Creates a [`GraphColoring`] with an explicit palette size `k`.
    ///
    /// # Panics
    ///
    /// Panics if `k == 0`.
    pub fn with_palette(graph: Graph, k: usize) -> Self {
        assert!(k >= 1, "palette size k must be >= 1");
        let penalty_weight = graph.len() as i64 + 1;
        Self {
            graph,
            k,
            penalty_weight,
        }
    }

    /// Loads a [`GraphColoring`] instance from a file in the `N M / i j w`
    /// edge-list format (weights are ignored; coloring is unweighted).
    pub fn load_file(path: impl AsRef<std::path::Path>) -> Result<Self, crate::error::OptError> {
        Graph::load_from_file(path).map(Self::new)
    }

    /// Builds a [`GraphColoringSolution`] from a color assignment, recomputing
    /// all cached state (`gamma`, `class_size`, `colors_used`, `conflicts`,
    /// `objective`) from scratch.
    pub fn solution_from_colors(&self, colors: Vec<usize>) -> GraphColoringSolution {
        let n = self.graph.len();
        let k = self.k;
        let mut gamma = vec![0u32; n * k];
        let mut class_size = vec![0usize; k];
        for &c in &colors {
            class_size[c] += 1;
        }
        let colors_used = class_size.iter().filter(|&&s| s > 0).count();
        let mut conflicts = 0usize;
        for (i, j, _w) in self.graph.edges() {
            if colors[i] == colors[j] {
                conflicts += 1;
            }
            gamma[i * k + colors[j]] += 1;
            gamma[j * k + colors[i]] += 1;
        }
        let objective = colors_used as i64 + self.penalty_weight * conflicts as i64;
        GraphColoringSolution {
            colors,
            gamma,
            class_size,
            colors_used,
            conflicts,
            objective,
        }
    }

    /// The single incremental-update primitive: recolor vertex `v` to
    /// `new_color`, updating every cached field in O(degree). Shared by the
    /// recolor and swap moves. A no-op if `v` already has `new_color`.
    pub(crate) fn recolor(&self, sol: &mut GraphColoringSolution, v: usize, new_color: usize) {
        let cur = sol.colors[v];
        if cur == new_color {
            return;
        }
        let k = self.k;

        // Conflict delta: neighbors currently equal to `cur` stop conflicting,
        // neighbors equal to `new_color` start conflicting.
        let removed = sol.gamma[v * k + cur] as usize;
        let added = sol.gamma[v * k + new_color] as usize;
        sol.conflicts = sol.conflicts + added - removed;

        // Neighbors see `v` change color.
        for &(u, _w) in self.graph.iter_on_adjacency(v) {
            sol.gamma[u * k + cur] -= 1;
            sol.gamma[u * k + new_color] += 1;
        }

        // Color-class bookkeeping.
        sol.class_size[cur] -= 1;
        if sol.class_size[cur] == 0 {
            sol.colors_used -= 1;
        }
        if sol.class_size[new_color] == 0 {
            sol.colors_used += 1;
        }
        sol.class_size[new_color] += 1;

        sol.colors[v] = new_color;
        sol.objective = sol.colors_used as i64 + self.penalty_weight * sol.conflicts as i64;
    }

    /// O(1) change in `objective` if vertex `v` is recolored to `new_color`.
    pub(crate) fn recolor_gain(
        &self,
        sol: &GraphColoringSolution,
        v: usize,
        new_color: usize,
    ) -> i64 {
        let cur = sol.colors[v];
        let k = self.k;
        let d_conflicts = sol.gamma[v * k + new_color] as i64 - sol.gamma[v * k + cur] as i64;
        let mut d_colors: i64 = 0;
        if sol.class_size[cur] == 1 {
            d_colors -= 1; // `cur` becomes empty
        }
        if sol.class_size[new_color] == 0 {
            d_colors += 1; // `new_color` becomes non-empty
        }
        d_colors + self.penalty_weight * d_conflicts
    }

    /// O(1) change in `objective` if the colors of `i` and `j` are exchanged.
    ///
    /// Requires `colors[i] != colors[j]`. The number of colors used is
    /// unchanged by a swap, so only the conflict term contributes.
    pub(crate) fn swap_gain(&self, sol: &GraphColoringSolution, i: usize, j: usize) -> i64 {
        let k = self.k;
        let ci = sol.colors[i];
        let cj = sol.colors[j];
        // If `i` and `j` are adjacent, each `gamma` row counts the other vertex,
        // producing two phantom conflicts (they move simultaneously) — correct
        // by subtracting 2.
        let adjacent = if self.graph.has_edge(i, j) { 1i64 } else { 0 };
        let d_conflicts = (sol.gamma[i * k + cj] as i64 - sol.gamma[i * k + ci] as i64)
            + (sol.gamma[j * k + ci] as i64 - sol.gamma[j * k + cj] as i64)
            - 2 * adjacent;
        self.penalty_weight * d_conflicts
    }
}

// === ProblemTrait: problem interface ===
impl ProblemTrait for GraphColoring {
    type Solution = GraphColoringSolution;

    fn new_solution(&self, rng: &mut impl Rng) -> Self::Solution {
        let n = self.graph.len();
        let colors = (0..n).map(|_| rng.random_range(0..self.k)).collect();
        self.solution_from_colors(colors)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::problem::graph_coloring::{GraphColoringRecolorNeighbor, GraphColoringSwapNeighbor};
    use crate::search_state::MoveToNeighbor;
    use rand::SeedableRng;
    use rand::rngs::SmallRng;

    /// Triangle + a pendant vertex: forces at least 3 colors on the triangle.
    fn sample() -> GraphColoring {
        let g = Graph::from_edges([(0, 1, 1.0), (1, 2, 1.0), (2, 0, 1.0), (2, 3, 1.0)]);
        GraphColoring::new(g)
    }

    fn recompute(prob: &GraphColoring, sol: &GraphColoringSolution) -> GraphColoringSolution {
        prob.solution_from_colors(sol.colors.clone())
    }

    fn assert_consistent(prob: &GraphColoring, sol: &GraphColoringSolution) {
        let fresh = recompute(prob, sol);
        assert_eq!(sol.gamma, fresh.gamma, "gamma drifted");
        assert_eq!(sol.class_size, fresh.class_size, "class_size drifted");
        assert_eq!(sol.colors_used, fresh.colors_used, "colors_used drifted");
        assert_eq!(sol.conflicts, fresh.conflicts, "conflicts drifted");
        assert_eq!(sol.objective, fresh.objective, "objective drifted");
    }

    #[test]
    fn test_palette_from_max_degree() {
        let prob = sample();
        // vertex 2 has degree 3 -> k = 4
        assert_eq!(prob.k, 4);
        assert_eq!(prob.penalty_weight, prob.graph.len() as i64 + 1);
    }

    #[test]
    fn test_new_solution_valid_and_consistent() {
        let prob = sample();
        let mut rng = SmallRng::seed_from_u64(42);
        let sol = prob.new_solution(&mut rng);
        assert_eq!(sol.colors.len(), prob.graph.len());
        assert!(sol.colors.iter().all(|&c| c < prob.k));
        assert_consistent(&prob, &sol);
    }

    #[test]
    fn test_proper_coloring_zero_conflicts() {
        let prob = sample();
        // 0->0, 1->1, 2->2, 3->0 : proper, 3 colors used.
        let sol = prob.solution_from_colors(vec![0, 1, 2, 0]);
        assert_eq!(sol.conflicts, 0);
        assert_eq!(sol.colors_used, 3);
        assert_eq!(sol.objective, 3);
    }

    #[test]
    fn test_recolor_incremental_matches_recompute_and_gain() {
        let prob = sample();
        let mut rng = SmallRng::seed_from_u64(7);
        let mut sol = prob.new_solution(&mut rng);

        for _ in 0..200 {
            let moves: Vec<_> = GraphColoringRecolorNeighbor::iter(&prob, &sol).collect();
            let m = moves[rng.random_range(0..moves.len())];
            let before = sol.objective;
            let gain = m.gain;
            m.apply_to_solution(&prob, &mut sol).unwrap();
            assert_eq!(before + gain, sol.objective, "recolor gain mismatch");
            assert_consistent(&prob, &sol);
        }
    }

    #[test]
    fn test_swap_incremental_matches_recompute_and_gain() {
        let prob = sample();
        let mut rng = SmallRng::seed_from_u64(9);
        let mut sol = prob.new_solution(&mut rng);

        for _ in 0..200 {
            let moves: Vec<_> = GraphColoringSwapNeighbor::iter(&prob, &sol).collect();
            if moves.is_empty() {
                break;
            }
            let m = moves[rng.random_range(0..moves.len())];
            let before = sol.objective;
            let colors_used_before = sol.colors_used;
            let gain = m.gain;
            m.apply_to_solution(&prob, &mut sol).unwrap();
            assert_eq!(before + gain, sol.objective, "swap gain mismatch");
            assert_eq!(
                colors_used_before, sol.colors_used,
                "swap must not change colors_used"
            );
            assert_consistent(&prob, &sol);
        }
    }

    #[test]
    fn test_crossover_child_valid() {
        use crate::problem::graph_coloring::GraphColoringUniformCrossover;
        use crate::search_state::Crossover;

        let prob = sample();
        let p1 = prob.solution_from_colors(vec![0, 1, 2, 0]);
        let p2 = prob.solution_from_colors(vec![1, 2, 0, 1]);
        let mut rng = SmallRng::seed_from_u64(3);
        let mut xover = GraphColoringUniformCrossover;
        let child = xover.crossover(&prob, &p1, &p2, &mut rng).unwrap();
        for v in 0..prob.graph.len() {
            assert!(child.colors[v] == p1.colors[v] || child.colors[v] == p2.colors[v]);
        }
        assert_consistent(&prob, &child);
    }
}
