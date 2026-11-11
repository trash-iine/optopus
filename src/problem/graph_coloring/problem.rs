use crate::building_blocks::instance::Graph;
use crate::building_blocks::representation::{IntVar, IntVars};
use crate::error::OptError;
use crate::search_state::{Distance, Evaluate, ProblemTrait};
use crate::trait_defs::{IntAssignment, IntSwap, KeyDelta, Rankable};
use rand::Rng;

/// The Graph Coloring problem in a self-contained, penalty-augmented form.
///
/// Every vertex is assigned a color in `0..k`, where the palette size `k` is
/// derived from the graph (`max_degree + 1`, Brooks' bound, so a proper
/// coloring always exists). Solutions rank by their conflicts first, the
/// edges whose endpoints share a color, and by the colors used second, so
/// the best solution is a proper coloring that uses as few colors as
/// possible.
///
/// The number a solution reports, minimized, is
///
/// ```text
/// objective = colors_used + penalty_weight * conflicts
/// ```
///
/// with `penalty_weight = n + 1`. At most `n` colors are used, so removing one
/// conflict always outweighs any change in the colors used, and this number
/// orders solutions exactly as the ranking does (as in
/// [`crate::problem::VertexCover`]). The searches that weigh how much worse a
/// candidate is read this number.
///
/// It is an [`IntAssignment`] over the two keys, one variable per vertex
/// ranging over the palette, so its moves are the generic ones.
/// [`IntChangeNeighbor<2>`](crate::problem::IntChangeNeighbor) recolors a
/// vertex and [`IntSwapNeighbor<2>`](crate::problem::IntSwapNeighbor)
/// exchanges the colors of two, each priced in O(1) from the solution's
/// caches.
pub struct GraphColoring {
    graph: Graph,
    k: usize,
    /// One variable per vertex over `0..k`, what the generic moves walk.
    domains: IntVars,
}

/// A solution for the Graph Coloring problem.
#[derive(Debug, Clone, Rankable, Evaluate, Distance)]
pub struct GraphColoringSolution {
    /// `colors[v]` = color assigned to vertex `v`, in `0..k`.
    #[distance(hamming)]
    pub colors: Vec<usize>,
    /// Flat `n * k` matrix: `gamma[v * k + c]` = number of neighbors of `v`
    /// currently colored `c` (the TabuCol Γ matrix; drives O(1) gain deltas).
    gamma: Vec<u32>,
    /// `class_size[c]` = number of vertices currently colored `c`.
    class_size: Vec<usize>,
    /// Number of edges whose endpoints share a color, the first thing
    /// solutions are ranked by.
    #[rankable(minimize)]
    pub conflicts: usize,
    /// Number of non-empty color classes, ranked by after the conflicts.
    #[rankable(minimize)]
    pub colors_used: usize,
    /// Penalty-augmented objective: `colors_used + penalty_weight * conflicts`.
    #[objective(minimize)]
    pub objective: i64,
}

impl GraphColoring {
    /// Creates a [`GraphColoring`] from a [`Graph`], deriving the palette size
    /// as `max_degree + 1`.
    pub fn new(graph: Graph) -> Self {
        let max_degree = (0..graph.len()).map(|v| graph.degree(v)).max().unwrap_or(0);
        let k = max_degree + 1;
        let domains = (0..graph.len())
            .map(|_| IntVar::new(0, k as i64 - 1))
            .collect();
        Self { graph, k, domains }
    }

    /// The underlying graph.
    pub fn graph(&self) -> &Graph {
        &self.graph
    }

    /// The palette size, the number of available colors.
    pub fn k(&self) -> usize {
        self.k
    }

    /// Returns the penalty weight applied to each conflicting edge (`graph.len() + 1`).
    pub fn penalty_weight(&self) -> i64 {
        self.graph.len() as i64 + 1
    }

    /// Loads a [`GraphColoring`] instance from a file in the `N M / i j w`
    /// edge-list format (weights are ignored; coloring is unweighted).
    pub fn load_file(path: impl AsRef<std::path::Path>) -> Result<Self, crate::error::OptError> {
        Graph::load_file(path).map(Self::new)
    }

    /// Builds a [`GraphColoringSolution`] from a color assignment, recomputing
    /// all cached state (`gamma`, `class_size`, `colors_used`, `conflicts`,
    /// `objective`) from scratch, for starting a search somewhere other than a
    /// random coloring.
    ///
    /// Fails unless `colors` gives every vertex a color in `0..k`.
    pub fn solution_from_colors(
        &self,
        colors: Vec<usize>,
    ) -> Result<GraphColoringSolution, OptError> {
        if colors.len() != self.graph.len() {
            return Err(OptError::InvalidState(format!(
                "{} colors given for {} vertices",
                colors.len(),
                self.graph.len()
            )));
        }
        if let Some((v, &c)) = colors.iter().enumerate().find(|&(_, &c)| c >= self.k) {
            return Err(OptError::InvalidState(format!(
                "vertex {v} has color {c}, outside the palette 0..{}",
                self.k
            )));
        }
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
        let objective = colors_used as i64 + self.penalty_weight() * conflicts as i64;
        Ok(GraphColoringSolution {
            colors,
            gamma,
            class_size,
            colors_used,
            conflicts,
            objective,
        })
    }

    /// O(1) change in `conflicts` if the colors of `i` and `j` are exchanged.
    ///
    /// Requires `colors[i] != colors[j]`. The number of colors used is
    /// unchanged by a swap.
    fn swap_conflicts_delta(&self, sol: &GraphColoringSolution, i: usize, j: usize) -> i64 {
        let k = self.k;
        let ci = sol.colors[i];
        let cj = sol.colors[j];
        // If `i` and `j` are adjacent, each `gamma` row counts the other vertex,
        // producing two phantom conflicts (they move simultaneously).
        let edge_correction = if self.graph.has_edge(i, j) { 2 } else { 0 };
        (sol.gamma[i * k + cj] as i64 - sol.gamma[i * k + ci] as i64)
            + (sol.gamma[j * k + ci] as i64 - sol.gamma[j * k + cj] as i64)
            - edge_correction
    }

    /// The change a move makes that changes `conflicts` by `d_conflicts` and
    /// `colors_used` by `d_colors`, as the two keys and the change in the
    /// penalized objective, which is minimized.
    #[inline]
    fn key_delta(&self, d_conflicts: i64, d_colors: i64) -> KeyDelta<2> {
        let (d_conflicts, d_colors) = (d_conflicts as f64, d_colors as f64);
        KeyDelta {
            keys: [d_conflicts, d_colors],
            cost: d_colors + self.penalty_weight() as f64 * d_conflicts,
        }
    }
}

impl IntAssignment<2> for GraphColoring {
    fn domains(&self) -> &IntVars {
        &self.domains
    }

    #[inline]
    fn get(sol: &GraphColoringSolution, i: usize) -> i64 {
        sol.colors[i] as i64
    }

    /// Recolors vertex `v` and updates every cached field in O(degree), and
    /// does nothing if `v` already has that color. The default `assign_swap`
    /// calls it twice.
    #[inline]
    fn assign(&self, sol: &mut GraphColoringSolution, v: usize, new_color: i64) {
        let new_color = new_color as usize;
        let cur = sol.colors[v];
        if cur == new_color {
            return;
        }
        let k = self.k;

        // Neighbors colored `cur` stop conflicting with `v`, and those colored
        // `new_color` start.
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
        sol.objective = sol.colors_used as i64 + self.penalty_weight() * sol.conflicts as i64;
    }

    /// Prices a recolor of `v` in O(1) from its row of `gamma` and the class
    /// sizes.
    #[inline]
    fn assign_delta(&self, sol: &GraphColoringSolution, v: usize, new_color: i64) -> KeyDelta<2> {
        let new_color = new_color as usize;
        let cur = sol.colors[v];
        let k = self.k;
        let d_conflicts = sol.gamma[v * k + new_color] as i64 - sol.gamma[v * k + cur] as i64;
        // `new_color` becomes non-empty, `cur` becomes empty.
        let d_colors =
            i64::from(sol.class_size[new_color] == 0) - i64::from(sol.class_size[cur] == 1);
        self.key_delta(d_conflicts, d_colors)
    }
}

/// Exchanging the colors of two vertices. Reversing a range of vertices means
/// nothing on a coloring, so `GraphColoring` is not an `IntReverse`.
impl IntSwap<2> for GraphColoring {
    /// A swap of two differently colored vertices leaves the colors used as
    /// they are, and corrects for the two being adjacent.
    #[inline]
    fn assign_swap_delta(&self, sol: &GraphColoringSolution, i: usize, j: usize) -> KeyDelta<2> {
        self.key_delta(self.swap_conflicts_delta(sol, i, j), 0)
    }
}

// === ProblemTrait: problem interface ===
impl ProblemTrait for GraphColoring {
    type Solution = GraphColoringSolution;

    fn new_solution(&self, rng: &mut impl Rng) -> Self::Solution {
        let n = self.graph.len();
        let colors = (0..n).map(|_| rng.random_range(0..self.k)).collect();
        self.solution_from_colors(colors)
            .expect("random colors lie in the palette")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::problem::{IntChangeNeighbor, IntSwapNeighbor};
    use crate::search_state::MoveToNeighbor;
    use rand::SeedableRng;
    use rand::rngs::SmallRng;

    /// Triangle + a pendant vertex: forces at least 3 colors on the triangle.
    fn sample() -> GraphColoring {
        let g = Graph::from_edges([(0, 1, 1.0), (1, 2, 1.0), (2, 0, 1.0), (2, 3, 1.0)]);
        GraphColoring::new(g)
    }

    fn assert_consistent(prob: &GraphColoring, sol: &GraphColoringSolution) {
        let fresh = prob.solution_from_colors(sol.colors.clone()).unwrap();
        assert_eq!(sol.gamma, fresh.gamma, "gamma drifted");
        assert_eq!(sol.class_size, fresh.class_size, "class_size drifted");
        assert_eq!(sol.colors_used, fresh.colors_used, "colors_used drifted");
        assert_eq!(sol.conflicts, fresh.conflicts, "conflicts drifted");
        assert_eq!(sol.objective, fresh.objective, "objective drifted");
    }

    /// A coloring of the wrong length or with a color outside the palette is
    /// an error, not an index panic.
    #[test]
    fn solution_from_colors_rejects_bad_colorings() {
        let prob = GraphColoring::new(Graph::from_edges([(0, 1, 1.0), (1, 2, 1.0)]));
        assert!(prob.solution_from_colors(vec![0, 1, 0]).is_ok());
        assert!(prob.solution_from_colors(vec![0, 1]).is_err());
        assert!(prob.solution_from_colors(vec![0, 1, prob.k]).is_err());
    }

    #[test]
    fn test_palette_from_max_degree() {
        let prob = sample();
        // vertex 2 has degree 3 -> k = 4
        assert_eq!(prob.k, 4);
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
        let sol = prob.solution_from_colors(vec![0, 1, 2, 0]).unwrap();
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
            let moves: Vec<IntChangeNeighbor<2>> = IntChangeNeighbor::iter(&prob, &sol).collect();
            let m = moves[rng.random_range(0..moves.len())];
            let before = sol.objective;
            let (conflicts_before, colors_before) = (sol.conflicts, sol.colors_used);
            m.apply_to_solution(&prob, &mut sol).unwrap();
            assert_eq!(
                before as f64 + m.cost,
                sol.objective as f64,
                "recolor cost mismatch"
            );
            assert_eq!(
                sol.conflicts as f64 - conflicts_before as f64,
                m.keys[0],
                "recolor conflicts key mismatch"
            );
            assert_eq!(
                sol.colors_used as f64 - colors_before as f64,
                m.keys[1],
                "recolor colors key mismatch"
            );
            assert_consistent(&prob, &sol);
        }
    }

    #[test]
    fn test_swap_incremental_matches_recompute_and_gain() {
        let prob = sample();
        let mut rng = SmallRng::seed_from_u64(9);
        let mut sol = prob.new_solution(&mut rng);

        for _ in 0..200 {
            let moves: Vec<IntSwapNeighbor<2>> = IntSwapNeighbor::iter(&prob, &sol).collect();
            if moves.is_empty() {
                break;
            }
            let m = moves[rng.random_range(0..moves.len())];
            let before = sol.objective;
            let colors_used_before = sol.colors_used;
            let conflicts_before = sol.conflicts;
            m.apply_to_solution(&prob, &mut sol).unwrap();
            assert_eq!(
                before as f64 + m.cost,
                sol.objective as f64,
                "swap cost mismatch"
            );
            assert_eq!(
                sol.conflicts as f64 - conflicts_before as f64,
                m.keys[0],
                "swap conflicts key mismatch"
            );
            assert_eq!(m.keys[1], 0.0);
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
        let p1 = prob.solution_from_colors(vec![0, 1, 2, 0]).unwrap();
        let p2 = prob.solution_from_colors(vec![1, 2, 0, 1]).unwrap();
        let mut rng = SmallRng::seed_from_u64(3);
        let mut xover = GraphColoringUniformCrossover;
        let child = xover.crossover(&prob, &p1, &p2, &mut rng).unwrap();
        for v in 0..prob.graph.len() {
            assert!(child.colors[v] == p1.colors[v] || child.colors[v] == p2.colors[v]);
        }
        assert_consistent(&prob, &child);
    }
}
