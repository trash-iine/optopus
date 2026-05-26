use crate::common::Graph;
use crate::search_state::{Distance, ProblemTrait, Rankable};

/// The Minimum Vertex Cover problem.
///
/// Given an undirected graph, Vertex Cover seeks the smallest subset of vertices
/// such that every edge has at least one endpoint in the subset.
///
/// Hard feasibility is handled with a penalty: the augmented objective is
/// `|cover| + penalty_weight * uncovered_edges`, with `penalty_weight = n + 1`
/// so any optimum is feasible.
pub struct VertexCover {
    /// The underlying graph.
    pub graph: Graph,
}

/// A solution for the Vertex Cover problem.
#[derive(Debug, Clone)]
pub struct VertexCoverSolution {
    /// `cover[v] = true` iff vertex `v` is selected.
    pub cover: Vec<bool>,
    /// `gain[v]` = change in `objective` if vertex `v` is flipped (negative = improving).
    pub gain: Vec<i32>,
    /// Penalty-augmented objective: `cover_size + penalty_weight * uncovered_edges`.
    pub objective: i32,
    /// Number of vertices currently in the cover.
    pub cover_size: usize,
    /// Number of edges currently uncovered (both endpoints out of the cover).
    pub uncovered_edges: usize,
}

impl Rankable for VertexCoverSolution {
    fn is_better_than(&self, other: &Self) -> bool {
        self.objective < other.objective
    }
}

impl Distance for VertexCoverSolution {
    fn distance(&self, other: &Self) -> usize {
        self.cover
            .iter()
            .zip(other.cover.iter())
            .filter(|(a, b)| a != b)
            .count()
    }
}

impl VertexCover {
    /// Creates a [`VertexCover`] from a [`Graph`].
    pub fn new(graph: Graph) -> Self {
        Self { graph }
    }

    /// Loads a [`VertexCover`] instance from a file in the `N M / i j w` format.
    ///
    /// Wraps [`Graph::load_from_file`] and constructs the problem with
    /// [`VertexCover::new`]. Edge weights in the file are loaded into the
    /// graph but ignored by the vertex-cover objective (which counts
    /// uncovered edges, not their weights).
    pub fn load_file(path: impl AsRef<std::path::Path>) -> Result<Self, crate::error::OptError> {
        Graph::load_from_file(path).map(Self::new)
    }

    /// Returns the penalty weight applied to each uncovered edge (`graph.len() + 1`).
    pub fn penalty_weight(&self) -> i32 {
        (self.graph.len() as i32) + 1
    }

    /// Recomputes `(gain, objective, cover_size, uncovered_edges)` from scratch
    /// for the given assignment slice.
    pub(crate) fn calculate_state(&self, cover: &[bool]) -> (Vec<i32>, i32, usize, usize) {
        let n = self.graph.len();
        let pw = self.penalty_weight();
        let mut gain = vec![0i32; n];
        let mut cover_size: usize = 0;
        let mut uncovered_edges: usize = 0;

        for &v in self.graph.iter_on_vertices() {
            if cover[v] {
                cover_size += 1;
            }
        }

        // Count uncovered edges (each edge counted once via i < j).
        for &i in self.graph.iter_on_vertices() {
            for &(j, _w) in self.graph.iter_on_adjacency(i) {
                if i < j && !cover[i] && !cover[j] {
                    uncovered_edges += 1;
                }
            }
        }

        for &i in self.graph.iter_on_vertices() {
            // gain[i] = Δobjective if `i` is flipped.
            //   ΔcoverSize = +1 (insertion) or -1 (removal)
            //   For each neighbor j with !cover[j]: edge (i,j) flips between
            //     covered ↔ uncovered, contributing ±penalty_weight.
            let bi = cover[i];
            let mut neigh_uncovered_now: i32 = 0;
            for &(j, _w) in self.graph.iter_on_adjacency(i) {
                if !cover[j] {
                    neigh_uncovered_now += 1;
                }
            }
            gain[i] = if bi {
                // Removing i: +pw for each neighbor not in cover (these edges become uncovered),
                // minus 1 from cover size.
                -1 + pw * neigh_uncovered_now
            } else {
                // Inserting i: -pw for each neighbor not in cover (these edges become covered),
                // plus 1 in cover size.
                1 - pw * neigh_uncovered_now
            };
        }

        let objective = (cover_size as i32) + pw * (uncovered_edges as i32);
        (gain, objective, cover_size, uncovered_edges)
    }

    /// Builds a [`VertexCoverSolution`] from a boolean assignment (same length as `len()`).
    pub fn solution_from_assignment(&self, assignment: &[bool]) -> VertexCoverSolution {
        let n = self.graph.len();
        let cover = if assignment.len() >= n {
            assignment[..n].to_vec()
        } else {
            let mut c = assignment.to_vec();
            c.resize(n, false);
            c
        };
        let (gain, objective, cover_size, uncovered_edges) = self.calculate_state(&cover);
        VertexCoverSolution {
            cover,
            gain,
            objective,
            cover_size,
            uncovered_edges,
        }
    }
}

impl ProblemTrait for VertexCover {
    type Solution = VertexCoverSolution;
    fn new_solution(&self, rng: &mut impl rand::Rng) -> Self::Solution {
        let n = self.graph.len();
        let mut cover = vec![false; n];
        for &i in self.graph.iter_on_vertices() {
            cover[i] = rng.random_bool(0.5);
        }
        let (gain, objective, cover_size, uncovered_edges) = self.calculate_state(&cover);
        VertexCoverSolution {
            cover,
            gain,
            objective,
            cover_size,
            uncovered_edges,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::problem::vertex_cover::VertexCoverFlipNeighbor;
    use crate::search_state::MoveToNeighbor;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    /// Triangle graph: vertices 0,1,2 with edges (0,1), (0,2), (1,2). Min VC = 2.
    fn make_triangle() -> VertexCover {
        let mut vc = VertexCover::new(Graph::new());
        vc.graph.add_edge(0, 1);
        vc.graph.add_edge(0, 2);
        vc.graph.add_edge(1, 2);
        vc
    }

    /// Path P4: 0-1-2-3 with edges (0,1), (1,2), (2,3). Min VC = 2 (e.g. {1, 2}).
    fn make_path4() -> VertexCover {
        let mut vc = VertexCover::new(Graph::new());
        vc.graph.add_edge(0, 1);
        vc.graph.add_edge(1, 2);
        vc.graph.add_edge(2, 3);
        vc
    }

    #[test]
    fn test_load_file_roundtrip() {
        use std::io::Write;
        let mut path = std::env::temp_dir();
        path.push(format!(
            "optopus_vc_{}_{}.txt",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        {
            let mut f = std::fs::File::create(&path).unwrap();
            writeln!(f, "3 3").unwrap();
            writeln!(f, "1 2 1").unwrap();
            writeln!(f, "1 3 1").unwrap();
            writeln!(f, "2 3 1").unwrap();
        }
        let vc = VertexCover::load_file(&path).expect("load_file should succeed");
        assert_eq!(vc.graph.num_vertices(), 3);
        assert_eq!(vc.graph.num_edges(), 3);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_blank_graph() {
        let vc = VertexCover::new(Graph::new());
        assert_eq!(vc.graph.len(), 0);
    }

    #[test]
    fn test_add_edge_and_has_edge() {
        let vc = make_triangle();
        assert_eq!(vc.graph.len(), 3);
        assert!(vc.graph.has_edge(0, 1));
        assert!(vc.graph.has_edge(1, 2));
        assert!(vc.graph.has_edge(0, 2));
        assert!(!vc.graph.has_edge(0, 0));
    }

    #[test]
    fn test_calculate_state_triangle_all_false() {
        let vc = make_triangle();
        let (gain, obj, cover_size, uncov) = vc.calculate_state(&[false, false, false]);
        assert_eq!(cover_size, 0);
        assert_eq!(uncov, 3);
        // pw = 4. Inserting any vertex covers its 2 neighbors → gain = 1 - 4*2 = -7.
        let pw = vc.penalty_weight();
        assert_eq!(pw, 4);
        for &g in &gain {
            assert_eq!(g, 1 - 4 * 2);
        }
        assert_eq!(obj, (4 * 3));
    }

    #[test]
    fn test_calculate_state_triangle_one_in() {
        let vc = make_triangle();
        // cover {0}: edges (0,1) and (0,2) covered, edge (1,2) still uncovered.
        let (gain, obj, cover_size, uncov) = vc.calculate_state(&[true, false, false]);
        assert_eq!(cover_size, 1);
        assert_eq!(uncov, 1);
        let pw = vc.penalty_weight();
        // gain[0]: removing 0 → covers vanish. Both 1 and 2 are not in cover, so the
        //   2 covered edges (0,1) and (0,2) become uncovered. gain = -1 + pw*2 = 7.
        assert_eq!(gain[0], -1 + pw * 2);
        // gain[1]: inserting 1 → only neighbor not in cover is 2 (edge (1,2) becomes covered);
        //   neighbor 0 is already in cover. gain = 1 - pw*1 = -3.
        assert_eq!(gain[1], 1 - pw);
        // gain[2]: symmetric.
        assert_eq!(gain[2], 1 - pw);
        // obj = 1 + pw*1 = 5.
        assert_eq!(obj, 1 + pw);
    }

    #[test]
    fn test_random_walk_gain_consistency() {
        // Apply 200 random flips and ensure the incrementally-maintained solution
        // matches a from-scratch recomputation at every step.
        let vc = make_path4();
        let mut rng = StdRng::seed_from_u64(42);
        let mut sol = vc.new_solution(&mut rng);
        for _ in 0..200 {
            let i = rand::Rng::random_range(&mut rng, 0..vc.graph.len());
            let neighbor = VertexCoverFlipNeighbor {
                i,
                gain: sol.gain[i],
            };
            neighbor.apply_to_solution(&vc, &mut sol).unwrap();

            let (gain, obj, cs, ue) = vc.calculate_state(&sol.cover);
            assert_eq!(sol.gain, gain, "gain drift");
            assert_eq!(sol.objective, obj, "objective drift");
            assert_eq!(sol.cover_size, cs, "cover_size drift");
            assert_eq!(sol.uncovered_edges, ue, "uncovered_edges drift");
        }
    }

    #[test]
    fn test_local_search_finds_min_cover_triangle() {
        use crate::heuristic::{Heuristic, LocalSearch, StopCondition};
        use crate::search_state::SearchState;

        let vc = make_triangle();
        let mut state = SearchState::new(&vc);
        let mut ls = LocalSearch::<VertexCoverFlipNeighbor>::new(StopCondition::iterations(1000));
        ls.run(&mut state).unwrap();

        // The triangle's minimum vertex cover has size 2; LocalSearch should find a feasible one.
        assert_eq!(state.best_solution.uncovered_edges, 0);
        assert_eq!(state.best_solution.cover_size, 2);
        assert_eq!(state.best_solution.objective, 2);
    }

    #[test]
    fn test_local_search_finds_min_cover_path4() {
        use crate::heuristic::{Heuristic, LocalSearch, StopCondition};
        use crate::search_state::SearchState;

        let vc = make_path4();
        let mut state = SearchState::new(&vc);
        let mut ls = LocalSearch::<VertexCoverFlipNeighbor>::new(StopCondition::iterations(1000));
        ls.run(&mut state).unwrap();

        assert_eq!(state.best_solution.uncovered_edges, 0);
        assert_eq!(state.best_solution.cover_size, 2);
    }
}
