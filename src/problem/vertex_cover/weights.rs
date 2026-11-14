//! Edge weights, the [`WeightedConstraints`] side of [`VertexCover`].

use super::{VertexCover, VertexCoverFlipNeighbor, VertexCoverSolution};
use crate::building_blocks::instance::EdgeIndex;
use crate::building_blocks::search::IndexSet;
use crate::error::OptError;
use crate::trait_defs::{MoveToNeighbor, WeightedConstraints};
use rand::Rng;

/// One weight per edge of a [`VertexCover`], with what each vertex would
/// leave uncovered under them.
///
/// The constraints are the edges, and the weighted cost of a vertex set is its
/// size plus the total weight of the edges it leaves uncovered. A loop from a
/// vertex to itself is no constraint here, as it is none in the uncovered
/// count of the solution.
#[derive(Debug, Clone)]
pub struct VertexCoverWeights {
    /// `weight[e]` is the weight of edge `e`.
    weight: Vec<u32>,
    edges: EdgeIndex,
    /// `open[v]` is the total weight of the edges at `v` whose other end is
    /// out of the set, the weight `v` alone stands between and uncovered.
    open: Vec<i64>,
    /// The edges with both ends out of the set.
    uncovered: IndexSet,
}

impl VertexCoverWeights {
    /// The weight of the edge between `i` and `j`, or `None` when they are
    /// not adjacent.
    pub fn weight(&self, i: usize, j: usize) -> Option<u32> {
        self.edges.between(i, j).map(|e| self.weight[e])
    }

    /// How many edges the set leaves uncovered.
    pub fn uncovered_count(&self) -> usize {
        self.uncovered.len()
    }
}

impl WeightedConstraints for VertexCover {
    type Move = VertexCoverFlipNeighbor;
    type Weights = VertexCoverWeights;

    fn unit_weights(&self, sol: &VertexCoverSolution) -> VertexCoverWeights {
        let n = self.graph.len();
        let edges = EdgeIndex::new(&self.graph);

        let mut uncovered = IndexSet::new(edges.len());
        let mut open = vec![0i64; n];
        for e in 0..edges.len() {
            let (i, j) = edges.ends(e);
            if !sol.x[i] {
                open[j] += 1;
            }
            if !sol.x[j] {
                open[i] += 1;
            }
            if !sol.x[i] && !sol.x[j] {
                uncovered.insert(e);
            }
        }
        VertexCoverWeights {
            weight: vec![1; edges.len()],
            edges,
            open,
            uncovered,
        }
    }

    fn constraint_count(&self) -> usize {
        self.graph.num_edges()
    }

    /// Leaving the set saves one vertex and uncovers `open`, entering it does
    /// the reverse.
    fn weighted_gain(
        &self,
        sol: &VertexCoverSolution,
        weights: &VertexCoverWeights,
        mv: &VertexCoverFlipNeighbor,
    ) -> f64 {
        let open = weights.open[mv.i];
        (if sol.x[mv.i] { 1 - open } else { open - 1 }) as f64
    }

    fn apply_weighted(
        &self,
        sol: &mut VertexCoverSolution,
        weights: &mut VertexCoverWeights,
        mv: &VertexCoverFlipNeighbor,
    ) -> Result<(), OptError> {
        mv.apply_to_solution(self, sol)?;
        let v = mv.i;
        let entered = sol.x[v];
        for at in 0..weights.edges.at(v).len() {
            let (e, u) = weights.edges.at(v)[at];
            let w = weights.weight[e as usize] as i64;
            if entered {
                weights.open[u] -= w;
                weights.uncovered.remove(e as usize);
            } else {
                weights.open[u] += w;
                if !sol.x[u] {
                    weights.uncovered.insert(e as usize);
                }
            }
        }
        Ok(())
    }

    fn raise_violated(&self, _: &VertexCoverSolution, weights: &mut VertexCoverWeights, cap: u32) {
        for &e in weights.uncovered.as_slice() {
            if weights.weight[e] < cap {
                weights.weight[e] += 1;
                // Both ends are out, so each stands open to the other.
                let (i, j) = weights.edges.ends(e);
                weights.open[i] += 1;
                weights.open[j] += 1;
            }
        }
    }

    fn lower_satisfied(&self, sol: &VertexCoverSolution, weights: &mut VertexCoverWeights) {
        for e in 0..weights.weight.len() {
            if weights.weight[e] > 1 && !weights.uncovered.contains(e) {
                weights.weight[e] -= 1;
                let (i, j) = weights.edges.ends(e);
                if !sol.x[j] {
                    weights.open[i] -= 1;
                }
                if !sol.x[i] {
                    weights.open[j] -= 1;
                }
            }
        }
    }

    /// The two ends of one uncovered edge, either of which covers it by
    /// entering. Once every edge is covered, the vertices of the cover, any of
    /// which shrinks it by leaving.
    fn escape_moves(
        &self,
        sol: &VertexCoverSolution,
        weights: &VertexCoverWeights,
        rng: &mut rand::rngs::SmallRng,
        out: &mut Vec<VertexCoverFlipNeighbor>,
    ) {
        let uncovered = weights.uncovered.as_slice();
        if uncovered.is_empty() {
            out.extend(
                self.graph
                    .iter_on_vertices()
                    .filter(|&&v| sol.x[v])
                    .map(|&v| VertexCoverFlipNeighbor::new(self, sol, v)),
            );
            return;
        }
        let (i, j) = weights
            .edges
            .ends(uncovered[rng.random_range(0..uncovered.len())]);
        out.push(VertexCoverFlipNeighbor::new(self, sol, i));
        out.push(VertexCoverFlipNeighbor::new(self, sol, j));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::building_blocks::instance::Graph;
    use crate::trait_defs::ProblemTrait;
    use rand::SeedableRng;
    use rand::rngs::SmallRng;

    fn wheel() -> VertexCover {
        VertexCover::new(Graph::from_edges([
            (0, 1, 1.0),
            (1, 2, 1.0),
            (2, 3, 1.0),
            (3, 4, 1.0),
            (4, 0, 1.0),
            (5, 0, 1.0),
            (5, 1, 1.0),
            (5, 2, 1.0),
            (5, 3, 1.0),
            (5, 4, 1.0),
        ]))
    }

    /// The weighted cost, from scratch.
    fn cost(x: &[bool], w: &VertexCoverWeights) -> i64 {
        let size = x.iter().filter(|&&b| b).count() as i64;
        let open: i64 = (0..w.edges.len())
            .filter(|&e| !x[w.edges.ends(e).0] && !x[w.edges.ends(e).1])
            .map(|e| w.weight[e] as i64)
            .sum();
        size + open
    }

    fn assert_priced(vc: &VertexCover, sol: &VertexCoverSolution, w: &VertexCoverWeights) {
        let now = cost(&sol.x, w);
        for v in 0..vc.graph.len() {
            let mut x = sol.x.clone();
            x[v] = !x[v];
            let mv = VertexCoverFlipNeighbor::new(vc, sol, v);
            assert_eq!(
                vc.weighted_gain(sol, w, &mv),
                (now - cost(&x, w)) as f64,
                "gain of {v}"
            );
        }
        let mut got = w.uncovered.as_slice().to_vec();
        got.sort_unstable();
        let expected: Vec<usize> = (0..w.edges.len())
            .filter(|&e| !sol.x[w.edges.ends(e).0] && !sol.x[w.edges.ends(e).1])
            .collect();
        assert_eq!(got, expected);
        assert_eq!(w.uncovered_count(), sol.uncovered_edges);
    }

    #[test]
    fn prices_follow_flips_and_weight_changes() {
        let vc = wheel();
        let mut rng = SmallRng::seed_from_u64(2);
        let mut sol = vc.new_solution(&mut rng);
        let mut w = vc.unit_weights(&sol);
        assert_eq!(w.edges.len(), vc.constraint_count());
        assert_priced(&vc, &sol, &w);
        let mut raised = false;
        for step in 0..300 {
            match step % 5 {
                1 => {
                    vc.raise_violated(&sol, &mut w, 5);
                    raised |= w.weight.iter().any(|&x| x > 1);
                }
                4 => vc.lower_satisfied(&sol, &mut w),
                _ => {
                    let mv = VertexCoverFlipNeighbor::new(&vc, &sol, rng.random_range(0..6));
                    vc.apply_weighted(&mut sol, &mut w, &mv).unwrap();
                }
            }
            assert_priced(&vc, &sol, &w);
            assert!(w.weight.iter().all(|&x| (1..=5).contains(&x)));
        }
        assert!(raised, "some weight was raised");
        assert_eq!(w.weight(0, 5), w.weight(5, 0));
        assert_eq!(w.weight(0, 2), None);
    }

    #[test]
    fn escape_covers_an_edge_or_shrinks_a_cover() {
        let vc = wheel();
        let mut rng = SmallRng::seed_from_u64(9);
        let mut out = Vec::new();

        let open = vc.solution_from_assignment(&[false; 6]).unwrap();
        let w = vc.unit_weights(&open);
        vc.escape_moves(&open, &w, &mut rng, &mut out);
        assert_eq!(out.len(), 2);
        assert!(vc.graph.has_edge(out[0].i, out[1].i));

        out.clear();
        let cover = vc
            .solution_from_assignment(&[true, false, true, true, false, true])
            .unwrap();
        let w = vc.unit_weights(&cover);
        assert_eq!(w.uncovered_count(), 0);
        vc.escape_moves(&cover, &w, &mut rng, &mut out);
        let mut leaving: Vec<usize> = out.iter().map(|mv| mv.i).collect();
        leaving.sort_unstable();
        assert_eq!(leaving, vec![0, 2, 3, 5]);
    }
}
