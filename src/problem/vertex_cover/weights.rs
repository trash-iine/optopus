//! Edge prices, the [`WeightedConstraints`] side of [`VertexCover`].

use super::{VertexCover, VertexCoverFlipNeighbor, VertexCoverSolution};
use crate::building_blocks::instance::EdgeIndex;
use crate::building_blocks::search::IndexSet;
use crate::error::OptError;
use crate::trait_defs::{MoveToNeighbor, WeightedConstraints};

/// What a [`VertexCover`] keeps to price a flip under edge weights, which
/// each vertex would leave uncovered.
///
/// The constraints are the edges, numbered by [`EdgeIndex`], and the weighted
/// cost of a vertex set is its size plus the total weight of the edges it
/// leaves uncovered. A loop from a vertex to itself is no constraint here, as
/// it is none in the uncovered count of the solution.
#[derive(Debug, Clone)]
pub struct VertexCoverPrices {
    edges: EdgeIndex,
    /// `open[v]` is the total weight of the edges at `v` whose other end is
    /// out of the set, the weight `v` alone stands between and uncovered.
    open: Vec<i64>,
}

impl VertexCoverPrices {
    /// The numbering of the edges, the constraints the weights are on.
    pub fn edges(&self) -> &EdgeIndex {
        &self.edges
    }
}

impl WeightedConstraints for VertexCover {
    type Move = VertexCoverFlipNeighbor;
    type Prices = VertexCoverPrices;

    fn unit_prices(&self, sol: &VertexCoverSolution) -> (VertexCoverPrices, IndexSet) {
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
        (VertexCoverPrices { edges, open }, uncovered)
    }

    /// Leaving the set saves one vertex and uncovers `open`, entering it does
    /// the reverse.
    fn weighted_gain(
        &self,
        sol: &VertexCoverSolution,
        prices: &VertexCoverPrices,
        mv: &VertexCoverFlipNeighbor,
    ) -> f64 {
        let open = prices.open[mv.i];
        (if sol.x[mv.i] { 1 - open } else { open - 1 }) as f64
    }

    fn apply_weighted(
        &self,
        sol: &mut VertexCoverSolution,
        mv: &VertexCoverFlipNeighbor,
        weight: &[u32],
        prices: &mut VertexCoverPrices,
        uncovered: &mut IndexSet,
    ) -> Result<(), OptError> {
        mv.apply_to_solution(self, sol)?;
        let v = mv.i;
        let entered = sol.x[v];
        for &(e, u) in prices.edges.at(v) {
            let w = weight[e] as i64;
            if entered {
                prices.open[u] -= w;
                uncovered.remove(e);
            } else {
                prices.open[u] += w;
                if !sol.x[u] {
                    uncovered.insert(e);
                }
            }
        }
        Ok(())
    }

    /// Each end that is out of the set stands open to the other by the
    /// weight of the edge.
    #[inline]
    fn reweight(
        &self,
        sol: &VertexCoverSolution,
        prices: &mut VertexCoverPrices,
        e: usize,
        delta: i64,
    ) {
        let (i, j) = prices.edges.ends(e);
        if !sol.x[j] {
            prices.open[i] += delta;
        }
        if !sol.x[i] {
            prices.open[j] += delta;
        }
    }

    /// The two ends of an uncovered edge, either of which covers it by
    /// entering.
    fn repairs(
        &self,
        sol: &VertexCoverSolution,
        prices: &VertexCoverPrices,
        e: usize,
        out: &mut Vec<VertexCoverFlipNeighbor>,
    ) {
        let (i, j) = prices.edges.ends(e);
        out.push(VertexCoverFlipNeighbor::new(self, sol, i));
        out.push(VertexCoverFlipNeighbor::new(self, sol, j));
    }

    /// The vertices of the cover, any of which shrinks it by leaving.
    fn objective_escapes(&self, sol: &VertexCoverSolution, out: &mut Vec<VertexCoverFlipNeighbor>) {
        out.extend(
            self.graph
                .iter_on_vertices()
                .filter(|&&v| sol.x[v])
                .map(|&v| VertexCoverFlipNeighbor::new(self, sol, v)),
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::building_blocks::instance::Graph;
    use crate::heuristic::ConstraintWeights;
    use crate::trait_defs::ProblemTrait;
    use rand::Rng;
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

    type Weights = ConstraintWeights<VertexCover>;

    /// The weighted cost, from scratch.
    fn cost(x: &[bool], w: &Weights) -> i64 {
        let edges = w.prices().edges();
        let size = x.iter().filter(|&&b| b).count() as i64;
        let open: i64 = (0..edges.len())
            .filter(|&e| !x[edges.ends(e).0] && !x[edges.ends(e).1])
            .map(|e| w.weight(e) as i64)
            .sum();
        size + open
    }

    fn assert_priced(vc: &VertexCover, sol: &VertexCoverSolution, w: &Weights) {
        let edges = w.prices().edges();
        let now = cost(&sol.x, w);
        for v in 0..vc.graph.len() {
            let mut x = sol.x.clone();
            x[v] = !x[v];
            let mv = VertexCoverFlipNeighbor::new(vc, sol, v);
            assert_eq!(
                vc.weighted_gain(sol, w.prices(), &mv),
                (now - cost(&x, w)) as f64,
                "gain of {v}"
            );
        }
        let mut got = w.violated().as_slice().to_vec();
        got.sort_unstable();
        let expected: Vec<usize> = (0..edges.len())
            .filter(|&e| !sol.x[edges.ends(e).0] && !sol.x[edges.ends(e).1])
            .collect();
        assert_eq!(got, expected);
        assert_eq!(w.violated().len(), sol.uncovered_edges);
    }

    #[test]
    fn prices_follow_flips_and_weight_changes() {
        let vc = wheel();
        let mut rng = SmallRng::seed_from_u64(2);
        let mut sol = vc.new_solution(&mut rng);
        let mut w = Weights::new(&vc, &sol, 5);
        assert_eq!(w.len(), vc.graph.num_edges());
        assert_eq!(w.cap(), 10, "the cap is the edge count when that is larger");
        assert_priced(&vc, &sol, &w);
        let mut raised = false;
        for step in 0..300 {
            match step % 5 {
                1 => {
                    w.raise_violated(&vc, &sol);
                    raised |= (0..w.len()).any(|e| w.weight(e) > 1);
                }
                4 => w.lower_satisfied(&vc, &sol),
                _ => {
                    let mv = VertexCoverFlipNeighbor::new(&vc, &sol, rng.random_range(0..6));
                    w.apply(&vc, &mut sol, &mv).unwrap();
                }
            }
            assert_priced(&vc, &sol, &w);
            assert!((0..w.len()).all(|e| (1..=w.cap()).contains(&w.weight(e))));
        }
        assert!(raised, "some weight was raised");
    }

    #[test]
    fn escape_covers_an_edge_or_shrinks_a_cover() {
        let vc = wheel();
        let mut rng = SmallRng::seed_from_u64(9);
        let mut out = Vec::new();

        let open = vc.solution_from_assignment(&[false; 6]).unwrap();
        let w = Weights::new(&vc, &open, 5);
        w.escape_moves(&vc, &open, &mut rng, &mut out);
        assert_eq!(out.len(), 2);
        assert!(vc.graph.has_edge(out[0].i, out[1].i));

        out.clear();
        let cover = vc
            .solution_from_assignment(&[true, false, true, true, false, true])
            .unwrap();
        let w = Weights::new(&vc, &cover, 5);
        assert!(w.violated().is_empty());
        w.escape_moves(&vc, &cover, &mut rng, &mut out);
        let mut leaving: Vec<usize> = out.iter().map(|mv| mv.i).collect();
        leaving.sort_unstable();
        assert_eq!(leaving, vec![0, 2, 3, 5]);
    }
}
