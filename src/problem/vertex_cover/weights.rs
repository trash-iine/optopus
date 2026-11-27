//! Edge prices, the [`WeightedConstraints`] side of [`VertexCover`].

use super::{VertexCover, VertexCoverFlipNeighbor, VertexCoverSolution};
use crate::building_blocks::instance::EdgeIndex;
use crate::building_blocks::representation::uniform_binary_crossover;
use crate::building_blocks::search::IndexSet;
use crate::error::OptError;
use crate::trait_defs::{MoveToNeighbor, ProblemTrait, WeightedConstraints};
use rand::rngs::SmallRng;

/// What a [`VertexCover`] keeps to price a flip under edge weights, which
/// each vertex would leave uncovered.
///
/// The constraints are the edges, numbered by [`EdgeIndex`], and the
/// objective is the size of the set. A loop from a vertex to itself is no constraint here, as
/// it is none in the uncovered count of the solution.
#[derive(Debug, Clone)]
pub struct VertexCoverPrices {
    edges: EdgeIndex,
    /// `open[v]` is the total weight of the edges at `v` whose other end is
    /// out of the set. When `v` is in the set these are the edges only `v`
    /// covers, and when it is out they are the uncovered edges it would cover.
    open: Vec<i64>,
}

impl WeightedConstraints for VertexCover {
    type Move = VertexCoverFlipNeighbor;
    type Prices = VertexCoverPrices;

    fn unit_prices(&self, sol: &VertexCoverSolution) -> (VertexCoverPrices, IndexSet) {
        let n = self.graph.len();
        let edges = EdgeIndex::new(&self.graph);

        let mut uncovered = IndexSet::new(edges.len());
        let mut open = vec![0; n];
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

    /// Leaving the set uncovers `open`, entering it covers `open`.
    fn weight_gain(
        &self,
        sol: &VertexCoverSolution,
        prices: &VertexCoverPrices,
        mv: &VertexCoverFlipNeighbor,
    ) -> i64 {
        let open = prices.open[mv.i];
        if sol.x[mv.i] { -open } else { open }
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
        for &(e, u) in prices.edges.incident(v) {
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
    /// entering, and the two ends of an edge both ends of which are in the
    /// set, either of which can leave it covered. An edge with one end in has
    /// none.
    fn tight_moves(
        &self,
        sol: &VertexCoverSolution,
        prices: &VertexCoverPrices,
        e: usize,
        out: &mut Vec<VertexCoverFlipNeighbor>,
    ) {
        let (i, j) = prices.edges.ends(e);
        if sol.x[i] == sol.x[j] {
            out.push(VertexCoverFlipNeighbor::new(self, sol, i));
            out.push(VertexCoverFlipNeighbor::new(self, sol, j));
        }
    }

    /// The vertices of the set, any of which shrinks it by leaving.
    fn objective_escapes(&self, sol: &VertexCoverSolution, out: &mut Vec<VertexCoverFlipNeighbor>) {
        out.extend(
            self.graph
                .iter_on_vertices()
                .filter(|&&v| sol.x[v])
                .map(|&v| VertexCoverFlipNeighbor::new(self, sol, v)),
        );
    }

    fn objective(&self, sol: &VertexCoverSolution) -> f64 {
        sol.cover_size as f64
    }

    fn objective_change(&self, sol: &VertexCoverSolution, mv: &VertexCoverFlipNeighbor) -> f64 {
        if sol.x[mv.i] { -1.0 } else { 1.0 }
    }

    /// Each vertex in or out as in `best` or as in a fresh set, with even odds.
    fn restart_from(&self, best: &VertexCoverSolution, rng: &mut SmallRng) -> VertexCoverSolution {
        let fresh = self.new_solution(rng);
        uniform_binary_crossover(self, best, &fresh, rng).expect("a uniform crossover cannot fail")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::building_blocks::instance::Graph;
    use crate::building_blocks::search::ConstraintWeights;
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

    /// The weight of the uncovered edges, from scratch.
    fn open_weight(x: &[bool], w: &Weights) -> i64 {
        let edges = &w.prices().edges;
        (0..edges.len())
            .filter(|&e| !x[edges.ends(e).0] && !x[edges.ends(e).1])
            .map(|e| w.weight(e) as i64)
            .sum()
    }

    fn assert_priced(vc: &VertexCover, sol: &VertexCoverSolution, w: &Weights) {
        let edges = &w.prices().edges;
        let now = open_weight(&sol.x, w);
        for v in 0..vc.graph.len() {
            let mut x = sol.x.clone();
            x[v] = !x[v];
            let mv = VertexCoverFlipNeighbor::new(vc, sol, v);
            assert_eq!(
                vc.weight_gain(sol, w.prices(), &mv),
                now - open_weight(&x, w),
                "gain of {v}"
            );
            let size = |x: &[bool]| x.iter().filter(|&&b| b).count() as f64;
            assert_eq!(vc.objective_change(sol, &mv), size(&x) - size(&sol.x));
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
    fn tight_moves_cover_an_edge_or_leave_it_covered() {
        let vc = wheel();
        let sol = vc
            .solution_from_assignment(&[true, false, false, true, false, true])
            .unwrap();
        let w = Weights::new(&vc, &sol, 5);
        let edges = &w.prices().edges;
        for e in 0..edges.len() {
            let (i, j) = edges.ends(e);
            let mut out = Vec::new();
            vc.tight_moves(&sol, w.prices(), e, &mut out);
            let flipped: Vec<usize> = out.iter().map(|mv| mv.i).collect();
            if sol.x[i] == sol.x[j] {
                assert_eq!(flipped, vec![i, j], "edge {e}");
            } else {
                assert!(flipped.is_empty(), "edge {e}");
            }
        }
        let mut out = Vec::new();
        vc.objective_escapes(&sol, &mut out);
        let leaving: Vec<usize> = out.iter().map(|mv| mv.i).collect();
        assert_eq!(leaving, vec![0, 3, 5]);
    }
}
