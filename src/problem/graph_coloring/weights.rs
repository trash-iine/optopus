//! Edge prices, the [`WeightedConstraints`] side of [`GraphColoring`].

use super::{GraphColoring, GraphColoringSolution, GraphColoringUniformCrossover};
use crate::building_blocks::instance::EdgeIndex;
use crate::building_blocks::search::IndexSet;
use crate::error::OptError;
use crate::problem::IntChangeNeighbor;
use crate::trait_defs::{Crossover, MoveToNeighbor, ProblemTrait, WeightedConstraints};
use rand::rngs::SmallRng;

/// What a [`GraphColoring`] keeps to price a recoloring under edge weights,
/// the weight each vertex meets under each color.
///
/// The constraints are the edges, numbered by [`EdgeIndex`], each one kept by
/// two ends of different colors, and the objective is the number of colors
/// used.
#[derive(Debug, Clone)]
pub struct GraphColoringPrices {
    edges: EdgeIndex,
    /// `met[v * k + c]`, with `k` the palette size, is the total weight of the
    /// edges at `v` whose other end has color `c`, what `v` would conflict
    /// with under that color.
    met: Vec<i64>,
}

impl WeightedConstraints for GraphColoring {
    type Move = IntChangeNeighbor<2>;
    type Prices = GraphColoringPrices;

    fn unit_prices(&self, sol: &GraphColoringSolution) -> (GraphColoringPrices, IndexSet) {
        let k = self.k();
        let edges = EdgeIndex::new(self.graph());
        let mut met = vec![0; self.graph().len() * k];
        let mut conflicting = IndexSet::new(edges.len());
        for e in 0..edges.len() {
            let (i, j) = edges.ends(e);
            met[i * k + sol.colors[j]] += 1;
            met[j * k + sol.colors[i]] += 1;
            if sol.colors[i] == sol.colors[j] {
                conflicting.insert(e);
            }
        }
        (GraphColoringPrices { edges, met }, conflicting)
    }

    /// The weight the vertex stops meeting, less the weight it starts to
    /// meet.
    fn weight_gain(
        &self,
        sol: &GraphColoringSolution,
        prices: &GraphColoringPrices,
        mv: &IntChangeNeighbor<2>,
    ) -> i64 {
        let row = mv.var * self.k();
        prices.met[row + sol.colors[mv.var]] - prices.met[row + mv.value as usize]
    }

    fn apply_weighted(
        &self,
        sol: &mut GraphColoringSolution,
        mv: &IntChangeNeighbor<2>,
        weight: &[u32],
        prices: &mut GraphColoringPrices,
        conflicting: &mut IndexSet,
    ) -> Result<(), OptError> {
        let v = mv.var;
        let (old, new) = (sol.colors[v], mv.value as usize);
        if old == new {
            return Ok(());
        }
        mv.apply_to_solution(self, sol)?;
        let k = self.k();
        for &(e, u) in prices.edges.incident(v) {
            let w = weight[e] as i64;
            prices.met[u * k + old] -= w;
            prices.met[u * k + new] += w;
            if sol.colors[u] == old {
                conflicting.remove(e);
            } else if sol.colors[u] == new {
                conflicting.insert(e);
            }
        }
        Ok(())
    }

    /// Each end meets the other under the other's color, by the weight of the
    /// edge.
    #[inline]
    fn reweight(
        &self,
        sol: &GraphColoringSolution,
        prices: &mut GraphColoringPrices,
        e: usize,
        delta: i64,
    ) {
        let k = self.k();
        let (i, j) = prices.edges.ends(e);
        prices.met[i * k + sol.colors[j]] += delta;
        prices.met[j * k + sol.colors[i]] += delta;
    }

    /// The recolorings of the two ends of a conflicting edge, any of which
    /// ends that conflict, and the recolorings of the two ends of an edge
    /// whose ends differ to a color neither end has, which keep them apart.
    fn tight_moves(
        &self,
        sol: &GraphColoringSolution,
        prices: &GraphColoringPrices,
        e: usize,
        out: &mut Vec<IntChangeNeighbor<2>>,
    ) {
        let (i, j) = prices.edges.ends(e);
        let (ci, cj) = (sol.colors[i], sol.colors[j]);
        for v in [i, j] {
            out.extend(
                (0..self.k())
                    .filter(|&c| c != ci && c != cj)
                    .map(|c| IntChangeNeighbor::new(self, sol, v, c as i64)),
            );
        }
    }

    /// The recolorings that take a vertex of the smallest color class into
    /// another class in use, which empty that class one vertex at a time. A
    /// coloring with one color has none.
    fn objective_escapes(&self, sol: &GraphColoringSolution, out: &mut Vec<IntChangeNeighbor<2>>) {
        let in_use = |c: &usize| sol.class_size(*c) > 0;
        let Some(smallest) = (0..self.k())
            .filter(in_use)
            .min_by_key(|&c| sol.class_size(c))
        else {
            return;
        };
        for v in (0..sol.colors.len()).filter(|&v| sol.colors[v] == smallest) {
            out.extend(
                (0..self.k())
                    .filter(|c| *c != smallest && in_use(c))
                    .map(|c| IntChangeNeighbor::new(self, sol, v, c as i64)),
            );
        }
    }

    fn objective(&self, sol: &GraphColoringSolution) -> f64 {
        sol.colors_used as f64
    }

    /// The change in the colors used, which the move carries as its second
    /// key.
    fn objective_change(&self, _: &GraphColoringSolution, mv: &IntChangeNeighbor<2>) -> f64 {
        mv.keys[1]
    }

    /// Each vertex colored as in `best` or as in a fresh coloring, with even
    /// odds.
    fn restart_from(
        &self,
        best: &GraphColoringSolution,
        rng: &mut SmallRng,
    ) -> GraphColoringSolution {
        let fresh = self.new_solution(rng);
        GraphColoringUniformCrossover
            .crossover(self, best, &fresh, rng)
            .expect("a uniform crossover cannot fail")
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

    fn wheel() -> GraphColoring {
        GraphColoring::new(Graph::from_edges([
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

    type Weights = ConstraintWeights<GraphColoring>;

    /// The weight of the conflicting edges and the colors used, from scratch.
    fn cost(colors: &[usize], w: &Weights) -> (i64, f64) {
        let edges = &w.prices().edges;
        let mut used: Vec<usize> = colors.to_vec();
        used.sort_unstable();
        used.dedup();
        let conflicts: i64 = (0..edges.len())
            .filter(|&e| colors[edges.ends(e).0] == colors[edges.ends(e).1])
            .map(|e| w.weight(e) as i64)
            .sum();
        (conflicts, used.len() as f64)
    }

    fn assert_priced(gc: &GraphColoring, sol: &GraphColoringSolution, w: &Weights) {
        let edges = &w.prices().edges;
        let now = cost(&sol.colors, w);
        for v in 0..sol.colors.len() {
            for c in (0..gc.k()).filter(|&c| c != sol.colors[v]) {
                let mut colors = sol.colors.clone();
                colors[v] = c;
                let mv = IntChangeNeighbor::new(gc, sol, v, c as i64);
                let after = cost(&colors, w);
                assert_eq!(
                    gc.weight_gain(sol, w.prices(), &mv),
                    now.0 - after.0,
                    "gain of {v} to {c}"
                );
                assert_eq!(gc.objective_change(sol, &mv), after.1 - now.1);
            }
        }
        let mut got = w.violated().as_slice().to_vec();
        got.sort_unstable();
        let expected: Vec<usize> = (0..edges.len())
            .filter(|&e| sol.colors[edges.ends(e).0] == sol.colors[edges.ends(e).1])
            .collect();
        assert_eq!(got, expected);
        assert_eq!(w.violated().len(), sol.conflicts);
    }

    #[test]
    fn prices_follow_recolorings_and_weight_changes() {
        let gc = wheel();
        let mut rng = SmallRng::seed_from_u64(2);
        let mut sol = gc.new_solution(&mut rng);
        let mut w = Weights::new(&gc, &sol, 5);
        assert_eq!(w.len(), gc.graph().num_edges());
        assert_priced(&gc, &sol, &w);
        let mut raised = false;
        for step in 0..400 {
            match step % 5 {
                1 => {
                    w.raise_violated(&gc, &sol);
                    raised |= (0..w.len()).any(|e| w.weight(e) > 1);
                }
                4 => w.lower_satisfied(&gc, &sol),
                _ => {
                    let v = rng.random_range(0..6);
                    let c = rng.random_range(0..gc.k());
                    let mv = IntChangeNeighbor::new(&gc, &sol, v, c as i64);
                    w.apply(&gc, &mut sol, &mv).unwrap();
                }
            }
            assert_priced(&gc, &sol, &w);
            assert!((0..w.len()).all(|e| (1..=w.cap()).contains(&w.weight(e))));
        }
        assert!(raised, "some weight was raised");
    }

    #[test]
    fn tight_moves_end_a_conflict_or_keep_two_ends_apart() {
        let gc = wheel();
        let sol = gc.solution_from_colors(vec![0, 0, 1, 2, 1, 3]).unwrap();
        let w = Weights::new(&gc, &sol, 5);
        assert_eq!(w.violated().len(), 1);
        let edges = &w.prices().edges;
        for e in 0..edges.len() {
            let (i, j) = edges.ends(e);
            let mut out = Vec::new();
            gc.tight_moves(&sol, w.prices(), e, &mut out);
            let distinct = if sol.colors[i] == sol.colors[j] { 1 } else { 2 };
            assert_eq!(out.len(), 2 * (gc.k() - distinct), "edge {e}");
            for mv in &out {
                assert!(mv.var == i || mv.var == j);
                let other = if mv.var == i { j } else { i };
                assert_ne!(mv.value as usize, sol.colors[other], "edge {e}");
                assert_ne!(mv.value as usize, sol.colors[mv.var], "edge {e}");
            }
        }
    }

    #[test]
    fn objective_escapes_empty_the_smallest_class() {
        let gc = wheel();
        let proper = gc.solution_from_colors(vec![0, 1, 0, 1, 2, 3]).unwrap();
        let mut out = Vec::new();
        gc.objective_escapes(&proper, &mut out);
        // Color 2 is the first of the two smallest classes, held by vertex 4.
        let mut targets: Vec<i64> = out.iter().map(|mv| mv.value).collect();
        targets.sort_unstable();
        assert!(out.iter().all(|mv| mv.var == 4));
        assert_eq!(targets, vec![0, 1, 3]);
    }
}
