//! Edge prices, the [`WeightedConstraints`] side of [`GraphColoring`].

use super::{GraphColoring, GraphColoringSolution};
use crate::building_blocks::instance::EdgeIndex;
use crate::building_blocks::search::IndexSet;
use crate::error::OptError;
use crate::problem::IntChangeNeighbor;
use crate::trait_defs::{MoveToNeighbor, WeightedConstraints};

/// What a [`GraphColoring`] keeps to price a recoloring under edge weights,
/// the weight each vertex meets under each color.
///
/// The constraints are the edges, numbered by [`EdgeIndex`], and the weighted
/// cost of a coloring is the number of colors it uses plus the total weight of
/// the edges whose two ends share a color.
#[derive(Debug, Clone)]
pub struct GraphColoringPrices {
    edges: EdgeIndex,
    /// `met[v * k + c]`, with `k` the palette size, is the total weight of the
    /// edges at `v` whose other end has color `c`, what `v` would conflict
    /// with under that color.
    met: Vec<i64>,
}

impl GraphColoringPrices {
    /// The numbering of the edges, the constraints the weights are on.
    pub fn edges(&self) -> &EdgeIndex {
        &self.edges
    }
}

impl WeightedConstraints for GraphColoring {
    type Move = IntChangeNeighbor<2>;
    type Prices = GraphColoringPrices;

    fn unit_prices(&self, sol: &GraphColoringSolution) -> (GraphColoringPrices, IndexSet) {
        let k = self.k();
        let edges = EdgeIndex::new(self.graph());
        let mut met = vec![0i64; self.graph().len() * k];
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
    /// meet, less the change in the colors used, which the move carries as its
    /// second key.
    fn weighted_gain(
        &self,
        sol: &GraphColoringSolution,
        prices: &GraphColoringPrices,
        mv: &IntChangeNeighbor<2>,
    ) -> f64 {
        let row = mv.var * self.k();
        let left = prices.met[row + sol.colors[mv.var]];
        let entered = prices.met[row + mv.value as usize];
        (left - entered) as f64 - mv.keys[1]
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
        mv.apply_to_solution(self, sol)?;
        if old == new {
            return Ok(());
        }
        let k = self.k();
        for &(e, u) in prices.edges.at(v) {
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
    /// ends that conflict.
    fn repairs(
        &self,
        sol: &GraphColoringSolution,
        prices: &GraphColoringPrices,
        e: usize,
        out: &mut Vec<IntChangeNeighbor<2>>,
    ) {
        let (i, j) = prices.edges.ends(e);
        for v in [i, j] {
            out.extend(
                (0..self.k())
                    .filter(|&c| c != sol.colors[v])
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

    /// The weighted cost, from scratch.
    fn cost(colors: &[usize], w: &Weights) -> i64 {
        let edges = w.prices().edges();
        let mut used: Vec<usize> = colors.to_vec();
        used.sort_unstable();
        used.dedup();
        let conflicts: i64 = (0..edges.len())
            .filter(|&e| colors[edges.ends(e).0] == colors[edges.ends(e).1])
            .map(|e| w.weight(e) as i64)
            .sum();
        used.len() as i64 + conflicts
    }

    fn assert_priced(gc: &GraphColoring, sol: &GraphColoringSolution, w: &Weights) {
        let edges = w.prices().edges();
        let now = cost(&sol.colors, w);
        for v in 0..sol.colors.len() {
            for c in (0..gc.k()).filter(|&c| c != sol.colors[v]) {
                let mut colors = sol.colors.clone();
                colors[v] = c;
                let mv = IntChangeNeighbor::new(gc, sol, v, c as i64);
                assert_eq!(
                    gc.weighted_gain(sol, w.prices(), &mv),
                    (now - cost(&colors, w)) as f64,
                    "gain of {v} to {c}"
                );
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
    fn escape_ends_a_conflict_or_empties_the_smallest_class() {
        let gc = wheel();
        let mut rng = SmallRng::seed_from_u64(9);
        let mut out = Vec::new();

        let clash = gc.solution_from_colors(vec![0, 0, 1, 2, 1, 3]).unwrap();
        let w = Weights::new(&gc, &clash, 5);
        assert_eq!(w.violated().len(), 1);
        w.escape_moves(&gc, &clash, &mut rng, &mut out);
        assert_eq!(out.len(), 2 * (gc.k() - 1));
        assert!(out.iter().all(|mv| mv.var <= 1 && mv.value != 0));

        out.clear();
        let proper = gc.solution_from_colors(vec![0, 1, 0, 1, 2, 3]).unwrap();
        let w = Weights::new(&gc, &proper, 5);
        assert!(w.violated().is_empty());
        w.escape_moves(&gc, &proper, &mut rng, &mut out);
        // Color 2 is the first of the two smallest classes, held by vertex 4.
        let mut targets: Vec<i64> = out.iter().map(|mv| mv.value).collect();
        targets.sort_unstable();
        assert!(out.iter().all(|mv| mv.var == 4));
        assert_eq!(targets, vec![0, 1, 3]);
    }
}
