//! Edge weights, the [`WeightedConstraints`] side of [`GraphColoring`].

use super::{GraphColoring, GraphColoringSolution};
use crate::building_blocks::instance::EdgeIndex;
use crate::building_blocks::search::IndexSet;
use crate::error::OptError;
use crate::problem::IntChangeNeighbor;
use crate::trait_defs::{MoveToNeighbor, WeightedConstraints};
use rand::Rng;

/// One weight per edge of a [`GraphColoring`], with the weight each vertex
/// meets under each color.
///
/// The constraints are the edges, and the weighted cost of a coloring is the
/// number of colors it uses plus the total weight of the edges whose two ends
/// share a color.
#[derive(Debug, Clone)]
pub struct GraphColoringWeights {
    /// `weight[e]` is the weight of edge `e`.
    weight: Vec<u32>,
    edges: EdgeIndex,
    /// The palette size, the width of a row of `met`.
    k: usize,
    /// `met[v * k + c]` is the total weight of the edges at `v` whose other
    /// end has color `c`, what `v` would conflict with under that color.
    met: Vec<i64>,
    /// The edges whose two ends share a color.
    conflicting: IndexSet,
}

impl GraphColoringWeights {
    /// The weight of the edge between `i` and `j`, or `None` when they are
    /// not adjacent.
    pub fn weight(&self, i: usize, j: usize) -> Option<u32> {
        self.edges.between(i, j).map(|e| self.weight[e])
    }

    /// How many edges have two ends of one color.
    pub fn conflicting_count(&self) -> usize {
        self.conflicting.len()
    }
}

impl WeightedConstraints for GraphColoring {
    type Move = IntChangeNeighbor<2>;
    type Weights = GraphColoringWeights;

    fn unit_weights(&self, sol: &GraphColoringSolution) -> GraphColoringWeights {
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
        GraphColoringWeights {
            weight: vec![1; edges.len()],
            edges,
            k,
            met,
            conflicting,
        }
    }

    fn constraint_count(&self) -> usize {
        self.graph().num_edges()
    }

    /// The weight the vertex stops meeting, less the weight it starts to
    /// meet, less the change in the colors used, which the move carries as its
    /// second key.
    fn weighted_gain(
        &self,
        sol: &GraphColoringSolution,
        weights: &GraphColoringWeights,
        mv: &IntChangeNeighbor<2>,
    ) -> f64 {
        let row = mv.var * weights.k;
        let left = weights.met[row + sol.colors[mv.var]];
        let entered = weights.met[row + mv.value as usize];
        (left - entered) as f64 - mv.keys[1]
    }

    fn apply_weighted(
        &self,
        sol: &mut GraphColoringSolution,
        weights: &mut GraphColoringWeights,
        mv: &IntChangeNeighbor<2>,
    ) -> Result<(), OptError> {
        let v = mv.var;
        let (old, new) = (sol.colors[v], mv.value as usize);
        mv.apply_to_solution(self, sol)?;
        if old == new {
            return Ok(());
        }
        let k = weights.k;
        for at in 0..weights.edges.at(v).len() {
            let (e, u) = weights.edges.at(v)[at];
            let w = weights.weight[e as usize] as i64;
            weights.met[u * k + old] -= w;
            weights.met[u * k + new] += w;
            if sol.colors[u] == old {
                weights.conflicting.remove(e as usize);
            } else if sol.colors[u] == new {
                weights.conflicting.insert(e as usize);
            }
        }
        Ok(())
    }

    fn raise_violated(
        &self,
        sol: &GraphColoringSolution,
        weights: &mut GraphColoringWeights,
        cap: u32,
    ) {
        let k = weights.k;
        for &e in weights.conflicting.as_slice() {
            if weights.weight[e] < cap {
                weights.weight[e] += 1;
                // Both ends have the one color, so each meets the other in it.
                let (i, j) = weights.edges.ends(e);
                let c = sol.colors[i];
                weights.met[i * k + c] += 1;
                weights.met[j * k + c] += 1;
            }
        }
    }

    fn lower_satisfied(&self, sol: &GraphColoringSolution, weights: &mut GraphColoringWeights) {
        let k = weights.k;
        for e in 0..weights.weight.len() {
            if weights.weight[e] > 1 && !weights.conflicting.contains(e) {
                weights.weight[e] -= 1;
                let (i, j) = weights.edges.ends(e);
                weights.met[i * k + sol.colors[j]] -= 1;
                weights.met[j * k + sol.colors[i]] -= 1;
            }
        }
    }

    /// The recolorings of the two ends of one conflicting edge, any of which
    /// ends that conflict. On a proper coloring, the recolorings that take a
    /// vertex of the smallest color class into another class in use, which
    /// empty that class one vertex at a time. A coloring with one color has
    /// none.
    fn escape_moves(
        &self,
        sol: &GraphColoringSolution,
        weights: &GraphColoringWeights,
        rng: &mut rand::rngs::SmallRng,
        out: &mut Vec<IntChangeNeighbor<2>>,
    ) {
        let conflicting = weights.conflicting.as_slice();
        if !conflicting.is_empty() {
            let (i, j) = weights
                .edges
                .ends(conflicting[rng.random_range(0..conflicting.len())]);
            for v in [i, j] {
                out.extend(
                    (0..self.k())
                        .filter(|&c| c != sol.colors[v])
                        .map(|c| IntChangeNeighbor::new(self, sol, v, c as i64)),
                );
            }
            return;
        }
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
    use crate::trait_defs::ProblemTrait;
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

    /// The weighted cost, from scratch.
    fn cost(colors: &[usize], w: &GraphColoringWeights) -> i64 {
        let mut used: Vec<usize> = colors.to_vec();
        used.sort_unstable();
        used.dedup();
        let conflicts: i64 = (0..w.edges.len())
            .filter(|&e| colors[w.edges.ends(e).0] == colors[w.edges.ends(e).1])
            .map(|e| w.weight[e] as i64)
            .sum();
        used.len() as i64 + conflicts
    }

    fn assert_priced(gc: &GraphColoring, sol: &GraphColoringSolution, w: &GraphColoringWeights) {
        let now = cost(&sol.colors, w);
        for v in 0..sol.colors.len() {
            for c in (0..gc.k()).filter(|&c| c != sol.colors[v]) {
                let mut colors = sol.colors.clone();
                colors[v] = c;
                let mv = IntChangeNeighbor::new(gc, sol, v, c as i64);
                assert_eq!(
                    gc.weighted_gain(sol, w, &mv),
                    (now - cost(&colors, w)) as f64,
                    "gain of {v} to {c}"
                );
            }
        }
        let mut got = w.conflicting.as_slice().to_vec();
        got.sort_unstable();
        let expected: Vec<usize> = (0..w.edges.len())
            .filter(|&e| sol.colors[w.edges.ends(e).0] == sol.colors[w.edges.ends(e).1])
            .collect();
        assert_eq!(got, expected);
        assert_eq!(w.conflicting_count(), sol.conflicts);
    }

    #[test]
    fn prices_follow_recolorings_and_weight_changes() {
        let gc = wheel();
        let mut rng = SmallRng::seed_from_u64(2);
        let mut sol = gc.new_solution(&mut rng);
        let mut w = gc.unit_weights(&sol);
        assert_eq!(w.edges.len(), gc.constraint_count());
        assert_priced(&gc, &sol, &w);
        let mut raised = false;
        for step in 0..400 {
            match step % 5 {
                1 => {
                    gc.raise_violated(&sol, &mut w, 5);
                    raised |= w.weight.iter().any(|&x| x > 1);
                }
                4 => gc.lower_satisfied(&sol, &mut w),
                _ => {
                    let v = rng.random_range(0..6);
                    let c = rng.random_range(0..gc.k());
                    let mv = IntChangeNeighbor::new(&gc, &sol, v, c as i64);
                    gc.apply_weighted(&mut sol, &mut w, &mv).unwrap();
                }
            }
            assert_priced(&gc, &sol, &w);
            assert!(w.weight.iter().all(|&x| (1..=5).contains(&x)));
        }
        assert!(raised, "some weight was raised");
        assert_eq!(w.weight(0, 5), w.weight(5, 0));
        assert_eq!(w.weight(0, 2), None);
    }

    #[test]
    fn escape_ends_a_conflict_or_empties_the_smallest_class() {
        let gc = wheel();
        let mut rng = SmallRng::seed_from_u64(9);
        let mut out = Vec::new();

        let clash = gc.solution_from_colors(vec![0, 0, 1, 2, 1, 3]).unwrap();
        let w = gc.unit_weights(&clash);
        assert_eq!(w.conflicting_count(), 1);
        gc.escape_moves(&clash, &w, &mut rng, &mut out);
        assert_eq!(out.len(), 2 * (gc.k() - 1));
        assert!(out.iter().all(|mv| mv.var <= 1 && mv.value != 0));

        out.clear();
        let proper = gc.solution_from_colors(vec![0, 1, 0, 1, 2, 3]).unwrap();
        let w = gc.unit_weights(&proper);
        assert_eq!(w.conflicting_count(), 0);
        gc.escape_moves(&proper, &w, &mut rng, &mut out);
        // Color 2 is the first of the two smallest classes, held by vertex 4.
        let mut targets: Vec<i64> = out.iter().map(|mv| mv.value).collect();
        targets.sort_unstable();
        assert!(out.iter().all(|mv| mv.var == 4));
        assert_eq!(targets, vec![0, 1, 3]);
    }
}
