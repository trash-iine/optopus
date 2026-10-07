use crate::building_blocks::instance::Graph;
use crate::trait_defs::{Evaluable, FixVariables, FixedVariables, Placed};

use super::problem::VertexCover;

crate::binary_branch_space!(VertexCover);

impl FixVariables for VertexCover {
    /// The free vertices, numbered in order, over the edges neither end of
    /// which is in the cover.
    ///
    /// A free vertex next to one fixed out of the cover is put in it as well,
    /// since leaving it out costs more than the vertex. The offset counts the
    /// vertices in the cover and charges the edges both of whose ends are
    /// fixed out. The folded instance charges an uncovered edge less than the
    /// whole one does, its penalty following its own size, so the two agree on
    /// the best solution rather than on every one.
    fn fix(&self, fixed: &[Option<bool>]) -> FixedVariables<VertexCover> {
        let mut forced = vec![false; fixed.len()];
        for (u, v, _) in self.graph.edges() {
            match (fixed[u], fixed[v]) {
                (Some(false), None) => forced[v] = true,
                (None, Some(false)) => forced[u] = true,
                _ => {}
            }
        }
        let mut placed = vec![Placed::Fixed(false); fixed.len()];
        let mut free = 0;
        for &v in self.graph.iter_on_vertices() {
            placed[v] = match fixed[v] {
                Some(value) => Placed::Fixed(value),
                None if forced[v] => Placed::Fixed(true),
                None => {
                    free += 1;
                    Placed::Free(free - 1)
                }
            };
        }
        let mut graph = Graph::new();
        let mut uncovered = 0;
        for (u, v, _) in self.graph.edges() {
            match (placed[u], placed[v]) {
                (Placed::Free(a), Placed::Free(b)) => graph.add_edge(a, b),
                (Placed::Fixed(false), Placed::Fixed(false)) => uncovered += 1,
                _ => {}
            }
        }
        let in_cover = placed.iter().filter(|&&p| p == Placed::Fixed(true)).count();
        FixedVariables {
            target: VertexCover::new(graph),
            placed,
            offset: in_cover as f64 + f64::from(self.penalty_weight()) * f64::from(uncovered),
            reference: None,
        }
    }

    /// The size of a maximal matching, found greedily. Every edge of a
    /// matching needs its own vertex in any cover.
    fn trivial_bound(&self) -> Evaluable<f64> {
        let mut matched = vec![false; self.graph.len()];
        let mut size = 0;
        for (u, v, _) in self.graph.edges() {
            if !matched[u] && !matched[v] {
                matched[u] = true;
                matched[v] = true;
                size += 1;
            }
        }
        Evaluable::Minimize(f64::from(size))
    }
}
