use crate::building_blocks::instance::Graph;
use crate::trait_defs::{Evaluable, FixVariables, FixedVariables, Placed};

use super::problem::MaxCut;

crate::binary_branch_space!(MaxCut);

impl FixVariables for MaxCut {
    /// The free vertices, numbered in order, and one reference vertex standing
    /// for every fixed one.
    ///
    /// An edge from a free vertex to a vertex fixed on side `false` is cut when
    /// the free vertex sits apart from the reference, and becomes an edge to
    /// it. One to a vertex fixed on side `true` is cut when they sit together,
    /// which is its weight in the offset less an edge to the reference of the
    /// negated weight. Flipping every vertex changes no cut, so reading the
    /// free vertices relative to the reference makes the folding exact for
    /// every solution.
    fn fix(&self, fixed: &[Option<bool>]) -> FixedVariables<MaxCut> {
        let mut placed = vec![Placed::Fixed(false); fixed.len()];
        let mut free = 0;
        for &v in self.graph.iter_on_vertices() {
            placed[v] = match fixed[v] {
                Some(side) => Placed::Fixed(side),
                None => {
                    free += 1;
                    Placed::Free(free - 1)
                }
            };
        }
        let reference = free;
        let mut graph = Graph::new();
        let mut offset = 0.0;
        for (u, v, w) in self.graph.edges() {
            match (placed[u], placed[v]) {
                (Placed::Free(a), Placed::Free(b)) => graph.add_weight(a, b, w),
                (Placed::Free(a), Placed::Fixed(side)) | (Placed::Fixed(side), Placed::Free(a)) => {
                    if side {
                        offset += f64::from(w);
                        graph.add_weight(a, reference, -w);
                    } else {
                        graph.add_weight(a, reference, w);
                    }
                }
                (Placed::Fixed(x), Placed::Fixed(y)) => {
                    if x != y {
                        offset += f64::from(w);
                    }
                }
            }
        }
        FixedVariables {
            target: MaxCut::new(graph),
            placed,
            offset,
            reference: Some(reference),
        }
    }

    /// Every edge of positive weight cut.
    fn trivial_bound(&self) -> Evaluable<f64> {
        Evaluable::Maximize(
            self.graph
                .edges()
                .map(|(_, _, w)| f64::from(w.max(0.0)))
                .sum(),
        )
    }
}
