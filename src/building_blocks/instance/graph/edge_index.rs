//! The edges of a graph, numbered.

use super::Graph;

/// The edges of a [`Graph`] numbered from zero, with the edges at each vertex.
///
/// A graph stores its edges as adjacency lists, which name no edge. Something
/// kept per edge, such as a weight on each one, needs a number for it, and
/// needs to walk the numbers of the edges at a vertex. A loop from a vertex to
/// itself is left out.
///
/// ```
/// use optopus::building_blocks::instance::{EdgeIndex, Graph};
///
/// let graph = Graph::from_edges([(0, 1, 1.0), (1, 2, 1.0)]);
/// let edges = EdgeIndex::new(&graph);
/// assert_eq!(edges.len(), 2);
/// assert_eq!(edges.ends(0), (0, 1));
/// assert_eq!(edges.at(1), &[(0, 0), (1, 2)]);
/// assert_eq!(edges.between(2, 1), Some(1));
/// ```
#[derive(Debug, Clone)]
pub struct EdgeIndex {
    /// The two ends of each edge, the smaller first.
    ends: Vec<(usize, usize)>,
    /// `incident[first[v]..first[v + 1]]` lists the edges at vertex `v`, each
    /// with the vertex at its other end.
    incident: Vec<(usize, usize)>,
    first: Vec<usize>,
}

impl EdgeIndex {
    /// Numbers the edges of `graph` in the order [`Graph::edges`] lists them.
    ///
    /// Each vertex's edges come out in the order of the vertex at their other
    /// end, which is the order of its adjacency list.
    pub fn new(graph: &Graph) -> Self {
        let n = graph.len();
        let ends: Vec<(usize, usize)> = graph.edges().map(|(i, j, _)| (i, j)).collect();
        let mut first = vec![0; n + 1];
        for &(i, j) in &ends {
            first[i + 1] += 1;
            first[j + 1] += 1;
        }
        for v in 0..n {
            first[v + 1] += first[v];
        }
        let mut next = first[..n].to_vec();
        let mut incident = vec![(0, 0); first[n]];
        for (e, &(i, j)) in ends.iter().enumerate() {
            incident[next[i]] = (e, j);
            next[i] += 1;
            incident[next[j]] = (e, i);
            next[j] += 1;
        }
        Self {
            ends,
            incident,
            first,
        }
    }

    /// How many edges there are.
    pub fn len(&self) -> usize {
        self.ends.len()
    }

    /// Whether there is no edge.
    pub fn is_empty(&self) -> bool {
        self.ends.is_empty()
    }

    /// The two ends of edge `e`, the smaller first.
    #[inline]
    pub fn ends(&self, e: usize) -> (usize, usize) {
        self.ends[e]
    }

    /// The edges at vertex `v`, each as its number and the vertex at its
    /// other end.
    #[inline]
    pub fn at(&self, v: usize) -> &[(usize, usize)] {
        &self.incident[self.first[v]..self.first[v + 1]]
    }

    /// The number of the edge between `i` and `j`, or `None` when they are
    /// not adjacent.
    pub fn between(&self, i: usize, j: usize) -> Option<usize> {
        if i + 1 >= self.first.len() {
            return None;
        }
        self.at(i)
            .iter()
            .find(|&&(_, other)| other == j)
            .map(|&(e, _)| e)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_edge_is_numbered_once_and_listed_at_both_ends() {
        let graph = Graph::from_edges([
            (0, 1, 1.0),
            (1, 2, 1.0),
            (2, 0, 1.0),
            (3, 0, 1.0),
            (4, 4, 1.0),
        ]);
        let edges = EdgeIndex::new(&graph);
        assert_eq!(edges.len(), 4, "the loop at 4 is left out");
        for e in 0..edges.len() {
            let (i, j) = edges.ends(e);
            assert!(i < j && graph.has_edge(i, j));
            assert!(edges.at(i).contains(&(e, j)));
            assert!(edges.at(j).contains(&(e, i)));
            assert_eq!(edges.between(i, j), Some(e));
            assert_eq!(edges.between(j, i), Some(e));
        }
        assert_eq!(edges.between(1, 3), None);
        assert_eq!(edges.between(9, 0), None);
        assert!(edges.at(4).is_empty());
    }
}
