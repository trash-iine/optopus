//! Maximum Cut (MaxCut) problem definition and neighborhood structures.
//!
//! Given an undirected weighted graph, MaxCut seeks a partition of the vertices
//! into two sets that maximizes the total weight of edges crossing the
//! partition.
//!
//! ```
//! use optopus::prelude::*;
//!
//! let mc = MaxCut::from_edges([(0, 1, 1.0), (0, 2, 1.0), (1, 2, 1.0)]);
//!
//! let mut state = SearchState::new(&mc);
//! let mut ls = LocalSearch::<MaxCutFlipNeighbor>::new(
//!     StopCondition::iterations(10_000),
//! );
//! ls.run(&mut state).unwrap();
//! println!("best = {}", state.best_solution.objective);
//! ```
//!
//! [`MaxCut`] is the instance and wraps a [`Graph`](crate::common::Graph),
//! which is where the graph builders and accessors are documented.
//! [`MaxCutSolution`] is the partition, carrying `x`, `gain` and `objective`.
//!
//! | Move | Description | Iteration cost |
//! |---|---|---|
//! | [`MaxCutFlipNeighbor`] | move one vertex to the opposite side | `+1` |
//! | [`MaxCutSwapNeighbor`] | exchange two vertices on opposite sides | `+2` |
//!
//! [`MaxCutUniformCrossover`] recombines two partitions per vertex, and
//! [`MaxCutKernel`] reduces an instance exactly before a search runs on it.
//! The [MaxCut page](https://trash-iine.github.io/optopus/problems/max_cut/)
//! covers the file format, the planted-optimum suites and worked examples.

mod crossover;
mod kernel;
mod neighbor;
mod planted;
mod problem;

pub use crossover::MaxCutUniformCrossover;
pub use kernel::MaxCutKernel;
pub use neighbor::{MaxCutFlipNeighbor, MaxCutSwapNeighbor};
pub use planted::{PlantedMaxCut, TileProbs2d, TileProbs3d, WishartCouplers};
pub use problem::{MaxCut, MaxCutSolution};

/// Instances the crate's own tests search over.
///
/// They live with the problem rather than with any one search, because three
/// modules in two layers want the same graph and a fixture is instance data
/// like any other.
#[cfg(test)]
pub(crate) mod test_fixtures {
    use super::MaxCut;

    /// A small toroidal-like graph (degree 4, unit weights) that keeps both
    /// partition sides populated throughout a search, and whose plateaus are
    /// wide enough that tie-breaking shows up.
    pub(crate) fn small_instance() -> MaxCut {
        let n = 30usize;
        let mut edges = Vec::new();
        for i in 0..n {
            edges.push((i, (i + 1) % n, 1.0));
            edges.push((i, (i + 2) % n, 1.0));
        }
        MaxCut::from_edges(edges)
    }
}
