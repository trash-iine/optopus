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
