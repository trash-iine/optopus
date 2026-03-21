//! Maximum Cut (MaxCut) problem definition and neighborhood structures.
//!
//! Given an undirected weighted graph, MaxCut seeks a partition of the vertices into
//! two sets that maximizes the total weight of edges crossing the partition.

mod neighbor;
mod problem;

pub use neighbor::{MaxCutFlipNeighbor, MaxCutSwapNeighbor};
pub use problem::{MaxCut, MaxCutSolution};
