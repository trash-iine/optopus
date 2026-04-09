//! Minimum Vertex Cover problem definition and neighborhood structures.
//!
//! Given an undirected graph, Vertex Cover seeks the smallest subset of vertices
//! such that every edge has at least one endpoint in the subset.
//!
//! Hard feasibility (every edge covered) is enforced via a penalty term:
//! `objective = |cover| + penalty_weight * uncovered_edges`, with
//! `penalty_weight = n + 1` so the global optimum is always feasible.

mod crossover;
mod neighbor;
mod problem;

pub use crossover::VertexCoverUniformCrossover;
pub use neighbor::{VertexCoverFlipNeighbor, VertexCoverSwapNeighbor};
pub use problem::{VertexCover, VertexCoverSolution};
