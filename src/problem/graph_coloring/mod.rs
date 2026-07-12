//! Graph Coloring problem definition and neighborhood structures.
//!
//! Given an undirected graph, assign each vertex a color in `0..k` so that
//! adjacent vertices differ, using as few distinct colors as possible.
//!
//! This is a self-contained, penalty-augmented formulation: the palette size
//! `k` is derived from the graph (`max_degree + 1`), and hard feasibility (a
//! proper coloring) is enforced via a penalty term:
//! `objective = colors_used + penalty_weight * conflicts`, with
//! `penalty_weight = n + 1` so the global optimum is always a proper coloring.

mod crossover;
mod neighbor;
mod problem;

pub use crossover::GraphColoringUniformCrossover;
pub use neighbor::{GraphColoringRecolorNeighbor, GraphColoringSwapNeighbor};
pub use problem::{GraphColoring, GraphColoringSolution};
