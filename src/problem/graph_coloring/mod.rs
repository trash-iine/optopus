//! Graph Coloring problem definition and its crossover.
//!
//! Given an undirected graph, assign each vertex a color in `0..k` so that
//! adjacent vertices differ, using as few distinct colors as possible.
//!
//! This is a self-contained, penalty-augmented formulation: the palette size
//! `k` is derived from the graph (`max_degree + 1`), and hard feasibility (a
//! proper coloring) is enforced via a penalty term:
//! `objective = colors_used + penalty_weight * conflicts`, with
//! `penalty_weight = n + 1` so the global optimum is always a proper coloring.
//!
//! The moves are the generic ones of [`IntAssignment`](crate::trait_defs::IntAssignment),
//! [`IntChangeNeighbor<2>`](crate::problem::IntChangeNeighbor) and
//! [`IntSwapNeighbor<2>`](crate::problem::IntSwapNeighbor).

mod crossover;
mod problem;
mod weights;

pub use crossover::GraphColoringUniformCrossover;
pub use problem::{GraphColoring, GraphColoringSolution};
pub use weights::GraphColoringWeights;
