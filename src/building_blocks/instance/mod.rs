//! What a problem instance is built from and read with.
//!
//! The graph that MaxCut, VertexCover and GraphColoring are stated on, the
//! distances TSP and VRP route over, and the line reader every instance-file
//! loader parses through.

mod distance_store;
mod graph;
mod parse;

pub use distance_store::{DistanceStore, EdgeWeightType};
pub use graph::{Graph, seeded_rng};
pub use parse::{InstanceLines, InstanceTokens};
