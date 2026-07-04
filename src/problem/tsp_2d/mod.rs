//! Traveling Salesman Problem (TSP) definition and neighborhood structures.
//!
//! Given a set of cities with 2D coordinates, TSP seeks a tour that visits every city
//! exactly once and minimizes the total Euclidean distance.

mod crossover;
mod neighbor;
mod problem;

pub use crossover::TspOrderCrossover;
pub use neighbor::{TspRelocateNeighbor, TspTwoOptNeighbor};
pub use problem::{EdgeWeightType, TspSolution, TspTour, TspWithCoordinates};
