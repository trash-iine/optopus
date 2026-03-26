//! Travelling Salesman Problem (TSP) definition and neighborhood structures.
//!
//! Given a set of cities with 2D coordinates, TSP seeks a tour that visits every city
//! exactly once and minimizes the total Euclidean distance.

mod neighbor;
mod problem;

pub use neighbor::{TspRelocateNeighbor, TspTwoOptNeighbor};
pub use problem::{TspSolution, TspTour, TspWithCoordinates};
