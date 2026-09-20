//! Traveling Salesman Problem (TSP) definition and neighborhood structures.
//!
//! Given the pairwise distances between a set of cities, TSP seeks a tour that
//! visits every city exactly once and minimizes the total distance. An instance
//! is built from 2D coordinates and a TSPLIB distance formula, keeping either
//! the full distance matrix or the nearest neighbours of every city, or from a
//! distance matrix given directly.

mod crossover;
mod neighbor;
mod problem;
mod ruin;

pub use crossover::TspOrderCrossover;
pub use neighbor::{TspRelocateNeighbor, TspTwoOptNeighbor};
pub use problem::{EdgeWeightType, NeighborLists, Tsp, TspSolution, TspTour};
pub use ruin::{AnchoredTourDescent, TspPartial};
