//! Capacitated Vehicle Routing Problem (CVRP) definition and neighborhoods.
//!
//! A depot and `n` customers with 2D coordinates and demands are served by a
//! fixed fleet of capacity-limited vehicles. The goal is to minimize total
//! travel distance such that every customer is visited exactly once and no
//! route exceeds capacity (enforced with a penalty, see [`VrpSolution`]).

mod crossover;
mod neighbor;
mod problem;

pub use crossover::VrpOrderCrossover;
pub use neighbor::{VrpRelocateNeighbor, VrpSwapNeighbor, VrpTwoOptNeighbor};
pub use problem::{VRP_DIST_MATRIX_MAX_N, Vrp, VrpSolution};
