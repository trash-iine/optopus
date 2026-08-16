//! Capacitated Vehicle Routing Problem (CVRP) definition and neighborhoods.
//!
//! A depot and `n` customers with 2D coordinates and demands are served by a
//! fixed fleet of capacity-limited vehicles. The goal is to minimize total
//! travel distance such that every customer is visited exactly once and no
//! route exceeds capacity (enforced with a penalty, see [`VrpSolution`]).

mod adjacency;
mod crossover;
mod neighbor;
mod problem;
mod split;

pub use crossover::VrpOrderCrossover;
pub use neighbor::{VrpRelocateNeighbor, VrpSwapNeighbor, VrpTwoOptNeighbor};
pub use problem::{VRP_DIST_MATRIX_MAX_N, Vrp, VrpSolution};
pub use split::split_giant_tour;

/// The capacity overflow of a route load, shared with the VRP heuristics so the
/// penalty they search under is computed exactly as [`VrpSolution`]'s is.
pub(crate) use problem::overload_of;

/// Who each customer is served between, shared with the VRP heuristics so that
/// solution diversity means one thing in this crate — see
/// [`Distance for VrpSolution`](VrpSolution#impl-Distance-for-VrpSolution).
pub(crate) use adjacency::RouteAdjacency;
