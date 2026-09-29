//! Vehicle Routing Problem definitions and neighborhoods.
//!
//! A depot and `n` customers with 2D coordinates, demands and service times
//! are served by a fleet of one or more [`VehicleType`]s, each with its own
//! capacity, speed, costs, minimum usage and route-time limit. The
//! homogeneous CVRP is the one-type instance. The objective is a time term,
//! the total of all route durations or the longest one, plus the fleet's
//! cost, with capacity, route time and minimum fleet usage enforced through a
//! penalty, see [`Vrp`].

mod adjacency;
mod crossover;
mod neighbor;
pub(crate) mod ops;
mod problem;
mod ruin;
mod split;

pub use crossover::VrpOrderCrossover;
pub use neighbor::{VrpRelocateNeighbor, VrpSwapNeighbor, VrpTwoOptNeighbor};
pub use problem::{ObjectiveMode, VehicleType, Vrp, VrpSolution};
pub use ruin::{AnchoredRouteDescent, VrpPartial};
pub use split::split_giant_tour;

/// Who each customer is served between, shared with the VRP heuristics so that
/// solution diversity means one thing in this crate, see
/// [`Distance for VrpSolution`](VrpSolution#impl-Distance-for-VrpSolution).
pub(crate) use adjacency::RouteAdjacency;
