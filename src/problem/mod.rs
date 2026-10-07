//! Problem definitions and neighborhood structures for combinatorial optimization.
//!
//! Each sub-module provides:
//! - A problem struct implementing [`crate::search_state::ProblemTrait`]
//! - A solution struct implementing [`crate::search_state::Rankable`]
//! - One or more neighborhood move types implementing [`crate::search_state::MoveToNeighbor`]
//!
//! # Available Problems
//!
//! | Module | Problem | Objective |
//! |--------|---------|-----------|
//! | [`max_cut`] | Maximum Cut | Maximize cut weight |
//! | [`qubo`] | Quadratic Unconstrained Binary Optimization | Minimize energy |
//! | [`qap`] | Quadratic Assignment | Minimize assignment cost |
//! | [`sat`] | Maximum Satisfiability (MaxSAT) | Maximize satisfied clauses |
//! | [`tsp`] | Traveling Salesman Problem | Minimize tour length |
//! | [`vertex_cover`] | Minimum Vertex Cover | Minimize cover size |
//! | [`job_shop_scheduling`] | Job Shop Scheduling | Minimize makespan |
//! | [`flow_shop`] | Permutation Flow Shop Scheduling | Minimize makespan |
//! | [`vrp`] | Capacitated + heterogeneous-fleet Vehicle Routing | Minimize total distance / time + cost |
//! | [`graph_coloring`] | Graph Coloring | Minimize colors used |
//! | [`integer`] | Any problem over bounded integer variables, by a closure or a formula | Configurable |
//!
//! [`BranchSpace`](crate::trait_defs::BranchSpace), what
//! [`BranchAndBound`](crate::heuristic::BranchAndBound) asks of a problem, is
//! implemented for the integer problems and for the binary ones.

pub mod flow_shop;
pub mod graph_coloring;
pub mod integer;
pub mod job_shop_scheduling;
pub mod max_cut;
pub mod qap;
pub mod qubo;
pub mod sat;
pub mod tsp;
pub mod vertex_cover;
pub mod vrp;

pub use flow_shop::{
    FlowShop, FlowShopInsertNeighbor, FlowShopOrderCrossover, FlowShopSolution,
    FlowShopSwapNeighbor,
};
pub use graph_coloring::{
    GraphColoring, GraphColoringRecolorNeighbor, GraphColoringSolution, GraphColoringSwapNeighbor,
    GraphColoringUniformCrossover,
};
pub use integer::{
    Constraint, ConstraintRel, Expr, FormulaProblem, FormulaSolution, IntChangeNeighbor,
    IntCrossover, IntReverseNeighbor, IntSolution, IntSwapNeighbor, IntegerProblem,
    IntervalRelaxation,
};
pub use job_shop_scheduling::{
    JobShopPpxCrossover, JobShopRelocateNeighbor, JobShopScheduling, JobShopSolution,
    JobShopSwapNeighbor,
};
pub use max_cut::{
    EigenvalueRelaxation, MaxCut, MaxCutFlipNeighbor, MaxCutKernel, MaxCutSolution,
    MaxCutSwapNeighbor, MaxCutUniformCrossover, PlantedMaxCut, TileProbs2d, TileProbs3d,
    WishartCouplers,
};
pub use qap::{Qap, QapOrderCrossover, QapSolution, QapSwapNeighbor};
pub use qubo::{Qubo, QuboFlipNeighbor, QuboSolution, QuboSwapNeighbor, QuboUniformCrossover};
pub use sat::{Sat, SatFlipNeighbor, SatSolution, SatSwapNeighbor, SatUniformCrossover};
pub use tsp::{
    Tsp, TspOrderCrossover, TspRelocateNeighbor, TspSolution, TspTour, TspTwoOptNeighbor,
};
pub use vertex_cover::{
    VertexCover, VertexCoverFlipNeighbor, VertexCoverSolution, VertexCoverSwapNeighbor,
    VertexCoverUniformCrossover,
};
pub use vrp::{
    ObjectiveMode, VehicleType, Vrp, VrpOrderCrossover, VrpRelocateNeighbor, VrpSolution,
    VrpSwapNeighbor, VrpTwoOptNeighbor, split_giant_tour,
};
