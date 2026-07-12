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
//! | [`sat`] | Maximum Satisfiability (MaxSAT) | Maximize satisfied clauses |
//! | [`tsp_2d`] | Traveling Salesman Problem | Minimize tour length |
//! | [`vertex_cover`] | Minimum Vertex Cover | Minimize cover size |
//! | [`job_shop_scheduling`] | Job Shop Scheduling | Minimize makespan |
//! | [`vrp`] | Capacitated Vehicle Routing | Minimize total distance |
//! | [`binary_optimization`] | Formula-based binary optimization | Configurable |

pub mod binary_optimization;
pub mod job_shop_scheduling;
pub mod max_cut;
pub mod qubo;
pub mod sat;
pub mod tsp_2d;
pub mod vertex_cover;
pub mod vrp;

pub use binary_optimization::{
    Constraint, ConstraintRel, Expr, FormulaFlipNeighbor, FormulaProblem, FormulaSolution,
    FormulaSwapNeighbor, FormulaUniformCrossover, OptDirection,
};
pub use job_shop_scheduling::{
    JobShopPpxCrossover, JobShopRelocateNeighbor, JobShopScheduling, JobShopSolution,
    JobShopSwapNeighbor,
};
pub use max_cut::{
    MaxCut, MaxCutFlipNeighbor, MaxCutSolution, MaxCutSwapNeighbor, MaxCutUniformCrossover,
};
pub use qubo::{Qubo, QuboFlipNeighbor, QuboSolution, QuboSwapNeighbor, QuboUniformCrossover};
pub use sat::{Sat, SatFlipNeighbor, SatSolution, SatSwapNeighbor, SatUniformCrossover};
pub use tsp_2d::{
    TspOrderCrossover, TspRelocateNeighbor, TspSolution, TspTour, TspTwoOptNeighbor,
    TspWithCoordinates,
};
pub use vertex_cover::{
    VertexCover, VertexCoverFlipNeighbor, VertexCoverSolution, VertexCoverSwapNeighbor,
    VertexCoverUniformCrossover,
};
pub use vrp::{
    Vrp, VrpOrderCrossover, VrpRelocateNeighbor, VrpSolution, VrpSwapNeighbor, VrpTwoOptNeighbor,
};
