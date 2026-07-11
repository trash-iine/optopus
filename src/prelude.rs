//! Convenience re-exports of the most commonly used types and traits.
//!
//! Import everything with:
//!
//! ```rust
//! use optopus::prelude::*;
//! ```

// Common types
pub use crate::common::Graph;

// Error type
pub use crate::error::OptError;

// Search state
pub use crate::search_state::{SearchState, SearchStateCloneType, TrajectoryPoint};

// Heuristics
pub use crate::heuristic::{
    BangBangSimulatedAnnealing, BeamSearch, BreakoutLocalSearchForMaxCut, GeneticAlgorithm,
    Heuristic, Iterated, LateAcceptanceHillClimbing, LinKernighanHelsgaunForTsp, LocalSearch,
    ParentSelection, RandomWalk, Restart, RewardShaping, RlBreakoutLocalSearchForMaxCut, RlSearch,
    Sequential, SimulatedAnnealing, StopCondition, SubProblemBasedCrossover, TabuSearch,
    boltzmann_accept,
};

// Traits
pub use crate::trait_defs::{
    Crossover, Distance, EnabledTabu, Evaluable, Evaluate, MoveToNeighbor, ProblemTrait, Rankable,
    SubProblemExtractable,
};

// Problem and neighbor types
pub use crate::problem::{
    // Formula
    Constraint,
    ConstraintRel,
    Expr,
    FormulaFlipNeighbor,
    FormulaProblem,
    FormulaSolution,
    FormulaSwapNeighbor,
    FormulaUniformCrossover,
    // Job Shop Scheduling
    JobShopPpxCrossover,
    JobShopRelocateNeighbor,
    JobShopScheduling,
    JobShopSolution,
    JobShopSwapNeighbor,
    // MaxCut
    MaxCut,
    MaxCutFlipNeighbor,
    MaxCutSolution,
    MaxCutSwapNeighbor,
    OptDirection,
    // QUBO
    Qubo,
    QuboFlipNeighbor,
    QuboSolution,
    QuboSwapNeighbor,
    // SAT
    Sat,
    SatFlipNeighbor,
    SatSolution,
    SatSwapNeighbor,
    // TSP
    TspRelocateNeighbor,
    TspSolution,
    TspTour,
    TspTwoOptNeighbor,
    TspWithCoordinates,
    // Vertex Cover
    VertexCover,
    VertexCoverFlipNeighbor,
    VertexCoverSolution,
    VertexCoverSwapNeighbor,
};
