//! Convenience re-exports of the most commonly used types and traits.
//!
//! Import everything with:
//!
//! ```rust
//! use optopus::prelude::*;
//! ```

// Error type
pub use crate::error::OptError;

// Search state
pub use crate::search_state::{SearchState, SearchStateCloneType};

// Heuristics
pub use crate::heuristic::{
    BangBangSimulatedAnnealing, BeamSearch, BreakoutLocalSearchForMaxCut, Heuristic, LocalSearch,
    ParallelHeuristic, RandomWalk, Sequential, SimulatedAnnealing, StopCondition, TabuSearch,
};

// Traits
pub use crate::search_state::{
    EnabledTabu, Evaluable, MoveToNeigbor, ProblemTrait, Rankable,
};

// Problem and neighbor types
pub use crate::problem::{
    // MaxCut
    MaxCut,
    MaxCutFlipNeighbor,
    MaxCutSwapNeighbor,
    // SAT
    Sat,
    SatFlipNeighbor,
    SatSolution,
    SatSwapNeighbor,
    // QUBO
    Qubo,
    QuboFlipNeighbour,
    QuboSolution,
    QuboSwapNeighbour,
    // TSP
    TspRelocateNeighbor,
    TspSolution,
    TspTour,
    TspTwoOptNeighbor,
    TspWithCoordinates,
    // Formula
    Constraint,
    ConstraintRel,
    Expr,
    FormulaProblem,
    FormulaSolution,
    FormulaFlipNeighbor,
    FormulaSwapNeighbor,
    OptDirection,
};
