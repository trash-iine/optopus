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
pub use crate::search_state::{EnabledTabu, Evaluable, MoveToNeighbor, ProblemTrait, Rankable};

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
    // MaxCut
    MaxCut,
    MaxCutFlipNeighbor,
    MaxCutSwapNeighbor,
    OptDirection,
    // QUBO
    Qubo,
    QuboFlipNeighbour,
    QuboSolution,
    QuboSwapNeighbour,
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
};
