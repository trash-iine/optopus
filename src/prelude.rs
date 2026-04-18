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
    BangBangSimulatedAnnealing, BeamSearch, BreakoutLocalSearchForMaxCut, GeneticAlgorithm,
    Heuristic, Iterated, LateAcceptanceHillClimbing, LocalSearch, ParallelHeuristic, RandomWalk,
    Restart, Sequential, SimulatedAnnealing, StopCondition, SubProblemBasedCrossover, TabuSearch,
    boltzmann_accept,
};

// Traits
pub use crate::search_state::{
    Crossover, EnabledTabu, Evaluable, Evaluate, MoveToNeighbor, ProblemTrait, Rankable,
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
    // MaxCut
    MaxCut,
    MaxCutSolution,
    MaxCutFlipNeighbor,
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
