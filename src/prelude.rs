//! Convenience re-exports of the most commonly used types and traits.
//!
//! Import everything with:
//!
//! ```rust
//! use optopus::prelude::*;
//! ```

// Common types
pub use crate::common::{Graph, seeded_rng};

// Error type
pub use crate::error::OptError;

// Search state
pub use crate::search_state::{SearchState, SearchStateCloneType, TrajectoryPoint};

// Heuristics
pub use crate::heuristic::{
    AdaptiveLargeNeighborhoodSearch, AdaptivePerturbation, BangBangSimulatedAnnealing, BeamSearch,
    BranchAndBound, BreakoutLocalSearch, BreakoutLocalSearchForMaxCut, GeneticAlgorithm, Heuristic,
    HybridGeneticSearchForVrp, Iterated, LateAcceptanceHillClimbing, LinKernighanHelsgaunForTsp,
    LocalSearch, MaxCutPerturbation, ParentSelection, PerturbationSchedule, RandomWalk,
    ReinforcementLearningSearch, Restart, RewardShaping, Sequential, SimulatedAnnealing,
    StopCondition, SubProblemBasedCrossover, TabuSearch, VariableNeighborhoodSearch, WalkSatForSat,
    alns_for_tsp, alns_for_vrp, bls_for_max_cut, boltzmann_accept, max_cut_descent,
    max_cut_perturbation,
};

// Traits
pub use crate::trait_defs::{
    Crossover, Distance, EnabledTabu, Evaluable, Evaluate, MoveToNeighbor, ProblemTrait, Rankable,
    SubProblemExtractable,
};

// Problem and neighbor types
pub use crate::problem::{
    // Branch-and-bound
    BinaryRelaxation,
    Branchable,
    // Formula
    Constraint,
    ConstraintRel,
    Expr,
    FormulaProblem,
    FormulaSolution,
    // Graph Coloring
    GraphColoring,
    GraphColoringRecolorNeighbor,
    GraphColoringSolution,
    GraphColoringSwapNeighbor,
    // Integer variables
    IntAssignment,
    IntChangeNeighbor,
    IntCrossover,
    IntReverseNeighbor,
    IntSolution,
    IntSwapNeighbor,
    IntVar,
    IntVars,
    IntegerProblem,
    IntervalRelaxation,
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
    // VRP objective selector
    ObjectiveMode,
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
    Tsp,
    TspRelocateNeighbor,
    TspSolution,
    TspTour,
    TspTwoOptNeighbor,
    // VRP fleet description
    VehicleType,
    // Vertex Cover
    VertexCover,
    VertexCoverFlipNeighbor,
    VertexCoverSolution,
    VertexCoverSwapNeighbor,
    // VRP
    Vrp,
    VrpRelocateNeighbor,
    VrpSolution,
    VrpSwapNeighbor,
    VrpTwoOptNeighbor,
};

// The eigenvalue bound of MaxCut, a relaxation for BranchAndBound
pub use crate::problem::EigenvalueRelaxation;
