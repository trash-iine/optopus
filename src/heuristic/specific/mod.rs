//! Problem-specific heuristic algorithms.
//!
//! This module contains heuristics that are tailored to a particular problem type
//! and cannot be expressed generically through the [`Heuristic`] trait alone.
//!
//! MaxCut has enough of them — and enough shared machinery between them — to
//! warrant its own directory; TSP and SAT have one each and stay flat.

mod lkh_for_tsp;
mod max_cut;
mod walksat_for_sat;

// This is where heuristics for different problems meet, so this is where the
// problem suffix goes on: the modules below name their types for what they do,
// not for what they do it to.
pub use lkh_for_tsp::LinKernighanHelsgaun as LinKernighanHelsgaunForTsp;
pub use max_cut::{
    BreakoutLocalSearch as BreakoutLocalSearchForMaxCut, NUM_CONTEXT_FEATURES,
    NUM_PERTURBATION_TYPES, PopulationAnnealing as PopulationAnnealingForMaxCut,
    RlBreakoutLocalSearch as RlBreakoutLocalSearchForMaxCut,
};
pub use walksat_for_sat::WalkSat as WalkSatForSat;
