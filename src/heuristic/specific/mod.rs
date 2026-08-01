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

pub use lkh_for_tsp::LinKernighanHelsgaun as LinKernighanHelsgaunForTsp;
pub use max_cut::{
    BreakoutLocalSearchForMaxCut, NUM_CONTEXT_FEATURES, NUM_PERTURBATION_TYPES,
    PopulationAnnealingForMaxCut, RlBreakoutLocalSearchForMaxCut,
};
pub use walksat_for_sat::WalkSatForSat;
