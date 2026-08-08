//! Problem-specific heuristic algorithms.
//!
//! This module contains heuristics that are tailored to a particular problem type
//! and cannot be expressed generically through the [`Heuristic`] trait alone.

mod lkh_for_tsp;
mod max_cut;
mod walksat_for_sat;

pub use lkh_for_tsp::LinKernighanHelsgaun as LinKernighanHelsgaunForTsp;
pub use max_cut::{
    BreakoutLocalSearch as BreakoutLocalSearchForMaxCut, NUM_CONTEXT_FEATURES,
    NUM_PERTURBATION_TYPES, PopulationAnnealing as PopulationAnnealingForMaxCut,
    RlBreakoutLocalSearch as RlBreakoutLocalSearchForMaxCut,
};
pub use walksat_for_sat::WalkSat as WalkSatForSat;
