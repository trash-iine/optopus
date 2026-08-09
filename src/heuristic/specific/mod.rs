//! Problem-specific heuristic algorithms.
//!
//! This module contains heuristics that are tailored to a particular problem type
//! and cannot be expressed generically through the [`Heuristic`] trait alone.

mod alns_for_vrp;
mod hgs_for_vrp;
mod lkh_for_tsp;
mod max_cut;
mod walksat_for_sat;

pub use alns_for_vrp::AdaptiveLargeNeighborhoodSearch as AdaptiveLargeNeighborhoodSearchForVrp;
pub use hgs_for_vrp::HybridGeneticSearch as HybridGeneticSearchForVrp;
pub use lkh_for_tsp::LinKernighanHelsgaun as LinKernighanHelsgaunForTsp;
pub use max_cut::{
    BreakoutLocalSearch as BreakoutLocalSearchForMaxCut,
    KernelizedSearch as KernelizedSearchForMaxCut, NUM_CONTEXT_FEATURES, NUM_PERTURBATION_TYPES,
    PopulationAnnealing as PopulationAnnealingForMaxCut,
    RlBreakoutLocalSearch as RlBreakoutLocalSearchForMaxCut,
};
pub use walksat_for_sat::WalkSat as WalkSatForSat;
