//! Problem-specific heuristic algorithms.
//!
//! This module contains heuristics that are tailored to a particular problem type
//! and cannot be expressed generically through the [`Heuristic`] trait alone.

mod lkh_for_tsp;
mod max_cut;
mod vrp;
mod walksat_for_sat;

pub use lkh_for_tsp::LinKernighanHelsgaun as LinKernighanHelsgaunForTsp;
pub use max_cut::{
    BreakoutLocalSearchForMaxCut, ExternallyDrivenBlsForMaxCut,
    PerturbationType as MaxCutPerturbation, breakout_local_search_for_max_cut,
    externally_driven_bls_for_max_cut,
};
pub use vrp::HybridGeneticSearch as HybridGeneticSearchForVrp;
pub use walksat_for_sat::WalkSat as WalkSatForSat;
