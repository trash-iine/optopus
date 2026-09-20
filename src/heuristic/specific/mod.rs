//! Problem-specific heuristic algorithms.
//!
//! This module contains heuristics that are tailored to a particular problem type
//! and cannot be expressed generically through the [`Heuristic`] trait alone.

mod max_cut;
mod tsp;
mod vrp;
mod walksat_for_sat;

pub use max_cut::{
    BreakoutLocalSearchForMaxCut, PerturbationType as MaxCutPerturbation, bls_for_max_cut,
    max_cut_descent, max_cut_perturbation,
};
pub use tsp::{LinKernighanHelsgaun as LinKernighanHelsgaunForTsp, alns_for_tsp};
pub use vrp::{HybridGeneticSearch as HybridGeneticSearchForVrp, alns_for_vrp};
pub use walksat_for_sat::WalkSat as WalkSatForSat;
