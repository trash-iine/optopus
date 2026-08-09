//! Problem-specific heuristic algorithms.
//!
//! This module contains heuristics that are tailored to a particular problem type
//! and cannot be expressed generically through the [`Heuristic`] trait alone.

mod alns_for_vrp;
mod bls_for_max_cut;
mod hgs_for_vrp;
mod lkh_for_tsp;
mod population_annealing_for_max_cut;
mod rl_bls_for_max_cut;
mod walksat_for_sat;

pub use alns_for_vrp::AdaptiveLargeNeighborhoodSearch as AdaptiveLargeNeighborhoodSearchForVrp;
pub use bls_for_max_cut::BreakoutLocalSearch as BreakoutLocalSearchForMaxCut;
pub use hgs_for_vrp::HybridGeneticSearch as HybridGeneticSearchForVrp;
pub use lkh_for_tsp::LinKernighanHelsgaun as LinKernighanHelsgaunForTsp;
pub use population_annealing_for_max_cut::PopulationAnnealing as PopulationAnnealingForMaxCut;
pub use rl_bls_for_max_cut::RlBreakoutLocalSearch as RlBreakoutLocalSearchForMaxCut;
pub use rl_bls_for_max_cut::{NUM_CONTEXT_FEATURES, NUM_PERTURBATION_TYPES};
pub use walksat_for_sat::WalkSatForSat;
