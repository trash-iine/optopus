//! Problem-specific heuristic algorithms.
//!
//! This module contains heuristics that are tailored to a particular problem type
//! and cannot be expressed generically through the [`Heuristic`] trait alone.

mod bls_for_max_cut;
mod lkh_for_tsp;
mod population_annealing_for_max_cut;
mod rl_bls_for_max_cut;
mod walksat_for_sat;

pub use bls_for_max_cut::BreakoutLocalSearch as BreakoutLocalSearchForMaxCut;
pub use lkh_for_tsp::LinKernighanHelsgaun as LinKernighanHelsgaunForTsp;
pub use population_annealing_for_max_cut::PopulationAnnealing as PopulationAnnealingForMaxCut;
pub use rl_bls_for_max_cut::RlBreakoutLocalSearch as RlBreakoutLocalSearchForMaxCut;
pub use rl_bls_for_max_cut::{NUM_CONTEXT_FEATURES, NUM_PERTURBATION_TYPES};
pub use walksat_for_sat::WalkSatForSat;
