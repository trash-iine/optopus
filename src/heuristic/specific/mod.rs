//! Problem-specific heuristic algorithms.
//!
//! This module contains heuristics that are tailored to a particular problem type
//! and cannot be expressed generically through the [`Heuristic`] trait alone.

mod bls_for_max_cut;
mod lkh_for_tsp;

pub use bls_for_max_cut::BreakoutLocalSearch as BreakoutLocalSearchForMaxCut;
pub use lkh_for_tsp::LinKernighanHelsgaun as LinKernighanHelsgaunForTsp;
