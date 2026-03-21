//! Problem-specific heuristic algorithms.
//!
//! This module contains heuristics that are tailored to a particular problem type
//! and cannot be expressed generically through the [`Heuristic`] trait alone.

mod bls_for_max_cut;

pub use bls_for_max_cut::BreakoutLocalSearch as BreakoutLocalSearchForMaxCut;
