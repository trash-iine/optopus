//! MaxCut-specific heuristics.
//!
//! [`best_swap::BestSwap`] is the one operator here with no generic equivalent
//! in the library. `M2` moves one vertex per partition side in a single move,
//! which a pair of independent one-step searches cannot express. Everything else a
//! Breakout Local Search round does is a generic heuristic driven from
//! [`bls`], a [`LocalSearch`](crate::heuristic::LocalSearch) descent, a
//! [`RandomWalk`](crate::heuristic::RandomWalk) strong kick and a
//! [`TabuSearch`](crate::heuristic::TabuSearch) weak flip, all sharing the
//! tabu memory of the `SearchState` they are handed.
//!
//! [`max_cut_descent`] and [`max_cut_perturbation`] hand those pieces out one
//! at a time, so a [`PerturbationSchedule`](crate::heuristic::PerturbationSchedule)
//! of your own reuses the operators rather than rebuilding them.
//! `examples/rl_bls.rs` writes one around a contextual bandit.

mod best_swap;
mod bls;

pub use bls::{
    BreakoutLocalSearchForMaxCut, PerturbationType, bls_for_max_cut, max_cut_descent,
    max_cut_perturbation,
};
