//! MaxCut-specific heuristics.
//!
//! [`best_swap`] is the one operator here with no generic equivalent in the
//! library: `M2` moves one vertex per partition side in a *single* move, which
//! a pair of independent one-step searches cannot express. Everything else a
//! Breakout Local Search round does is a generic heuristic driven from
//! [`bls`] — a [`LocalSearch`](crate::heuristic::LocalSearch) descent, a
//! [`RandomWalk`](crate::heuristic::RandomWalk) strong kick and a
//! [`TabuSearch`](crate::heuristic::TabuSearch) weak flip — all sharing the
//! tabu memory of the `SearchState` they are handed.
//!
//! [`BreakoutLocalSearch`] additionally exposes its round as two halves
//! (`descend` / `kick`), so a controller outside the library can supply its own
//! perturbation rule: `examples/rl_bls.rs` drives it with a learned one.

mod best_swap;
mod bls;
mod population_annealing;

pub use bls::{BreakoutLocalSearch, PerturbationType};
pub use population_annealing::PopulationAnnealing;
