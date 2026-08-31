//! MaxCut-specific heuristics.
//!
//! [`ops`] holds the shared search engine — the tabu map, the gain-indexed
//! descent and the perturbation operators — that the heuristics here drive.
//! Each of the other modules is one heuristic built on top of it.
//!
//! [`BreakoutLocalSearch`] additionally exposes its round as two halves
//! (`descend` / `kick`), so a controller outside the library can supply its own
//! perturbation rule: `examples/rl_bls.rs` drives it with a learned one.

mod bls;
mod ops;
mod population_annealing;

pub use bls::{BreakoutLocalSearch, PerturbationType};
pub use population_annealing::PopulationAnnealing;
