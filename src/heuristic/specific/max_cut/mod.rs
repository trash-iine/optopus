//! MaxCut-specific heuristics.
//!
//! [`ops`] holds the operators these heuristics share that the library has no
//! generic equivalent for — the tabu walk and the two-sided swap. What they
//! share beyond the code is the tabu memory of the `SearchState` they are
//! handed. Each of the other modules is one heuristic built on top of that.
//!
//! [`BreakoutLocalSearch`] additionally exposes its round as two halves
//! (`descend` / `kick`), so a controller outside the library can supply its own
//! perturbation rule: `examples/rl_bls.rs` drives it with a learned one.

mod bls;
mod ops;
mod population_annealing;

pub use bls::{BreakoutLocalSearch, PerturbationType};
pub use population_annealing::PopulationAnnealing;
