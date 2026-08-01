//! MaxCut-specific heuristics.
//!
//! [`ops`] holds the shared search engine — the tabu map, the gain-indexed
//! descent and the perturbation operators — that [`bls`] and [`rl_bls`] both
//! drive. Each of the other modules is one heuristic built on top of it.

mod bls;
mod ops;
mod population_annealing;
mod rl_bls;

pub use bls::BreakoutLocalSearch as BreakoutLocalSearchForMaxCut;
pub use population_annealing::PopulationAnnealing as PopulationAnnealingForMaxCut;
pub use rl_bls::RlBreakoutLocalSearch as RlBreakoutLocalSearchForMaxCut;
pub use rl_bls::{NUM_CONTEXT_FEATURES, NUM_PERTURBATION_TYPES};
