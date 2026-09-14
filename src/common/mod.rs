//! Common data structures shared across problem types.

pub mod biased_fitness;
pub mod binary;
pub mod graph;
pub mod parse;
pub mod permutation;
pub mod tabu;

pub use biased_fitness::{BiasedFitnessPopulation, CostFn, DistanceFn, binary_tournament};
pub use binary::{
    apply_swap_as_two_flips, hamming_distance, lift_binary_solution, lift_compact_binary_solution,
    uniform_binary_crossover,
};
pub use graph::{Graph, seeded_rng};
pub use parse::InstanceLines;
pub use permutation::order_crossover;
pub use tabu::{TabuKey, TabuMemory};
