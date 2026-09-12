//! Common data structures shared across problem types.

pub mod adaptive_weights;
pub mod biased_fitness;
pub mod binary;
pub mod graph;
pub mod parse;
pub mod permutation;
pub mod ruin_recreate;
pub mod tabu;

pub use adaptive_weights::AdaptiveWeights;
pub use biased_fitness::{BiasedFitnessPopulation, binary_tournament};
pub use binary::{
    apply_swap_as_two_flips, hamming_distance, lift_binary_solution, lift_compact_binary_solution,
    uniform_binary_crossover,
};
pub use graph::{Graph, seeded_rng};
pub use parse::InstanceLines;
pub use permutation::order_crossover;
pub use ruin_recreate::{
    best_two_insertions, greedy_insertion, random_removal, regret2_insertion, shaw_removal,
    worst_removal,
};
pub use tabu::{TabuKey, TabuMemory};
