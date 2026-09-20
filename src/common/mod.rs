//! Common data structures shared across problem types.

pub mod adaptive_weights;
pub mod anchored_sweep;
pub mod biased_fitness;
pub mod binary;
pub mod distance_store;
pub mod graph;
pub mod parse;
pub mod permutation;
pub mod ruin_recreate;
pub mod tabu;

pub use adaptive_weights::AdaptiveWeights;
pub use anchored_sweep::AnchoredSweep;
pub use biased_fitness::{BiasedFitnessPopulation, CostFn, DistanceFn, binary_tournament};
pub use binary::{
    apply_swap_as_two_flips, hamming_distance, lift_binary_solution, lift_compact_binary_solution,
    uniform_binary_crossover,
};
pub use distance_store::{DistanceStore, EdgeWeightType};
pub use graph::{Graph, seeded_rng};
pub use parse::InstanceLines;
pub use permutation::order_crossover;
pub use ruin_recreate::{
    best_two_insertions, greedy_insertion, random_removal, regret2_insertion, shaw_removal,
    worst_removal,
};
pub use tabu::{TabuKey, TabuMemory};

/// The smallest objective change a search treats as a real improvement.
///
/// A descent that accepted any negative delta would cycle on ties, since two
/// moves that undo each other can each show a delta of minus one rounding
/// error. Lin-Kernighan's closing gain, the granular route descent, the
/// anchored tour descent and Hybrid Genetic Search's new-best test all ask
/// the same question of the same kind of number, so they share the answer.
/// It is not a general epsilon. Simulated annealing's exponent floor and the
/// strict-inequality margin of `FormulaProblem` mean something else and stay
/// where they are reasoned out.
pub const MIN_IMPROVEMENT: f64 = 1e-10;
