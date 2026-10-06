//! Search machinery that knows nothing about any problem.
//!
//! The tabu memory every move writes to, the adaptive roulette and the ruin
//! and recreate operators of ALNS, the biased-fitness population of the
//! genetic searches, and the anchored sweep the ruin repairs descend with.
//! Heuristics, `SearchState` and the problems' own moves and descents all use
//! these, and none of them is about a particular problem.

mod adaptive_weights;
mod anchored_sweep;
mod biased_fitness;
mod ruin_recreate;
mod tabu;

pub use adaptive_weights::AdaptiveWeights;
pub use anchored_sweep::AnchoredSweep;
pub use biased_fitness::{BiasedFitnessPopulation, CostFn, DistanceFn, binary_tournament};
pub use ruin_recreate::{
    best_two_insertions, greedy_insertion, random_removal, regret2_insertion, shaw_removal,
    worst_removal,
};
pub(crate) use tabu::assert_valid_tenure;
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
