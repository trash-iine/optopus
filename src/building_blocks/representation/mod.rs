//! Helpers shared by the problems whose solutions are laid out the same way.
//!
//! One module per kind of decision variable. Binary vectors get crossover,
//! variable fixing and the paired moves, bounded integers get the ranges that
//! integer problems and branch and bound are stated in, and permutations get
//! order crossover.

mod binary;
mod integer;
mod permutation;

pub(crate) use binary::fixed_by;
pub use binary::{
    BinaryFixing, BinaryRelaxation, apply_swap_as_two_flips, binary_node, binary_ranges,
    binary_solution_with, binary_value, differing_pairs, hamming_distance, lift_binary_solution,
    lift_compact_binary_solution, random_differing_pair, uniform_binary_crossover, variable_slots,
};
pub use integer::{DomainRestriction, IntVar, IntVars};
pub use permutation::{order_crossover, random_distinct_pair};
