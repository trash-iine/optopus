//! Generic machinery shared by the binary-variable problems
//! (MaxCut / QUBO / MaxSAT / VertexCover).

mod crossover;
mod fix;
mod neighbor;

pub use crossover::{lift_binary_solution, lift_compact_binary_solution, uniform_binary_crossover};
pub(crate) use fix::fixed_by;
pub use fix::{
    BinaryFixing, BinaryRelaxation, binary_node, binary_ranges, binary_solution_with, binary_value,
    variable_slots,
};
pub use neighbor::{apply_swap_as_two_flips, differing_pairs, random_differing_pair};
