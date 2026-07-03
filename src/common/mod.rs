//! Common data structures shared across problem types.

pub mod binary;
pub mod gain_index;
pub mod graph;
pub mod parse;
pub mod tabu;

pub use binary::{
    apply_swap_as_two_flips, hamming_distance, lift_binary_solution, lift_compact_binary_solution,
    uniform_binary_crossover,
};
pub use gain_index::GainIndex;
pub use graph::Graph;
pub use parse::InstanceLines;
pub use tabu::{VarTabuMap, add_var_to_tabu, is_var_enabled};
