//! Common data structures shared across problem types.

pub mod binary;
pub mod gain_index;
pub mod graph;
pub mod tabu;

pub use binary::uniform_binary_crossover;
pub use gain_index::GainIndex;
pub use graph::Graph;
pub use tabu::{VarTabuMap, add_var_to_tabu, is_var_enabled};
