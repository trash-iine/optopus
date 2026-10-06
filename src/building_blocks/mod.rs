//! The pieces problems and heuristics are built from, in three groups.
//!
//! - [`instance`] holds what a problem instance is built from and read with,
//!   such as [`Graph`] and [`DistanceStore`].
//! - [`representation`] holds the helpers shared by problems whose solutions
//!   are laid out the same way, binary vectors, bounded integers or
//!   permutations.
//! - [`search`] holds the search machinery that knows nothing about any
//!   problem, such as [`TabuMemory`] and [`BiasedFitnessPopulation`].
//!
//! Every item is also re-exported here, so `building_blocks::Graph` and
//! `building_blocks::instance::Graph` name the same type.

pub mod instance;
pub mod representation;
pub mod search;

pub use instance::*;
pub use representation::*;
pub use search::*;
