//! The pieces problems and heuristics are built from, in three groups.
//!
//! - [`instance`] holds what a problem instance is built from and read with,
//!   such as [`Graph`](instance::Graph) and
//!   [`DistanceStore`](instance::DistanceStore).
//! - [`representation`] holds the helpers shared by problems whose solutions
//!   are laid out the same way, binary vectors, bounded integers or
//!   permutations.
//! - [`search`] holds the search machinery that knows nothing about any
//!   problem, such as [`TabuMemory`](search::TabuMemory) and
//!   [`BiasedFitnessPopulation`](search::BiasedFitnessPopulation).
//!
//! Items are reached through their group, as in
//! `building_blocks::search::TabuMemory`, so a path says which layer a piece
//! serves.

pub mod instance;
pub mod representation;
pub mod search;
