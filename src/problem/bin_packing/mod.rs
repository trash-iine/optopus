//! One-dimensional Bin Packing.
//!
//! Items of given sizes go into bins of one capacity, none over it, and the
//! number of bins is minimized, with Falkenauer's fill term breaking the ties
//! that count leaves. It is the first [`Ruinable`](crate::trait_defs::Ruinable)
//! whose containers open and close as the search goes.

mod crossover;
mod neighbor;
mod problem;
mod ruin;

pub use crossover::BinPackingGroupCrossover;
pub use neighbor::{BinPackingRelocateNeighbor, BinPackingSwapNeighbor};
pub use problem::{BinPacking, BinPackingSolution};
pub use ruin::BinPackingPartial;
