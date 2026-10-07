//! Permutation Flow Shop Scheduling (PFSP), `Fm | prmu | Cmax`.
//!
//! `n_jobs` jobs pass through `n_machines` machines in the same order, and
//! every machine takes the jobs in one common sequence, the solution. The
//! makespan, when the last job leaves the last machine, is minimized. For three
//! machines or more it is strongly NP-hard.
//!
//! The moves and the [`Ruinable`](crate::trait_defs::Ruinable) implementation
//! all price edits with Taillard's heads and tails, so inserting a job at
//! every place in the sequence costs O(nm) together rather than O(nm) each.

mod crossover;
mod neighbor;
mod problem;
mod ruin;
mod taillard;

pub use crossover::FlowShopOrderCrossover;
pub use neighbor::{FlowShopInsertNeighbor, FlowShopSwapNeighbor};
pub use problem::{FlowShop, FlowShopSolution};
pub use ruin::{FlowShopInsertionDescent, FlowShopPartial};
