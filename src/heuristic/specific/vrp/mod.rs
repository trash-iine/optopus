//! VRP-specific heuristics.
//!
//! [`ops`] holds what both heuristics work on: the route arithmetic every move
//! is priced with, the granular candidate lists, and the descent that turns a
//! route partition into a local optimum. [`alns`] and [`hgs`] are the two
//! searches built on top of it — one ruins and recreates, the other recombines
//! giant tours — and they differ in what they do *around* the descent, not in
//! how a route is measured.

mod alns;
mod hgs;
mod ops;

pub use alns::AdaptiveLargeNeighborhoodSearch;
pub use hgs::HybridGeneticSearch;
