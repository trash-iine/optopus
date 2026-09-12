//! VRP-specific heuristics.
//!
//! Both work on [`problem::vrp::ops`](crate::problem::vrp::ops), which holds
//! the route arithmetic every move is priced with, the granular candidate
//! lists, and the descent that turns a route partition into a local optimum.
//! [`alns`] and [`hgs`] are the two searches built on top of it, one ruins and
//! recreates, the other recombines giant tours, and they differ in what they
//! do around the descent, not in how a route is measured.

mod hgs;

pub use hgs::HybridGeneticSearch;
