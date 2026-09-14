//! VRP-specific heuristics.
//!
//! [`hgs`] is what is left here. Ruin-and-recreate turned out to need nothing
//! of VRP beyond what [`Ruinable`](crate::trait_defs::Ruinable) asks any
//! problem, so it generalized into [`AdaptiveLargeNeighborhoodSearch`](crate::heuristic::AdaptiveLargeNeighborhoodSearch), and the
//! route machinery both searches read moved to
//! [`problem::vrp::ops`](crate::problem::vrp::ops) with the problem whose
//! routes it measures.
//!
//! What keeps Hybrid Genetic Search here is the giant-tour recombination, a
//! decode from a customer permutation into a route partition that no other
//! problem has.

mod hgs;

pub use hgs::HybridGeneticSearch;
