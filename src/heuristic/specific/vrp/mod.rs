//! VRP-specific heuristics.
//!
//! [`hgs`] is the one search left here. Ruin-and-recreate turned out to need
//! nothing of VRP beyond what [`Ruinable`](crate::trait_defs::Ruinable) asks
//! any problem, so it generalized into
//! [`AdaptiveLargeNeighborhoodSearch`](crate::heuristic::AdaptiveLargeNeighborhoodSearch),
//! and the route machinery both searches read moved to
//! [`problem::vrp::ops`](crate::problem::vrp::ops) with the problem whose
//! routes it measures. What stayed behind is [`alns_for_vrp`], the wiring that
//! hands that generic search the route descent it needs, the same shape
//! `breakout_local_search_for_max_cut` has on the MaxCut side.
//!
//! What keeps Hybrid Genetic Search here is the giant-tour recombination, a
//! decode from a customer permutation into a route partition that no other
//! problem has.

mod hgs;

pub use hgs::HybridGeneticSearch;

use crate::heuristic::{AdaptiveLargeNeighborhoodSearch, StopCondition};
use crate::problem::Vrp;
use crate::problem::vrp::AnchoredRouteDescent;

/// Ruin-and-recreate over CVRP, with the anchored descent that makes it pay.
///
/// [`AdaptiveLargeNeighborhoodSearch`] takes its local repair as an option,
/// because a problem with no anchored local search still runs plain
/// ruin-and-recreate. On routes it is not optional in practice. Greedy
/// re-insertion puts each customer where it is cheapest at the time and never
/// repairs the edges that choice spoils, and the gap widens with the instance,
/// from nothing around 350 customers to 8% at 1000. Naming the pair here keeps
/// the good search the one a caller gets by default, rather than the one they
/// get if they remember
/// [`with_local_repair`](AdaptiveLargeNeighborhoodSearch::with_local_repair).
///
/// The tuning parameters stay on the builders. This is the wiring, not a
/// second place to set them, so a caller who needs one of them writes the pair
/// out and chains it:
///
/// ```rust,ignore
/// AdaptiveLargeNeighborhoodSearch::<Vrp>::new(stop_condition, 0.15, 0.9995)
///     .with_local_repair(Box::new(AnchoredRouteDescent::new()))
///     .with_max_removal(20)
/// ```
///
/// # Panics
///
/// Panics if `removal_fraction` or `cooling_rate` is outside `(0, 1]`, which
/// [`AdaptiveLargeNeighborhoodSearch::new`] checks.
pub fn alns_for_vrp(
    stop_condition: StopCondition,
    removal_fraction: f64,
    cooling_rate: f64,
) -> AdaptiveLargeNeighborhoodSearch<Vrp> {
    AdaptiveLargeNeighborhoodSearch::<Vrp>::new(stop_condition, removal_fraction, cooling_rate)
        .with_local_repair(Box::new(AnchoredRouteDescent::new()))
}
