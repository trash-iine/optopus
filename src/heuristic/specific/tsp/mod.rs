//! TSP-specific heuristics.
//!
//! [`lkh`] is the one search here. Ruin-and-recreate runs on a tour through
//! [`Ruinable`](crate::trait_defs::Ruinable) like on any other problem, so what
//! stays behind is [`alns_for_tsp`], the wiring that hands the generic search
//! the anchored tour descent, the same shape `alns_for_vrp` has on the VRP
//! side.

mod lkh;

pub use lkh::LinKernighanHelsgaun;

use crate::heuristic::{AdaptiveLargeNeighborhoodSearch, StopCondition};
use crate::problem::TspWithCoordinates;
use crate::problem::tsp_2d::AnchoredTourDescent;

/// Ruin-and-recreate over a tour, with the anchored Or-opt and 2-opt descent.
///
/// [`AdaptiveLargeNeighborhoodSearch`] takes its local repair as an option.
/// This names the pair so the search a caller gets by default is the one with
/// the descent, rather than the one they get if they remember
/// [`with_local_repair`](AdaptiveLargeNeighborhoodSearch::with_local_repair).
/// The tuning parameters stay on the builders, as `alns_for_vrp` explains.
///
/// # Panics
///
/// Panics if `removal_fraction` or `cooling_rate` is outside `(0, 1]`, which
/// [`AdaptiveLargeNeighborhoodSearch::new`] checks.
pub fn alns_for_tsp(
    stop_condition: StopCondition,
    removal_fraction: f64,
    cooling_rate: f64,
) -> AdaptiveLargeNeighborhoodSearch<TspWithCoordinates> {
    AdaptiveLargeNeighborhoodSearch::<TspWithCoordinates>::new(
        stop_condition,
        removal_fraction,
        cooling_rate,
    )
    .with_local_repair(Box::new(AnchoredTourDescent::new()))
}
