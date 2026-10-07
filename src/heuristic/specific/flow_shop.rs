//! Flow shop wiring for the generic searches over
//! [`Ruinable`](crate::trait_defs::Ruinable).
//!
//! Nothing here searches. Iterated Greedy and ALNS run on the flow shop like on
//! any other `Ruinable`, and what stays behind is what the problem has to hand
//! them, its insertion descent and, for Iterated Greedy, a temperature in
//! makespan units.

use crate::heuristic::{AdaptiveLargeNeighborhoodSearch, IteratedGreedy, StopCondition};
use crate::problem::FlowShop;
use crate::problem::flow_shop::FlowShopInsertionDescent;

/// Iterated Greedy as Ruiz and Stützle run it on the flow shop.
///
/// The temperature is their `T · Σp / (n · m · 10)`, a tenth of the mean
/// processing time scaled by `temperature_factor`, and the local search is the
/// full insertion descent. Their tuned values are `removal_count = 4` and
/// `temperature_factor = 0.4`.
///
/// # Panics
///
/// Panics if `removal_count` is zero or `temperature_factor` is not positive.
pub fn iterated_greedy_for_flow_shop(
    stop_condition: StopCondition,
    removal_count: usize,
    temperature_factor: f64,
) -> IteratedGreedy<FlowShop> {
    assert!(
        temperature_factor > 0.0,
        "temperature_factor must be positive"
    );
    let temperature_of = move |prob: &FlowShop| {
        let operations = (prob.n_jobs * prob.n_machines).max(1) as f64;
        temperature_factor * prob.total_processing_time() as f64 / (operations * 10.0)
    };
    IteratedGreedy::new(stop_condition, removal_count, temperature_of)
        .with_local_repair(Box::new(FlowShopInsertionDescent::new()))
}

/// Ruin-and-recreate over a flow shop sequence, with the insertion descent.
///
/// The same shape as `alns_for_tsp`.
///
/// # Panics
///
/// Panics if `removal_fraction` or `cooling_rate` is outside `(0, 1]`, which
/// [`AdaptiveLargeNeighborhoodSearch::new`] checks.
pub fn alns_for_flow_shop(
    stop_condition: StopCondition,
    removal_fraction: f64,
    cooling_rate: f64,
) -> AdaptiveLargeNeighborhoodSearch<FlowShop> {
    AdaptiveLargeNeighborhoodSearch::<FlowShop>::new(stop_condition, removal_fraction, cooling_rate)
        .with_local_repair(Box::new(FlowShopInsertionDescent::new()))
}
