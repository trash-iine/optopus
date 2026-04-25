//! Profiling: Lin-Kernighan (Helsgaun) × TSP (Tier 1, LKH baseline)
//!
//! Hot paths:
//!   - ensure_candidates(): one-time O(n²) all-pairs distance computation +
//!     sort (candidate-list construction)
//!   - find_lk_move(): depth-bounded DFS — walks succ/pred along the tour
//!     to search for improving edge-exchange sequences
//!   - prob.distance(): f64::sqrt (recomputed from coordinates each call)
//!   - apply_to_solution(): partial tour reversal + incremental update of the gain HashMap
//!
//! LKH self-terminates with `no_improvement = true` at a local optimum,
//! so wrap in Restart to run for a fixed time (berlin52 converges very fast).
//!
//! How to run:
//! ```
//! cargo build --profile profiling --example prof_lkh_tsp
//! samply record ./target/profiling/examples/prof_lkh_tsp
//! ```

use std::time::Duration;

use optopus::prelude::*;

fn main() {
    let tsp = TspWithCoordinates::load_file("data/tsp/berlin52.tsp").unwrap();
    let mut state = SearchState::new(&tsp);

    Restart::new(
        StopCondition::duration(Duration::from_secs(5)),
        Box::new(LinKernighanHelsgottForTsp::new(
            StopCondition::iterations(u64::MAX),
            5,
            5,
        )),
        StopCondition::failed_updates(1),
    )
    .run(&mut state)
    .unwrap();
}
