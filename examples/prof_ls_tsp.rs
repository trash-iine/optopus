//! Profiling: LocalSearch × TSP × TwoOpt + Relocate (Tier 1)
//!
//! Hot paths (TwoOpt):
//!   - O(n²) iter: 4 distance-matrix reads per candidate
//!     (calc_2opt_gain_cities)
//!   - apply_to_solution: tour segment reversal, linear in segment
//!     length (j - i + 1)
//!
//! Hot paths (Relocate):
//!   - O(n²) lazy iter: 6 distance-matrix reads per (pos, ins) pair
//!
//! Distances come from the lazily built distance matrix
//! (n ≤ DIST_MATRIX_MAX_N), so no sqrt appears in the loop.
//!
//! LocalSearch halts at a local optimum, so wrap in Restart to run for a fixed time.
//!
//! How to run:
//! ```
//! cargo build --profile profiling --example prof_ls_tsp
//! samply record ./target/profiling/examples/prof_ls_tsp
//! ```

use std::time::Duration;

use optopus::prelude::*;

fn main() {
    let tsp = TspWithCoordinates::load_file("data/instances/tsp/berlin52.tsp").unwrap();

    // --- TwoOpt ---
    let mut state = SearchState::new(&tsp);
    Restart::new(
        StopCondition::duration(Duration::from_secs(3)),
        Box::new(LocalSearch::<TspTwoOptNeighbor>::new(
            StopCondition::failed_updates(1),
        )),
        StopCondition::failed_updates(1),
    )
    .run(&mut state)
    .unwrap();

    // --- Relocate ---
    let mut state2 = SearchState::new(&tsp);
    Restart::new(
        StopCondition::duration(Duration::from_secs(3)),
        Box::new(LocalSearch::<TspRelocateNeighbor>::new(
            StopCondition::failed_updates(1),
        )),
        StopCondition::failed_updates(1),
    )
    .run(&mut state2)
    .unwrap();
}
