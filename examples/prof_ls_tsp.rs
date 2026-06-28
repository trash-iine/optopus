//! Profiling: LocalSearch × TSP × TwoOpt + Relocate (Tier 1)
//!
//! Hot paths (TwoOpt):
//!   - O(n²) iter: lookup into `sol.gain: HashMap<(Edge, Edge), f64>`
//!     keyed by normalize_edge_pair ((Edge, Edge))
//!   - apply_to_solution: update_gains_for_removed/added_edge is
//!     linear in segment length (j - i + 1)
//!
//! Hot paths (Relocate):
//!   - iter() eagerly collects the O(n²) enumeration into a Vec (no Arc)
//!   - For each (pos, ins) pair, prob.distance() invokes f64::sqrt 6 times
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
