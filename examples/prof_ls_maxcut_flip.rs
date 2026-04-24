//! Profiling: LocalSearch × MaxCut × Flip (Tier 1 baseline reference)
//!
//! Hot paths:
//!   - MaxCutFlipNeighbor::iter (Vec iteration over Graph::vertices)
//!   - MaxCutSolution.gain: read Vec<f32> → filter → max_by
//!   - apply_to_solution: refresh gain[] only for adjacent vertices (O(degree))
//!
//! LocalSearch halts at a local optimum, so wrap in Restart to run for
//! a fixed time and ensure sufficient sample density.
//!
//! How to run (samply):
//! ```
//! cargo build --profile profiling --example prof_ls_maxcut_flip
//! samply record ./target/profiling/examples/prof_ls_maxcut_flip
//! ```

use std::time::Duration;

use optopus::prelude::*;

fn main() {
    let mc = MaxCut::new(Graph::load_from_file("data/max_cut/G22").unwrap());
    let mut state = SearchState::new(&mc);

    Restart::new(
        StopCondition::duration(Duration::from_secs(5)),
        Box::new(LocalSearch::<MaxCutFlipNeighbor>::new(
            StopCondition::failed_updates(1),
        )),
        StopCondition::failed_updates(1),
    )
    .run(&mut state)
    .unwrap();
}
