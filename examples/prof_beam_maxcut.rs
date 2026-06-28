//! Profiling: BeamSearch × MaxCut × Flip (Tier 1)
//!
//! Hot paths:
//!   - beam_width × n MaxCutSolution::clone() calls per iteration
//!     (memcpy of Vec<bool> cut + Vec<f32> gain — linear in n)
//!   - apply_to_solution per candidate (O(degree))
//!   - beam pruning via select_nth_unstable_by
//!
//! For allocation analysis, cargo-instruments works well:
//! ```
//! cargo instruments --profile profiling --example prof_beam_maxcut -t Allocations
//! ```
//!
//! How to run (samply):
//! ```
//! cargo build --profile profiling --example prof_beam_maxcut
//! samply record ./target/profiling/examples/prof_beam_maxcut
//! ```

use std::time::Duration;

use optopus::prelude::*;

fn main() {
    // G22: n=2000 — beam_width × n = 5 × 2000 clones/iter maximizes allocation pressure
    let mc = MaxCut::new(Graph::load_from_file("data/instances/max_cut/G22").unwrap());
    let mut state = SearchState::new(&mc);
    BeamSearch::<MaxCut, MaxCutFlipNeighbor>::new(
        StopCondition::duration(Duration::from_secs(5)),
        5,
    )
    .run(&mut state)
    .unwrap();
}
