//! Profiling: SimulatedAnnealing × MaxCut × Flip + Swap (Tier 1)
//!
//! Hot paths (Flip):
//!   - iter().choose() consumes O(n) via reservoir sampling
//!   - boltzmann_accept: f64::exp + rng.random() (only on worsening moves)
//!   - MaxCutSolution.gain: indexed access into Vec<f32>
//!
//! Hot paths (Swap):
//!   - iter() collects all (i, j) pairs into a Vec → O(n²) alloc
//!   - choose() then consumes O(n²) — compare scaling against Flip
//!
//! How to run:
//! ```
//! cargo build --profile profiling --example prof_sa_maxcut
//! samply record ./target/profiling/examples/prof_sa_maxcut
//! ```

use std::time::Duration;

use optopus::prelude::*;

fn main() {
    let mc = MaxCut::new(Graph::load_from_file("data/instances/max_cut/G22").unwrap());

    // --- SA × Flip: O(n) iter().choose() ---
    let mut state = SearchState::new(&mc);
    SimulatedAnnealing::<MaxCutFlipNeighbor>::new(
        StopCondition::duration(Duration::from_secs(3)),
        100.0,
        0.9999,
    )
    .run(&mut state)
    .unwrap();

    // --- SA × Swap: O(n²) iter().choose() ---
    let mut state2 = SearchState::new(&mc);
    SimulatedAnnealing::<MaxCutSwapNeighbor>::new(
        StopCondition::duration(Duration::from_secs(3)),
        100.0,
        0.9999,
    )
    .run(&mut state2)
    .unwrap();
}
