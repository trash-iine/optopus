//! Profiling: TabuSearch × MaxCut × Flip (Tier 1)
//!
//! Hot paths:
//!   - filter_best() reallocates Vec via vec![r] every call
//!   - is_move_enabled() does TabuMap (HashMap<usize, u64>)::get × n
//!   - add_to_tabu_map() HashMap::insert
//!   - MaxCutSolution.gain: full scan over Vec<f32> for all vertices
//!
//! How to run:
//! ```
//! cargo build --profile profiling --example prof_ts_maxcut_flip
//! samply record ./target/profiling/examples/prof_ts_maxcut_flip
//! ```

use std::time::Duration;

use optopus::prelude::*;

fn main() {
    let mc = MaxCut::new(Graph::load_from_file("data/instances/max_cut/G22").unwrap());
    let mut state = SearchState::new(&mc);
    TabuSearch::<MaxCutFlipNeighbor>::new(
        StopCondition::duration(Duration::from_secs(5)),
        (3, 10),
        None,
    )
    .run(&mut state)
    .unwrap();
}
